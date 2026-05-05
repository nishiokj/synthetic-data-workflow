from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import OpenAIClient, PlanAuditor, ProviderError, QualityGate, RubricGate, SampleGenerator, Strategist
from config import RuntimeConfig
from models import (
    AgentRole,
    CandidateSample,
    CertifiedSample,
    CheckResult,
    ContextPolicy,
    PlanVerdict,
    RouteCode,
    RoutingDecision,
    SampleVerdict,
    SeedSpec,
    StageKind,
    StageRecord,
    Verdict,
    stable_hash,
)
from observability import StageLogWriter, emit_event
from router import route_after
from rules import deterministic_sample_verdict, validate_seed_plan
from services.corpus_index import CorpusIndex
from services.coverage_ledger import CoverageLedger
from services.rejection_archive import RejectionArchive
from services.validation_ledger import ValidationLedger


class PipelineState(TypedDict, total=False):
    run_id: str
    target_n: int
    max_plan_retries: int
    plan_round: int
    plan_retry_route_code: RouteCode | None
    plan_retry_subcodes: list[str]
    seeds_queue: list[SeedSpec]
    seed: SeedSpec | None
    gen_attempt: int
    gen_retry_route_code: RouteCode | None
    gen_retry_subcodes: list[str]
    candidate: CandidateSample | None
    det_checks: list[CheckResult]
    det_accepted: bool
    quality_verdict: SampleVerdict | None
    rubric_verdict: SampleVerdict | None
    last_decision: RoutingDecision | None
    committed_count: int
    dropped_count: int


def route_from_decision(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision is None or decision.terminal:
        return END
    return {
        StageKind.STRATEGY: "strategy",
        StageKind.PLAN_AUDIT: "audit_seed_plan",
        StageKind.GENERATION: "generate",
        StageKind.VALIDATION: "validate_det",
        StageKind.CURATION: "curate",
    }[decision.next_stage]


def after_validate_seed_plan_det(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision and decision.verdict == Verdict.ACCEPT:
        return "select_next_seed"
    return route_from_decision(state)


def after_curate(state: PipelineState) -> str:
    if state["committed_count"] >= state["target_n"]:
        return END
    if state["seeds_queue"]:
        return "select_next_seed"
    if state["plan_round"] > state["max_plan_retries"]:
        return END
    return "strategy"


def after_terminal_seed(state: PipelineState) -> str:
    if state["committed_count"] >= state["target_n"]:
        return END
    if state["seeds_queue"]:
        return "select_next_seed"
    if state["plan_round"] > state["max_plan_retries"]:
        return END
    return "strategy"


def after_select_next_seed(state: PipelineState) -> str:
    return "audit_seed_plan" if state.get("seed") else "strategy"


def after_audit_seed_plan(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision and decision.verdict == Verdict.ACCEPT:
        return "generate"
    if state["seeds_queue"]:
        return "select_next_seed"
    if state["plan_round"] > state["max_plan_retries"]:
        return END
    return "strategy"


def after_validate_det(state: PipelineState) -> str | list[str]:
    if state["det_accepted"]:
        return ["quality_gate", "rubric_gate"]
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_seed(state)
    return route_from_decision(state)


def after_generate(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_seed(state)
    return route_from_decision(state)


def after_gate_join(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_seed(state)
    return route_from_decision(state)


class PipelineRunner:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.client = OpenAIClient(config.models)
        self.writer = StageLogWriter(config.logs_dir, config.run_id)
        self.coverage = CoverageLedger(config.data_dir, config.domain)
        self.validation_ledger = ValidationLedger(self.writer)
        self.rejections = RejectionArchive(self.writer)
        self.corpus = CorpusIndex(config.data_dir, config.domain, self.client, config.run_id)
        self.strategist = Strategist(self.client, config.domain)
        self.plan_auditor = PlanAuditor(self.client, config.domain)
        self.generator = SampleGenerator(self.client, config.domain)
        self.quality_gate = QualityGate(self.client, config.domain)
        self.rubric_gate = RubricGate(self.client, config.domain)
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(PipelineState)
        graph.add_node("strategy", self.node_strategy)
        graph.add_node("validate_seed_plan_det", self.node_validate_seed_plan_det)
        graph.add_node("select_next_seed", self.node_select_next_seed)
        graph.add_node("audit_seed_plan", self.node_audit_seed_plan)
        graph.add_node("generate", self.node_generate)
        graph.add_node("validate_det", self.node_validate_det)
        graph.add_node("quality_gate", self.node_quality_gate)
        graph.add_node("rubric_gate", self.node_rubric_gate)
        graph.add_node("join_gates", self.node_join_gates)
        graph.add_node("curate", self.node_curate)
        graph.add_edge(START, "strategy")
        graph.add_edge("strategy", "validate_seed_plan_det")
        graph.add_conditional_edges("validate_seed_plan_det", after_validate_seed_plan_det)
        graph.add_conditional_edges("select_next_seed", after_select_next_seed)
        graph.add_conditional_edges("audit_seed_plan", after_audit_seed_plan)
        graph.add_conditional_edges("generate", after_generate)
        graph.add_conditional_edges("validate_det", after_validate_det)
        graph.add_edge(["quality_gate", "rubric_gate"], "join_gates")
        graph.add_conditional_edges("join_gates", after_gate_join)
        graph.add_conditional_edges("curate", after_curate)
        return graph

    def run(self) -> dict[str, Any]:
        self._progress(
            "run",
            "start",
            target=self.config.target_n,
            domain=self.config.domain.domain_id,
            model=self.config.models.model,
            graph_limit=_graph_recursion_limit(self.config),
        )
        initial: PipelineState = {
            "run_id": self.config.run_id,
            "target_n": self.config.target_n,
            "max_plan_retries": self.config.domain.max_plan_retries,
            "plan_round": 0,
            "plan_retry_subcodes": [],
            "seeds_queue": [],
            "gen_attempt": 0,
            "gen_retry_subcodes": [],
            "det_checks": [],
            "det_accepted": False,
            "committed_count": 0,
            "dropped_count": 0,
        }
        final = self.graph.invoke(initial, config={"recursion_limit": _graph_recursion_limit(self.config)})
        return {
            "run_id": self.config.run_id,
            "committed": final["committed_count"],
            "dropped": final["dropped_count"],
        }

    def node_strategy(self, state: PipelineState) -> PipelineState:
        plan_round = state["plan_round"] + 1
        count = max(1, self.config.target_n * 2)
        self._progress(
            "strategy",
            "start",
            round=plan_round,
            requested_seeds=count,
            retry=state.get("plan_retry_route_code"),
        )
        seeds, meta = self.strategist.plan(
            run_id=f"{self.config.run_id}-r{plan_round}",
            target_n=count,
            coverage_snapshot=self.coverage.snapshot(),
            retry_route_code=state.get("plan_retry_route_code"),
            retry_subcodes=state.get("plan_retry_subcodes"),
        )
        verdict = Verdict.ACCEPT if seeds else Verdict.REJECT
        route_code = RouteCode.ACCEPT if seeds else RouteCode.RETRY_PROVIDER_EMPTY
        self._record(
            stage_kind=StageKind.STRATEGY,
            role="plan_strategy_batch",
            agent_role=AgentRole.STRATEGIST,
            artifact_id=f"{self.config.run_id}-strategy-{plan_round}",
            parent_artifact_id=None,
            verdict=verdict,
            route_code=route_code,
            context_policy=_producer_context_policy(state.get("plan_retry_route_code")),
            meta=meta,
            retry_index=plan_round - 1,
        )
        return {
            "plan_round": plan_round,
            "seeds_queue": seeds,
            "seed": None,
            "plan_retry_route_code": None,
            "plan_retry_subcodes": [],
            "last_decision": None,
        }

    def node_validate_seed_plan_det(self, state: PipelineState) -> PipelineState:
        seeds = state["seeds_queue"]
        self._progress("plan_det", "start", round=state["plan_round"], seeds=len(seeds))
        verdict, route_code, subcodes = validate_seed_plan(seeds, self.config.domain)
        retry_index = state["plan_round"] - 1
        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.PLAN_AUDIT,
            verdict=verdict,
            route_code=route_code,
            retry_index=retry_index,
            max_plan_retries=self.config.domain.max_plan_retries,
            subcodes=subcodes,
        )
        self._record(
            stage_kind=StageKind.PLAN_AUDIT,
            role="validate_seed_plan_deterministically",
            agent_role=None,
            artifact_id=f"{self.config.run_id}-seed-plan-{state['plan_round']}-deterministic-verdict",
            parent_artifact_id=f"{self.config.run_id}-strategy-{state['plan_round']}",
            verdict=verdict,
            route_code=decision.route_code,
            subcodes=subcodes,
            reason_codes=subcodes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=_local_meta(),
            retry_index=retry_index,
        )
        update: PipelineState = {"last_decision": decision}
        if verdict == Verdict.REJECT:
            self.rejections.append({"seeds": [seed.model_dump(mode="json") for seed in seeds]}, decision)
            update["plan_retry_route_code"] = decision.route_code
            update["plan_retry_subcodes"] = decision.subcodes
        return update

    def node_select_next_seed(self, state: PipelineState) -> PipelineState:
        seeds = list(state["seeds_queue"])
        seed = seeds.pop(0) if seeds else None
        if seed:
            self._progress(
                "seed",
                "select",
                id=seed.id,
                case_type=seed.cell.case_type,
                difficulty=seed.cell.difficulty,
                scenario=seed.cell.scenario,
                remaining=len(seeds),
            )
        return {
            "seeds_queue": seeds,
            "seed": seed,
            "gen_attempt": 0,
            "gen_retry_route_code": None,
            "gen_retry_subcodes": [],
            "candidate": None,
            "det_checks": [],
            "det_accepted": False,
            "quality_verdict": None,
            "rubric_verdict": None,
            "last_decision": None,
        }

    def node_audit_seed_plan(self, state: PipelineState) -> PipelineState:
        seed = _require(state.get("seed"), "seed")
        self._progress("plan_audit", "start", seed=seed.id, case_type=seed.cell.case_type)
        verdict, route_code, subcodes = validate_seed_plan([seed], self.config.domain)
        if verdict == Verdict.REJECT:
            plan_verdict = _local_plan_verdict(seed, route_code, subcodes)
            meta = _local_meta()
        else:
            plan_verdict, meta = self.plan_auditor.audit(seed)
        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.PLAN_AUDIT,
            verdict=plan_verdict.verdict,
            route_code=plan_verdict.route_code,
            retry_index=0,
            max_plan_retries=self.config.domain.max_plan_retries,
            subcodes=plan_verdict.subcodes,
            reason_codes=plan_verdict.reason_codes,
        )
        self._record(
            stage_kind=StageKind.PLAN_AUDIT,
            role="audit_seed_plan",
            agent_role=AgentRole.PLAN_AUDITOR if meta["provider"] != "local" else None,
            artifact_id=f"{seed.id}-plan-verdict",
            parent_artifact_id=seed.id,
            verdict=plan_verdict.verdict,
            route_code=plan_verdict.route_code,
            subcodes=plan_verdict.subcodes,
            reason_codes=plan_verdict.reason_codes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=meta,
        )
        update: PipelineState = {"last_decision": decision}
        if plan_verdict.verdict == Verdict.REJECT:
            self.rejections.append(seed, decision)
            update["dropped_count"] = state["dropped_count"] + 1
        return update

    def node_generate(self, state: PipelineState) -> PipelineState:
        seed = _require(state.get("seed"), "seed")
        retry_index = state["gen_attempt"]
        self._progress(
            "generation",
            "start",
            seed=seed.id,
            attempt=retry_index + 1,
            retry=state.get("gen_retry_route_code"),
        )
        try:
            candidate, gen_meta = self.generator.generate(
                run_id=self.config.run_id,
                seed=seed,
                attempt=retry_index + 1,
                retry_route_code=state.get("gen_retry_route_code"),
                retry_subcodes=state.get("gen_retry_subcodes"),
            )
        except ProviderError as exc:
            route_code = RouteCode.RETRY_PARSE if "invalid JSON" in str(exc) else RouteCode.RETRY_INFRA
            decision = route_after(
                run_id=self.config.run_id,
                from_stage=StageKind.GENERATION,
                verdict=Verdict.REJECT,
                route_code=route_code,
                retry_index=retry_index,
                max_generation_retries=self.config.domain.max_generation_retries,
                subcodes=["provider_error"],
                reason_codes=["provider_error"],
            )
            self._record(
                stage_kind=StageKind.GENERATION,
                role="generate_candidate_sample",
                agent_role=AgentRole.SAMPLE_GENERATOR,
                artifact_id=f"{seed.id}-generation-error-{retry_index}",
                parent_artifact_id=seed.id,
                verdict=Verdict.REJECT,
                route_code=decision.route_code,
                subcodes=["provider_error"],
                reason_codes=["provider_error"],
                context_policy=_producer_context_policy(state.get("gen_retry_route_code")),
                meta=_local_meta(error=str(exc)),
                retry_index=retry_index,
            )
            update: PipelineState = {"last_decision": decision}
            if decision.terminal:
                update["dropped_count"] = state["dropped_count"] + 1
            else:
                update["gen_attempt"] = retry_index + 1
                update["gen_retry_route_code"] = decision.route_code
                update["gen_retry_subcodes"] = decision.subcodes
            return update

        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.GENERATION,
            verdict=Verdict.ACCEPT,
            route_code=RouteCode.ACCEPT,
            retry_index=retry_index,
            max_generation_retries=self.config.domain.max_generation_retries,
        )
        self._record(
            stage_kind=StageKind.GENERATION,
            role="generate_candidate_sample",
            agent_role=AgentRole.SAMPLE_GENERATOR,
            artifact_id=candidate.id,
            parent_artifact_id=seed.id,
            verdict=Verdict.ACCEPT,
            route_code=RouteCode.ACCEPT,
            context_policy=_producer_context_policy(state.get("gen_retry_route_code")),
            meta=gen_meta,
            retry_index=retry_index,
        )
        self._progress("candidate", "generated", **_candidate_progress(candidate))
        return {
            "candidate": candidate,
            "det_accepted": False,
            "quality_verdict": None,
            "rubric_verdict": None,
            "last_decision": decision,
            "gen_retry_route_code": None,
            "gen_retry_subcodes": [],
        }

    def node_validate_det(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        self._progress("validation_det", "start", candidate=candidate.id)
        det_verdict, checks = deterministic_sample_verdict(candidate, self.config.domain)
        self.validation_ledger.append(det_verdict)
        self._record(
            stage_kind=StageKind.VALIDATION,
            role="validate_candidate_deterministically",
            agent_role=None,
            artifact_id=f"{candidate.id}-deterministic-verdict",
            parent_artifact_id=candidate.id,
            verdict=det_verdict.verdict,
            route_code=det_verdict.route_code,
            subcodes=det_verdict.subcodes,
            reason_codes=det_verdict.reason_codes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=_local_meta(),
            retry_index=state["gen_attempt"],
        )
        if det_verdict.verdict == Verdict.ACCEPT:
            return {"det_checks": checks, "det_accepted": True, "last_decision": None}

        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.VALIDATION,
            verdict=det_verdict.verdict,
            route_code=det_verdict.route_code,
            retry_index=state["gen_attempt"],
            max_generation_retries=self.config.domain.max_generation_retries,
            subcodes=det_verdict.subcodes,
            reason_codes=det_verdict.reason_codes,
        )
        self.rejections.append(candidate, decision)
        self._progress(
            "candidate",
            "rejected",
            **_candidate_progress(candidate),
            route=decision.route_code,
            codes=decision.reason_codes or decision.subcodes,
        )
        update: PipelineState = {"det_checks": checks, "det_accepted": False, "last_decision": decision}
        if decision.terminal:
            update["dropped_count"] = state["dropped_count"] + 1
        else:
            update["gen_attempt"] = state["gen_attempt"] + 1
            update["gen_retry_route_code"] = decision.route_code
            update["gen_retry_subcodes"] = decision.subcodes
        return update

    def node_quality_gate(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        self._progress("quality_gate", "start", candidate=candidate.id)
        quality_verdict, quality_meta = self.quality_gate.validate(candidate)
        self.validation_ledger.append(quality_verdict)
        self._record(
            stage_kind=StageKind.VALIDATION,
            role="quality_gate_candidate",
            agent_role=AgentRole.QUALITY_GATE,
            artifact_id=f"{candidate.id}-quality-verdict",
            parent_artifact_id=candidate.id,
            verdict=quality_verdict.verdict,
            route_code=quality_verdict.route_code,
            subcodes=quality_verdict.subcodes,
            reason_codes=quality_verdict.reason_codes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=quality_meta,
            retry_index=state["gen_attempt"],
        )
        return {"quality_verdict": quality_verdict}

    def node_rubric_gate(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        self._progress("rubric_gate", "start", candidate=candidate.id)
        rubric_verdict, rubric_meta = self.rubric_gate.validate(candidate)
        self.validation_ledger.append(rubric_verdict)
        self._record(
            stage_kind=StageKind.VALIDATION,
            role="rubric_gate_candidate",
            agent_role=AgentRole.RUBRIC_GATE,
            artifact_id=f"{candidate.id}-rubric-verdict",
            parent_artifact_id=candidate.id,
            verdict=rubric_verdict.verdict,
            route_code=rubric_verdict.route_code,
            subcodes=rubric_verdict.subcodes,
            reason_codes=rubric_verdict.reason_codes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=rubric_meta,
            retry_index=state["gen_attempt"],
        )
        return {"rubric_verdict": rubric_verdict}

    def node_join_gates(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        quality_verdict = _require(state.get("quality_verdict"), "quality_verdict")
        rubric_verdict = _require(state.get("rubric_verdict"), "rubric_verdict")
        verdict = quality_verdict if quality_verdict.verdict == Verdict.REJECT else rubric_verdict
        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.VALIDATION,
            verdict=verdict.verdict,
            route_code=verdict.route_code,
            retry_index=state["gen_attempt"],
            max_generation_retries=self.config.domain.max_generation_retries,
            subcodes=verdict.subcodes,
            reason_codes=verdict.reason_codes,
        )
        update: PipelineState = {"last_decision": decision}
        if verdict.verdict == Verdict.REJECT:
            self.rejections.append(candidate, decision)
            self._progress(
                "candidate",
                "rejected",
                **_candidate_progress(candidate),
                route=decision.route_code,
                codes=decision.reason_codes or decision.subcodes,
            )
            if decision.terminal:
                update["dropped_count"] = state["dropped_count"] + 1
            else:
                update["gen_attempt"] = state["gen_attempt"] + 1
                update["gen_retry_route_code"] = decision.route_code
                update["gen_retry_subcodes"] = decision.subcodes
        return update

    def node_curate(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        quality_verdict = _require(state.get("quality_verdict"), "quality_verdict")
        rubric_verdict = _require(state.get("rubric_verdict"), "rubric_verdict")
        self._progress("curation", "start", candidate=candidate.id)
        certified = CertifiedSample(
            id=f"{candidate.id}-certified",
            candidate_id=candidate.id,
            content_hash=stable_hash(candidate.model_dump(mode="json")),
            candidate=candidate,
            deterministic_checks=state["det_checks"],
            semantic_checks=[quality_verdict, rubric_verdict],
        )
        committed, cur_verdict, cur_meta = self.corpus.curate(
            certified_id=certified.id,
            candidate=candidate,
            deterministic_checks=certified.deterministic_checks,
            semantic_checks=certified.semantic_checks,
            run_id=self.config.run_id,
        )
        self.validation_ledger.append(cur_verdict)
        self._record(
            stage_kind=StageKind.CURATION,
            role="curate_committed_sample",
            agent_role=None,
            artifact_id=committed.id if committed else f"{candidate.id}-curation-reject",
            parent_artifact_id=certified.id,
            verdict=cur_verdict.verdict,
            route_code=cur_verdict.route_code,
            subcodes=cur_verdict.subcodes,
            reason_codes=cur_verdict.reason_codes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=cur_meta,
            retry_index=state["gen_attempt"],
        )
        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.CURATION,
            verdict=cur_verdict.verdict,
            route_code=cur_verdict.route_code,
            retry_index=state["gen_attempt"],
            subcodes=cur_verdict.subcodes,
            reason_codes=cur_verdict.reason_codes,
        )
        if committed:
            self.coverage.increment(candidate.cell)
            self._progress(
                "candidate",
                "committed",
                **_candidate_progress(candidate),
                route=decision.route_code,
                codes=cur_verdict.reason_codes or cur_verdict.subcodes,
            )
            return {
                "committed_count": state["committed_count"] + 1,
                "last_decision": decision,
                "candidate": None,
                "quality_verdict": None,
                "rubric_verdict": None,
                "det_checks": [],
            }

        self.rejections.append(candidate, decision)
        self._progress(
            "candidate",
            "rejected",
            **_candidate_progress(candidate),
            route=decision.route_code,
            codes=decision.reason_codes or decision.subcodes,
        )
        return {
            "dropped_count": state["dropped_count"] + 1,
            "last_decision": decision,
            "candidate": None,
            "quality_verdict": None,
            "rubric_verdict": None,
            "det_checks": [],
        }

    def _record(
        self,
        *,
        stage_kind: StageKind,
        role: str,
        agent_role: AgentRole | None,
        artifact_id: str,
        parent_artifact_id: str | None,
        verdict: Verdict,
        route_code: RouteCode,
        context_policy: ContextPolicy,
        meta: dict[str, Any],
        subcodes: list[str] | None = None,
        reason_codes: list[str] | None = None,
        retry_index: int = 0,
    ) -> None:
        record = StageRecord(
            run_id=self.config.run_id,
            stage_id=f"{stage_kind.value}:{artifact_id}",
            role=role,
            stage_kind=stage_kind,
            agent_role=agent_role,
            parent_artifact_id=parent_artifact_id,
            artifact_id=artifact_id,
            model=str(meta.get("model", "none")),
            provider=str(meta.get("provider", "local")),
            prompt_hash=str(meta.get("prompt_hash", "")),
            input_tokens=int(meta.get("input_tokens", 0)),
            output_tokens=int(meta.get("output_tokens", 0)),
            latency_ms=int(meta.get("latency_ms", 0)),
            cost_usd=float(meta.get("cost_usd", 0.0)),
            reasoning_effort=None if meta.get("reasoning_effort") is None else str(meta.get("reasoning_effort")),
            text_normalization_replacements=int(meta.get("text_normalization_replacements", 0)),
            verdict=verdict,
            route_code=route_code,
            subcodes=subcodes or [],
            reason_codes=reason_codes or [],
            criteria_hash=stable_hash(self.config.domain.model_dump(mode="json")),
            context_policy=context_policy,
            retry_index=retry_index,
        )
        self.writer.write_stage_record(record)
        self._progress_record(record)

    def _progress_record(self, record: StageRecord) -> None:
        self._progress(
            _stage_label(record),
            "result",
            verdict=record.verdict,
            route=record.route_code,
            subcodes=record.subcodes,
            attempt=record.retry_index + 1,
            model=record.model,
            latency=f"{record.latency_ms}ms",
            tokens=f"{record.input_tokens}/{record.output_tokens}",
            artifact=_short_id(record.artifact_id),
        )

    def _progress(self, stage: str, event: str, **fields: Any) -> None:
        if not self.config.console_progress:
            return
        if emit_event("stage_progress", {"stage": stage, "event": event, **fields}):
            return
        parts = [f"[{self.config.run_id}]", event, stage]
        for key, value in fields.items():
            formatted = _format_progress_value(value)
            if formatted:
                parts.append(f"{key}={formatted}")
        print(" ".join(parts), flush=True)


def _producer_context_policy(retry_route_code: RouteCode | None) -> ContextPolicy:
    if retry_route_code is None:
        return ContextPolicy.FRESH
    if retry_route_code in {RouteCode.RETRY_INFRA, RouteCode.RETRY_PARSE, RouteCode.RETRY_PROVIDER_EMPTY}:
        return ContextPolicy.SAME_INPUT_RETRY
    return ContextPolicy.CRITERIA_PLUS_ROUTE_CODE


def _graph_recursion_limit(config: RuntimeConfig) -> int:
    plan_rounds = config.domain.max_plan_retries + 1
    seeds_per_round = max(1, config.target_n * 2)
    generation_attempts = config.domain.max_generation_retries + 1
    per_seed_steps = 2 + (generation_attempts * 4) + 1
    plan_steps = 2
    return 10 + plan_rounds * (plan_steps + seeds_per_round * per_seed_steps)


def _stage_label(record: StageRecord) -> str:
    if record.role == "validate_seed_plan_deterministically":
        return "plan_det"
    if record.role == "audit_seed_plan":
        return "plan_audit"
    if record.role == "generate_candidate_sample":
        return "generation"
    if record.role == "validate_candidate_deterministically":
        return "validation_det"
    if record.role == "quality_gate_candidate":
        return "quality_gate"
    if record.role == "rubric_gate_candidate":
        return "rubric_gate"
    if record.role == "curate_committed_sample":
        return "curation"
    return record.stage_kind.value


def _format_progress_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(_format_progress_value(item) for item in value if item is not None) or "-"
    if hasattr(value, "value"):
        return str(value.value)
    text = str(value)
    if not text:
        return ""
    return text.replace(" ", "_")


def _short_id(value: str) -> str:
    if len(value) <= 48:
        return value
    return f"{value[:22]}...{value[-22:]}"


def _candidate_progress(candidate: CandidateSample) -> dict[str, Any]:
    ability = candidate.ability_z.get("name") if isinstance(candidate.ability_z, dict) else None
    prompt = candidate.benchmark_case.get("prompt") if isinstance(candidate.benchmark_case, dict) else None
    return {
        "id": candidate.id,
        "case_type": candidate.case_type,
        "ability": ability,
        "prompt": prompt,
        "proxy": candidate.proxy_claim,
    }


def _local_plan_verdict(seed: SeedSpec, route_code: RouteCode, subcodes: list[str]) -> PlanVerdict:
    return PlanVerdict(
        seed_id=seed.id,
        verdict=Verdict.REJECT,
        route_code=route_code,
        subcodes=subcodes,
        reason_codes=subcodes,
    )


def _require(value: Any | None, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"pipeline state missing {name}")
    return value


def _local_meta(error: str | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "provider": "local",
        "model": "deterministic",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "cost_usd": 0.0,
        "prompt_hash": stable_hash({"local": True}),
    }
    if error:
        meta["error"] = error
    return meta
