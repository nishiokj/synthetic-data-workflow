from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import Adversary, ModelClient, DesignAuditor, ProviderError, QualityGate, RubricGate, SampleGenerator, Designer
from config import RuntimeConfig
from models import (
    AgentRole,
    AdversaryReport,
    CandidateSample,
    CertifiedSample,
    CheckResult,
    ContextPolicy,
    DesignVerdict,
    GenerationEnvelope,
    GenerationPipelineInput,
    GenerationPipelineResult,
    RouteCode,
    RoutingDecision,
    SampleVerdict,
    DesignBrief,
    StageKind,
    StageRecord,
    Verdict,
    stable_hash,
)
from observability import StageLogWriter, emit_event, trace_hash
from router import route_after
from rules import deterministic_sample_verdict, validate_design_batch
from services.corpus_index import CorpusIndex
from services.coverage_ledger import CoverageLedger
from services.rejection_archive import RejectionArchive
from services.validation_ledger import ValidationLedger
from services.workspace_export import WorkspaceExport


class PipelineState(TypedDict, total=False):
    run_id: str
    target_n: int
    max_design_retries: int
    design_round: int
    design_retry_route_code: RouteCode | None
    design_retry_subcodes: list[str]
    designs_queue: list[DesignBrief]
    design: DesignBrief | None
    generation_envelope: GenerationEnvelope | None
    gen_attempt: int
    gen_retry_route_code: RouteCode | None
    gen_retry_subcodes: list[str]
    candidate: CandidateSample | None
    det_checks: list[CheckResult]
    det_accepted: bool
    adversary_done: bool
    adversary_report: AdversaryReport | None
    quality_verdict: SampleVerdict | None
    rubric_verdict: SampleVerdict | None
    last_decision: RoutingDecision | None
    last_candidate_id: str | None
    committed_count: int
    dropped_count: int


def route_from_decision(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision is None or decision.terminal:
        return END
    return {
        StageKind.DESIGN: "design",
        StageKind.DESIGN_AUDIT: "audit_design",
        StageKind.GENERATION: "generate",
        StageKind.VALIDATION: "validate_det",
        StageKind.CURATION: "curate",
    }[decision.next_stage]


def after_validate_design_batch_det(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision and decision.verdict == Verdict.ACCEPT:
        return "select_next_design"
    return route_from_decision(state)


def after_curate(state: PipelineState) -> str:
    if state["committed_count"] >= state["target_n"]:
        return END
    if state["designs_queue"]:
        return "select_next_design"
    if state["design_round"] > state["max_design_retries"]:
        return END
    return "design"


def after_terminal_design(state: PipelineState) -> str:
    if state["committed_count"] >= state["target_n"]:
        return END
    if state["designs_queue"]:
        return "select_next_design"
    if state["design_round"] > state["max_design_retries"]:
        return END
    return "design"


def after_select_next_design(state: PipelineState) -> str:
    return "audit_design" if state.get("design") else "design"


def after_audit_design(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision and decision.verdict == Verdict.ACCEPT:
        return "generate"
    if state["designs_queue"]:
        return "select_next_design"
    if state["design_round"] > state["max_design_retries"]:
        return END
    return "design"


def after_validate_det(state: PipelineState) -> str | list[str]:
    if state["det_accepted"]:
        if not state.get("adversary_done"):
            return "adversary"
        return "quality_gate"
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_design(state)
    return route_from_decision(state)


def after_generate(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_design(state)
    return route_from_decision(state)


def after_adversary(state: PipelineState) -> str | list[str]:
    decision = state.get("last_decision")
    if decision:
        if decision.terminal:
            return after_terminal_design(state)
        return route_from_decision(state)
    if state.get("adversary_done"):
        return "quality_gate"
    return "revise_from_adversary"


def after_gate_join(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_design(state)
    return route_from_decision(state)


def after_generate_entrypoint(state: PipelineState) -> str:
    return route_from_decision(state)


def after_validate_det_entrypoint(state: PipelineState) -> str | list[str]:
    if state["det_accepted"]:
        if not state.get("adversary_done"):
            return "adversary"
        return "quality_gate"
    return route_from_decision(state)


def after_adversary_entrypoint(state: PipelineState) -> str | list[str]:
    decision = state.get("last_decision")
    if decision:
        return route_from_decision(state)
    if state.get("adversary_done"):
        return "quality_gate"
    return "revise_from_adversary"


def after_gate_join_entrypoint(state: PipelineState) -> str:
    return route_from_decision(state)


def after_curate_entrypoint(state: PipelineState) -> str:
    return END


class PipelineRunner:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.client = ModelClient(config.models)
        self.writer = StageLogWriter(config.logs_dir, config.run_id)
        self.coverage = CoverageLedger(config.data_dir, config.domain)
        self.validation_ledger = ValidationLedger(self.writer)
        self.rejections = RejectionArchive(self.writer)
        self.corpus = CorpusIndex(config.data_dir, config.domain, self.client, config.run_id)
        self.workspace_export = WorkspaceExport(logs_dir=config.logs_dir, data_dir=config.data_dir, run_id=config.run_id)
        self.designer = Designer(self.client, config.domain)
        self.design_auditor = DesignAuditor(self.client, config.domain)
        self.generator = SampleGenerator(
            self.client,
            config.domain,
            system_prompt_override=config.generator_system_prompt_override,
            system_prompt_append=config.generator_system_prompt_append,
        )
        self.adversary = Adversary(self.client, config.domain)
        self.quality_gate = QualityGate(self.client, config.domain)
        self.rubric_gate = RubricGate(self.client, config.domain)
        self.quality_gate_ensemble = [
            (f"gate_ensemble_{index}", QualityGate(ModelClient(model_config), config.domain))
            for index, model_config in enumerate(config.gate_ensemble_models, start=1)
        ]
        self.rubric_gate_ensemble = [
            (f"gate_ensemble_{index}", RubricGate(ModelClient(model_config), config.domain))
            for index, model_config in enumerate(config.gate_ensemble_models, start=1)
        ]
        self.graph = self._build_graph().compile()
        self.generation_graph = self._build_generation_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(PipelineState)
        graph.add_node("design", self.node_design)
        graph.add_node("validate_design_batch_det", self.node_validate_design_batch_det)
        graph.add_node("select_next_design", self.node_select_next_design)
        graph.add_node("audit_design", self.node_audit_design)
        graph.add_node("generate", self.node_generate)
        graph.add_node("validate_det", self.node_validate_det)
        graph.add_node("adversary", self.node_adversary)
        graph.add_node("revise_from_adversary", self.node_revise_from_adversary)
        graph.add_node("quality_gate", self.node_quality_gate)
        graph.add_node("rubric_gate", self.node_rubric_gate)
        graph.add_node("join_gates", self.node_join_gates)
        graph.add_node("curate", self.node_curate)
        graph.add_edge(START, "design")
        graph.add_edge("design", "validate_design_batch_det")
        graph.add_conditional_edges("validate_design_batch_det", after_validate_design_batch_det)
        graph.add_conditional_edges("select_next_design", after_select_next_design)
        graph.add_conditional_edges("audit_design", after_audit_design)
        graph.add_conditional_edges("generate", after_generate)
        graph.add_conditional_edges("validate_det", after_validate_det)
        graph.add_conditional_edges("adversary", after_adversary)
        graph.add_edge("revise_from_adversary", "validate_det")
        graph.add_edge("quality_gate", "rubric_gate")
        graph.add_edge("rubric_gate", "join_gates")
        graph.add_conditional_edges("join_gates", after_gate_join)
        graph.add_conditional_edges("curate", after_curate)
        return graph

    def _build_generation_graph(self) -> StateGraph:
        graph = StateGraph(PipelineState)
        graph.add_node("generate", self.node_generate)
        graph.add_node("validate_det", self.node_validate_det)
        graph.add_node("adversary", self.node_adversary)
        graph.add_node("revise_from_adversary", self.node_revise_from_adversary)
        graph.add_node("quality_gate", self.node_quality_gate)
        graph.add_node("rubric_gate", self.node_rubric_gate)
        graph.add_node("join_gates", self.node_join_gates)
        graph.add_node("curate", self.node_curate)
        graph.add_edge(START, "generate")
        graph.add_conditional_edges("generate", after_generate_entrypoint)
        graph.add_conditional_edges("validate_det", after_validate_det_entrypoint)
        graph.add_conditional_edges("adversary", after_adversary_entrypoint)
        graph.add_edge("revise_from_adversary", "validate_det")
        graph.add_edge("quality_gate", "rubric_gate")
        graph.add_edge("rubric_gate", "join_gates")
        graph.add_conditional_edges("join_gates", after_gate_join_entrypoint)
        graph.add_conditional_edges("curate", after_curate_entrypoint)
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
            "max_design_retries": self.config.domain.max_design_retries,
            "design_round": 0,
            "design_retry_subcodes": [],
            "designs_queue": [],
            "gen_attempt": 0,
            "gen_retry_subcodes": [],
            "det_checks": [],
            "det_accepted": False,
            "adversary_done": False,
            "committed_count": 0,
            "dropped_count": 0,
        }
        final = self.graph.invoke(initial, config={"recursion_limit": _graph_recursion_limit(self.config)})
        return {
            "run_id": self.config.run_id,
            "committed": final["committed_count"],
            "dropped": final["dropped_count"],
        }

    def run_from_generation(self, request: GenerationPipelineInput) -> GenerationPipelineResult:
        envelope = request.envelope
        self._progress(
            "run",
            "start_from_generation",
            envelope=envelope.id,
            design=envelope.design.id,
            model=self.config.models.model,
            graph_limit=_graph_recursion_limit(self.config),
        )
        initial: PipelineState = {
            "run_id": self.config.run_id,
            "target_n": 1,
            "max_design_retries": 0,
            "design_round": 0,
            "design_retry_subcodes": [],
            "designs_queue": [],
            "design": envelope.design,
            "generation_envelope": envelope,
            "gen_attempt": 0,
            "gen_retry_subcodes": [],
            "det_checks": [],
            "det_accepted": False,
            "adversary_done": False,
            "committed_count": 0,
            "dropped_count": 0,
            "last_candidate_id": None,
        }
        final = self.generation_graph.invoke(initial, config={"recursion_limit": _graph_recursion_limit(self.config)})
        result = self._generation_result(envelope, final, request.output_dir)
        if request.output_dir is not None:
            _write_generation_result(result)
        return result

    def node_design(self, state: PipelineState) -> PipelineState:
        design_round = state["design_round"] + 1
        count = max(1, self.config.target_n * 2)
        self._progress(
            "design",
            "start",
            round=design_round,
            requested_designs=count,
            retry=state.get("design_retry_route_code"),
        )
        coverage_snapshot = self.coverage.snapshot()
        stage_input = {
            "run_id": f"{self.config.run_id}-r{design_round}",
            "target_n": count,
            "coverage_snapshot": coverage_snapshot,
            "retry_route_code": state.get("design_retry_route_code"),
            "retry_subcodes": state.get("design_retry_subcodes"),
        }
        designs, meta = self.designer.design(
            run_id=f"{self.config.run_id}-r{design_round}",
            target_n=count,
            coverage_snapshot=coverage_snapshot,
            retry_route_code=state.get("design_retry_route_code"),
            retry_subcodes=state.get("design_retry_subcodes"),
        )
        verdict = Verdict.ACCEPT if designs else Verdict.REJECT
        route_code = RouteCode.ACCEPT if designs else RouteCode.RETRY_PROVIDER_EMPTY
        self._record(
            stage_kind=StageKind.DESIGN,
            role="design_batch",
            agent_role=AgentRole.DESIGNER,
            artifact_id=f"{self.config.run_id}-design-{design_round}",
            parent_artifact_id=None,
            verdict=verdict,
            route_code=route_code,
            context_policy=_producer_context_policy(state.get("design_retry_route_code")),
            meta=meta,
            retry_index=design_round - 1,
            stage_input=stage_input,
            stage_output={"designs": designs},
        )
        return {
            "design_round": design_round,
            "designs_queue": designs,
            "design": None,
            "design_retry_route_code": None,
            "design_retry_subcodes": [],
            "last_decision": None,
        }

    def node_validate_design_batch_det(self, state: PipelineState) -> PipelineState:
        designs = state["designs_queue"]
        self._progress("design_det", "start", round=state["design_round"], designs=len(designs))
        accepted_designs, rejected_designs = self._partition_design_batch(designs)
        if accepted_designs:
            verdict = Verdict.ACCEPT
            route_code = RouteCode.ACCEPT
            subcodes: list[str] = []
        elif rejected_designs:
            _, route_code, subcodes = rejected_designs[0]
            verdict = Verdict.REJECT
        else:
            verdict = Verdict.REJECT
            route_code = RouteCode.RETRY_PROVIDER_EMPTY
            subcodes = ["provider_error"]
        retry_index = state["design_round"] - 1
        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.DESIGN_AUDIT,
            verdict=verdict,
            route_code=route_code,
            retry_index=retry_index,
            max_design_retries=self.config.domain.max_design_retries,
            subcodes=subcodes,
        )
        self._record(
            stage_kind=StageKind.DESIGN_AUDIT,
            role="validate_design_batch_deterministically",
            agent_role=None,
            artifact_id=f"{self.config.run_id}-design-batch-{state['design_round']}-deterministic-verdict",
            parent_artifact_id=f"{self.config.run_id}-design-{state['design_round']}",
            verdict=verdict,
            route_code=decision.route_code,
            subcodes=subcodes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=_local_meta(),
            retry_index=retry_index,
            stage_input={"designs": designs},
            stage_output={
                "accepted_designs": accepted_designs,
                "rejected_designs": [
                    {"design": rejected_design, "route_code": rejected_route_code, "subcodes": rejected_subcodes}
                    for rejected_design, rejected_route_code, rejected_subcodes in rejected_designs
                ],
                "decision": decision,
            },
        )
        update: PipelineState = {"last_decision": decision, "designs_queue": accepted_designs}
        for rejected_design, rejected_route_code, rejected_subcodes in rejected_designs:
            rejected_decision = RoutingDecision(
                run_id=self.config.run_id,
                from_stage=StageKind.DESIGN_AUDIT,
                verdict=Verdict.REJECT,
                route_code=rejected_route_code,
                subcodes=rejected_subcodes,
                next_stage=None,
                context_policy=ContextPolicy.CRITERIA_ONLY,
                retry_index=retry_index,
                terminal=True,
            )
            self.rejections.append(rejected_design, rejected_decision)
        if verdict == Verdict.REJECT:
            update["design_retry_route_code"] = decision.route_code
            update["design_retry_subcodes"] = decision.subcodes
        return update

    def _partition_design_batch(
        self,
        designs: list[DesignBrief],
    ) -> tuple[list[DesignBrief], list[tuple[DesignBrief, RouteCode, list[str]]]]:
        accepted: list[DesignBrief] = []
        rejected: list[tuple[DesignBrief, RouteCode, list[str]]] = []
        seen: set[str] = set()
        for design in designs:
            if design.content_hash in seen:
                rejected.append((design, RouteCode.REJECT_DUPLICATE, ["duplicate_design"]))
                continue
            verdict, route_code, subcodes = validate_design_batch([design], self.config.domain)
            if verdict == Verdict.ACCEPT:
                accepted.append(design)
                seen.add(design.content_hash)
            else:
                rejected.append((design, route_code, subcodes))
        return accepted, rejected

    def node_select_next_design(self, state: PipelineState) -> PipelineState:
        designs = list(state["designs_queue"])
        design = designs.pop(0) if designs else None
        if design:
            self._progress(
                "design_cursor",
                "select",
                id=design.id,
                case_type=design.cell.case_type,
                difficulty=design.cell.difficulty,
                scenario=design.cell.scenario,
                remaining=len(designs),
            )
        return {
            "designs_queue": designs,
            "design": design,
            "gen_attempt": 0,
            "gen_retry_route_code": None,
            "gen_retry_subcodes": [],
            "candidate": None,
            "det_checks": [],
            "det_accepted": False,
            "adversary_done": False,
            "adversary_report": None,
            "quality_verdict": None,
            "rubric_verdict": None,
            "last_decision": None,
        }

    def node_audit_design(self, state: PipelineState) -> PipelineState:
        design = _require(state.get("design"), "design")
        self._progress("design_audit", "start", design=design.id, case_type=design.cell.case_type)
        verdict, route_code, subcodes = validate_design_batch([design], self.config.domain)
        if verdict == Verdict.REJECT:
            design_verdict = _local_design_verdict(design, route_code, subcodes)
            meta = _local_meta()
        else:
            design_verdict, meta = self.design_auditor.audit(design)
        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.DESIGN_AUDIT,
            verdict=design_verdict.verdict,
            route_code=design_verdict.route_code,
            retry_index=0,
            max_design_retries=self.config.domain.max_design_retries,
            subcodes=design_verdict.subcodes,
        )
        self._record(
            stage_kind=StageKind.DESIGN_AUDIT,
            role="audit_design",
            agent_role=AgentRole.DESIGN_AUDITOR if meta["provider"] != "local" else None,
            artifact_id=f"{design.id}-design-verdict",
            parent_artifact_id=design.id,
            verdict=design_verdict.verdict,
            route_code=design_verdict.route_code,
            subcodes=design_verdict.subcodes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=meta,
            stage_input={"design": design},
            stage_output={"design_verdict": design_verdict, "decision": decision},
        )
        update: PipelineState = {"last_decision": decision}
        if design_verdict.verdict == Verdict.REJECT:
            self.rejections.append(design, decision)
            update["dropped_count"] = state["dropped_count"] + 1
        return update

    def node_generate(self, state: PipelineState) -> PipelineState:
        design = _require(state.get("design"), "design")
        retry_index = state["gen_attempt"]
        self._progress(
            "generation",
            "start",
            design=design.id,
            attempt=retry_index + 1,
            retry=state.get("gen_retry_route_code"),
        )
        try:
            envelope = state.get("generation_envelope") or GenerationEnvelope.from_design(design)
            self._append_generation_envelope(
                envelope,
                role="generate_candidate_sample",
                retry_index=retry_index,
                retry_route_code=state.get("gen_retry_route_code"),
                retry_subcodes=state.get("gen_retry_subcodes"),
            )
            candidate, gen_meta = self.generator.generate_from_envelope(
                run_id=self.config.run_id,
                envelope=envelope,
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
            )
            self._record(
                stage_kind=StageKind.GENERATION,
                role="generate_candidate_sample",
                agent_role=AgentRole.SAMPLE_GENERATOR,
                artifact_id=f"{design.id}-generation-error-{retry_index}",
                parent_artifact_id=design.id,
                verdict=Verdict.REJECT,
                route_code=decision.route_code,
                subcodes=["provider_error"],
                context_policy=_producer_context_policy(state.get("gen_retry_route_code")),
                meta=_local_meta(error=str(exc)),
                retry_index=retry_index,
                stage_input={
                    "envelope": envelope,
                    "attempt": retry_index + 1,
                    "retry_route_code": state.get("gen_retry_route_code"),
                    "retry_subcodes": state.get("gen_retry_subcodes"),
                },
                stage_output={"error": str(exc), "decision": decision},
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
            parent_artifact_id=design.id,
            verdict=Verdict.ACCEPT,
            route_code=RouteCode.ACCEPT,
            context_policy=_producer_context_policy(state.get("gen_retry_route_code")),
            meta=gen_meta,
            retry_index=retry_index,
            stage_input={
                "envelope": envelope,
                "attempt": retry_index + 1,
                "retry_route_code": state.get("gen_retry_route_code"),
                "retry_subcodes": state.get("gen_retry_subcodes"),
            },
            stage_output={"candidate": candidate, "decision": decision},
        )
        self._append_candidate_snapshot(
            candidate,
            phase="generated",
            role="generate_candidate_sample",
            retry_index=retry_index,
        )
        self._progress("candidate", "generated", **_candidate_progress(candidate))
        return {
            "candidate": candidate,
            "det_accepted": False,
            "adversary_done": False,
            "adversary_report": None,
            "quality_verdict": None,
            "rubric_verdict": None,
            "last_decision": decision,
            "gen_retry_route_code": None,
            "gen_retry_subcodes": [],
            "last_candidate_id": candidate.id,
        }

    def node_validate_det(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        self._progress("validation_det", "start", candidate=candidate.id)
        det_verdict, checks = deterministic_sample_verdict(
            candidate,
            self.config.domain,
            workspace_validation_executor=self.config.workspace_validation_executor,
        )
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
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=_local_meta(),
            retry_index=state["gen_attempt"],
            stage_input={"candidate": candidate},
            stage_output={"deterministic_verdict": det_verdict, "checks": checks},
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
        )
        self.rejections.append(candidate, decision)
        self.workspace_export.export_rejection(candidate, decision)
        self._progress(
            "candidate",
            "rejected",
            **_candidate_progress(candidate),
            route=decision.route_code,
            codes=decision.subcodes,
        )
        update: PipelineState = {"det_checks": checks, "det_accepted": False, "last_decision": decision}
        if decision.terminal:
            update["dropped_count"] = state["dropped_count"] + 1
        else:
            update["gen_attempt"] = state["gen_attempt"] + 1
            update["gen_retry_route_code"] = decision.route_code
            update["gen_retry_subcodes"] = decision.subcodes
        return update

    def node_adversary(self, state: PipelineState) -> PipelineState:
        design = _require(state.get("design"), "design")
        candidate = _require(state.get("candidate"), "candidate")
        self._progress("adversary", "start", candidate=candidate.id)
        report, meta = self.adversary.attack(candidate, design)
        self.writer.append_adversary_report(report)
        self._record(
            stage_kind=StageKind.VALIDATION,
            role="adversary_attack_report",
            agent_role=AgentRole.ADVERSARY,
            artifact_id=f"{candidate.id}-adversary-report",
            parent_artifact_id=candidate.id,
            verdict=Verdict.ACCEPT,
            route_code=RouteCode.ACCEPT,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=meta,
            retry_index=state["gen_attempt"],
            stage_input={"design": design, "candidate": candidate},
            stage_output={"adversary_report": report},
        )
        self._progress(
            "adversary",
            "reported",
            candidate=candidate.id,
            attacks=len(report.attacks),
            disposition=report.revision_disposition,
        )
        if report.revision_disposition == "nuke":
            decision = route_after(
                run_id=self.config.run_id,
                from_stage=StageKind.VALIDATION,
                verdict=Verdict.REJECT,
                route_code=RouteCode.REJECT_SEMANTIC_MISMATCH,
                retry_index=state["gen_attempt"],
                max_generation_retries=self.config.domain.max_generation_retries,
                subcodes=["adversary_nuke"],
            )
            self.rejections.append(candidate, decision)
            self.workspace_export.export_rejection(candidate, decision)
            self._progress(
                "candidate",
                "rejected",
                **_candidate_progress(candidate),
                route=decision.route_code,
                codes=decision.subcodes,
            )
            update: PipelineState = {
                "adversary_report": report,
                "adversary_done": True,
                "det_accepted": False,
                "last_decision": decision,
            }
            if decision.terminal:
                update["dropped_count"] = state["dropped_count"] + 1
            else:
                update["gen_attempt"] = state["gen_attempt"] + 1
                update["gen_retry_route_code"] = decision.route_code
                update["gen_retry_subcodes"] = decision.subcodes
            return update
        if report.revision_disposition == "pass":
            return {"adversary_report": report, "adversary_done": True, "last_decision": None}
        return {"adversary_report": report, "last_decision": None}

    def node_revise_from_adversary(self, state: PipelineState) -> PipelineState:
        design = _require(state.get("design"), "design")
        candidate = _require(state.get("candidate"), "candidate")
        report = _require(state.get("adversary_report"), "adversary_report")
        attempt = state["gen_attempt"] + 1
        self._progress(
            "generation",
            "revise",
            design=design.id,
            candidate=candidate.id,
            attacks=len(report.attacks),
        )
        try:
            revised, meta = self.generator.revise_from_attack(
                run_id=self.config.run_id,
                design=design,
                candidate=candidate,
                report=report,
                attempt=attempt,
            )
        except ProviderError as exc:
            decision = route_after(
                run_id=self.config.run_id,
                from_stage=StageKind.GENERATION,
                verdict=Verdict.REJECT,
                route_code=RouteCode.RETRY_INFRA,
                retry_index=state["gen_attempt"],
                max_generation_retries=self.config.domain.max_generation_retries,
                subcodes=["provider_error"],
            )
            self._record(
                stage_kind=StageKind.GENERATION,
                role="revise_candidate_from_adversary",
                agent_role=AgentRole.SAMPLE_GENERATOR,
                artifact_id=f"{candidate.id}-adversary-revision-error",
                parent_artifact_id=candidate.id,
                verdict=Verdict.REJECT,
                route_code=decision.route_code,
                subcodes=["provider_error"],
                context_policy=ContextPolicy.CRITERIA_PLUS_ROUTE_CODE,
                meta=_local_meta(error=str(exc)),
                retry_index=state["gen_attempt"],
                stage_input={"design": design, "candidate": candidate, "adversary_report": report, "attempt": attempt},
                stage_output={"error": str(exc), "decision": decision},
            )
            update: PipelineState = {"last_decision": decision, "adversary_done": True}
            if decision.terminal:
                update["dropped_count"] = state["dropped_count"] + 1
            else:
                update["gen_attempt"] = state["gen_attempt"] + 1
                update["gen_retry_route_code"] = decision.route_code
                update["gen_retry_subcodes"] = decision.subcodes
            return update

        self._record(
            stage_kind=StageKind.GENERATION,
            role="revise_candidate_from_adversary",
            agent_role=AgentRole.SAMPLE_GENERATOR,
            artifact_id=revised.id,
            parent_artifact_id=candidate.id,
            verdict=Verdict.ACCEPT,
            route_code=RouteCode.ACCEPT,
            context_policy=ContextPolicy.CRITERIA_PLUS_ROUTE_CODE,
            meta=meta,
            retry_index=state["gen_attempt"],
            stage_input={"design": design, "candidate": candidate, "adversary_report": report, "attempt": attempt},
            stage_output={"candidate": revised},
        )
        self._append_candidate_snapshot(
            revised,
            phase="adversary_revision",
            role="revise_candidate_from_adversary",
            retry_index=state["gen_attempt"],
            parent_candidate_id=candidate.id,
            adversary_report_id=f"{candidate.id}-adversary-report",
        )
        self._progress("candidate", "revised", **_candidate_progress(revised))
        return {
            "candidate": revised,
            "det_accepted": False,
            "adversary_done": True,
            "quality_verdict": None,
            "rubric_verdict": None,
            "last_decision": None,
            "last_candidate_id": revised.id,
        }

    def _append_candidate_snapshot(
        self,
        candidate: CandidateSample,
        *,
        phase: str,
        role: str,
        retry_index: int,
        parent_candidate_id: str | None = None,
        adversary_report_id: str | None = None,
    ) -> None:
        self.writer.append_candidate(
            {
                "run_id": self.config.run_id,
                "phase": phase,
                "role": role,
                "candidate_id": candidate.id,
                "design_id": candidate.design_id,
                "parent_candidate_id": parent_candidate_id,
                "adversary_report_id": adversary_report_id,
                "retry_index": retry_index,
                "candidate": candidate.model_dump(mode="json"),
            }
        )
        self.workspace_export.export_snapshot(
            candidate,
            phase=phase,
            role=role,
            retry_index=retry_index,
            parent_candidate_id=parent_candidate_id,
            adversary_report_id=adversary_report_id,
        )

    def _append_generation_envelope(
        self,
        envelope: GenerationEnvelope,
        *,
        role: str,
        retry_index: int,
        retry_route_code: RouteCode | None,
        retry_subcodes: list[str] | None,
    ) -> None:
        self.writer.append_generation_envelope(
            {
                "run_id": self.config.run_id,
                "role": role,
                "envelope_id": envelope.id,
                "design_id": envelope.design.id,
                "retry_index": retry_index,
                "retry_route_code": retry_route_code,
                "retry_subcodes": retry_subcodes or [],
                "envelope": envelope.model_dump(mode="json"),
            }
        )

    def node_quality_gate(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        self._progress("quality_gate", "start", candidate=candidate.id)
        self._progress(
            "quality_gate",
            "provider_start",
            candidate=candidate.id,
            model=self.config.models.model,
            provider=self.config.models.provider,
        )
        quality_verdict, quality_meta = self.quality_gate.validate(candidate)
        self._progress(
            "quality_gate",
            "provider_end",
            candidate=candidate.id,
            model=quality_meta.get("model"),
            provider=quality_meta.get("provider"),
            latency=f"{quality_meta.get('latency_ms', 0)}ms",
        )
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
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=quality_meta,
            retry_index=state["gen_attempt"],
            stage_input={"candidate": candidate},
            stage_output={"quality_verdict": quality_verdict},
        )
        self._run_gate_ensemble(
            gate_kind="quality",
            gates=self.quality_gate_ensemble,
            candidate=candidate,
            retry_index=state["gen_attempt"],
        )
        return {"quality_verdict": quality_verdict}

    def node_rubric_gate(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        self._progress("rubric_gate", "start", candidate=candidate.id)
        self._progress(
            "rubric_gate",
            "provider_start",
            candidate=candidate.id,
            model=self.config.models.model,
            provider=self.config.models.provider,
        )
        rubric_verdict, rubric_meta = self.rubric_gate.validate(candidate)
        self._progress(
            "rubric_gate",
            "provider_end",
            candidate=candidate.id,
            model=rubric_meta.get("model"),
            provider=rubric_meta.get("provider"),
            latency=f"{rubric_meta.get('latency_ms', 0)}ms",
        )
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
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=rubric_meta,
            retry_index=state["gen_attempt"],
            stage_input={"candidate": candidate},
            stage_output={"rubric_verdict": rubric_verdict},
        )
        self._run_gate_ensemble(
            gate_kind="rubric",
            gates=self.rubric_gate_ensemble,
            candidate=candidate,
            retry_index=state["gen_attempt"],
        )
        return {"rubric_verdict": rubric_verdict}

    def _run_gate_ensemble(
        self,
        *,
        gate_kind: str,
        gates: list[tuple[str, QualityGate | RubricGate]],
        candidate: CandidateSample,
        retry_index: int,
    ) -> None:
        for gate_id, gate in gates:
            self._progress(f"{gate_kind}_gate:{gate_id}", "start", candidate=candidate.id)
            verdict, meta = gate.validate(candidate)
            role = f"{gate_kind}_gate_candidate_ensemble"
            agent_role = AgentRole.QUALITY_GATE if gate_kind == "quality" else AgentRole.RUBRIC_GATE
            self._record(
                stage_kind=StageKind.VALIDATION,
                role=role,
                agent_role=agent_role,
                artifact_id=f"{candidate.id}-{gate_kind}-verdict-{gate_id}",
                parent_artifact_id=candidate.id,
                verdict=verdict.verdict,
                route_code=verdict.route_code,
                subcodes=verdict.subcodes,
                context_policy=ContextPolicy.CRITERIA_ONLY,
                meta={**meta, "gate_ensemble_id": gate_id},
                retry_index=retry_index,
                stage_input={"candidate": candidate, "gate_ensemble_id": gate_id},
                stage_output={f"{gate_kind}_verdict": verdict, "gate_ensemble_id": gate_id},
            )

    def node_join_gates(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        quality_verdict = _require(state.get("quality_verdict"), "quality_verdict")
        rubric_verdict = _require(state.get("rubric_verdict"), "rubric_verdict")
        self._progress(
            "join_gates",
            "start",
            candidate=candidate.id,
            quality=quality_verdict.verdict,
            rubric=rubric_verdict.verdict,
        )
        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.VALIDATION,
            verdict=Verdict.ACCEPT,
            route_code=RouteCode.ACCEPT,
            retry_index=state["gen_attempt"],
            max_generation_retries=self.config.domain.max_generation_retries,
            subcodes=_gate_caveat_subcodes(quality_verdict, rubric_verdict),
        )
        self._record(
            stage_kind=StageKind.VALIDATION,
            role="join_quality_rubric_gates",
            agent_role=None,
            artifact_id=f"{candidate.id}-gate-join",
            parent_artifact_id=candidate.id,
            verdict=decision.verdict,
            route_code=decision.route_code,
            subcodes=decision.subcodes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=_local_meta(),
            retry_index=state["gen_attempt"],
            stage_input={
                "candidate": candidate,
                "quality_verdict": quality_verdict,
                "rubric_verdict": rubric_verdict,
            },
            stage_output={"decision": decision},
        )
        return {"last_decision": decision}

    def node_curate(self, state: PipelineState) -> PipelineState:
        candidate = _require(state.get("candidate"), "candidate")
        quality_verdict = state.get("quality_verdict") or _bypass_gate_verdict(candidate, "quality")
        rubric_verdict = state.get("rubric_verdict") or _bypass_gate_verdict(candidate, "rubric")
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
        decision = route_after(
            run_id=self.config.run_id,
            from_stage=StageKind.CURATION,
            verdict=cur_verdict.verdict,
            route_code=cur_verdict.route_code,
            retry_index=state["gen_attempt"],
            subcodes=cur_verdict.subcodes,
        )
        self._record(
            stage_kind=StageKind.CURATION,
            role="curate_committed_sample",
            agent_role=None,
            artifact_id=committed.id if committed else f"{candidate.id}-curation-reject",
            parent_artifact_id=certified.id,
            verdict=cur_verdict.verdict,
            route_code=cur_verdict.route_code,
            subcodes=cur_verdict.subcodes,
            context_policy=ContextPolicy.CRITERIA_ONLY,
            meta=cur_meta,
            retry_index=state["gen_attempt"],
            stage_input={
                "certified": certified,
                "quality_verdict": quality_verdict,
                "rubric_verdict": rubric_verdict,
            },
            stage_output={"committed": committed, "curation_verdict": cur_verdict, "decision": decision},
        )
        if committed:
            self.workspace_export.export_committed(committed)
            self.coverage.increment(candidate.cell)
            self._progress(
                "candidate",
                "committed",
                **_candidate_progress(candidate),
                route=decision.route_code,
                codes=cur_verdict.subcodes,
            )
            return {
                "committed_count": state["committed_count"] + 1,
                "last_decision": decision,
                "candidate": None,
                "quality_verdict": None,
                "rubric_verdict": None,
                "det_checks": [],
                "last_candidate_id": candidate.id,
            }

        self.rejections.append(candidate, decision)
        self.workspace_export.export_rejection(candidate, decision)
        self._progress(
            "candidate",
            "rejected",
            **_candidate_progress(candidate),
            route=decision.route_code,
            codes=decision.subcodes,
        )
        return {
            "dropped_count": state["dropped_count"] + 1,
            "last_decision": decision,
            "candidate": None,
            "quality_verdict": None,
            "rubric_verdict": None,
            "det_checks": [],
            "last_candidate_id": candidate.id,
        }

    def _generation_result(
        self,
        envelope: GenerationEnvelope,
        final: PipelineState,
        output_dir: Path | None,
    ) -> GenerationPipelineResult:
        last_decision = final.get("last_decision")
        committed = final.get("committed_count", 0)
        dropped = final.get("dropped_count", 0)
        if committed:
            final_status = "committed"
        elif dropped:
            final_status = "dropped"
        else:
            final_status = "incomplete"
        candidate = final.get("candidate")
        candidate_id = final.get("last_candidate_id")
        if candidate_id is None and candidate is not None:
            candidate_id = candidate.id
        result_path = None if output_dir is None else output_dir / "generation_result.json"
        return GenerationPipelineResult(
            run_id=self.config.run_id,
            envelope_id=envelope.id,
            design_id=envelope.design.id,
            final_status=final_status,
            committed=committed,
            dropped=dropped,
            candidate_id=candidate_id,
            route_code=None if last_decision is None else last_decision.route_code,
            subcodes=[] if last_decision is None else list(last_decision.subcodes),
            logs_dir=self.config.logs_dir / self.config.run_id,
            corpus_path=self.config.data_dir / "corpus" / "benchmark" / f"{self.config.run_id}.jsonl",
            materialized_dir=self.config.data_dir / "materialized" / "benchmark" / self.config.run_id,
            result_path=result_path,
        )

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
        retry_index: int = 0,
        stage_input: Any | None = None,
        stage_output: Any | None = None,
    ) -> None:
        stage_id = f"{stage_kind.value}:{artifact_id}"
        record = StageRecord(
            run_id=self.config.run_id,
            stage_id=stage_id,
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
            error=None if meta.get("error") is None else str(meta.get("error")),
            verdict=verdict,
            route_code=route_code,
            subcodes=subcodes or [],
            criteria_hash=stable_hash(self.config.domain.model_dump(mode="json")),
            context_policy=context_policy,
            retry_index=retry_index,
            input_hash=None if stage_input is None else trace_hash(stage_input),
            output_hash=None if stage_output is None else trace_hash(stage_output),
            trace_ref=None if stage_input is None and stage_output is None else "stage_io.jsonl",
        )
        self.writer.write_stage_record(record, stage_input=stage_input, stage_output=stage_output)
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
        self.writer.append_event(
            "stage_progress",
            {
                "run_id": self.config.run_id,
                "stage": stage,
                "stage_event": event,
                **_event_fields(fields),
            },
        )
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


def _write_generation_result(result: GenerationPipelineResult) -> None:
    if result.result_path is None:
        return
    result.result_path.parent.mkdir(parents=True, exist_ok=True)
    result.result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _bypass_gate_verdict(candidate: CandidateSample, check_kind: str) -> SampleVerdict:
    return SampleVerdict(
        candidate_id=candidate.id,
        check_kind=check_kind,  # type: ignore[arg-type]
        verdict=Verdict.ACCEPT,
        route_code=RouteCode.ACCEPT,
        subcodes=["adversary_only_test_bypass"],
        rationale="Quality and rubric gates bypassed for the temporary adversary-only test run.",
    )


def _gate_caveat_subcodes(*verdicts: SampleVerdict) -> list[str]:
    subcodes: list[str] = []
    for verdict in verdicts:
        if verdict.verdict != Verdict.REJECT:
            continue
        label = f"{verdict.check_kind}_gate_rejected"
        if label not in subcodes:
            subcodes.append(label)
        for subcode in verdict.subcodes:
            if subcode not in subcodes:
                subcodes.append(subcode)
    return subcodes


def _graph_recursion_limit(config: RuntimeConfig) -> int:
    design_rounds = config.domain.max_design_retries + 1
    designs_per_round = max(1, config.target_n * 2)
    generation_attempts = config.domain.max_generation_retries + 1
    per_design_steps = 2 + (generation_attempts * 4) + 1
    design_steps = 2
    return 10 + design_rounds * (design_steps + designs_per_round * per_design_steps)


def _stage_label(record: StageRecord) -> str:
    if record.role == "validate_design_batch_deterministically":
        return "design_det"
    if record.role == "audit_design":
        return "design_audit"
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


def _event_fields(fields: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        if key in {"prompt", "proxy", "candidate", "design", "stage_input", "stage_output"}:
            continue
        if hasattr(value, "value"):
            safe[key] = value.value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [item.value if hasattr(item, "value") else item for item in value if isinstance(item, (str, int, float, bool)) or hasattr(item, "value")]
        else:
            safe[key] = str(value)
    return safe


def _short_id(value: str) -> str:
    if len(value) <= 48:
        return value
    return f"{value[:22]}...{value[-22:]}"


def _candidate_progress(candidate: CandidateSample) -> dict[str, Any]:
    ability = candidate.ability_z.get("name") if isinstance(candidate.ability_z, dict) else None
    prompt = candidate.agent_artifact.benchmark_case.get("prompt")
    return {
        "id": candidate.id,
        "case_type": candidate.case_type,
        "ability": ability,
        "prompt": prompt,
        "proxy": candidate.judge_artifact.proxy_claim,
    }


def _local_design_verdict(design: DesignBrief, route_code: RouteCode, subcodes: list[str]) -> DesignVerdict:
    return DesignVerdict(
        design_id=design.id,
        verdict=Verdict.REJECT,
        route_code=route_code,
        subcodes=subcodes,
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
