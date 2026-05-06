from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from config import DomainConfig, ModelConfig
from models import (
    CandidateSample,
    EvidenceRef,
    PlanVerdict,
    RouteCode,
    SampleVerdict,
    SeedSpec,
    TaxonomyCell,
    Verdict,
    stable_hash,
)
from text_hygiene import normalize_text_tree


class ProviderError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required for the live POC demo")

    def complete_json(self, *, system: str, user: str, temperature: float = 0.4) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.config.reasoning_effort and _supports_reasoning_effort(self.config.model):
            body["reasoning_effort"] = self.config.reasoning_effort
        if not self.config.model.startswith("gpt-5"):
            body["temperature"] = temperature
        response = self._post("/chat/completions", body)
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response["choices"][0]["message"]["content"]
        if not content:
            raise ProviderError("provider returned empty JSON content")
        usage = response.get("usage", {})
        meta = {
            "provider": self.config.provider,
            "model": self.config.model,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "reasoning_effort": self.config.reasoning_effort,
        }
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"provider returned invalid JSON: {exc}") from exc
        payload, replacements = normalize_text_tree(payload)
        meta["text_normalization_replacements"] = replacements
        return payload, meta

    def complete_text(self, *, system: str, user: str, temperature: float = 0.7) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.config.reasoning_effort and _supports_reasoning_effort(self.config.model):
            body["reasoning_effort"] = self.config.reasoning_effort
        if not self.config.model.startswith("gpt-5"):
            body["temperature"] = temperature
        response = self._post("/chat/completions", body)
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response["choices"][0]["message"]["content"]
        if not content:
            raise ProviderError("provider returned empty text content")
        usage = response.get("usage", {})
        content, replacements = normalize_text_tree(content)
        return content, {
            "provider": self.config.provider,
            "model": self.config.model,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "reasoning_effort": self.config.reasoning_effort,
            "text_normalization_replacements": replacements,
        }

    def embed(self, text: str) -> tuple[list[float], dict[str, Any]]:
        started = time.perf_counter()
        response = self._post(
            "/embeddings",
            {"model": self.config.embedding_model, "input": text},
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = response.get("usage", {})
        return response["data"][0]["embedding"], {
            "provider": self.config.provider,
            "model": self.config.embedding_model,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": 0,
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
        }

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"OpenAI connection error: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderError(f"OpenAI read timed out after {self.config.request_timeout_seconds:g}s") from exc


class Strategist:
    role_name = "Strategist"

    def __init__(self, client: OpenAIClient, domain: DomainConfig) -> None:
        self.client = client
        self.domain = domain

    def plan(
        self,
        *,
        run_id: str,
        target_n: int,
        coverage_snapshot: dict[str, int],
        retry_route_code: RouteCode | None = None,
        retry_subcodes: list[str] | None = None,
    ) -> tuple[list[SeedSpec], dict[str, Any]]:
        system = (
            "You are the Strategist for a benchmark-generation pipeline. Produce diverse benchmark seed specs only. "
            "Do not generate benchmark cases. Return JSON only."
        )
        payload: dict[str, Any] = {
            "task": "Create seed specs for benchmark cases where score_x should proxy ability_z in environment_y.",
            "target_count": target_n,
            "case_types": self.domain.case_types,
            "difficulties": self.domain.difficulties,
            "scenarios": self.domain.scenarios,
            "abilities": self.domain.abilities,
            "environments": self.domain.environments,
            "diagnostic_pressure_types": self.domain.diagnostic_pressure_types,
            "scoring_methods": self.domain.scoring_methods,
            "coverage_snapshot": coverage_snapshot,
            "required_json_shape": {
                "seeds": [
                    {
                        "case_type": "one allowed case type",
                        "difficulty": "integer 1..5",
                        "scenario": "one allowed scenario",
                        "ability": "one allowed ability or specific sub-ability",
                        "environment": "one allowed environment or environment slice",
                        "environment_seed": {
                            "product_context": "specific realistic software setting",
                            "codebase_shape": "compact description of modules/files/components",
                            "state_model": "objects, data flow, lifecycle, or invariant-bearing state",
                            "core_invariant": "behavior that must remain true",
                            "failure_surface": "observable symptom or user-facing failure",
                            "tempting_wrong_fix": "plausible shallow repair path",
                            "actual_causal_region": "where the real cause should live",
                            "required_depth": "why resolving it requires nontrivial reasoning",
                            "non_goals": ["toy bug shapes to avoid"],
                        },
                        "diagnostic_pressure": "specific pressure this case should apply",
                        "scoring_strategy": "how score_x should be determined",
                        "leakage_risk": "most likely way the benchmark could be passed without the target ability",
                        "intent": "specific generation brief, 12-40 words",
                    }
                ]
            },
        }
        if retry_route_code is not None:
            payload["prior_plan_rejection"] = {
                "route_code": retry_route_code.value,
                "subcodes": retry_subcodes or [],
            }
        user = json.dumps(payload, sort_keys=True)
        payload, meta = self.client.complete_json(system=system, user=user, temperature=0.7)
        seeds: list[SeedSpec] = []
        for index, raw in enumerate(payload.get("seeds", [])):
            cell = TaxonomyCell(
                case_type=raw["case_type"],
                difficulty=int(raw["difficulty"]),
                scenario=raw["scenario"],
            )
            seeds.append(
                SeedSpec.create(
                    seed_id=f"{run_id}-seed-{index + 1}",
                    cell=cell,
                    intent=raw["intent"],
                    ability=str(raw["ability"]),
                    environment=str(raw["environment"]),
                    environment_seed=raw["environment_seed"] if isinstance(raw.get("environment_seed"), dict) else {},
                    diagnostic_pressure=str(raw["diagnostic_pressure"]),
                    scoring_strategy=str(raw["scoring_strategy"]),
                    leakage_risk=str(raw["leakage_risk"]),
                    parent_plan_id=f"{run_id}-plan-1",
                )
            )
        return seeds, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}


class PlanAuditor:
    role_name = "PlanAuditor"

    def __init__(self, client: OpenAIClient, domain: DomainConfig) -> None:
        self.client = client
        self.domain = domain

    def audit(self, seed: SeedSpec) -> tuple[PlanVerdict, dict[str, Any]]:
        system = (
            "You are a stateless Plan Auditor. Judge the seed against the domain criteria. "
            "Return verdict metadata plus a concise public rationale. "
            "Do not reveal hidden chain-of-thought. Do not rewrite, improve, or repair the seed. Return JSON only."
        )
        user = json.dumps(
            {
                "seed": seed.model_dump(mode="json"),
                "criteria": {
                    "allowed_case_types": self.domain.case_types,
                    "allowed_scenarios": self.domain.scenarios,
                    "allowed_abilities": self.domain.abilities,
                    "allowed_environments": self.domain.environments,
                    "diagnostic_pressure_types": self.domain.diagnostic_pressure_types,
                    "scoring_methods": self.domain.scoring_methods,
                    "difficulty_range": self.domain.difficulties,
                    "code_seed_standard": {
                        "required_when_domain": "benchmark_code_debug",
                        "environment_seed_must_define": [
                            "product_context",
                            "codebase_shape",
                            "state_model",
                            "core_invariant",
                            "failure_surface",
                            "tempting_wrong_fix",
                            "actual_causal_region",
                            "required_depth",
                            "non_goals",
                        ],
                        "reject_if": [
                            "the seed can naturally instantiate as a one-line patch",
                            "the environment is just a file/function label",
                            "the tempting wrong fix is not plausible",
                            "the failure has no state, invariant, or causal depth",
                        ],
                    },
                    "route_codes": self.domain.route_codes,
                    "subcodes": self.domain.subcodes,
                },
                "required_json_shape": {
                    "verdict": "accept or reject",
                    "route_code": "accept or a reject route code",
                    "subcodes": ["descriptive labels only"],
                    "reason_codes": ["descriptive labels only"],
                    "evidence": [{"source": "criteria", "path": "field", "value": "short quote"}],
                    "rationale": "2-5 sentence public justification for the verdict, citing concrete evidence; no hidden chain-of-thought",
                },
            },
            sort_keys=True,
        )
        payload, meta = self.client.complete_json(system=system, user=user, temperature=0.2)
        verdict = _verdict(payload.get("verdict"))
        route_code = _route_code(payload.get("route_code"), default=RouteCode.ACCEPT if verdict == Verdict.ACCEPT else RouteCode.REJECT_CRITERIA_MISMATCH)
        plan_verdict = PlanVerdict(
            seed_id=seed.id,
            verdict=verdict,
            route_code=route_code,
            subcodes=list(payload.get("subcodes", [])),
            reason_codes=list(payload.get("reason_codes", [])),
            evidence=_evidence(payload.get("evidence", [])),
            rationale=str(payload.get("rationale", "")),
        )
        return plan_verdict, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}


_GENERATOR_PRINCIPLES = """
You are a Benchmark Case Generator. Your output is training data: a benchmark case used to
evaluate an LLM's ability. Every case you produce begins as a plausible benchmark
candidate and must earn promotion into the corpus. Do not claim admission quality.
Return JSON only.

Your job is not to produce a merely valid benchmark case. Plausible, well-formed,
or rubric-shaped is not enough. Assume every generated case costs real money,
evaluation time, and user trust. A useful benchmark candidate should include the
evidence needed to promote it from plausible to a defensible proxy for ability_z
in environment_y.

Design the case so a neutral, critical gate can see why it is more than checklist
compliance. If a weak but careful model could pass by following visible rules,
making token-level substitutions, adding decorative details, or satisfying
checklists without showing the target ability, the candidate has not earned
promotion.

Aim above the smallest plausible task. Favor cases with meaningful structure:
multiple interacting signals, a realistic but compact environment, a tempting
wrong path, and a scoring setup that can separate adequate from excellent
outputs. Do not create complexity by adding noise, verbosity, irrelevant files,
or confusing wording. Create complexity through causal depth, tradeoffs,
coverage breadth, and non-obvious failure modes.

Avoid safe benchmark templates unless the seed explicitly requires one. In code
debugging, do not default to trivial off-by-one, typo, missing import, wrong
operator, or single visible failing assertion cases. If you use a familiar bug
shape, add a substantive twist: an upstream producer/consumer mismatch, an
invariant that only fails under an edge case, a misleading product constraint,
or an explanation burden that reveals real causal reasoning.

Never leak the answer in candidate-facing material. If the benchmark asks an
agent to infer, diagnose, judge, repair, or discover something, the prompt,
inputs, code comments, labels, filenames, visible outputs, fixtures, and setup
must not reveal or strongly hint at the intended answer. A case that gives away
its own answer cannot be promoted no matter how strong the rubric looks.

Prefer benchmark designs that create pressure toward excellent outputs, not only
filters against bad outputs. The case should make a strong model reveal taste,
judgment, control, strategy, or depth that an adequate model would not show.
Avoid converging on familiar safe templates. If the first design is a standard
constraint-following prompt, improve it before returning it by adding meaningful
tradeoff, transformation, preservation, comparison, revision, or other structure
that creates ceiling pressure.

Hard checks may disqualify bad outputs, but table-stakes compliance is not the same as
high ability. A useful benchmark should create evidence about ability, including signals
that can separate adequate from excellent behavior when the domain supports that.

Return only a case you would be willing to defend as a promoted corpus item under
its stated assumptions and limits.
""".strip()


def _format_generator_guidance(domain: DomainConfig) -> str:
    guidance = domain.generator_guidance
    if not guidance:
        return ""
    parts: list[str] = []
    parts.append("\nDOMAIN-SPECIFIC GENERATOR GUIDANCE")
    _section(parts, "Goal", guidance.get("goal"))
    _section(parts, "Scoring contract standard", guidance.get("scoring_contract_bar"))
    _section(parts, "Proxy claim standard", guidance.get("proxy_claim_bar"))
    _section(parts, "Diagnostic pressure in this domain", guidance.get("diagnostic_pressure_notes"))
    return "\n".join(parts)


def _format_gate_guidance(domain: DomainConfig, rules_attr: str) -> str:
    parts: list[str] = []
    rules = getattr(domain, rules_attr)
    if rules:
        parts.append("\nDOMAIN GATE RULES")
        for rule in rules:
            parts.append(f"  - {rule}")
    _format_probe_principles(parts, domain.general_probe_principles)
    _format_anti_overfit_policy(parts, domain.anti_overfit_policy)
    guidance = domain.generator_guidance
    patterns = guidance.get("common_rejection_patterns", [])
    if patterns:
        parts.append("\nCOMMON REJECTION PATTERNS")
        for p in patterns:
            parts.append(f"  - {p['name']}: {str(p['description']).strip()}")
    return "\n".join(parts)


def _format_probe_principles(parts: list[str], principles: dict[str, Any]) -> None:
    if not principles:
        return
    parts.append("\nGENERAL PROBE PRINCIPLES")
    for name, value in principles.items():
        if not isinstance(value, dict):
            parts.append(f"\n{name}:\n{value}")
            continue
        parts.append(f"\n{name}:")
        for key in ("definition", "test_question", "bad_example", "good_example"):
            if value.get(key):
                parts.append(f"  {key}: {str(value[key]).strip()}")
        shortcuts = value.get("shortcuts", [])
        if shortcuts:
            parts.append("  shortcuts:")
            for shortcut in shortcuts:
                parts.append(f"    - {shortcut}")


def _format_anti_overfit_policy(parts: list[str], policy: list[str]) -> None:
    if not policy:
        return
    parts.append("\nANTI-OVERFIT POLICY")
    for item in policy:
        parts.append(f"  - {item}")


def _section(parts: list[str], title: str, body: Any) -> None:
    if body:
        parts.append(f"\n{title}:\n{str(body).strip()}")


class SampleGenerator:
    role_name = "SampleGenerator"

    def __init__(self, client: OpenAIClient, domain: DomainConfig) -> None:
        self.client = client
        self.domain = domain
        self._system = _GENERATOR_PRINCIPLES + _format_generator_guidance(domain)

    def generate(
        self,
        *,
        run_id: str,
        seed: SeedSpec,
        attempt: int,
        retry_route_code: RouteCode | None = None,
        retry_subcodes: list[str] | None = None,
    ) -> tuple[CandidateSample, dict[str, Any]]:
        system = self._system
        payload: dict[str, Any] = {
            "seed": seed.model_dump(mode="json"),
            "domain": {
                "output_schema": self.domain.output_schema,
                "benchmark_case_schema": self.domain.benchmark_case_schema,
                "abilities": self.domain.abilities,
                "environments": self.domain.environments,
                "diagnostic_pressure_types": self.domain.diagnostic_pressure_types,
                "scoring_methods": self.domain.scoring_methods,
            },
            "required_json_schema": self.domain.output_schema,
            "fallback_required_json_shape": {
                "benchmark_case": {"prompt": "benchmark prompt string", "setup": "optional setup", "inputs": {}, "environment": {}},
                "score_x": {
                    "score_type": "one allowed scoring method",
                    "range": [0, 1],
                    "dimensions": [
                        {
                            "name": "dimension name",
                            "weight": 0.5,
                            "high_score_criterion": "concrete observable behavior in the agent output that earns full credit",
                            "low_score_criterion": "concrete observable behavior in the agent output that earns zero credit",
                        }
                    ],
                },
                "ability_z": {"name": "target ability", "sub_abilities": ["specific sub-ability"]},
                "environment_y": {"name": "target environment", "assumptions": ["assumption"]},
                "proxy_claim": "why score_x should indicate ability_z in environment_y",
                "diagnostic_pressure": ["specific pressure exerted by this case"],
                "scoring_contract": {
                    "credit": ["observable behavior that earns credit"],
                    "penalties": ["shallow or bad behavior that loses credit"],
                    "uncertainty_policy": "when judges should mark uncertainty",
                },
                "leakage_risks": ["how the case or scorer can be gamed"],
                "known_limits": ["what this benchmark case does not prove"],
                "coverage_tags": ["short coverage tags"],
                "negative_controls": [{"output": "known-bad agent output", "should_fail_because": "why score_x should penalize it"}],
            },
        }
        if retry_route_code is not None:
            payload["prior_generation_rejection"] = {
                "route_code": retry_route_code.value,
                "subcodes": _generator_safe_retry_subcodes(retry_subcodes or []),
            }
        user = json.dumps(payload, sort_keys=True)
        payload, meta = self.client.complete_json(system=system, user=user, temperature=0.8)
        content = {
            "seed_id": seed.id,
            "cell": seed.cell.model_dump(),
            "output": payload,
            "benchmark_case": payload.get("benchmark_case", {}),
            "score_x": payload.get("score_x", {}),
            "ability_z": payload.get("ability_z", {}),
            "environment_y": payload.get("environment_y", {}),
            "proxy_claim": payload.get("proxy_claim", ""),
            "diagnostic_pressure": list(payload.get("diagnostic_pressure", [])),
            "scoring_contract": payload.get("scoring_contract", {}),
            "leakage_risks": list(payload.get("leakage_risks", [])),
            "known_limits": list(payload.get("known_limits", [])),
            "coverage_tags": list(payload.get("coverage_tags", [])),
            "negative_controls": list(payload.get("negative_controls", [])),
        }
        candidate = CandidateSample(
            id=f"{run_id}-candidate-{seed.id}-{attempt}",
            seed_id=seed.id,
            content_hash=stable_hash(content),
            cell=seed.cell,
            output=dict(payload),
            benchmark_case=dict(payload.get("benchmark_case", {})),
            score_x=dict(payload.get("score_x", {})),
            ability_z=dict(payload.get("ability_z", {})),
            environment_y=dict(payload.get("environment_y", {})),
            proxy_claim=str(payload.get("proxy_claim", "")),
            diagnostic_pressure=list(payload.get("diagnostic_pressure", [])),
            scoring_contract=dict(payload.get("scoring_contract", {})),
            leakage_risks=list(payload.get("leakage_risks", [])),
            known_limits=list(payload.get("known_limits", [])),
            coverage_tags=list(payload.get("coverage_tags", [])),
            negative_controls=list(payload.get("negative_controls", [])),
            difficulty=seed.cell.difficulty,
            case_type=seed.cell.case_type,
            provenance={"seed_id": seed.id, "generator": self.role_name},
        )
        return candidate, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}


class _GateValidator:
    role_name = "GateValidator"
    check_kind = "semantic"
    system_prompt = ""
    rules_attr = "semantic_rules"

    def __init__(self, client: OpenAIClient, domain: DomainConfig) -> None:
        self.client = client
        self.domain = domain
        self._system = self.system_prompt + _format_gate_guidance(domain, self.rules_attr)

    def validate(self, candidate: CandidateSample) -> tuple[SampleVerdict, dict[str, Any]]:
        system = self._system
        rules = getattr(self.domain, self.rules_attr)
        user = json.dumps(
            {
                "candidate": candidate.model_dump(mode="json"),
                "criteria": {
                    "gate_rules": rules,
                    "general_probe_principles": self.domain.general_probe_principles,
                    "anti_overfit_policy": self.domain.anti_overfit_policy,
                    "route_codes": self.domain.route_codes,
                    "subcodes": self.domain.subcodes,
                    "reason_codes": self.domain.reason_codes,
                },
                "required_json_shape": {
                    "verdict": "accept or reject",
                    "route_code": "accept or reject_semantic_mismatch",
                    "subcodes": ["descriptive labels only"],
                    "reason_codes": ["descriptive labels only"],
                    "evidence": [{"source": "candidate", "path": "field", "value": "short span"}],
                    "rationale": "2-5 sentence public justification for the verdict, citing concrete candidate fields; no hidden chain-of-thought",
                },
            },
            sort_keys=True,
        )
        payload, meta = self.client.complete_json(system=system, user=user, temperature=0.2)
        verdict = _verdict(payload.get("verdict"))
        subcodes = list(payload.get("subcodes", []))
        reason_codes = list(payload.get("reason_codes", []))
        route_code = _route_code(
            payload.get("route_code"),
            default=RouteCode.ACCEPT if verdict == Verdict.ACCEPT else RouteCode.REJECT_SEMANTIC_MISMATCH,
        )
        verdict, route_code, subcodes, reason_codes = _coerce_gate_verdict(
            verdict=verdict,
            route_code=route_code,
            subcodes=subcodes,
            reason_codes=reason_codes,
        )
        sample_verdict = SampleVerdict(
            candidate_id=candidate.id,
            check_kind=self.check_kind,
            verdict=verdict,
            route_code=route_code,
            subcodes=subcodes,
            reason_codes=reason_codes,
            evidence=_evidence(payload.get("evidence", [])),
            rationale=str(payload.get("rationale", "")),
        )
        return sample_verdict, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}


class QualityGate(_GateValidator):
    role_name = "QualityGate"
    check_kind = "quality"
    rules_attr = "quality_gate_rules"
    system_prompt = (
        "You are QualityGate for a benchmark-generation pipeline. "
        "You are an adversarial quality gate. Your job is to prevent weak, toy, low-effort, or benchmark-shaped-but-empty cases from entering the corpus. "
        "Treat every candidate as rejected by default until the concrete artifact earns promotion. "
        "Judge whether the benchmark case is a strong proxy for ability_z in environment_y. "
        "Focus on diagnostic pressure, proxy validity, difficulty, environment relevance, and leakage mitigation. "
        "Inspect the candidate-facing prompt, files, tests, visible symptoms, shallow fixes, negative controls, and scoring contract before trusting any proxy_claim. Treat hidden tests or oracle material as scoring support only, not quality evidence. "
        "Reject if the local artifact is weak even when the wrapper language sounds rigorous. "
        "Reject benchmarks whose core task is an obvious one-line local patch, typo, missing import, trivial off-by-one, simple slice change, literal/operator swap, or direct traceback repair. Hidden tests, oracle text, negative controls, or rubric language do not rescue a toy core task. "
        "Reject if hidden tests or oracle requirements are the only source of difficulty but the candidate-facing requirements do not establish that behavior. "
        "Reject if negative controls are straw men, hidden tests do not discriminate, or weak models can pass through visible-test compliance, pattern matching, mechanical substitution, or generic defensive guards. "
        "Actively inspect all candidate-facing material for answer leakage: prompts, inputs, code comments, labels, filenames, tests, fixtures, visible outputs, and setup. "
        "If the benchmark reveals or strongly hints at the intended answer, root cause, fix, or scoring target, reject it even if the rubric is otherwise strong. "
        "Do not reject merely because scoring is subjective or imperfect; RubricGate judges scoring reliability. "
        "Do not accept merely plausible, tidy, coherent, or low-difficulty cases. "
        "Accept only when success would be meaningful evidence of the claimed ability and failure modes are concretely exercised by the artifact. "
        "Return verdict metadata plus a concise public rationale. Do not reveal hidden chain-of-thought. "
        "Do not rewrite or repair anything. Return JSON only."
    )


class RubricGate(_GateValidator):
    role_name = "RubricGate"
    check_kind = "rubric"
    rules_attr = "rubric_gate_rules"
    system_prompt = (
        "You are RubricGate for a benchmark-generation pipeline. "
        "You are an adversarial scoring gate. Treat the scoring setup as rejected by default until it proves it can grade the artifact reliably. "
        "Judge whether score_x and scoring_contract are reliable enough to grade outputs for this benchmark case. "
        "Focus on observable criteria, negative controls, boundary handling, judge variance, and whether known-bad outputs are penalized. "
        "Inspect whether score_x is grounded in the actual artifact evidence, visible tests, shallow fixes, and patch behavior. "
        "Reject if the rubric rewards benchmark theater, explanation polish, or visible-test passing without executable or inspectable evidence of the claimed ability. "
        "Reject if known-bad or shallow patches could receive high scores, if negative controls are straw men, or if graders must invent missing ground truth. "
        "If candidate-facing materials leak the answer so badly that high scores cannot distinguish true ability from clue-following, reject the scoring setup as unreliable. "
        "Do not reject merely because the benchmark is subjective, hard, or not a perfect proxy; QualityGate judges benchmark quality. "
        "Do not accept vague, permissive, or merely plausible scoring contracts. "
        "Accept only when the scoring contract would reliably punish shallow fixes and reward the intended capability in the actual benchmark artifact. Do not accept a scoring setup that makes a toy benchmark look rigorous by adding hidden oracle machinery. "
        "Return verdict metadata plus a concise public rationale. Do not reveal hidden chain-of-thought. "
        "Do not rewrite or repair anything. Return JSON only."
    )


def _supports_reasoning_effort(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))



def _verdict(value: Any) -> Verdict:
    try:
        return Verdict(str(value).lower())
    except ValueError:
        return Verdict.REJECT


def _route_code(value: Any, *, default: RouteCode) -> RouteCode:
    try:
        return RouteCode(str(value))
    except ValueError:
        return default


_REJECT_SIGNAL_CODES = {
    "weak_proxy_validity",
    "unreliable_score",
    "weak_diagnostic_pressure",
    "shortcut_leakage",
    "vague_scoring_contract",
    "fake_difficulty",
    "irrelevant_environment",
    "ambiguous_success_criteria",
    "overbroad_proxy_claim",
    "missing_known_limits",
    "missing_negative_control",
    "missing_oracle",
    "schema_violation",
    "near_duplicate",
}


_GENERATOR_RETRY_CODE_MAP = {
    "missing_private_oracle": "weak_judge_confidence",
    "missing_oracle": "weak_judge_confidence",
    "private_oracle_integrity": "weak_judge_confidence",
}


def _generator_safe_retry_subcodes(subcodes: list[str]) -> list[str]:
    safe: list[str] = []
    for code in subcodes:
        mapped = _GENERATOR_RETRY_CODE_MAP.get(code, code)
        if mapped not in safe:
            safe.append(mapped)
    return safe


def _coerce_gate_verdict(
    *,
    verdict: Verdict,
    route_code: RouteCode,
    subcodes: list[str],
    reason_codes: list[str],
) -> tuple[Verdict, RouteCode, list[str], list[str]]:
    labels = {str(code) for code in [*subcodes, *reason_codes]}
    reject_labels = sorted(labels & _REJECT_SIGNAL_CODES)
    if verdict == Verdict.ACCEPT and reject_labels:
        merged_subcodes = _dedupe([*subcodes, *reject_labels])
        merged_reason_codes = _dedupe([*reason_codes, *reject_labels])
        route = RouteCode.REJECT_LEAKAGE if "shortcut_leakage" in reject_labels else RouteCode.REJECT_SEMANTIC_MISMATCH
        return Verdict.REJECT, route, merged_subcodes, merged_reason_codes
    return verdict, route_code, subcodes, reason_codes


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _evidence(values: Any) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    if not isinstance(values, list):
        return refs
    for value in values:
        if isinstance(value, dict):
            refs.append(
                EvidenceRef(
                    source=str(value.get("source", "llm")),
                    path=str(value.get("path", "")),
                    value=None if value.get("value") is None else str(value.get("value")),
                )
            )
    return refs
