from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class StageKind(str, Enum):
    STRATEGY = "strategy"
    PLAN_AUDIT = "plan_audit"
    GENERATION = "generation"
    VALIDATION = "validation"
    CURATION = "curation"


class AgentRole(str, Enum):
    STRATEGIST = "strategist"
    PLAN_AUDITOR = "plan_auditor"
    SAMPLE_GENERATOR = "sample_generator"
    SEMANTIC_VALIDATOR = "semantic_validator"
    QUALITY_GATE = "quality_gate"
    RUBRIC_GATE = "rubric_gate"


class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


class RouteCode(str, Enum):
    ACCEPT = "accept"
    REJECT_CRITERIA_MISMATCH = "reject_criteria_mismatch"
    REJECT_SCHEMA = "reject_schema"
    REJECT_LEAKAGE = "reject_leakage"
    REJECT_DUPLICATE = "reject_duplicate"
    REJECT_COVERAGE_MISMATCH = "reject_coverage_mismatch"
    REJECT_SEMANTIC_MISMATCH = "reject_semantic_mismatch"
    REJECT_UPSTREAM_INVARIANT = "reject_upstream_invariant"
    RETRY_INFRA = "retry_infra"
    RETRY_PARSE = "retry_parse"
    RETRY_PROVIDER_EMPTY = "retry_provider_empty"
    DROP_RETRY_EXHAUSTED = "drop_retry_exhausted"
    DROP_TIMEOUT = "drop_timeout"
    DROP_POLICY_CEILING = "drop_policy_ceiling"


class ContextPolicy(str, Enum):
    FRESH = "fresh"
    SAME_INPUT_RETRY = "same_input_retry"
    CRITERIA_ONLY = "criteria_only"
    ROUTE_CODE_ONLY = "route_code_only"
    CRITERIA_PLUS_ROUTE_CODE = "criteria_plus_route_code"


class TaxonomyCell(BaseModel):
    case_type: str
    difficulty: int = Field(ge=1, le=5)
    scenario: str

    def key(self) -> str:
        return f"{self.case_type}|{self.difficulty}|{self.scenario}"


class EvidenceRef(BaseModel):
    source: str
    path: str
    value: str | None = None


class CheckResult(BaseModel):
    check_id: str
    passed: bool
    route_code: RouteCode = RouteCode.ACCEPT
    subcode: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class InnerInput(BaseModel):
    question: str
    claimed_answer: str
    context: str | None = None


class InnerCriteria(BaseModel):
    rules: list[str]
    requires_context: bool = False


class SeedSpec(BaseModel):
    id: str
    target_stage: Literal["benchmark"] = "benchmark"
    cell: TaxonomyCell
    intent: str
    ability: str
    environment: str
    environment_seed: dict[str, Any] = Field(default_factory=dict)
    diagnostic_pressure: str
    scoring_strategy: str
    leakage_risk: str
    parent_plan_id: str | None = None
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        seed_id: str,
        cell: TaxonomyCell,
        intent: str,
        ability: str,
        environment: str,
        environment_seed: dict[str, Any] | None = None,
        diagnostic_pressure: str,
        scoring_strategy: str,
        leakage_risk: str,
        parent_plan_id: str | None = None,
    ) -> "SeedSpec":
        content_hash = stable_hash(
            {
                "target_stage": "benchmark",
                "cell": cell.model_dump(),
                "intent": intent,
                "ability": ability,
                "environment": environment,
                "environment_seed": environment_seed or {},
                "diagnostic_pressure": diagnostic_pressure,
                "scoring_strategy": scoring_strategy,
                "leakage_risk": leakage_risk,
                "parent_plan_id": parent_plan_id,
            }
        )
        return cls(
            id=seed_id,
            cell=cell,
            intent=intent,
            ability=ability,
            environment=environment,
            environment_seed=environment_seed or {},
            diagnostic_pressure=diagnostic_pressure,
            scoring_strategy=scoring_strategy,
            leakage_risk=leakage_risk,
            parent_plan_id=parent_plan_id,
            content_hash=content_hash,
        )


class PlanVerdict(BaseModel):
    seed_id: str
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str = ""


class CandidateSample(BaseModel):
    id: str
    seed_id: str
    content_hash: str
    cell: TaxonomyCell
    output: dict[str, Any] = Field(default_factory=dict)
    benchmark_case: dict[str, Any]
    score_x: dict[str, Any]
    ability_z: dict[str, Any]
    environment_y: dict[str, Any]
    proxy_claim: str
    diagnostic_pressure: list[str] = Field(default_factory=list)
    scoring_contract: dict[str, Any]
    leakage_risks: list[str] = Field(default_factory=list)
    known_limits: list[str] = Field(default_factory=list)
    coverage_tags: list[str] = Field(default_factory=list)
    negative_controls: list[dict[str, Any]] = Field(default_factory=list)
    difficulty: int = Field(ge=1, le=5)
    case_type: str
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def difficulty_matches_cell(self) -> "CandidateSample":
        if self.difficulty != self.cell.difficulty:
            raise ValueError("difficulty must match taxonomy cell difficulty")
        if self.case_type != self.cell.case_type:
            raise ValueError("case_type must match taxonomy cell case_type")
        if not self.output:
            self.output = {
                "benchmark_case": self.benchmark_case,
                "score_x": self.score_x,
                "ability_z": self.ability_z,
                "environment_y": self.environment_y,
                "proxy_claim": self.proxy_claim,
                "diagnostic_pressure": self.diagnostic_pressure,
                "scoring_contract": self.scoring_contract,
                "leakage_risks": self.leakage_risks,
                "known_limits": self.known_limits,
                "coverage_tags": self.coverage_tags,
                "negative_controls": self.negative_controls,
            }
        return self


class SampleVerdict(BaseModel):
    candidate_id: str
    check_kind: Literal["deterministic", "semantic", "quality", "rubric", "curation"]
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str = ""


class CertifiedSample(BaseModel):
    id: str
    candidate_id: str
    content_hash: str
    candidate: CandidateSample
    deterministic_checks: list[CheckResult]
    semantic_checks: list[SampleVerdict]


class CommittedSample(BaseModel):
    id: str
    certified_id: str
    content_hash: str
    candidate: CandidateSample
    deterministic_checks: list[CheckResult]
    semantic_checks: list[SampleVerdict]
    embedding_ref: str
    nn_distance: float | None
    taxonomy_cell: TaxonomyCell


class RoutingDecision(BaseModel):
    run_id: str
    from_stage: StageKind
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    next_stage: StageKind | None
    context_policy: ContextPolicy
    retry_index: int
    attempt_of: str | None = None
    terminal: bool = False


class StageRecord(BaseModel):
    run_id: str
    stage_id: str
    role: str
    stage_kind: StageKind
    agent_role: AgentRole | None = None
    parent_artifact_id: str | None = None
    artifact_id: str
    model: str
    provider: str
    prompt_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    reasoning_effort: str | None = None
    text_normalization_replacements: int = 0
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    criteria_hash: str
    context_policy: ContextPolicy
    retry_index: int = 0
    attempt_of: str | None = None
    wallclock_ts: str = Field(default_factory=utc_now_iso)


class StageResult(BaseModel):
    artifact: Any
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    model: str = "none"
    provider: str = "local"
    prompt_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
