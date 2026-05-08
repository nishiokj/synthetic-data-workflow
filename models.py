from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class StageKind(str, Enum):
    DESIGN = "design"
    DESIGN_AUDIT = "design_audit"
    GENERATION = "generation"
    VALIDATION = "validation"
    CURATION = "curation"


class AgentRole(str, Enum):
    DESIGNER = "designer"
    DESIGN_AUDITOR = "design_auditor"
    SAMPLE_GENERATOR = "sample_generator"
    ADVERSARY = "adversary"
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
    value: Optional[str] = None


class CheckResult(BaseModel):
    check_id: str
    passed: bool
    route_code: RouteCode = RouteCode.ACCEPT
    subcode: Optional[str] = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class InnerInput(BaseModel):
    question: str
    claimed_answer: str
    context: Optional[str] = None


class InnerCriteria(BaseModel):
    rules: list[str]
    requires_context: bool = False


class DesignBrief(BaseModel):
    id: str
    target_stage: Literal["benchmark"] = "benchmark"
    cell: TaxonomyCell
    target_ability: str
    target_environment: str
    design_intent: str
    environment_premise: dict[str, Any] = Field(default_factory=dict)
    failure_mode_family: str
    diagnostic_pressure: list[str] = Field(default_factory=list)
    why_weak_agents_fail: list[str] = Field(default_factory=list)
    tempting_shallow_solutions: list[str] = Field(default_factory=list)
    success_evidence_required: list[str] = Field(default_factory=list)
    minimum_depth_requirements: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    parent_design_batch_id: Optional[str] = None
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        design_id: str,
        cell: TaxonomyCell,
        target_ability: str,
        target_environment: str,
        design_intent: str,
        environment_premise: Optional[dict[str, Any]] = None,
        failure_mode_family: str,
        diagnostic_pressure: list[str],
        why_weak_agents_fail: list[str],
        tempting_shallow_solutions: list[str],
        success_evidence_required: list[str],
        minimum_depth_requirements: list[str],
        forbidden_shortcuts: list[str],
        non_goals: list[str],
        parent_design_batch_id: Optional[str] = None,
    ) -> "DesignBrief":
        content_hash = stable_hash(
            {
                "target_stage": "benchmark",
                "cell": cell.model_dump(),
                "target_ability": target_ability,
                "target_environment": target_environment,
                "design_intent": design_intent,
                "environment_premise": environment_premise or {},
                "failure_mode_family": failure_mode_family,
                "diagnostic_pressure": diagnostic_pressure,
                "why_weak_agents_fail": why_weak_agents_fail,
                "tempting_shallow_solutions": tempting_shallow_solutions,
                "success_evidence_required": success_evidence_required,
                "minimum_depth_requirements": minimum_depth_requirements,
                "forbidden_shortcuts": forbidden_shortcuts,
                "non_goals": non_goals,
                "parent_design_batch_id": parent_design_batch_id,
            }
        )
        return cls(
            id=design_id,
            cell=cell,
            target_ability=target_ability,
            target_environment=target_environment,
            design_intent=design_intent,
            environment_premise=environment_premise or {},
            failure_mode_family=failure_mode_family,
            diagnostic_pressure=diagnostic_pressure,
            why_weak_agents_fail=why_weak_agents_fail,
            tempting_shallow_solutions=tempting_shallow_solutions,
            success_evidence_required=success_evidence_required,
            minimum_depth_requirements=minimum_depth_requirements,
            forbidden_shortcuts=forbidden_shortcuts,
            non_goals=non_goals,
            parent_design_batch_id=parent_design_batch_id,
            content_hash=content_hash,
        )


class DesignVerdict(BaseModel):
    design_id: str
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str = ""


class CandidateSample(BaseModel):
    id: str
    design_id: str
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


class AdversaryReport(BaseModel):
    candidate_id: str
    revision_disposition: Literal["pass", "revise", "nuke"] = "revise"
    disposition_rationale: str = ""
    attack_summary: str = ""
    attacks: list[dict[str, Any]] = Field(default_factory=list)
    cheap_pass_strategy: str = ""
    proxy_damage: str = ""
    survival_requirements: list[str] = Field(default_factory=list)


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
    nn_distance: Optional[float]
    taxonomy_cell: TaxonomyCell


class RoutingDecision(BaseModel):
    run_id: str
    from_stage: StageKind
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    next_stage: Optional[StageKind]
    context_policy: ContextPolicy
    retry_index: int
    attempt_of: Optional[str] = None
    terminal: bool = False


class StageRecord(BaseModel):
    run_id: str
    stage_id: str
    role: str
    stage_kind: StageKind
    agent_role: Optional[AgentRole] = None
    parent_artifact_id: Optional[str] = None
    artifact_id: str
    model: str
    provider: str
    prompt_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    reasoning_effort: Optional[str] = None
    text_normalization_replacements: int = 0
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    criteria_hash: str
    context_policy: ContextPolicy
    retry_index: int = 0
    attempt_of: Optional[str] = None
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
