from __future__ import annotations

import unicodedata
from typing import Any

from jsonschema import Draft202012Validator

from config import DomainConfig
from models import (
    CandidateSample,
    CheckResult,
    EvidenceRef,
    RouteCode,
    SampleVerdict,
    Verdict,
)
from services.environment_validation import validate_environment_artifact
from text_hygiene import find_disallowed_text


def validate_design_batch(designs: list[Any], domain: DomainConfig) -> tuple[Verdict, RouteCode, list[str]]:
    seen: set[str] = set()
    for design in designs:
        cell = design.cell
        if cell.case_type not in domain.case_types:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_case_type"]
        if cell.difficulty not in domain.difficulties:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_difficulty"]
        if cell.scenario not in domain.scenarios:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_scenario"]
        if design.content_hash in seen:
            return Verdict.REJECT, RouteCode.REJECT_DUPLICATE, ["duplicate_design"]
        design_environment_verdict = _validate_design_environment(design, domain)
        if design_environment_verdict is not None:
            return design_environment_verdict
        seen.add(design.content_hash)
    return Verdict.ACCEPT, RouteCode.ACCEPT, []


_CODE_ENV_DESIGN_REQUIRED_FIELDS = {
    "product_context",
    "codebase_shape",
    "state_model",
    "core_invariant",
    "failure_surface",
    "tempting_wrong_fix",
    "actual_causal_region",
    "required_depth",
}

_TOY_CODE_DESIGN_MARKERS = {
    "typo",
    "missing import",
    "off-by-one",
    "one-line",
    "single loop",
    "wrong operator",
}


def _validate_design_environment(design: Any, domain: DomainConfig) -> tuple[Verdict, RouteCode, list[str]] | None:
    if domain.domain_id != "benchmark_code_debug":
        return None

    env_design = getattr(design, "environment_premise", {}) or {}
    if not isinstance(env_design, dict):
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]

    missing = [field for field in sorted(_CODE_ENV_DESIGN_REQUIRED_FIELDS) if not env_design.get(field)]
    if missing:
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]
    list_fields = (
        "diagnostic_pressure",
        "why_weak_agents_fail",
        "tempting_shallow_solutions",
        "success_evidence_required",
        "minimum_depth_requirements",
        "forbidden_shortcuts",
        "non_goals",
    )
    if any(not getattr(design, field, []) for field in list_fields):
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]

    core_text = " ".join(str(env_design.get(field, "")) for field in _CODE_ENV_DESIGN_REQUIRED_FIELDS).lower()
    if any(marker in core_text for marker in _TOY_CODE_DESIGN_MARKERS):
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]

    return None


def deterministic_sample_verdict(candidate: CandidateSample, domain: DomainConfig) -> tuple[SampleVerdict, list[CheckResult]]:
    checks = [
        _text_hygiene_check(candidate),
        _output_schema_check(candidate, domain),
        _schema_check(candidate, domain),
        _taxonomy_check(candidate, domain),
        _benchmark_contract_check(candidate, domain),
        _benchmark_oracle_check(candidate, domain),
    ]
    if domain.domain_id == "benchmark_code_debug":
        checks.insert(3, validate_environment_artifact(candidate, domain))
        checks.insert(4, _candidate_facing_answer_leak_check(candidate))
    failed = [check for check in checks if not check.passed]
    if not failed:
        return (
            SampleVerdict(
                candidate_id=candidate.id,
                check_kind="deterministic",
                verdict=Verdict.ACCEPT,
                route_code=RouteCode.ACCEPT,
            ),
            checks,
        )

    first = failed[0]
    return (
        SampleVerdict(
            candidate_id=candidate.id,
            check_kind="deterministic",
            verdict=Verdict.REJECT,
            route_code=first.route_code,
            subcodes=[code for code in [first.subcode] if code],
            evidence=first.evidence,
        ),
        checks,
    )


def _schema_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    validator = Draft202012Validator(domain.benchmark_case_schema)
    errors = sorted(validator.iter_errors(candidate.agent_artifact.benchmark_case), key=lambda err: err.path)
    if errors:
        error = errors[0]
        return CheckResult(
            check_id="benchmark_case_schema",
            passed=False,
            route_code=RouteCode.REJECT_SCHEMA,
            subcode="schema_violation",
            evidence=[EvidenceRef(source="jsonschema", path=".".join(map(str, error.path)), value=error.message)],
        )
    return CheckResult(check_id="benchmark_case_schema", passed=True)


def _output_schema_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if not domain.output_schema:
        return CheckResult(check_id="output_schema", passed=True)
    validator = Draft202012Validator(domain.output_schema)
    errors = sorted(validator.iter_errors(candidate.output), key=lambda err: err.path)
    if errors:
        error = errors[0]
        return CheckResult(
            check_id="output_schema",
            passed=False,
            route_code=RouteCode.REJECT_SCHEMA,
            subcode="schema_violation",
            evidence=[
                EvidenceRef(
                    source="jsonschema",
                    path="output." + ".".join(map(str, error.path)),
                    value=error.message,
                )
            ],
        )
    return CheckResult(check_id="output_schema", passed=True)


def _text_hygiene_check(candidate: CandidateSample) -> CheckResult:
    issue = find_disallowed_text(candidate.model_dump(mode="json"))
    if issue is None:
        return CheckResult(check_id="text_hygiene", passed=True)
    path, char = issue
    codepoint = f"U+{ord(char):04X}"
    name = unicodedata.name(char, "UNKNOWN")
    return CheckResult(
        check_id="text_hygiene",
        passed=False,
        route_code=RouteCode.REJECT_SCHEMA,
        subcode="malformed_text",
        evidence=[EvidenceRef(source="deterministic_rule", path=path, value=f"{codepoint} {name}")],
    )


_ANSWER_LEAK_PATTERNS = (
    "bug:",
    "bug note",
    "bug (intentional",
    "bug intentionally",
    "intentionally buggy",
    "intentional bug",
    "root cause",
    "source of the bug",
    "source of this bug",
    "the intended fix",
    "exact fix",
    "faulty line",
    "current code incorrectly",
    "currently incorrectly",
    "incorrectly only",
    "should be keyed",
    "should use",
    "should update",
    "should set",
    "should invalidate",
    "before the transaction commits",
    "before tx.commit",
    "before committing",
    "before commit",
    "after commit",
    "causal chain",
    "causal interleaving",
    "hidden causal chain",
    "intended invariant",
    "intended correct",
    "intended behavior",
    "starter code",
    "starter workspace",
    "your fix should",
    "the task for",
    "acceptable solutions",
    "naive fixes",
    "shallow fixes",
    "regression test to catch",
    "detect naive",
    "benchmark",
)


def _candidate_facing_answer_leak_check(candidate: CandidateSample) -> CheckResult:
    for path, value in _candidate_facing_texts(candidate):
        lowered = value.lower()
        for pattern in _ANSWER_LEAK_PATTERNS:
            if pattern in lowered:
                return CheckResult(
                    check_id="answer_leakage",
                    passed=False,
                    route_code=RouteCode.REJECT_LEAKAGE,
                    subcode="answer_leak_in_candidate_materials",
                    evidence=[
                        EvidenceRef(
                            source="deterministic_rule",
                            path=path,
                            value=f"candidate-facing text contains answer-leak marker: {pattern}",
                        )
                    ],
                )
    return CheckResult(check_id="answer_leakage", passed=True)


def _candidate_facing_texts(candidate: CandidateSample) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    benchmark_case = candidate.agent_artifact.benchmark_case
    for key in ("prompt", "setup"):
        value = benchmark_case.get(key)
        if isinstance(value, str):
            texts.append((f"benchmark_case.{key}", value))
    environment = benchmark_case.get("environment")
    if isinstance(environment, dict):
        for key, value in environment.items():
            if isinstance(value, str):
                texts.append((f"benchmark_case.environment.{key}", value))

    artifact = candidate.agent_artifact.environment_artifact
    if artifact is not None and artifact.kind == "virtual_workspace":
        files = artifact.payload.get("files", [])
        if isinstance(files, list):
            for index, file_entry in enumerate(files):
                if not isinstance(file_entry, dict):
                    continue
                file_path = file_entry.get("path")
                content = file_entry.get("content")
                if isinstance(file_path, str):
                    texts.append((f"environment_artifact.payload.files.{index}.path", file_path))
                if isinstance(content, str):
                    texts.append((f"environment_artifact.payload.files.{index}.content", content))
    return texts


def _taxonomy_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if candidate.cell.case_type not in domain.case_types:
        return CheckResult(
            check_id="taxonomy_cell",
            passed=False,
            route_code=RouteCode.REJECT_COVERAGE_MISMATCH,
            subcode="unknown_case_type",
        )
    if candidate.cell.scenario not in domain.scenarios:
        return CheckResult(
            check_id="taxonomy_cell",
            passed=False,
            route_code=RouteCode.REJECT_COVERAGE_MISMATCH,
            subcode="unknown_scenario",
        )
    return CheckResult(check_id="taxonomy_cell", passed=True)


def _benchmark_contract_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    judge = candidate.judge_artifact
    if len(judge.proxy_claim.strip()) < int(domain.deterministic_rules.get("min_proxy_claim_chars", 1)):
        return _failed_contract("weak_proxy_validity", "proxy_claim is missing or too short")
    if len(judge.diagnostic_pressure) < int(domain.deterministic_rules.get("min_diagnostic_pressure_items", 1)):
        return _failed_contract("weak_diagnostic_pressure", "diagnostic_pressure has too few items")
    if len(judge.leakage_risks) < int(domain.deterministic_rules.get("min_leakage_risk_items", 0)):
        return _failed_contract("shortcut_leakage", "leakage_risks is missing")
    if len(judge.known_limits) < int(domain.deterministic_rules.get("min_known_limit_items", 0)):
        return _failed_contract("missing_known_limits", "known_limits is missing")
    if bool(domain.deterministic_rules.get("require_negative_control", False)) and not judge.negative_controls:
        return _failed_contract("missing_negative_control", "negative_controls is required")
    if not judge.score_x.get("score_type"):
        return _failed_contract("unreliable_score", "score_x.score_type is missing")
    if not judge.score_x.get("dimensions"):
        return _failed_contract("unreliable_score", "score_x.dimensions is missing")
    for dim in judge.score_x.get("dimensions", []):
        if not dim.get("high_score_criterion") or not dim.get("low_score_criterion"):
            return _failed_contract("vague_scoring_contract", f"dimension '{dim.get('name', '?')}' is missing high_score_criterion or low_score_criterion")
    if not judge.scoring_contract.get("credit") or not judge.scoring_contract.get("penalties"):
        return _failed_contract("vague_scoring_contract", "scoring_contract needs credit and penalties")
    if not candidate.ability_z.get("name"):
        return _failed_contract("weak_proxy_validity", "ability_z.name is missing")
    if not candidate.environment_y.get("name"):
        return _failed_contract("irrelevant_environment", "environment_y.name is missing")
    return CheckResult(check_id="benchmark_contract", passed=True)


def _benchmark_oracle_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if not bool(domain.deterministic_rules.get("require_benchmark_oracle", False)):
        return CheckResult(check_id="benchmark_oracle", passed=True)
    oracle = candidate.agent_artifact.benchmark_case.get("oracle")
    if not isinstance(oracle, dict):
        return _failed_oracle("benchmark_case.oracle is required")
    expected = oracle.get("expected_repair_characteristics")
    if not expected:
        return _failed_oracle("oracle.expected_repair_characteristics is required")
    hidden_tests = oracle.get("hidden_tests", [])
    if not isinstance(hidden_tests, list) or len(hidden_tests) < int(domain.deterministic_rules.get("min_hidden_tests", 1)):
        return _failed_oracle("oracle.hidden_tests has too few items")
    shallow_fix_failures = oracle.get("shallow_fix_failures", [])
    if not isinstance(shallow_fix_failures, list) or len(shallow_fix_failures) < int(domain.deterministic_rules.get("min_shallow_fix_failures", 1)):
        return _failed_oracle("oracle.shallow_fix_failures has too few items")
    return CheckResult(check_id="benchmark_oracle", passed=True)


def _failed_contract(subcode: str, value: str) -> CheckResult:
    route = RouteCode.REJECT_LEAKAGE if subcode == "shortcut_leakage" else RouteCode.REJECT_CRITERIA_MISMATCH
    return CheckResult(
        check_id="benchmark_contract",
        passed=False,
        route_code=route,
        subcode=subcode,
        evidence=[EvidenceRef(source="deterministic_rule", path="candidate", value=value)],
    )


def _failed_oracle(value: str) -> CheckResult:
    return CheckResult(
        check_id="benchmark_oracle",
        passed=False,
        route_code=RouteCode.REJECT_CRITERIA_MISMATCH,
        subcode="missing_oracle",
        evidence=[EvidenceRef(source="deterministic_rule", path="benchmark_case.oracle", value=value)],
    )
