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
from text_hygiene import find_disallowed_text


def validate_seed_plan(seeds: list[Any], domain: DomainConfig) -> tuple[Verdict, RouteCode, list[str]]:
    seen: set[str] = set()
    for seed in seeds:
        cell = seed.cell
        if cell.case_type not in domain.case_types:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_case_type"]
        if cell.difficulty not in domain.difficulties:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_difficulty"]
        if cell.scenario not in domain.scenarios:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_scenario"]
        if seed.content_hash in seen:
            return Verdict.REJECT, RouteCode.REJECT_DUPLICATE, ["duplicate_seed"]
        seed_environment_verdict = _validate_seed_environment(seed, domain)
        if seed_environment_verdict is not None:
            return seed_environment_verdict
        seen.add(seed.content_hash)
    return Verdict.ACCEPT, RouteCode.ACCEPT, []


_CODE_ENV_SEED_REQUIRED_FIELDS = {
    "product_context",
    "codebase_shape",
    "state_model",
    "core_invariant",
    "failure_surface",
    "tempting_wrong_fix",
    "actual_causal_region",
    "required_depth",
    "non_goals",
}

_TOY_CODE_SEED_MARKERS = {
    "typo",
    "missing import",
    "off-by-one",
    "one-line",
    "single loop",
    "wrong operator",
}


def _validate_seed_environment(seed: Any, domain: DomainConfig) -> tuple[Verdict, RouteCode, list[str]] | None:
    if domain.domain_id != "benchmark_code_debug":
        return None

    env_seed = getattr(seed, "environment_seed", {}) or {}
    if not isinstance(env_seed, dict):
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]

    missing = [field for field in sorted(_CODE_ENV_SEED_REQUIRED_FIELDS) if not env_seed.get(field)]
    if missing:
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]

    core_fields = _CODE_ENV_SEED_REQUIRED_FIELDS - {"non_goals"}
    core_text = " ".join(str(env_seed.get(field, "")) for field in core_fields).lower()
    if any(marker in core_text for marker in _TOY_CODE_SEED_MARKERS):
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
            reason_codes=[code for code in [first.subcode] if code],
            evidence=first.evidence,
        ),
        checks,
    )


def _schema_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    validator = Draft202012Validator(domain.benchmark_case_schema)
    errors = sorted(validator.iter_errors(candidate.benchmark_case), key=lambda err: err.path)
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
    if len(candidate.proxy_claim.strip()) < int(domain.deterministic_rules.get("min_proxy_claim_chars", 1)):
        return _failed_contract("weak_proxy_validity", "proxy_claim is missing or too short")
    if len(candidate.diagnostic_pressure) < int(domain.deterministic_rules.get("min_diagnostic_pressure_items", 1)):
        return _failed_contract("weak_diagnostic_pressure", "diagnostic_pressure has too few items")
    if len(candidate.leakage_risks) < int(domain.deterministic_rules.get("min_leakage_risk_items", 0)):
        return _failed_contract("shortcut_leakage", "leakage_risks is missing")
    if len(candidate.known_limits) < int(domain.deterministic_rules.get("min_known_limit_items", 0)):
        return _failed_contract("missing_known_limits", "known_limits is missing")
    if bool(domain.deterministic_rules.get("require_negative_control", False)) and not candidate.negative_controls:
        return _failed_contract("missing_negative_control", "negative_controls is required")
    if not candidate.score_x.get("score_type"):
        return _failed_contract("unreliable_score", "score_x.score_type is missing")
    if not candidate.score_x.get("dimensions"):
        return _failed_contract("unreliable_score", "score_x.dimensions is missing")
    for dim in candidate.score_x.get("dimensions", []):
        if not dim.get("high_score_criterion") or not dim.get("low_score_criterion"):
            return _failed_contract("vague_scoring_contract", f"dimension '{dim.get('name', '?')}' is missing high_score_criterion or low_score_criterion")
    if not candidate.scoring_contract.get("credit") or not candidate.scoring_contract.get("penalties"):
        return _failed_contract("vague_scoring_contract", "scoring_contract needs credit and penalties")
    if not candidate.ability_z.get("name"):
        return _failed_contract("weak_proxy_validity", "ability_z.name is missing")
    if not candidate.environment_y.get("name"):
        return _failed_contract("irrelevant_environment", "environment_y.name is missing")
    return CheckResult(check_id="benchmark_contract", passed=True)


def _benchmark_oracle_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if not bool(domain.deterministic_rules.get("require_benchmark_oracle", False)):
        return CheckResult(check_id="benchmark_oracle", passed=True)
    oracle = candidate.benchmark_case.get("oracle")
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
