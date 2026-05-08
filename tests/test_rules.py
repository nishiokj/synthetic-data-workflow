from __future__ import annotations

from config import load_domain
from models import CandidateSample, DesignBrief, TaxonomyCell, Verdict
from rules import deterministic_sample_verdict, validate_design_batch


def _candidate(**overrides) -> CandidateSample:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=3, scenario="adversarial")
    values = {
        "id": "candidate-1",
        "design_id": "design-1",
        "content_hash": "abc",
        "cell": cell,
        "benchmark_case": {"prompt": "Write a haiku that evokes layoffs as late autumn without saying work, loss, leaves, cold, or endings."},
        "score_x": {
            "score_type": "hard_checks_plus_rubric",
            "dimensions": [{"name": "constraint_adherence", "weight": 0.4, "high_score_criterion": "Output contains none of the forbidden terms and uses indirect imagery.", "low_score_criterion": "Output uses one or more forbidden terms directly."}],
        },
        "ability_z": {"name": "constrained_poetic_generation", "sub_abilities": ["metaphorical_transfer"]},
        "environment_y": {"name": "single_turn_creative_writing", "assumptions": ["No tools"]},
        "proxy_claim": "A model that succeeds here is showing more than line-count compliance because it must preserve emotional intent while avoiding the obvious vocabulary and imagery that would make a template answer pass superficially.",
        "diagnostic_pressure": ["forbids obvious imagery", "requires emotional transfer"],
        "scoring_contract": {
            "credit": ["preserves emotional intent", "obeys forbidden-word constraints"],
            "penalties": ["generic seasonal template", "mentions the source domain directly"],
            "uncertainty_policy": "Mark uncertainty when taste and constraint adherence conflict.",
        },
        "leakage_risks": ["A compliant but lifeless template may receive too much credit."],
        "known_limits": ["The case does not prove broad poetic taste."],
        "coverage_tags": ["anti_template", "emotional_transfer"],
        "negative_controls": [{"output": "dead leaves fall at work", "should_fail_because": "uses forbidden imagery and source domain"}],
        "difficulty": 3,
        "case_type": "proxy_strong",
    }
    values.update(overrides)
    return CandidateSample(**values)


def _code_design(**overrides) -> DesignBrief:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=4, scenario="edge")
    values = {
        "design_id": "code-design-1",
        "cell": cell,
        "target_ability": "fault_localization",
        "target_environment": "single_turn_debug_with_test",
        "design_intent": "Create a compact benchmark around a realistic stateful debugging failure.",
        "environment_premise": {
            "product_context": "billing reconciliation worker for subscription invoices",
            "codebase_shape": "parser, reconciliation engine, export adapter, and focused tests",
            "state_model": "invoice rows move through parsed, matched, adjusted, and exported states",
            "core_invariant": "recognized revenue totals must remain stable per billing period after adjustments",
            "failure_surface": "monthly summary is wrong only for partial refunds near timezone boundaries",
            "tempting_wrong_fix": "round the exported total or patch the summary formatter",
            "actual_causal_region": "refund normalization before grouping by billing period",
            "required_depth": "requires tracing a transformed value across parser, normalizer, and summarizer",
        },
        "failure_mode_family": "misleading downstream aggregate caused by upstream normalization",
        "diagnostic_pressure": ["misleading output error with upstream normalization cause"],
        "why_weak_agents_fail": ["they patch the formatter or rounded total without preserving the period invariant"],
        "tempting_shallow_solutions": ["formatter-only patch passes visible symptom without preserving invariant"],
        "success_evidence_required": ["trace transformed values across parser, normalizer, and summarizer"],
        "minimum_depth_requirements": ["requires at least two modules and a state invariant"],
        "forbidden_shortcuts": ["one-line arithmetic boundary patch", "test edit", "formatter mask"],
        "non_goals": ["typo", "missing import", "one-line loop patch"],
    }
    values.update(overrides)
    return DesignBrief.create(**values)


def test_benchmark_candidate_passes_deterministic_rules() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, _ = deterministic_sample_verdict(_candidate(), domain)

    assert verdict.verdict == Verdict.ACCEPT


def test_missing_negative_control_rejected() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, _ = deterministic_sample_verdict(_candidate(negative_controls=[]), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["schema_violation"]


def test_output_schema_violation_rejected() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, checks = deterministic_sample_verdict(_candidate(output={"benchmark_case": {"prompt": "too thin"}}), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["schema_violation"]
    assert checks[1].check_id == "output_schema"


def test_vague_scoring_contract_rejected() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, _ = deterministic_sample_verdict(_candidate(scoring_contract={"credit": ["good output"]}), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["schema_violation"]


def test_malformed_control_text_rejected_before_quality_gate() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, checks = deterministic_sample_verdict(
        _candidate(leakage_risks=["Rare-word stuffing that appears clich\u0000e9 but empty."]),
        domain,
    )

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["malformed_text"]
    assert checks[0].check_id == "text_hygiene"


def test_code_domain_does_not_require_oracle_during_generation_experiment() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")

    verdict, checks = deterministic_sample_verdict(_candidate(), domain)

    assert verdict.verdict == Verdict.ACCEPT
    assert checks[-1].check_id == "benchmark_oracle"


def test_code_domain_accepts_candidate_with_oracle() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = {
        "prompt": "Debug this compact Python case and provide a minimal patch with explanation.",
        "oracle": {
            "expected_repair_characteristics": ["producer-side minimal fix", "preserves public API"],
            "hidden_tests": [
                {"name": "edge_one", "input": "case A", "expected": "passes only with causal fix"},
                {"name": "edge_two", "input": "case B", "expected": "rejects shallow guard"},
            ],
            "shallow_fix_failures": [
                {"fix": "consumer-side masking", "fails_because": "does not repair producer invariant"},
                {"fix": "broad try/except", "fails_because": "swallows unrelated failures"},
            ],
        },
    }

    verdict, checks = deterministic_sample_verdict(_candidate(benchmark_case=benchmark_case), domain)

    assert verdict.verdict == Verdict.ACCEPT
    assert checks[-1].check_id == "benchmark_oracle"


def test_code_design_requires_environment_premise_depth() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    design = _code_design(environment_premise={})

    verdict, route_code, subcodes = validate_design_batch([design], domain)

    assert verdict == Verdict.REJECT
    assert route_code.value == "reject_criteria_mismatch"
    assert subcodes == ["weak_diagnostic_pressure"]


def test_code_design_rejects_toy_core_blueprint() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    design = _code_design(
        environment_premise={
            "product_context": "small list utility",
            "codebase_shape": "one module and one test file",
            "state_model": "single list input and output",
            "core_invariant": "return all items in order",
            "failure_surface": "one visible test fails",
            "tempting_wrong_fix": "change one loop bound",
            "actual_causal_region": "off-by-one in a single loop",
            "required_depth": "one-line patch",
            "non_goals": ["larger system debugging"],
        }
    )

    verdict, route_code, subcodes = validate_design_batch([design], domain)

    assert verdict == Verdict.REJECT
    assert route_code.value == "reject_criteria_mismatch"
    assert subcodes == ["weak_diagnostic_pressure"]


def test_code_design_accepts_environment_premise_blueprint() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")

    verdict, route_code, subcodes = validate_design_batch([_code_design()], domain)

    assert verdict == Verdict.ACCEPT
    assert route_code.value == "accept"
    assert subcodes == []
