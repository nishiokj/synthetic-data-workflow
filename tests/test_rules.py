from __future__ import annotations

from config import load_domain
from models import CandidateSample, TaxonomyCell, Verdict
from rules import deterministic_sample_verdict


def _candidate(**overrides) -> CandidateSample:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=3, scenario="adversarial")
    values = {
        "id": "candidate-1",
        "seed_id": "seed-1",
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
