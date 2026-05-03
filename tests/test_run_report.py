from __future__ import annotations

from run_report import _stage_is_current


def test_stage_is_current_for_benchmark_gate_roles() -> None:
    assert _stage_is_current({"role": "quality_gate_candidate"})
    assert _stage_is_current({"role": "rubric_gate_candidate"})
    assert _stage_is_current({"agent_role": "sample_generator"})


def test_stage_is_current_rejects_legacy_shape() -> None:
    assert not _stage_is_current({"role": "validate_candidate_semantically", "reason_codes": ["accept_clean"]})
