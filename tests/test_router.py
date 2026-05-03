from __future__ import annotations

from models import RouteCode, StageKind, Verdict
from router import route_after


def test_validation_reject_routes_back_to_generation_before_ceiling() -> None:
    decision = route_after(
        run_id="run",
        from_stage=StageKind.VALIDATION,
        verdict=Verdict.REJECT,
        route_code=RouteCode.REJECT_SEMANTIC_MISMATCH,
        retry_index=0,
        max_generation_retries=2,
    )

    assert decision.next_stage == StageKind.GENERATION
    assert decision.terminal is False


def test_validation_reject_drops_after_ceiling() -> None:
    decision = route_after(
        run_id="run",
        from_stage=StageKind.VALIDATION,
        verdict=Verdict.REJECT,
        route_code=RouteCode.REJECT_SEMANTIC_MISMATCH,
        retry_index=2,
        max_generation_retries=2,
    )

    assert decision.next_stage is None
    assert decision.route_code == RouteCode.DROP_RETRY_EXHAUSTED
    assert decision.terminal is True

