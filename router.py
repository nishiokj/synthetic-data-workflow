from __future__ import annotations

from models import ContextPolicy, RouteCode, RoutingDecision, StageKind, Verdict


def route_after(
    *,
    run_id: str,
    from_stage: StageKind,
    verdict: Verdict,
    route_code: RouteCode,
    retry_index: int = 0,
    max_design_retries: int = 2,
    max_generation_retries: int = 2,
    subcodes: list[str] | None = None,
    reason_codes: list[str] | None = None,
    attempt_of: str | None = None,
) -> RoutingDecision:
    subcodes = subcodes or []
    reason_codes = reason_codes or []

    next_stage: StageKind | None = None
    context_policy = ContextPolicy.FRESH
    terminal = False
    final_route = route_code
    final_verdict = verdict

    if from_stage == StageKind.DESIGN:
        next_stage = StageKind.DESIGN_AUDIT if verdict == Verdict.ACCEPT else None
        terminal = verdict == Verdict.REJECT

    elif from_stage == StageKind.DESIGN_AUDIT:
        if verdict == Verdict.ACCEPT:
            next_stage = StageKind.GENERATION
            context_policy = ContextPolicy.CRITERIA_ONLY
        elif retry_index < max_design_retries:
            next_stage = StageKind.DESIGN
            context_policy = ContextPolicy.CRITERIA_PLUS_ROUTE_CODE
        else:
            terminal = True
            next_stage = None
            final_route = RouteCode.DROP_RETRY_EXHAUSTED

    elif from_stage == StageKind.GENERATION:
        if verdict == Verdict.ACCEPT:
            next_stage = StageKind.VALIDATION
            context_policy = ContextPolicy.CRITERIA_ONLY
        elif route_code in {
            RouteCode.RETRY_INFRA,
            RouteCode.RETRY_PARSE,
            RouteCode.RETRY_PROVIDER_EMPTY,
        } and retry_index < max_generation_retries:
            next_stage = StageKind.GENERATION
            context_policy = ContextPolicy.SAME_INPUT_RETRY
        else:
            terminal = True
            next_stage = None
            final_route = RouteCode.DROP_RETRY_EXHAUSTED

    elif from_stage == StageKind.VALIDATION:
        if verdict == Verdict.ACCEPT:
            next_stage = StageKind.CURATION
            context_policy = ContextPolicy.CRITERIA_ONLY
        elif retry_index < max_generation_retries:
            next_stage = StageKind.GENERATION
            context_policy = ContextPolicy.CRITERIA_PLUS_ROUTE_CODE
        else:
            terminal = True
            next_stage = None
            final_route = RouteCode.DROP_RETRY_EXHAUSTED

    elif from_stage == StageKind.CURATION:
        terminal = True
        next_stage = None
        if verdict == Verdict.REJECT:
            final_route = RouteCode.REJECT_DUPLICATE

    return RoutingDecision(
        run_id=run_id,
        from_stage=from_stage,
        verdict=final_verdict,
        route_code=final_route,
        subcodes=subcodes,
        reason_codes=reason_codes,
        next_stage=next_stage,
        context_policy=context_policy,
        retry_index=retry_index,
        attempt_of=attempt_of,
        terminal=terminal,
    )

