from __future__ import annotations

from langgraph.graph import END

from config import build_runtime_config
from models import ContextPolicy, RouteCode, RoutingDecision, StageKind, Verdict
from pipeline import (
    PipelineRunner,
    after_audit_seed_plan,
    after_curate,
    after_generate,
    after_select_next_seed,
    after_terminal_seed,
    _graph_recursion_limit,
    route_from_decision,
)
from tests.test_pipeline_smoke import FakeOpenAIClient


def _decision(
    *,
    verdict: Verdict = Verdict.ACCEPT,
    next_stage: StageKind | None = StageKind.GENERATION,
    terminal: bool = False,
) -> RoutingDecision:
    return RoutingDecision(
        run_id="test",
        from_stage=StageKind.PLAN_AUDIT,
        verdict=verdict,
        route_code=RouteCode.ACCEPT,
        next_stage=next_stage,
        context_policy=ContextPolicy.FRESH,
        retry_index=0,
        terminal=terminal,
    )


def test_compiled_graph_contains_pipeline_nodes(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.OpenAIClient", FakeOpenAIClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="graph",
    )

    graph = PipelineRunner(config).graph.get_graph()

    assert set(graph.nodes) - {"__start__", "__end__"} == {
        "strategy",
        "validate_seed_plan_det",
        "select_next_seed",
        "audit_seed_plan",
        "generate",
        "validate_det",
        "quality_gate",
        "rubric_gate",
        "join_gates",
        "curate",
    }


def test_graph_recursion_limit_scales_with_run_policy() -> None:
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=5,
        seed=42,
        run_id="graph-limit",
    )

    assert _graph_recursion_limit(config) > 25


def test_route_from_decision_dispatches_by_next_stage() -> None:
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.STRATEGY)}) == "strategy"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.PLAN_AUDIT)}) == "audit_seed_plan"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.GENERATION)}) == "generate"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.VALIDATION)}) == "validate_det"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.CURATION)}) == "curate"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.GENERATION, terminal=True)}) == END


def test_after_curate_branches() -> None:
    assert (
        after_curate({"committed_count": 2, "target_n": 2, "seeds_queue": [], "plan_round": 1, "max_plan_retries": 2})
        == END
    )
    assert (
        after_curate(
            {"committed_count": 1, "target_n": 2, "seeds_queue": [object()], "plan_round": 1, "max_plan_retries": 2}
        )
        == "select_next_seed"
    )
    assert (
        after_curate({"committed_count": 1, "target_n": 2, "seeds_queue": [], "plan_round": 1, "max_plan_retries": 2})
        == "strategy"
    )
    assert (
        after_curate({"committed_count": 1, "target_n": 2, "seeds_queue": [], "plan_round": 3, "max_plan_retries": 2})
        == END
    )


def test_after_terminal_seed_branches() -> None:
    assert (
        after_terminal_seed({"committed_count": 2, "target_n": 2, "seeds_queue": [], "plan_round": 1, "max_plan_retries": 2})
        == END
    )
    assert (
        after_terminal_seed(
            {"committed_count": 1, "target_n": 2, "seeds_queue": [object()], "plan_round": 1, "max_plan_retries": 2}
        )
        == "select_next_seed"
    )
    assert (
        after_terminal_seed({"committed_count": 1, "target_n": 2, "seeds_queue": [], "plan_round": 1, "max_plan_retries": 2})
        == "strategy"
    )
    assert (
        after_terminal_seed({"committed_count": 1, "target_n": 2, "seeds_queue": [], "plan_round": 3, "max_plan_retries": 2})
        == END
    )


def test_after_generate_keeps_seed_terminal_from_ending_run() -> None:
    state = {
        "last_decision": _decision(next_stage=None, terminal=True),
        "committed_count": 0,
        "target_n": 2,
        "seeds_queue": [object()],
        "plan_round": 1,
        "max_plan_retries": 2,
    }

    assert after_generate(state) == "select_next_seed"


def test_after_select_next_seed_branches() -> None:
    assert after_select_next_seed({"seed": object()}) == "audit_seed_plan"
    assert after_select_next_seed({"seed": None}) == "strategy"


def test_after_audit_seed_plan_branches() -> None:
    assert (
        after_audit_seed_plan(
            {"last_decision": _decision(verdict=Verdict.ACCEPT), "seeds_queue": [], "plan_round": 1, "max_plan_retries": 2}
        )
        == "generate"
    )
    assert (
        after_audit_seed_plan(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "seeds_queue": [object()],
                "plan_round": 1,
                "max_plan_retries": 2,
            }
        )
        == "select_next_seed"
    )
    assert (
        after_audit_seed_plan(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "seeds_queue": [],
                "plan_round": 1,
                "max_plan_retries": 2,
            }
        )
        == "strategy"
    )
    assert (
        after_audit_seed_plan(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "seeds_queue": [],
                "plan_round": 3,
                "max_plan_retries": 2,
            }
        )
        == END
    )
