from __future__ import annotations

from langgraph.graph import END

from config import build_runtime_config
from models import ContextPolicy, RouteCode, RoutingDecision, SeedSpec, StageKind, TaxonomyCell, Verdict
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


def _code_seed(seed_id: str, *, environment_seed: dict | None = None) -> SeedSpec:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=4, scenario="edge")
    return SeedSpec.create(
        seed_id=seed_id,
        cell=cell,
        intent="Create a compact benchmark around a realistic stateful debugging failure.",
        ability="fault_localization",
        environment="single_turn_debug_with_test",
        environment_seed=environment_seed
        if environment_seed is not None
        else {
            "product_context": "billing reconciliation worker for subscription invoices",
            "codebase_shape": "parser, reconciliation engine, export adapter, and focused tests",
            "state_model": "invoice rows move through parsed, matched, adjusted, and exported states",
            "core_invariant": "recognized revenue totals must remain stable per billing period after adjustments",
            "failure_surface": "monthly summary is wrong only for partial refunds near timezone boundaries",
            "tempting_wrong_fix": "round the exported total or patch the summary formatter",
            "actual_causal_region": "refund normalization before grouping by billing period",
            "required_depth": "requires tracing a transformed value across parser, normalizer, and summarizer",
            "non_goals": ["typo", "missing import", "one-line loop patch"],
        },
        diagnostic_pressure="misleading output error with upstream normalization cause",
        scoring_strategy="hard_checks_plus_rubric",
        leakage_risk="formatter-only patch passes visible symptom without preserving invariant",
    )


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


def test_seed_plan_partition_keeps_valid_seeds_from_mixed_batch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.OpenAIClient", FakeOpenAIClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_code_debug.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="partition",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"
    runner = PipelineRunner(config)
    valid_seed = _code_seed("valid-seed")
    toy_seed = _code_seed(
        "toy-seed",
        environment_seed={
            "product_context": "small list utility",
            "codebase_shape": "one module and one test file",
            "state_model": "single list input and output",
            "core_invariant": "return all items in order",
            "failure_surface": "one visible test fails",
            "tempting_wrong_fix": "change one loop bound",
            "actual_causal_region": "off-by-one in a single loop",
            "required_depth": "one-line patch",
            "non_goals": ["larger system debugging"],
        },
    )

    accepted, rejected = runner._partition_seed_plan([toy_seed, valid_seed])

    assert [seed.id for seed in accepted] == ["valid-seed"]
    assert [(seed.id, route_code, subcodes) for seed, route_code, subcodes in rejected] == [
        ("toy-seed", RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"])
    ]


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
