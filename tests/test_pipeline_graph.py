from __future__ import annotations

from langgraph.graph import END

from config import build_runtime_config
from models import AdversaryReport, CandidateSample, ContextPolicy, RouteCode, RoutingDecision, DesignBrief, StageKind, TaxonomyCell, Verdict
from pipeline import (
    PipelineRunner,
    after_adversary,
    after_audit_design,
    after_curate,
    after_generate,
    after_validate_det,
    after_select_next_design,
    after_terminal_design,
    _graph_recursion_limit,
    route_from_decision,
)
from tests.test_pipeline_smoke import FakeOpenAIClient


def _code_design(design_id: str, *, environment_premise: dict | None = None) -> DesignBrief:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=4, scenario="edge")
    return DesignBrief.create(
        design_id=design_id,
        cell=cell,
        target_ability="fault_localization",
        target_environment="single_turn_debug_with_test",
        design_intent="Create a compact benchmark around a realistic stateful debugging failure.",
        environment_premise=environment_premise
        if environment_premise is not None
        else {
            "product_context": "billing reconciliation worker for subscription invoices",
            "codebase_shape": "parser, reconciliation engine, export adapter, and focused tests",
            "state_model": "invoice rows move through parsed, matched, adjusted, and exported states",
            "core_invariant": "recognized revenue totals must remain stable per billing period after adjustments",
            "failure_surface": "monthly summary is wrong only for partial refunds near timezone boundaries",
            "tempting_wrong_fix": "round the exported total or patch the summary formatter",
            "actual_causal_region": "refund normalization before grouping by billing period",
            "required_depth": "requires tracing a transformed value across parser, normalizer, and summarizer",
        },
        failure_mode_family="misleading downstream aggregate caused by upstream normalization",
        diagnostic_pressure=["misleading output error with upstream normalization cause"],
        why_weak_agents_fail=["they patch the formatter or rounded total without preserving the period invariant"],
        tempting_shallow_solutions=["formatter-only patch passes visible symptom without preserving invariant"],
        success_evidence_required=["trace transformed values across parser, normalizer, and summarizer"],
        minimum_depth_requirements=["requires at least two modules and a state invariant"],
        forbidden_shortcuts=["one-line arithmetic boundary patch", "test edit", "formatter mask"],
        non_goals=["typo", "missing import", "one-line loop patch"],
    )


def _decision(
    *,
    verdict: Verdict = Verdict.ACCEPT,
    next_stage: StageKind | None = StageKind.GENERATION,
    terminal: bool = False,
) -> RoutingDecision:
    return RoutingDecision(
        run_id="test",
        from_stage=StageKind.DESIGN_AUDIT,
        verdict=verdict,
        route_code=RouteCode.ACCEPT,
        next_stage=next_stage,
        context_policy=ContextPolicy.FRESH,
        retry_index=0,
        terminal=terminal,
    )


def _candidate(design: DesignBrief) -> CandidateSample:
    return CandidateSample(
        id=f"candidate-{design.id}",
        design_id=design.id,
        content_hash="abc",
        cell=design.cell,
        agent_artifact={"benchmark_case": {"prompt": "Debug the reconciliation worker and explain the invariant-preserving fix."}},
        judge_artifact={
            "score_x": {
                "score_type": "hard_checks_plus_rubric",
                "dimensions": [
                    {
                        "name": "causal_fix",
                        "weight": 1.0,
                        "high_score_criterion": "Identifies the upstream normalization cause and preserves period invariants.",
                        "low_score_criterion": "Patches only the displayed summary or visible assertion.",
                    }
                ],
            },
            "proxy_claim": "A strong score indicates debugging ability because the candidate must connect a misleading downstream symptom to upstream state normalization while preserving an invariant.",
            "diagnostic_pressure": ["misleading downstream symptom", "upstream state invariant"],
            "scoring_contract": {
                "credit": ["causal fix preserves invariant"],
                "penalties": ["formatter-only patch"],
            },
            "leakage_risks": ["Visible symptom may encourage formatter masking."],
            "known_limits": ["Single case does not prove broad debugging skill."],
            "coverage_tags": ["stateful_debugging"],
            "negative_controls": [{"output": "Round the summary total.", "should_fail_because": "Masks the symptom without fixing the cause."}],
        },
        ability_z={"name": "fault_localization"},
        environment_y={"name": "single_turn_debug_with_test"},
        difficulty=design.cell.difficulty,
        case_type=design.cell.case_type,
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
        "design",
        "validate_design_batch_det",
        "select_next_design",
        "audit_design",
        "generate",
        "validate_det",
        "adversary",
        "revise_from_adversary",
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


def test_design_batch_partition_keeps_valid_designs_from_mixed_batch(tmp_path, monkeypatch) -> None:
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
    valid_design = _code_design("valid-design")
    toy_design = _code_design(
        "toy-design",
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
        },
    )

    accepted, rejected = runner._partition_design_batch([toy_design, valid_design])

    assert [design.id for design in accepted] == ["valid-design"]
    assert [(design.id, route_code, subcodes) for design, route_code, subcodes in rejected] == [
        ("toy-design", RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"])
    ]


def test_route_from_decision_dispatches_by_next_stage() -> None:
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.DESIGN)}) == "design"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.DESIGN_AUDIT)}) == "audit_design"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.GENERATION)}) == "generate"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.VALIDATION)}) == "validate_det"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.CURATION)}) == "curate"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.GENERATION, terminal=True)}) == END


def test_after_curate_branches() -> None:
    assert (
        after_curate({"committed_count": 2, "target_n": 2, "designs_queue": [], "design_round": 1, "max_design_retries": 2})
        == END
    )
    assert (
        after_curate(
            {"committed_count": 1, "target_n": 2, "designs_queue": [object()], "design_round": 1, "max_design_retries": 2}
        )
        == "select_next_design"
    )
    assert (
        after_curate({"committed_count": 1, "target_n": 2, "designs_queue": [], "design_round": 1, "max_design_retries": 2})
        == "design"
    )
    assert (
        after_curate({"committed_count": 1, "target_n": 2, "designs_queue": [], "design_round": 3, "max_design_retries": 2})
        == END
    )


def test_after_terminal_design_branches() -> None:
    assert (
        after_terminal_design({"committed_count": 2, "target_n": 2, "designs_queue": [], "design_round": 1, "max_design_retries": 2})
        == END
    )
    assert (
        after_terminal_design(
            {"committed_count": 1, "target_n": 2, "designs_queue": [object()], "design_round": 1, "max_design_retries": 2}
        )
        == "select_next_design"
    )
    assert (
        after_terminal_design({"committed_count": 1, "target_n": 2, "designs_queue": [], "design_round": 1, "max_design_retries": 2})
        == "design"
    )
    assert (
        after_terminal_design({"committed_count": 1, "target_n": 2, "designs_queue": [], "design_round": 3, "max_design_retries": 2})
        == END
    )


def test_after_generate_keeps_design_terminal_from_ending_run() -> None:
    state = {
        "last_decision": _decision(next_stage=None, terminal=True),
        "committed_count": 0,
        "target_n": 2,
        "designs_queue": [object()],
        "design_round": 1,
        "max_design_retries": 2,
    }

    assert after_generate(state) == "select_next_design"


def test_after_select_next_design_branches() -> None:
    assert after_select_next_design({"design": object()}) == "audit_design"
    assert after_select_next_design({"design": None}) == "design"


def test_after_audit_design_branches() -> None:
    assert (
        after_audit_design(
            {"last_decision": _decision(verdict=Verdict.ACCEPT), "designs_queue": [], "design_round": 1, "max_design_retries": 2}
        )
        == "generate"
    )
    assert (
        after_audit_design(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "designs_queue": [object()],
                "design_round": 1,
                "max_design_retries": 2,
            }
        )
        == "select_next_design"
    )
    assert (
        after_audit_design(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "designs_queue": [],
                "design_round": 1,
                "max_design_retries": 2,
            }
        )
        == "design"
    )
    assert (
        after_audit_design(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "designs_queue": [],
                "design_round": 3,
                "max_design_retries": 2,
            }
        )
        == END
    )


def test_after_adversary_revise_still_routes_to_revision() -> None:
    assert after_adversary({"last_decision": None, "adversary_done": False}) == "revise_from_adversary"


def test_adversary_pass_skips_revision_and_routes_to_gates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.OpenAIClient", FakeOpenAIClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_code_debug.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="adv-pass",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"
    runner = PipelineRunner(config)
    design = _code_design("design-pass")
    candidate = _candidate(design)
    monkeypatch.setattr(
        runner.adversary,
        "attack",
        lambda candidate, design: (
            AdversaryReport(
                candidate_id=candidate.id,
                revision_disposition="pass",
                disposition_rationale="No blocking attack found.",
            ),
            {"provider": "test", "model": "fake", "prompt_hash": "fake"},
        ),
    )

    update = runner.node_adversary({"design": design, "candidate": candidate, "gen_attempt": 0})

    assert update["adversary_done"] is True
    assert update["last_decision"] is None
    assert after_adversary(update) == ["quality_gate", "rubric_gate"]
    assert after_validate_det({"det_accepted": True, "adversary_done": update["adversary_done"]}) == [
        "quality_gate",
        "rubric_gate",
    ]
