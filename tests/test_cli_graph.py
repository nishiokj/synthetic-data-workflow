from __future__ import annotations

from cli_graph import NODE_ORDER, _handle_progress, render_graph


def test_render_graph_contains_pipeline_nodes(monkeypatch) -> None:
    monkeypatch.setattr("cli_graph._term_width", lambda: 120)
    output = render_graph(
        {node: "pending" for node in NODE_ORDER},
        {"run_id": "graph", "target": 3, "committed": 0, "dropped": 0, "design": "-"},
    )

    assert "design" in output
    assert "design_det" in output
    assert "select_design" not in output
    assert "design_cursor" in output
    assert "design_audit" in output
    assert "generate" in output
    assert "validate_det" in output
    assert "adversary" in output
    assert "revise_adv" in output
    assert "quality_gate" in output
    assert "rubric_gate" in output
    assert "join_gates" in output
    assert "curate" in output
    assert "parallel" in output
    assert "retry sample" in output
    assert "next design" in output


def test_progress_events_highlight_runtime_nodes() -> None:
    node_status = {node: "pending" for node in NODE_ORDER}
    stats = {"run_id": "graph", "committed": 0, "dropped": 0, "design": "-"}
    recent: list[str] = []

    _handle_progress(
        {"stage": "generation", "event": "revise", "candidate": "candidate-1"},
        node_status,
        stats,
        recent,
    )
    assert node_status["revise_from_adversary"] == "running"
    assert node_status["generate"] == "pending"

    _handle_progress(
        {"stage": "join_gates", "event": "start", "candidate": "candidate-1"},
        node_status,
        stats,
        recent,
    )
    assert node_status["join_gates"] == "running"
    assert node_status["revise_from_adversary"] == "pending"
