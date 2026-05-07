from __future__ import annotations

from cli_graph import NODE_ORDER, render_graph


def test_render_graph_contains_pipeline_nodes(monkeypatch) -> None:
    monkeypatch.setattr("cli_graph._term_width", lambda: 120)
    output = render_graph(
        {node: "pending" for node in NODE_ORDER},
        {"run_id": "graph", "target": 3, "committed": 0, "dropped": 0, "seed": "-"},
    )

    assert "strategy" in output
    assert "plan_det" in output
    assert "select_seed" not in output
    assert "seed cursor" in output
    assert "plan_audit" in output
    assert "generate" in output
    assert "validate_det" in output
    assert "adversary" in output
    assert "revise_adv" in output
    assert "quality_gate" in output
    assert "rubric_gate" in output
    assert "join gates" in output
    assert "curate" in output
    assert "parallel" in output
    assert "retry sample" in output
    assert "next seed" in output
