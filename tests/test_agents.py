from __future__ import annotations

import socket
from typing import Any

import pytest

from agents import OpenAIClient, ProviderError, SampleGenerator, _coerce_gate_verdict
from config import ModelConfig, load_domain
from models import AdversaryReport, CandidateSample, DesignBrief, RouteCode, TaxonomyCell, Verdict


def test_reasoning_effort_is_sent_for_reasoning_models(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def fake_post(self: OpenAIClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["path"] = path
        captured["body"] = body
        return {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(OpenAIClient, "_post", fake_post)

    client = OpenAIClient(ModelConfig(model="gpt-5-mini", reasoning_effort="medium"))
    client.complete_json(system="Return JSON only.", user='{"task": "test"}')

    assert captured["path"] == "/chat/completions"
    assert captured["body"]["reasoning_effort"] == "medium"


def test_reasoning_effort_is_not_sent_for_non_reasoning_models(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def fake_post(self: OpenAIClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(OpenAIClient, "_post", fake_post)

    client = OpenAIClient(ModelConfig(model="gpt-4.1-mini", reasoning_effort="medium"))
    client.complete_json(system="Return JSON only.", user='{"task": "test"}')

    assert "reasoning_effort" not in captured["body"]


def test_complete_json_repairs_nul_hex_text_sequences(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_post(self: OpenAIClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"word": "clich\\u0000e9", "range": "3\\u0000E2\\u000080\\u00009110 words"}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(OpenAIClient, "_post", fake_post)

    client = OpenAIClient(ModelConfig(model="gpt-5-mini", reasoning_effort="medium"))
    payload, meta = client.complete_json(system="Return JSON only.", user='{"task": "test"}')

    assert payload == {"word": "cliché", "range": "3‑10 words"}
    assert meta["text_normalization_replacements"] == 2


def test_complete_text_returns_plain_model_output(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def fake_post(self: OpenAIClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
        captured["body"] = body
        return {
            "choices": [{"message": {"content": "A small test output."}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }

    monkeypatch.setattr(OpenAIClient, "_post", fake_post)

    client = OpenAIClient(ModelConfig(model="gpt-5-mini", reasoning_effort="medium"))
    output, meta = client.complete_text(system="Follow instructions.", user="Write.")

    assert output == "A small test output."
    assert captured["body"]["messages"][1]["content"] == "Write."
    assert meta["output_tokens"] == 4


def test_accept_with_reject_signal_code_is_coerced_to_reject() -> None:
    verdict, route_code, subcodes = _coerce_gate_verdict(
        verdict=Verdict.ACCEPT,
        route_code=RouteCode.ACCEPT,
        subcodes=["proxy_strong", "weak_diagnostic_pressure"],
    )

    assert verdict == Verdict.REJECT
    assert route_code == RouteCode.REJECT_SEMANTIC_MISMATCH
    assert "weak_diagnostic_pressure" in subcodes


def test_post_wraps_socket_read_timeout(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class TimeoutResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            raise socket.timeout("read timed out")

    def fake_urlopen(request, timeout):
        assert timeout == 12
        return TimeoutResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAIClient(ModelConfig(request_timeout_seconds=12))

    try:
        client._post("/chat/completions", {"model": "test"})
    except Exception as exc:
        assert type(exc).__name__ == "ProviderError"
        assert "read timed out after 12s" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def _code_design() -> DesignBrief:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=4, scenario="edge")
    return DesignBrief.create(
        design_id="design-1",
        cell=cell,
        target_ability="fault_localization",
        target_environment="single_turn_debug_with_test",
        design_intent="Create a compact benchmark around a realistic stateful debugging failure.",
        environment_premise={
            "product_context": "billing reconciliation worker",
            "codebase_shape": "parser, reconciliation engine, and tests",
            "state_model": "invoice rows flow through parse, normalize, summarize",
            "core_invariant": "refunds reduce revenue in the original period",
            "failure_surface": "partial refunds inflate the summary",
            "tempting_wrong_fix": "round or clamp the displayed total",
            "actual_causal_region": "refund normalization before aggregation",
            "required_depth": "requires tracing value semantics across modules",
        },
        environment_artifact_spec={"kind": "virtual_workspace"},
        failure_mode_family="misleading aggregate caused by upstream normalization",
        diagnostic_pressure=["misleading downstream symptom"],
        why_weak_agents_fail=["they patch only the formatter"],
        tempting_shallow_solutions=["clamp the output total"],
        success_evidence_required=["preserves refund invariant"],
        minimum_depth_requirements=["multi-file causal trace"],
        forbidden_shortcuts=["one-line display mask"],
        non_goals=["missing import"],
    )


def _code_candidate(design: DesignBrief) -> CandidateSample:
    return CandidateSample(
        id="candidate-1",
        design_id=design.id,
        content_hash="hash",
        cell=design.cell,
        agent_artifact={
            "benchmark_case": {
                "prompt": "Debug the reconciliation worker.",
                "setup": "Run pytest.",
                "inputs": {},
                "environment": {"runtime": "python"},
            },
            "environment_artifact": {
                "kind": "virtual_workspace",
                "payload": {
                    "files": [
                        {"path": "billing/parser.py", "content": "def parse_row(row):\n    return dict(row)\n"},
                        {
                            "path": "billing/reconcile.py",
                            "content": "from billing.parser import parse_row\n\n\ndef summarize(rows):\n    totals = {}\n    for row in rows:\n        parsed = parse_row(row)\n        totals[parsed['period']] = totals.get(parsed['period'], 0) + abs(parsed['amount'])\n    return totals\n",
                        },
                        {"path": "README.md", "content": "Patch the service without editing tests.\n"},
                        {
                            "path": "tests/test_reconcile.py",
                            "content": "from billing.reconcile import summarize\n\n\ndef test_refund_reduces_total():\n    assert summarize([{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]) == {'2026-03': 75}\n",
                        },
                    ],
                    "commands": {"test": "python -m pytest -q"},
                },
            },
        },
        judge_artifact={
            "score_x": {"score_type": "hard_checks_plus_rubric", "dimensions": [{"name": "causal_fix", "weight": 1.0}]},
            "proxy_claim": "The benchmark proxies debugging ability by requiring an invariant-preserving fix.",
            "diagnostic_pressure": ["misleading downstream symptom"],
            "scoring_contract": {"credit": ["fixes refund invariant"], "penalties": ["display-only mask"]},
            "leakage_risks": ["visible assertion may invite a clamp"],
            "known_limits": ["single case"],
            "coverage_tags": ["stateful_debugging"],
            "negative_controls": [{"output": "Clamp the total.", "should_fail_because": "masks the bug"}],
        },
        ability_z={"name": "fault_localization"},
        environment_y={"name": "single_turn_debug_with_test"},
        difficulty=design.cell.difficulty,
        case_type=design.cell.case_type,
    )


def _attack_report(candidate_id: str) -> AdversaryReport:
    return AdversaryReport(
        candidate_id=candidate_id,
        revision_disposition="revise",
        attack_summary="The task is too local and can be solved by clamping the aggregate.",
        cheap_pass_strategy="Change abs(parsed['amount']) to parsed['amount'] without understanding period invariants.",
        proxy_damage="The score would mostly reflect matching the visible assertion.",
        survival_requirements=["add cross-period refund pressure", "remove README boilerplate"],
    )


class _PatchClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def complete_json(self, *, system: str, user: str, temperature: float = 0.4):
        assert "REVISION MODE" in system
        assert "Return a revision patch JSON object only" in system
        assert "required_revision_patch_shape" in user
        return self.payload, {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
        }


def test_revision_applies_patch_to_prior_candidate_and_virtual_workspace() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "benchmark_case_updates": {
                    "prompt": "Debug the reconciliation worker across parser, normalizer, and summary behavior."
                },
                "metadata_updates": {
                    "proxy_claim": "The revised case proxies debugging ability by requiring the solver to preserve refund semantics across multiple periods, not merely satisfy one visible assertion."
                },
                "environment_ops": [
                    {"op": "delete_file", "path": "README.md"},
                    {
                        "op": "write_file",
                        "path": "billing/reconcile.py",
                        "content": "from billing.parser import parse_row\n\n\ndef summarize(rows):\n    totals = {}\n    for row in rows:\n        parsed = parse_row(row)\n        totals[parsed['period']] = totals.get(parsed['period'], 0) + parsed['amount']\n    return totals\n",
                    },
                    {
                        "op": "write_file",
                        "path": "tests/test_reconcile.py",
                        "content": "from billing.reconcile import summarize\n\n\ndef test_refund_reduces_total():\n    assert summarize([{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]) == {'2026-03': 75}\n\n\ndef test_multiple_periods_stay_separate():\n    rows = [{'period': '2026-03', 'amount': 100}, {'period': '2026-04', 'amount': -25}]\n    assert summarize(rows) == {'2026-03': 100, '2026-04': -25}\n",
                    },
                ],
                "revision_rationale": "Replace the local visible-assertion task with multi-period state pressure.",
            }
        ),
        domain,
    )

    revised, _ = generator.revise_from_attack(
        run_id="run",
        design=design,
        candidate=candidate,
        report=_attack_report(candidate.id),
        attempt=2,
    )

    assert revised.id == "run-candidate-design-1-2-rev"
    assert revised.provenance["revision_of"] == candidate.id
    assert revised.agent_artifact.benchmark_case["prompt"].startswith("Debug the reconciliation worker across")
    assert "multiple periods" in revised.judge_artifact.proxy_claim
    assert revised.agent_artifact.environment_artifact is not None
    paths = [item["path"] for item in revised.agent_artifact.environment_artifact.payload["files"]]
    assert "README.md" not in paths
    assert "billing/reconcile.py" in paths
    assert "revision_rationale" not in revised.output
    assert "environment_ops" not in revised.output
    assert revised.output["agent_artifact"]["environment_artifact"]["payload"]["commands"]["test"] == "python -m pytest -q"


def test_revision_rejects_old_full_candidate_output() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "benchmark_case": {"prompt": "old full-object style"},
                "score_x": {"score_type": "hard_checks_plus_rubric"},
                "environment_artifact": {"kind": "virtual_workspace", "payload": {}},
            }
        ),
        domain,
    )

    with pytest.raises(ProviderError, match="unsupported top-level keys"):
        generator.revise_from_attack(
            run_id="run",
            design=design,
            candidate=candidate,
            report=_attack_report(candidate.id),
            attempt=2,
        )
