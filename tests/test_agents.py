from __future__ import annotations

import json
import time
from http.client import IncompleteRead
from pathlib import Path
from typing import Any

import pytest
import yaml

from agents import (
    CodexClient,
    Designer,
    ModelClient,
    ProviderError,
    SampleGenerator,
    _coerce_gate_verdict,
)
from config import ModelConfig, load_domain
from models import AdversaryReport, CandidateSample, DesignBrief, GenerationEnvelope, RouteCode, TaxonomyCell, Verdict


class _FakeResponse:
    def __init__(
        self,
        content: Any,
        *,
        usage_metadata: dict[str, int] | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.usage_metadata = usage_metadata or {"input_tokens": 1, "output_tokens": 1}
        self.response_metadata = response_metadata or {"model_name": "fake-model"}


class _FakeLangChainModel:
    def __init__(self, response: _FakeResponse | Exception, capture: dict[str, Any] | None = None) -> None:
        self.response = response
        self.capture = capture

    def invoke(self, messages):
        if self.capture is not None:
            self.capture["messages"] = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_reasoning_effort_is_sent_to_langchain_for_openai_reasoning_models(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(_FakeResponse('{"ok": true}'))

    monkeypatch.setattr("agents._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(provider="openai", model="gpt-5-mini", reasoning_effort="medium"))
    client.complete_json(system="Return JSON only.", user='{"task": "test"}')

    assert captured["model"] == "gpt-5-mini"
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["reasoning_effort"] == "medium"
    assert "temperature" not in captured["kwargs"]


def test_reasoning_effort_is_not_sent_for_non_openai_models(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(_FakeResponse('{"ok": true}'))

    monkeypatch.setattr("agents._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(provider="anthropic", model="claude-sonnet-test", reasoning_effort="medium"))
    client.complete_json(system="Return JSON only.", user='{"task": "test"}')

    assert captured["kwargs"]["model_provider"] == "anthropic"
    assert captured["kwargs"]["temperature"] == 0.4
    assert "reasoning_effort" not in captured["kwargs"]


def test_complete_json_repairs_nul_hex_text_sequences(monkeypatch) -> None:
    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        return _FakeLangChainModel(
            _FakeResponse('{"word": "clich\\u0000e9", "range": "3\\u0000E2\\u000080\\u00009110 words"}')
        )

    monkeypatch.setattr("agents._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(model="gpt-5-mini", reasoning_effort="medium"))
    payload, meta = client.complete_json(system="Return JSON only.", user='{"task": "test"}')

    assert payload == {"word": "cliché", "range": "3‑10 words"}
    assert meta["text_normalization_replacements"] == 2


def test_complete_text_returns_plain_model_output(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(
            _FakeResponse("A small test output.", usage_metadata={"input_tokens": 3, "output_tokens": 4}),
            captured,
        )

    monkeypatch.setattr("agents._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(model="gpt-5-mini", reasoning_effort="medium"))
    output, meta = client.complete_text(system="Follow instructions.", user="Write.")

    assert output == "A small test output."
    assert captured["messages"][1] == ("user", "Write.")
    assert meta["output_tokens"] == 4


def test_local_embedding_does_not_initialize_provider_model(monkeypatch) -> None:
    def fail_init_embeddings():
        raise AssertionError("remote embedding provider should not be initialized")

    monkeypatch.setattr("agents._load_init_embeddings", fail_init_embeddings)

    client = ModelClient(ModelConfig(embedding_provider="local", embedding_model="local-hash-embedding"))
    vector, meta = client.embed("same text same text")

    assert len(vector) == 128
    assert meta["provider"] == "local"
    assert meta["model"] == "local-hash-embedding"


def test_accept_with_reject_signal_code_is_coerced_to_reject() -> None:
    verdict, route_code, subcodes = _coerce_gate_verdict(
        verdict=Verdict.ACCEPT,
        route_code=RouteCode.ACCEPT,
        subcodes=["proxy_strong", "weak_diagnostic_pressure"],
    )

    assert verdict == Verdict.REJECT
    assert route_code == RouteCode.REJECT_SEMANTIC_MISMATCH
    assert "weak_diagnostic_pressure" in subcodes


def test_model_invocation_errors_are_wrapped(monkeypatch) -> None:
    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        return _FakeLangChainModel(TimeoutError("read timed out"))

    monkeypatch.setattr("agents._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(request_timeout_seconds=12))

    try:
        client.complete_json(system="Return JSON only.", user='{"task": "test"}')
    except Exception as exc:
        assert type(exc).__name__ == "ProviderError"
        assert "model invocation failed" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_codex_client_uses_explicit_auth_file_and_streams_response(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 3600,
                    "account_id": "acct_test",
                }
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.chunks = [
                (
                    'data: {"type":"response.output_text.delta","delta":"{\\"ok\\": "}\n\n'
                    'data: {"type":"response.output_text.delta","delta":"true}"}\n\n'
                ).encode("utf-8"),
                (
                    'data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.5",'
                    '"usage":{"input_tokens":5,"output_tokens":2}}}\n\n'
                ).encode("utf-8"),
                b"ignored trailing bytes",
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return self.chunks.pop(0) if self.chunks else b""

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file, reasoning_effort="medium"))
    payload, meta = client.complete_json(system="Return JSON only.", user='{"task": "test"}')

    assert payload == {"ok": True}
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["headers"]["Authorization"] == "Bearer access-token"
    assert captured["headers"]["Chatgpt-account-id"] == "acct_test"
    assert captured["body"]["stream"] is True
    assert captured["body"]["store"] is False
    assert captured["body"]["instructions"] == "Return JSON only."
    assert captured["body"]["reasoning"] == {"effort": "medium"}
    assert meta["input_tokens"] == 5
    assert meta["output_tokens"] == 2


def test_codex_client_refreshes_expired_tokens(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "old-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() - 1,
                }
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, Any]] = []

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return self.body

    def fake_urlopen(request, timeout):
        requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": request.data})
        if request.full_url == CodexClient.token_endpoint:
            return FakeResponse(json.dumps({"access_token": "new-token", "expires_in": 3600}).encode("utf-8"))
        return FakeResponse(
            (
                'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                'data: {"type":"response.completed","response":{"model":"gpt-5.5","usage":{}}}\n\n'
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    output, _ = client.complete_text(system="Say hi.", user="Hi.")

    assert output == "hello"
    assert requests[0]["url"] == CodexClient.token_endpoint
    assert b"grant_type=refresh_token" in requests[0]["body"]
    assert requests[1]["headers"]["Authorization"] == "Bearer new-token"
    updated_auth = json.loads(auth_file.read_text(encoding="utf-8"))
    assert updated_auth["tokens"]["access_token"] == "new-token"
    assert updated_auth["tokens"]["refresh_token"] == "refresh-token"


def test_codex_client_tolerates_incomplete_read_after_completed_event(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 3600,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            raise IncompleteRead(
                (
                    'data: {"type":"response.output_text.delta","delta":"done"}\n\n'
                    'data: {"type":"response.completed","response":{"model":"gpt-5.5","usage":{}}}\n\n'
                ).encode("utf-8"),
                128,
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    output, _ = client.complete_text(system="Say done.", user="Go.")

    assert output == "done"


def test_codex_client_rejects_incomplete_read_before_completed_event(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 3600,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            raise IncompleteRead(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n', 128)

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    with pytest.raises(ProviderError, match="incomplete HTTP read before response.completed"):
        client.complete_text(system="Say partial.", user="Go.")


def test_codex_client_times_out_stream_without_completed_event(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 3600,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def readline(self):
            return b'data: {"type":"response.output_text.delta","delta":"still working"}\n\n'

    ticks = iter([0.0, 0.5, 1.1])
    monkeypatch.setattr("agents.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file, request_timeout_seconds=1))
    with pytest.raises(ProviderError, match="Codex stream timed out before response.completed"):
        client.complete_text(system="Say partial.", user="Go.")


def test_codex_client_missing_auth_file_has_clear_error(tmp_path) -> None:
    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=tmp_path / "missing.json"))

    with pytest.raises(ProviderError, match="Codex auth file not found"):
        client.complete_text(system="Say hi.", user="Hi.")


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
        runtime_requirements={
            "kind": "filesystem_task",
            "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
            "language": {"name": "python", "version": "3.11+"},
            "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
            "commands": {"test": "python -m pytest -q"},
            "network": "disabled_during_eval",
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
            "runtime_requirements": {
                "kind": "filesystem_task",
                "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
                "language": {"name": "python", "version": "3.11+"},
                "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
                "commands": {"test": "python -m pytest -q"},
                "network": "disabled_during_eval",
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


class _GenerateClient:
    def __init__(self, *, require_contract: bool = True) -> None:
        self.require_contract = require_contract
        self.system = ""
        self.user_payload: dict[str, Any] | None = None

    def complete_json(self, *, system: str, user: str, temperature: float = 0.4):
        if self.require_contract:
            assert "DESIGN IMPLEMENTATION CONTRACT" in system
        self.system = system
        self.user_payload = json.loads(user)
        return {
            "agent_artifact": {
                "benchmark_case": {
                    "prompt": "Debug the provided workspace.",
                    "setup": "Run pytest.",
                    "inputs": {},
                    "environment": {"runtime": "python"},
                },
                "runtime_requirements": {
                    "kind": "filesystem_task",
                    "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
                    "language": {"name": "python", "version": "3.11+"},
                    "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
                    "commands": {"test": "python -m pytest -q"},
                    "network": "disabled_during_eval",
                },
            },
            "judge_artifact": {
                "score_x": {
                    "score_type": "hard_checks_plus_rubric",
                    "dimensions": [
                        {
                            "name": "causal_fix",
                            "weight": 1.0,
                            "high_score_criterion": "Causal fix.",
                            "low_score_criterion": "Shallow fix.",
                        }
                    ],
                },
                "proxy_claim": "This case proxies debugging ability with a visible workspace and judge-facing scoring criteria.",
                "diagnostic_pressure": ["misleading symptom", "cross-module invariant"],
                "scoring_contract": {"credit": ["causal fix"], "penalties": ["test edits"]},
                "leakage_risks": ["Visible tests may invite shallow fixes."],
                "known_limits": ["Small synthetic workspace."],
                "coverage_tags": ["debugging"],
                "negative_controls": [{"output": "edit tests", "should_fail_because": "weakens benchmark"}],
            },
            "ability_z": {"name": "fault_localization"},
            "environment_y": {"name": "single_turn_debug_with_test"},
        }, {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
        }


class _DesignClient:
    def __init__(self) -> None:
        self.system = ""
        self.user_payload: dict[str, Any] | None = None

    def complete_json(self, *, system: str, user: str, temperature: float = 0.4):
        self.system = system
        self.user_payload = json.loads(user)
        return {"designs": []}, {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
        }


def test_design_retry_payload_explains_runtime_image_mode() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _DesignClient()
    designer = Designer(client, domain)

    designer.design(
        run_id="run",
        target_n=1,
        coverage_snapshot={},
        retry_route_code=RouteCode.REJECT_CRITERIA_MISMATCH,
        retry_subcodes=["unsupported_runtime_requirements"],
    )

    assert "never a combined value" in client.system
    rejection = client.user_payload["prior_design_rejection"]
    assert rejection["subcodes"] == ["unsupported_runtime_requirements"]
    assert any("task_image/container" in item for item in rejection["retry_guidance"])


def test_generation_retry_payload_includes_actionable_code_debug_guidance() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _GenerateClient()
    generator = SampleGenerator(client, domain)

    generator.generate(
        run_id="run",
        design=design,
        attempt=2,
        retry_route_code=RouteCode.REJECT_SCHEMA,
        retry_subcodes=["workspace_tests_do_not_reproduce_failure", "answer_leak_in_candidate_materials"],
    )

    rejection = client.user_payload["prior_generation_rejection"]
    assert rejection["subcodes"] == ["workspace_tests_do_not_reproduce_failure", "answer_leak_in_candidate_materials"]
    assert any("starter code has at least one deterministic failing pytest assertion" in item for item in rejection["retry_guidance"])
    assert any("Candidate-facing material leaked the answer" in item for item in rejection["retry_guidance"])
    assert client.user_payload["design_brief"]["runtime_requirements"]["kind"] == "filesystem_task"
    assert client.user_payload["example_output"]["agent_artifact"]["runtime_requirements"]["commands"]["test"] == "python -m pytest -q"


def test_generation_accepts_structured_envelope() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _GenerateClient()
    generator = SampleGenerator(client, domain)
    envelope = GenerationEnvelope.from_design(
        design,
        envelope_id="billing-ledger-v1",
        domain_ref="domains/benchmark_code_debug.yaml",
        seed_context={"seed_family": "billing-ledger"},
    )

    candidate, _ = generator.generate_from_envelope(run_id="run", envelope=envelope, attempt=1)

    assert client.user_payload["generation_envelope"]["id"] == "billing-ledger-v1"
    assert client.user_payload["generation_envelope"]["domain_ref"] == "domains/benchmark_code_debug.yaml"
    assert "generator_policy" not in client.user_payload["generation_envelope"]
    assert "tags" not in client.user_payload["generation_envelope"]
    assert client.user_payload["generation_envelope"]["seed_context"] == {"seed_family": "billing-ledger"}
    assert "experiment_context" not in client.user_payload
    assert candidate.provenance["generation_envelope_id"] == "billing-ledger-v1"
    assert "generator_variant" not in candidate.provenance


def test_generator_accepts_experiment_supplied_system_prompt_append() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    default_client = _GenerateClient()
    treatment_client = _GenerateClient()
    treatment_text = "This is experiment-owned independent-variable prompt text."

    SampleGenerator(default_client, domain).generate(run_id="run", design=design, attempt=1)
    SampleGenerator(treatment_client, domain, system_prompt_append=treatment_text).generate(
        run_id="run",
        design=design,
        attempt=1,
    )

    assert "EXPERIMENT-SUPPLIED GENERATOR INSTRUCTIONS" not in default_client.system
    assert "EXPERIMENT-SUPPLIED GENERATOR INSTRUCTIONS" in treatment_client.system
    assert treatment_text in treatment_client.system


def test_generator_accepts_experiment_supplied_system_prompt_override() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _GenerateClient(require_contract=False)
    override = "You are a minimal generator. Return JSON only."

    SampleGenerator(client, domain, system_prompt_override=override).generate(run_id="run", design=design, attempt=1)

    assert client.system.startswith(override)
    assert "Benchmark Case Generator" not in client.system
    assert "DESIGN IMPLEMENTATION CONTRACT" not in client.system
    assert client.user_payload["generation_envelope"]["design"]["id"] == design.id


def test_experiment_rubric_prompt_uses_domain_rubric_context_token() -> None:
    experiment_paths = [
        Path("experiments/adversary_awareness/experiment.yaml"),
        Path("experiments/adversary_awareness/experiment.smoke.yaml"),
    ]

    for experiment_path in experiment_paths:
        experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
        variants = {variant["variant_id"]: variant for variant in [experiment["baseline"]] + experiment["variant_plan"]}

        assert variants["generator_minimal"]["bindings"]["generator_system_prompt_append"] == ""
        assert "{{DOMAIN_RUBRIC_CONTEXT}}" not in variants["generator_minimal_adversary"]["bindings"][
            "generator_system_prompt_append"
        ]
        assert "{{DOMAIN_RUBRIC_CONTEXT}}" in variants["generator_minimal_rubric"]["bindings"][
            "generator_system_prompt_append"
        ]
        assert "{{DOMAIN_RUBRIC_CONTEXT}}" in variants["generator_minimal_rubric_adversary"]["bindings"][
            "generator_system_prompt_append"
        ]


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
    assert revised.output["agent_artifact"]["runtime_requirements"]["commands"]["test"] == "python -m pytest -q"
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


def test_revision_tolerates_unchanged_runtime_requirements_field() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "runtime_requirements": candidate.agent_artifact.runtime_requirements,
                "metadata_updates": {
                    "proxy_claim": "The revised metadata still preserves the same runtime contract while tightening the judge-facing proxy claim."
                },
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

    assert revised.agent_artifact.runtime_requirements == candidate.agent_artifact.runtime_requirements
    assert revised.output["agent_artifact"]["runtime_requirements"] == candidate.agent_artifact.runtime_requirements


def test_revision_rejects_runtime_contract_changes() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    changed_runtime = dict(candidate.agent_artifact.runtime_requirements)
    changed_runtime["commands"] = {"test": "pytest -q"}
    generator = SampleGenerator(
        _PatchClient(
            {
                "runtime_requirements": changed_runtime,
                "metadata_updates": {"proxy_claim": "Try to change runtime."},
            }
        ),
        domain,
    )

    with pytest.raises(ProviderError, match="cannot change runtime_requirements"):
        generator.revise_from_attack(
            run_id="run",
            design=design,
            candidate=candidate,
            report=_attack_report(candidate.id),
            attempt=2,
        )
