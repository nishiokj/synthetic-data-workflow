from __future__ import annotations

from typing import Any

from agents import OpenAIClient
from config import ModelConfig


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
