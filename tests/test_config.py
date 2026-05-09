from __future__ import annotations

import os

from config import build_runtime_config, load_domain, load_env_file


def test_load_env_file_sets_missing_values(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["OPENAI_API_KEY"] == "sk-test"


def test_load_env_file_strips_quotes_and_export(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text('export OPENAI_MODEL="gpt-test"\n', encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["OPENAI_MODEL"] == "gpt-test"


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["OPENAI_API_KEY"] == "from-shell"


def test_reasoning_effort_defaults_to_medium(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.reasoning_effort == "medium"


def test_reasoning_effort_can_be_disabled_with_empty_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.reasoning_effort is None


def test_domain_loads_output_schema_from_json_file() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    assert domain.output_schema["type"] == "object"
    assert "agent_artifact" in domain.output_schema["required"]
    assert "judge_artifact" in domain.output_schema["required"]
