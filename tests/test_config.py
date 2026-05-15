from __future__ import annotations

import os

from config import build_runtime_config, load_domain, load_env_file


def test_load_env_file_sets_missing_values(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("MODEL_API_KEY=sk-test\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["MODEL_API_KEY"] == "sk-test"


def test_load_env_file_strips_quotes_and_export(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text('export MODEL_NAME="gpt-test"\n', encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["MODEL_NAME"] == "gpt-test"


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "from-shell")
    env_path = tmp_path / ".env"
    env_path.write_text("MODEL_API_KEY=from-file\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["MODEL_API_KEY"] == "from-shell"


def test_reasoning_effort_defaults_to_medium(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_REASONING_EFFORT", raising=False)

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.reasoning_effort == "medium"


def test_model_defaults_to_gpt_5_5(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.model == "gpt-5.5"


def test_embedding_defaults_to_local_hash(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.embedding_provider == "local"
    assert config.models.embedding_model == "local-hash-embedding"


def test_openai_model_env_is_not_used_after_langchain_cutover(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.model == "gpt-5.5"


def test_reasoning_effort_can_be_disabled_with_empty_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_REASONING_EFFORT", "")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.reasoning_effort is None


def test_model_auth_file_can_be_set_with_env(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("MODEL_AUTH_FILE", str(auth_file))

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.auth_file == auth_file


def test_gate_ensemble_defaults_to_deepinfra_kimi_when_key_is_present(monkeypatch) -> None:
    monkeypatch.setenv("GATE_ENSEMBLE_1_API_KEY", "deepinfra-test-key")
    monkeypatch.delenv("GATE_ENSEMBLE_1_MODEL", raising=False)
    monkeypatch.delenv("GATE_ENSEMBLE_1_PROVIDER", raising=False)
    monkeypatch.delenv("GATE_ENSEMBLE_1_BASE_URL", raising=False)

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert len(config.gate_ensemble_models) == 1
    gate_model = config.gate_ensemble_models[0]
    assert gate_model.provider == "openai"
    assert gate_model.model == "moonshotai/Kimi-K2.6"
    assert gate_model.base_url == "https://api.deepinfra.com/v1/openai"
    assert gate_model.api_key is not None
    assert gate_model.api_key.get_secret_value() == "deepinfra-test-key"


def test_runtime_config_accepts_workspace_validation_executor_argument(monkeypatch) -> None:
    config = build_runtime_config(
        domain_path="domains/benchmark_code_debug.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=1,
        run_id="workspace-executor-test",
        workspace_validation_executor="local",
    )

    assert config.workspace_validation_executor == "local"


def test_domain_loads_output_schema_from_json_file() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    assert domain.output_schema["type"] == "object"
    assert "agent_artifact" in domain.output_schema["required"]
    assert "judge_artifact" in domain.output_schema["required"]
