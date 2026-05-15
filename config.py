from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr


class ModelConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-5.5"
    embedding_provider: str = "local"
    embedding_model: str = "local-hash-embedding"
    base_url: Optional[str] = None
    api_key: Optional[SecretStr] = Field(default=None, repr=False)
    auth_file: Optional[Path] = None
    reasoning_effort: Optional[str] = "medium"
    request_timeout_seconds: float = 180.0


class DomainConfig(BaseModel):
    domain_id: str
    case_types: list[str]
    difficulties: list[int]
    scenarios: list[str]
    abilities: list[str] = Field(default_factory=list)
    environments: list[str] = Field(default_factory=list)
    diagnostic_pressure_types: list[str] = Field(default_factory=list)
    scoring_methods: list[str] = Field(default_factory=list)
    route_codes: list[str]
    subcodes: list[str]
    novelty_threshold: float = 0.08
    max_design_retries: int = 2
    max_generation_retries: int = 2
    deterministic_rules: dict[str, Any] = Field(default_factory=dict)
    semantic_rules: list[str] = Field(default_factory=list)
    general_probe_principles: dict[str, Any] = Field(default_factory=dict)
    anti_overfit_policy: list[str] = Field(default_factory=list)
    quality_gate_rules: list[str] = Field(default_factory=list)
    rubric_gate_rules: list[str] = Field(default_factory=list)
    generator_guidance: dict[str, Any] = Field(default_factory=dict)
    output_schema_path: Optional[str] = None
    output_schema: dict[str, Any] = Field(default_factory=dict)
    benchmark_case_schema: dict[str, Any]


class RuntimeConfig(BaseModel):
    domain: DomainConfig
    domain_path: Path
    target_stage: str = "benchmark"
    target_n: int = 5
    seed: int = 42
    run_id: str
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    models: ModelConfig = Field(default_factory=ModelConfig)
    gate_ensemble_models: list[ModelConfig] = Field(default_factory=list)
    generator_system_prompt_override: str = ""
    generator_system_prompt_append: str = ""
    workspace_validation_executor: str = "docker"
    console_progress: bool = True


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_env_value(value.strip())
            if key and key not in os.environ:
                os.environ[key] = value


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_domain(path: str | Path) -> DomainConfig:
    domain_path = Path(path)
    with domain_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    schema_path = raw.get("output_schema_path")
    if schema_path:
        resolved = Path(schema_path)
        if not resolved.is_absolute():
            resolved = domain_path.parent / resolved
        with resolved.open("r", encoding="utf-8") as schema_handle:
            raw["output_schema"] = json.load(schema_handle)
    return DomainConfig.model_validate(raw)


def build_runtime_config(
    *,
    domain_path: str | Path,
    target_stage: str,
    target_n: int,
    seed: int,
    run_id: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    auth_file: Optional[str | Path] = None,
    embedding_model: Optional[str] = None,
    generator_system_prompt_override: Optional[str] = None,
    generator_system_prompt_append: Optional[str] = None,
    workspace_validation_executor: Optional[str] = None,
    console_progress: bool = True,
) -> RuntimeConfig:
    load_env_file()
    domain = load_domain(domain_path)
    models = ModelConfig(
        provider=provider or os.getenv("MODEL_PROVIDER", "openai"),
        model=model or os.getenv("MODEL_NAME", "gpt-5.5"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        embedding_model=embedding_model or os.getenv("EMBEDDING_MODEL", "local-hash-embedding"),
        base_url=os.getenv("MODEL_BASE_URL") or None,
        api_key=_optional_secret(os.getenv("MODEL_API_KEY")),
        auth_file=_resolve_optional_path(auth_file or os.getenv("MODEL_AUTH_FILE")),
        reasoning_effort=os.getenv("MODEL_REASONING_EFFORT", "medium") or None,
        request_timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "180")),
    )
    return RuntimeConfig(
        domain=domain,
        domain_path=Path(domain_path),
        target_stage=target_stage,
        target_n=target_n,
        seed=seed,
        run_id=run_id,
        models=models,
        gate_ensemble_models=_load_gate_ensemble_models(),
        generator_system_prompt_override=generator_system_prompt_override
        if generator_system_prompt_override is not None
        else os.getenv("GENERATOR_SYSTEM_PROMPT_OVERRIDE", ""),
        generator_system_prompt_append=generator_system_prompt_append
        if generator_system_prompt_append is not None
        else os.getenv("GENERATOR_SYSTEM_PROMPT_APPEND", ""),
        workspace_validation_executor=workspace_validation_executor
        if workspace_validation_executor is not None
        else "docker",
        console_progress=console_progress,
    )


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _optional_secret(value: str | None) -> SecretStr | None:
    if value is None or not value.strip():
        return None
    return SecretStr(value.strip())


def _load_gate_ensemble_models() -> list[ModelConfig]:
    models: list[ModelConfig] = []
    index = 1
    while True:
        prefix = f"GATE_ENSEMBLE_{index}_"
        has_any = any(key.startswith(prefix) for key in os.environ)
        if not has_any:
            break
        models.append(
            ModelConfig(
                provider=os.getenv(prefix + "PROVIDER", "openai"),
                model=os.getenv(prefix + "MODEL", "moonshotai/Kimi-K2.6"),
                base_url=os.getenv(prefix + "BASE_URL", "https://api.deepinfra.com/v1/openai"),
                api_key=_optional_secret(os.getenv(prefix + "API_KEY")),
                reasoning_effort=os.getenv(prefix + "REASONING_EFFORT") or None,
                request_timeout_seconds=float(os.getenv(prefix + "TIMEOUT_SECONDS", os.getenv("MODEL_TIMEOUT_SECONDS", "180"))),
                embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "local-hash-embedding"),
            )
        )
        index += 1
    return models
