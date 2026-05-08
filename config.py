from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    reasoning_effort: Optional[str] = "medium"
    request_timeout_seconds: float = 180.0


class DomainConfig(BaseModel):
    domain_id: str
    dataset_version: str = "poc-1"
    case_types: list[str]
    difficulties: list[int]
    scenarios: list[str]
    abilities: list[str] = Field(default_factory=list)
    environments: list[str] = Field(default_factory=list)
    diagnostic_pressure_types: list[str] = Field(default_factory=list)
    scoring_methods: list[str] = Field(default_factory=list)
    route_codes: list[str]
    subcodes: list[str]
    reason_codes: list[str]
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
    embedding_model: Optional[str] = None,
    console_progress: bool = True,
) -> RuntimeConfig:
    load_env_file()
    domain = load_domain(domain_path)
    models = ModelConfig(
        provider=provider or os.getenv("OPENAI_PROVIDER", "openai"),
        model=model or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        embedding_model=embedding_model
        or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium") or None,
        request_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "180")),
    )
    return RuntimeConfig(
        domain=domain,
        domain_path=Path(domain_path),
        target_stage=target_stage,
        target_n=target_n,
        seed=seed,
        run_id=run_id,
        models=models,
        console_progress=console_progress,
    )
