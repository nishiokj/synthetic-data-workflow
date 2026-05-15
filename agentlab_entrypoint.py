from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from agents import ADVERSARY_ATTACK_TYPE_TAXONOMY, ProviderError
from config import DomainConfig, build_runtime_config
from models import GenerationEnvelope, GenerationPipelineInput
from pipeline import PipelineRunner

DOMAIN_RUBRIC_CONTEXT_TOKEN = "{{DOMAIN_RUBRIC_CONTEXT}}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one AgentLab trial through the generator-first pipeline.")
    parser.add_argument("--domain", required=True, help="Path to the synth pipeline domain YAML.")
    parser.add_argument("--trial-input-path", default=os.getenv("AGENTLAB_TRIAL_INPUT_PATH"))
    parser.add_argument("--result-path", default=os.getenv("AGENTLAB_RESULT_PATH"))
    parser.add_argument("--output-dir", default=os.getenv("AGENTLAB_SESSION_CONTEXT_ROOT"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--auth-file", default=None)
    parser.add_argument("--workspace-validation-executor", choices=["docker", "local"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.trial_input_path:
        print("AGENTLAB_TRIAL_INPUT_PATH or --trial-input-path is required", file=sys.stderr)
        return 2
    if not args.result_path:
        print("AGENTLAB_RESULT_PATH or --result-path is required", file=sys.stderr)
        return 2

    trial_input_path = Path(args.trial_input_path)
    result_path = Path(args.result_path)
    output_dir = Path(args.output_dir) if args.output_dir else result_path.parent / "synth-pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        trial_input = _read_json(trial_input_path)
        ids = _trial_ids(trial_input)
        if os.getenv("AGENTLAB_PREFLIGHT_SMOKE") == "1":
            _write_json(result_path, _preflight_result(ids))
            return 0
        bindings = trial_input.get("bindings") if isinstance(trial_input.get("bindings"), dict) else {}
        envelope = _generation_envelope_from_trial(trial_input)
        domain_path = _resolve_domain_path(envelope.domain_ref, fallback=args.domain)
        pipeline_run_id = _pipeline_run_id(ids)
        generator_system_prompt_override = str(
            bindings.get("generator_system_prompt_override") or os.getenv("GENERATOR_SYSTEM_PROMPT_OVERRIDE", "")
        )
        raw_generator_system_prompt_append = str(
            bindings.get("generator_system_prompt_append") or os.getenv("GENERATOR_SYSTEM_PROMPT_APPEND", "")
        )
        os.environ.setdefault("EMBEDDING_PROVIDER", "local")

        config = build_runtime_config(
            domain_path=domain_path,
            target_stage="benchmark",
            target_n=1,
            seed=args.seed,
            run_id=pipeline_run_id,
            model=args.model or _optional_str(bindings.get("model")),
            provider=args.provider or _optional_str(bindings.get("model_provider") or bindings.get("provider")),
            auth_file=args.auth_file or _optional_str(bindings.get("auth_file")),
            generator_system_prompt_override=generator_system_prompt_override,
            generator_system_prompt_append=raw_generator_system_prompt_append,
            workspace_validation_executor=args.workspace_validation_executor
            or _optional_str(bindings.get("workspace_validation_executor")),
            console_progress=False,
        )
        generator_system_prompt_append = _expand_generator_system_prompt_append(
            raw_generator_system_prompt_append,
            domain=config.domain,
        )
        config.generator_system_prompt_append = generator_system_prompt_append
        config.logs_dir = output_dir / "logs"
        config.data_dir = output_dir / "data"

        result = PipelineRunner(config).run_from_generation(
            GenerationPipelineInput(envelope=envelope, output_dir=output_dir / "result")
        )
        run_dir = config.logs_dir / pipeline_run_id
        metrics = _metrics_from_run(result.model_dump(mode="json"), run_dir)
        metrics["domain_id"] = config.domain.domain_id
        metrics["domain_ref"] = envelope.domain_ref or str(config.domain_path)
        artifacts = _artifact_declarations(output_dir)
        agent_result = {
            "schema_version": "agent_result_v1",
            "ids": ids,
            "outcome": "success" if result.final_status == "committed" else "failure",
            "metrics": metrics,
            "objective": {"name": "committed", "value": float(metrics["committed"]), "direction": "maximize"},
            "answer": {
                "pipeline_run_id": pipeline_run_id,
                "final_status": result.final_status,
                "candidate_id": result.candidate_id,
                "domain_ref": envelope.domain_ref,
                "resolved_domain_path": str(config.domain_path),
                "resolved_domain_id": config.domain.domain_id,
                "generator_system_prompt_override_present": bool(generator_system_prompt_override.strip()),
                "generator_system_prompt_append": generator_system_prompt_append,
                "logs_dir": str(run_dir),
                "materialized_dir": str(result.materialized_dir),
            },
            "artifacts": artifacts,
        }
    except ProviderError as exc:
        agent_result = _error_result(_trial_ids_from_env(), "provider_error", str(exc))
    except Exception as exc:
        agent_result = _error_result(_trial_ids_from_env(), type(exc).__name__, str(exc))

    _write_json(result_path, agent_result)
    return 0 if agent_result["outcome"] == "success" else 1


def _generation_envelope_from_trial(trial_input: dict[str, Any]) -> GenerationEnvelope:
    task = trial_input.get("task") if isinstance(trial_input.get("task"), dict) else {}
    task_input = task.get("input") if isinstance(task.get("input"), dict) else {}
    raw = task_input.get("generation_envelope") or task.get("generation_envelope")
    if not isinstance(raw, dict):
        raise ValueError("trial input must include task.input.generation_envelope")
    return GenerationEnvelope.model_validate(raw)


def _resolve_domain_path(domain_ref: str | None, *, fallback: str) -> Path:
    if not domain_ref or not domain_ref.strip():
        return Path(fallback)
    ref = Path(domain_ref)
    if ref.is_absolute():
        return ref

    candidates = [
        Path.cwd() / ref,
        Path(__file__).resolve().parent / ref,
        Path(fallback).resolve().parent / ref,
        Path(fallback).resolve().parent / ref.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent / ref


def _expand_generator_system_prompt_append(raw: str, *, domain: DomainConfig) -> str:
    if DOMAIN_RUBRIC_CONTEXT_TOKEN not in raw:
        return raw
    return raw.replace(DOMAIN_RUBRIC_CONTEXT_TOKEN, _domain_rubric_context(domain))


def _domain_rubric_context(domain: DomainConfig) -> str:
    parts = [
        "Use these domain-authored benchmark quality and scoring rules while generating the candidate.",
        "Treat them as generator-facing rubric context; do not copy this text into candidate-facing materials.",
        "",
        "QUALITY GATE RULES:",
    ]
    parts.extend(f"- {rule}" for rule in domain.quality_gate_rules)
    parts.extend(["", "RUBRIC GATE RULES:"])
    parts.extend(f"- {rule}" for rule in domain.rubric_gate_rules)
    return "\n".join(parts).strip()


def _trial_ids(trial_input: dict[str, Any]) -> dict[str, Any]:
    raw_ids = trial_input.get("ids") if isinstance(trial_input.get("ids"), dict) else {}
    return {
        "run_id": str(raw_ids.get("run_id") or os.getenv("AGENTLAB_RUN_ID") or "unknown-run"),
        "trial_id": str(raw_ids.get("trial_id") or os.getenv("AGENTLAB_TRIAL_ID") or "unknown-trial"),
        "variant_id": str(raw_ids.get("variant_id") or os.getenv("AGENTLAB_VARIANT_ID") or "unknown-variant"),
        "task_id": str(raw_ids.get("task_id") or os.getenv("AGENTLAB_TASK_ID") or "unknown-task"),
        "repl_idx": int(raw_ids.get("repl_idx") or os.getenv("AGENTLAB_REPL_IDX") or 0),
    }


def _trial_ids_from_env() -> dict[str, Any]:
    return {
        "run_id": os.getenv("AGENTLAB_RUN_ID", "unknown-run"),
        "trial_id": os.getenv("AGENTLAB_TRIAL_ID", "unknown-trial"),
        "variant_id": os.getenv("AGENTLAB_VARIANT_ID", "unknown-variant"),
        "task_id": os.getenv("AGENTLAB_TASK_ID", "unknown-task"),
        "repl_idx": int(os.getenv("AGENTLAB_REPL_IDX", "0")),
    }


def _pipeline_run_id(ids: dict[str, Any]) -> str:
    parts = [ids["run_id"], ids["variant_id"], ids["task_id"], ids["trial_id"], str(ids["repl_idx"])]
    return "-".join(_safe_segment(part) for part in parts)


def _metrics_from_run(result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage_records = _read_jsonl(run_dir / "stage_records.jsonl")
    validations = _read_jsonl(run_dir / "validation.jsonl")
    adversary_reports = _read_jsonl(run_dir / "adversary.jsonl")
    quality = _last_by(validations, "check_kind", "quality")
    rubric = _last_by(validations, "check_kind", "rubric")
    final_subcodes = result.get("subcodes") if isinstance(result.get("subcodes"), list) else []
    adversary = adversary_reports[-1] if adversary_reports else {}
    attacks = adversary.get("attacks") if isinstance(adversary.get("attacks"), list) else []
    attack_group_counts = _attack_group_counts(attacks)
    attack_type_counts = _attack_type_counts(attacks)
    disposition = str(adversary.get("revision_disposition") or "missing")
    quality_ensemble = _records_by_role(stage_records, "quality_gate_candidate_ensemble")
    rubric_ensemble = _records_by_role(stage_records, "rubric_gate_candidate_ensemble")
    quality_ensemble_details = _gate_record_details(quality_ensemble)
    rubric_ensemble_details = _gate_record_details(rubric_ensemble)
    return {
        "committed": int(result.get("committed") or 0),
        "dropped": int(result.get("dropped") or 0),
        "final_status": str(result.get("final_status") or "unknown"),
        "candidate_id_present": bool(result.get("candidate_id")),
        "final_route_code": str(result.get("route_code") or ""),
        "final_subcode_count": len(final_subcodes),
        "final_subcodes": final_subcodes,
        "gate_caveat_count": len(final_subcodes),
        "adversary_disposition": disposition,
        "adversary_attack_count": len(attacks),
        "adversary_attack_type_counts": attack_type_counts,
        "adversary_explicitly_covered_attack_count": attack_group_counts["explicit"],
        "adversary_indirectly_covered_attack_count": attack_group_counts["indirect"],
        "adversary_uncovered_attack_count": attack_group_counts["uncovered"],
        "adversary_revision_required": disposition == "revise",
        "adversary_nuked": disposition == "nuke",
        "adversary_proxy_damage_present": bool(str(adversary.get("proxy_damage") or "").strip()),
        "adversary_cheap_pass_strategy_present": bool(str(adversary.get("cheap_pass_strategy") or "").strip()),
        "quality_gate_verdict": str(quality.get("verdict") or "missing"),
        "quality_gate_route_code": str(quality.get("route_code") or ""),
        "quality_gate_subcode_count": len(quality.get("subcodes") if isinstance(quality.get("subcodes"), list) else []),
        "quality_gate_subcodes": quality.get("subcodes") if isinstance(quality.get("subcodes"), list) else [],
        "quality_gate_rejected": quality.get("verdict") == "reject",
        "quality_gate_ensemble_count": len(quality_ensemble),
        "quality_gate_ensemble_reject_count": _reject_count(quality_ensemble),
        "quality_gate_ensemble_subcode_count": _subcode_count(quality_ensemble),
        "quality_gate_ensemble_details": quality_ensemble_details,
        "rubric_gate_verdict": str(rubric.get("verdict") or "missing"),
        "rubric_gate_route_code": str(rubric.get("route_code") or ""),
        "rubric_gate_subcode_count": len(rubric.get("subcodes") if isinstance(rubric.get("subcodes"), list) else []),
        "rubric_gate_subcodes": rubric.get("subcodes") if isinstance(rubric.get("subcodes"), list) else [],
        "rubric_gate_rejected": rubric.get("verdict") == "reject",
        "rubric_gate_ensemble_count": len(rubric_ensemble),
        "rubric_gate_ensemble_reject_count": _reject_count(rubric_ensemble),
        "rubric_gate_ensemble_subcode_count": _subcode_count(rubric_ensemble),
        "rubric_gate_ensemble_details": rubric_ensemble_details,
        "generation_attempts": _count_role(stage_records, "generate_candidate_sample"),
        "revision_count": _count_role(stage_records, "revise_candidate_from_adversary"),
        "input_tokens_total": sum(int(record.get("input_tokens") or 0) for record in stage_records),
        "output_tokens_total": sum(int(record.get("output_tokens") or 0) for record in stage_records),
        "latency_ms_total": sum(int(record.get("latency_ms") or 0) for record in stage_records),
        "cost_usd_total": float(sum(float(record.get("cost_usd") or 0.0) for record in stage_records)),
    }


def _artifact_declarations(output_dir: Path) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            artifacts.append({"path": str(path), "logical_name": str(path.relative_to(output_dir))})
    return artifacts


def _error_result(ids: dict[str, Any], error_type: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": "agent_result_v1",
        "ids": ids,
        "outcome": "error",
        "metrics": {"committed": 0, "dropped": 1, "final_status": "error"},
        "objective": {"name": "committed", "value": 0.0, "direction": "maximize"},
        "answer": {"final_status": "error"},
        "error": {"error_type": error_type, "message": message},
    }


def _preflight_result(ids: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "agent_result_v1",
        "ids": ids,
        "outcome": "success",
        "metrics": {
            "committed": 0,
            "dropped": 0,
            "final_status": "preflight_smoke",
            "adversary_attack_count": 0,
            "generation_attempts": 0,
            "revision_count": 0,
        },
        "objective": {"name": "committed", "value": 0.0, "direction": "maximize"},
        "answer": {"final_status": "preflight_smoke"},
    }


def _last_by(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get(key) == value]
    return matches[-1] if matches else {}


def _count_role(rows: list[dict[str, Any]], role: str) -> int:
    return sum(1 for row in rows if row.get("role") == role)


def _records_by_role(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("role") == role]


def _reject_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("verdict") == "reject")


def _subcode_count(rows: list[dict[str, Any]]) -> int:
    return sum(len(row.get("subcodes") if isinstance(row.get("subcodes"), list) else []) for row in rows)


def _gate_record_details(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        subcodes = row.get("subcodes") if isinstance(row.get("subcodes"), list) else []
        details.append(
            {
                "index": index,
                "provider": str(row.get("provider") or ""),
                "model": str(row.get("model") or ""),
                "verdict": str(row.get("verdict") or ""),
                "route_code": str(row.get("route_code") or ""),
                "subcodes": subcodes,
                "subcode_count": len(subcodes),
                "input_tokens": int(row.get("input_tokens") or 0),
                "output_tokens": int(row.get("output_tokens") or 0),
                "latency_ms": int(row.get("latency_ms") or 0),
                "cost_usd": float(row.get("cost_usd") or 0.0),
            }
        )
    return details


def _attack_type_counts(attacks: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attack in attacks:
        if not isinstance(attack, dict):
            attack_type = "unknown"
        else:
            attack_type = str(attack.get("attack_type") or "unknown").strip() or "unknown"
        counts[attack_type] = counts.get(attack_type, 0) + 1
    return counts


def _attack_group_counts(attacks: list[Any]) -> dict[str, int]:
    counts = {"explicit": 0, "indirect": 0, "uncovered": 0}
    for attack in attacks:
        attack_type = ""
        if isinstance(attack, dict):
            attack_type = str(attack.get("attack_type") or "").strip()
        group = ADVERSARY_ATTACK_TYPE_TAXONOMY.get(attack_type, ADVERSARY_ATTACK_TYPE_TAXONOMY["other"])[
            "coverage_group"
        ]
        if group not in counts:
            group = "uncovered"
        counts[group] += 1
    return counts


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_segment(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in text)
    return safe.strip(".-") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
