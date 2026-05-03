from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a synthetic-data pipeline run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir) / args.run_id
    corpus_path = Path(args.data_dir) / "corpus" / "benchmark" / f"{args.run_id}.jsonl"
    stage_records_path = logs_dir / "stage_records.jsonl"
    if not stage_records_path.exists():
        print(f"missing stage log: {stage_records_path}", file=sys.stderr)
        return 1
    records = _read_jsonl(stage_records_path)
    validations = _read_jsonl(logs_dir / "validation.jsonl")
    committed = _read_jsonl(corpus_path)

    route_counts = Counter(record.get("route_code") for record in records)
    stage_counts = Counter(record.get("stage_kind") for record in records)
    validation_counts = Counter(value.get("verdict") for value in validations)
    cells = Counter(_cell_key(value) for value in committed)
    candidates = [value.get("candidate", {}) for value in committed]
    metrics = {
        "run_id": args.run_id,
        "committed_count": len(committed),
        "stage_count": len(records),
        "route_code_distribution": dict(route_counts),
        "stage_distribution": dict(stage_counts),
        "validation_verdict_distribution": dict(validation_counts),
        "coverage_entropy": _entropy(cells),
        "coverage_cells": dict(cells),
        "case_type_distribution": dict(Counter(_cell(value).get("case_type") for value in committed)),
        "ability_distribution": dict(Counter(_name(candidate.get("ability_z")) for candidate in candidates)),
        "environment_distribution": dict(Counter(_name(candidate.get("environment_y")) for candidate in candidates)),
        "scoring_method_distribution": dict(Counter((candidate.get("score_x") or {}).get("score_type") for candidate in candidates)),
        "diagnostic_pressure_distribution": dict(Counter(tag for candidate in candidates for tag in candidate.get("diagnostic_pressure", []))),
        "leakage_risk_count_distribution": dict(Counter(len(candidate.get("leakage_risks", [])) for candidate in candidates)),
        "known_limit_count_distribution": dict(Counter(len(candidate.get("known_limits", [])) for candidate in candidates)),
        "deterministic_pass_rate": _pass_rate(validations, "deterministic"),
        "quality_gate_pass_rate": _pass_rate(validations, "quality"),
        "rubric_gate_pass_rate": _pass_rate(validations, "rubric"),
        "curator_accept_rate": _pass_rate(validations, "curation"),
    }

    metrics_path = logs_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(f"run_id={args.run_id}")
    print(f"committed_count={metrics['committed_count']}")
    print(f"stage_count={metrics['stage_count']}")
    print(f"coverage_entropy={metrics['coverage_entropy']:.4f}")
    print(f"deterministic_pass_rate={metrics['deterministic_pass_rate']:.4f}")
    print(f"quality_gate_pass_rate={metrics['quality_gate_pass_rate']:.4f}")
    print(f"rubric_gate_pass_rate={metrics['rubric_gate_pass_rate']:.4f}")
    print(f"curator_accept_rate={metrics['curator_accept_rate']:.4f}")
    print(f"metrics={metrics_path}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line))
    return values


def _cell_key(value: dict[str, Any]) -> str:
    cell = _cell(value)
    return f"{cell.get('case_type')}|{cell.get('difficulty')}|{cell.get('scenario')}"


def _cell(value: dict[str, Any]) -> dict[str, Any]:
    cell = value.get("taxonomy_cell", {})
    return cell if isinstance(cell, dict) else {}


def _name(value: Any) -> str | None:
    return value.get("name") if isinstance(value, dict) else None


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    return entropy


def _pass_rate(values: list[dict[str, Any]], check_kind: str) -> float:
    scoped = [value for value in values if value.get("check_kind") == check_kind]
    if not scoped:
        return 0.0
    passed = sum(1 for value in scoped if value.get("verdict") == "accept")
    return passed / len(scoped)


if __name__ == "__main__":
    raise SystemExit(main())
