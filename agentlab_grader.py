from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def main() -> int:
    grader_input_path = _required_env("AGENTLAB_GRADER_INPUT_PATH")
    mapped_output_path = _required_env("AGENTLAB_MAPPED_GRADER_OUTPUT_PATH")
    grader_input = _read_json(Path(grader_input_path))
    result_path = Path(grader_input["paths"]["result_path"])
    result = _read_json(result_path)
    committed = float(result.get("metrics", {}).get("committed") or 0.0)
    conclusion = {
        "schema_version": "trial_conclusion_v1",
        "reported_outcome": "success" if committed >= 1.0 else "failure",
        "primary_metric": {"name": "committed", "value": committed},
        "payload": {
            "task_id": grader_input.get("ids", {}).get("task_id"),
            "variant_id": grader_input.get("ids", {}).get("variant_id"),
            "agent_outcome": result.get("outcome"),
            "metrics": result.get("metrics", {}),
            "answer": result.get("answer", {}),
        },
        "grader": {"name": "synth_pipeline_passthrough", "strategy": "in_task_image", "version": "0.1.0"},
    }
    _write_json(Path(mapped_output_path), conclusion)
    return 0


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
