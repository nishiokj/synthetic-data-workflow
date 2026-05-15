from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    grader_input = _read_optional_json(os.environ.get("AGENTLAB_GRADER_INPUT_PATH")) or {}
    result_path = _result_path(grader_input)
    result = json.loads(result_path.read_text(encoding="utf-8"))
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
        "grader": {"name": "synth_pipeline_passthrough", "strategy": "in_task_runtime", "version": "0.1.0"},
    }
    Path(os.environ["AGENTLAB_MAPPED_GRADER_OUTPUT_PATH"]).write_text(
        json.dumps(conclusion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _read_optional_json(raw_path: str | None) -> dict:
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _result_path(grader_input: dict) -> Path:
    paths = grader_input.get("paths") if isinstance(grader_input.get("paths"), dict) else {}
    legacy = paths.get("result_path")
    if legacy:
        return Path(legacy)
    return Path(os.environ.get("AGENTLAB_RESULT_PATH", "/agentlab/out/result.json"))


if __name__ == "__main__":
    raise SystemExit(main())
