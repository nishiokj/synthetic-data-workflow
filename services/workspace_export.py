from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from models import CandidateSample, CommittedSample, RoutingDecision
from services.virtual_workspace import VirtualWorkspace, VirtualWorkspaceError, normalize_workspace_path


class WorkspaceExport:
    def __init__(self, *, logs_dir: Path, data_dir: Path, run_id: str) -> None:
        self.logs_dir = logs_dir
        self.data_dir = data_dir
        self.run_id = run_id

    def export_snapshot(
        self,
        candidate: CandidateSample,
        *,
        phase: str,
        role: str,
        retry_index: int,
        parent_candidate_id: str | None = None,
        adversary_report_id: str | None = None,
    ) -> Path | None:
        root = self.logs_dir / self.run_id / "workspaces" / _safe_segment(phase) / _safe_segment(candidate.id)
        metadata = {
            "run_id": self.run_id,
            "phase": phase,
            "role": role,
            "retry_index": retry_index,
            "candidate_id": candidate.id,
            "design_id": candidate.design_id,
            "parent_candidate_id": parent_candidate_id,
            "adversary_report_id": adversary_report_id,
        }
        return self._export_candidate(candidate, root, metadata)

    def export_rejection(self, candidate: CandidateSample, decision: RoutingDecision) -> Path | None:
        root = self.logs_dir / self.run_id / "workspaces" / "rejected" / _safe_segment(candidate.id)
        metadata = {
            "run_id": self.run_id,
            "phase": "rejected",
            "candidate_id": candidate.id,
            "design_id": candidate.design_id,
            "route": decision.model_dump(mode="json"),
        }
        return self._export_candidate(candidate, root, metadata)

    def export_committed(self, committed: CommittedSample) -> Path | None:
        candidate = committed.candidate
        root = self.data_dir / "materialized" / "benchmark" / self.run_id / _safe_segment(candidate.id)
        metadata = {
            "run_id": self.run_id,
            "phase": "committed",
            "committed_id": committed.id,
            "certified_id": committed.certified_id,
            "candidate_id": candidate.id,
            "design_id": candidate.design_id,
            "taxonomy_cell": committed.taxonomy_cell.model_dump(mode="json"),
            "nn_distance": committed.nn_distance,
        }
        return self._export_candidate(candidate, root, metadata)

    def _export_candidate(self, candidate: CandidateSample, root: Path, metadata: dict[str, Any]) -> Path | None:
        artifact = candidate.agent_artifact.environment_artifact
        if artifact is None or artifact.kind != "virtual_workspace":
            return None
        export_warning = None
        try:
            workspace = VirtualWorkspace.from_payload(artifact.payload)
            files = [(path, workspace.read_file(path)) for path in workspace.list_files()]
            commands = workspace.commands
        except VirtualWorkspaceError as exc:
            files = _raw_safe_files(artifact.payload)
            commands = _raw_commands(artifact.payload)
            export_warning = {"subcode": exc.subcode, "path": exc.path, "message": exc.message}
            if not files:
                return None

        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        for path, content in files:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        task_recipe = _task_image_recipe(candidate.agent_artifact.runtime_requirements, commands)
        if task_recipe is not None:
            task_dir = root / "task"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.json").write_text(json.dumps(task_recipe["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (task_dir / "Dockerfile").write_text(task_recipe["dockerfile"], encoding="utf-8")
        metadata_path = root / "_benchmark_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    **metadata,
                    "commands": commands,
                    "runtime_requirements": candidate.agent_artifact.runtime_requirements,
                    "export_warning": export_warning,
                    "benchmark_case": candidate.agent_artifact.benchmark_case,
                    "ability_z": candidate.ability_z,
                    "environment_y": candidate.environment_y,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return root


def _safe_segment(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    return safe.strip(".-") or "unnamed"


def _raw_safe_files(payload: dict[str, Any]) -> list[tuple[str, str]]:
    raw_files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(raw_files, list):
        return []
    files: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, file_entry in enumerate(raw_files):
        if not isinstance(file_entry, dict):
            continue
        try:
            path = normalize_workspace_path(file_entry.get("path"), f"files.{index}.path")
        except VirtualWorkspaceError:
            continue
        content = file_entry.get("content")
        if not isinstance(content, str) or path in seen:
            continue
        seen.add(path)
        files.append((path, content))
    return files


def _raw_commands(payload: dict[str, Any]) -> dict[str, str]:
    raw_commands = payload.get("commands") if isinstance(payload, dict) else None
    if not isinstance(raw_commands, dict):
        return {}
    return {str(key): str(value) for key, value in raw_commands.items()}


def _task_image_recipe(runtime_requirements: dict[str, Any] | None, workspace_commands: dict[str, str]) -> dict[str, Any] | None:
    if not isinstance(runtime_requirements, dict) or runtime_requirements.get("kind") != "filesystem_task":
        return None
    execution = runtime_requirements.get("execution")
    if not isinstance(execution, dict) or execution.get("mode") not in {"task_image", "container"}:
        return None
    base_image = execution.get("base_image")
    if not isinstance(base_image, str) or not base_image.strip():
        return None

    commands = dict(workspace_commands)
    runtime_commands = runtime_requirements.get("commands")
    if isinstance(runtime_commands, dict):
        commands.update({str(key): str(value) for key, value in runtime_commands.items()})

    test_command = commands.get("test", "")
    install_commands = _command_list(commands.get("install"))
    dockerfile_lines = [
        f"FROM {base_image.strip()}",
        "WORKDIR /workspace",
        "COPY . /workspace",
        "RUN rm -rf /workspace/task",
    ]
    for command in install_commands:
        dockerfile_lines.append(f"RUN {command}")
    if test_command:
        dockerfile_lines.append(f"CMD {json.dumps(['sh', '-lc', test_command])}")
    dockerfile = "\n".join(dockerfile_lines) + "\n"

    manifest = {
        "schema_version": "task-image.v1",
        "workspace_dir": "/workspace",
        "build": {
            "context": ".",
            "dockerfile": "task/Dockerfile",
            "base_image": base_image.strip(),
        },
        "commands": commands,
        "runtime_requirements": runtime_requirements,
        "network": runtime_requirements.get("network"),
    }
    return {"manifest": manifest, "dockerfile": dockerfile}


def _command_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []
