from __future__ import annotations

import json

from models import CandidateSample, TaxonomyCell
from services.workspace_export import WorkspaceExport


def _candidate() -> CandidateSample:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=3, scenario="nominal")
    return CandidateSample(
        id="candidate/with:unsafe chars",
        design_id="design-1",
        content_hash="abc",
        cell=cell,
        agent_artifact={
            "benchmark_case": {"prompt": "Debug the workspace.", "setup": "Run tests."},
            "runtime_requirements": {
                "kind": "filesystem_task",
                "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
                "commands": {"test": "python -m pytest -q"},
                "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
            },
            "environment_artifact": {
                "kind": "virtual_workspace",
                "payload": {
                    "files": [
                        {"path": "pkg/__init__.py", "content": ""},
                        {"path": "pkg/app.py", "content": "def value():\n    return 1\n"},
                        {"path": "tests/test_app.py", "content": "from pkg.app import value\n\n\ndef test_value():\n    assert value() == 1\n"},
                    ],
                    "commands": {"test": "python -m pytest -q"},
                },
            },
        },
        judge_artifact={
            "score_x": {
                "score_type": "hard_checks_plus_rubric",
                "dimensions": [
                    {
                        "name": "correctness",
                        "weight": 1.0,
                        "high_score_criterion": "Tests pass.",
                        "low_score_criterion": "Tests fail.",
                    }
                ],
            },
            "proxy_claim": "This candidate exercises a small code-debugging workspace with inspectable files.",
            "diagnostic_pressure": ["workspace execution", "minimal patch"],
            "scoring_contract": {"credit": ["tests pass"], "penalties": ["test edits"]},
            "leakage_risks": ["Visible tests may invite shallow fixes."],
            "known_limits": ["Small synthetic workspace."],
            "negative_controls": [{"output": "edit tests", "should_fail_because": "weakens benchmark"}],
        },
        ability_z={"name": "fault_localization"},
        environment_y={"name": "single_turn_debug_with_test"},
        difficulty=3,
        case_type="proxy_strong",
    )


def test_workspace_export_writes_candidate_files_and_metadata(tmp_path) -> None:
    exporter = WorkspaceExport(logs_dir=tmp_path / "logs", data_dir=tmp_path / "data", run_id="run")

    root = exporter.export_snapshot(_candidate(), phase="generated", role="generate", retry_index=0)

    assert root == tmp_path / "logs" / "run" / "workspaces" / "generated" / "candidate-with-unsafe-chars"
    assert (root / "pkg" / "__init__.py").read_text(encoding="utf-8") == ""
    assert (root / "pkg" / "app.py").read_text(encoding="utf-8").startswith("def value")
    metadata = json.loads((root / "_benchmark_metadata.json").read_text(encoding="utf-8"))
    assert metadata["candidate_id"] == "candidate/with:unsafe chars"
    assert metadata["commands"] == {"test": "python -m pytest -q"}
    assert metadata["runtime_requirements"]["kind"] == "filesystem_task"
    assert metadata["benchmark_case"]["prompt"] == "Debug the workspace."
    task_manifest = json.loads((root / "task" / "task.json").read_text(encoding="utf-8"))
    assert task_manifest["schema_version"] == "task-image.v1"
    assert task_manifest["build"]["base_image"] == "python:3.11-slim"
    dockerfile = (root / "task" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "RUN python -m pip install --no-cache-dir pytest" in dockerfile
    assert 'CMD ["sh", "-lc", "python -m pytest -q"]' in dockerfile


def test_workspace_export_installs_python_pinned_manifest(tmp_path) -> None:
    candidate = _candidate()
    runtime = candidate.agent_artifact.runtime_requirements
    runtime["dependencies"] = {"policy": "pinned_manifest", "manifest_path": "requirements.txt"}
    payload = candidate.agent_artifact.environment_artifact.payload
    payload["files"].append({"path": "requirements.txt", "content": "pytest==9.0.2\n"})
    exporter = WorkspaceExport(logs_dir=tmp_path / "logs", data_dir=tmp_path / "data", run_id="run")

    root = exporter.export_snapshot(candidate, phase="generated", role="generate", retry_index=0)

    dockerfile = (root / "task" / "Dockerfile").read_text(encoding="utf-8")
    assert "RUN python -m pip install --no-cache-dir -r requirements.txt" in dockerfile


def test_workspace_export_preserves_raw_files_for_invalid_workspace(tmp_path) -> None:
    candidate = _candidate()
    payload = candidate.agent_artifact.environment_artifact.payload
    payload["files"][1]["content"] = "..."
    exporter = WorkspaceExport(logs_dir=tmp_path / "logs", data_dir=tmp_path / "data", run_id="run")

    root = exporter.export_snapshot(candidate, phase="generated", role="generate", retry_index=0)

    assert (root / "pkg" / "app.py").read_text(encoding="utf-8") == "..."
    metadata = json.loads((root / "_benchmark_metadata.json").read_text(encoding="utf-8"))
    assert metadata["export_warning"]["subcode"] == "placeholder_workspace_file"
