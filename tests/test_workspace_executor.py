from __future__ import annotations

import subprocess

from services.workspace_executor import run_workspace_test, validate_supported_container_runtime


def _runtime(**overrides) -> dict:
    values = {
        "kind": "filesystem_task",
        "execution": {"mode": "container", "base_image": "python:3.12-slim", "os": "linux", "arch": "any"},
        "language": {"name": "python", "version": "3.12"},
        "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
        "commands": {"test": "python -m pytest -q"},
        "network": "disabled_during_eval",
    }
    values.update(overrides)
    return values


def test_container_runtime_allows_approved_python_image() -> None:
    assert validate_supported_container_runtime(_runtime()) is None


def test_container_runtime_rejects_unapproved_base_image() -> None:
    error = validate_supported_container_runtime(_runtime(execution={"mode": "container", "base_image": "python:latest"}))

    assert error is not None
    assert "base_image" in error


def test_container_executor_builds_image_and_disables_network(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["docker", "build", "-q"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:test-image\n", stderr="")
        if cmd[:3] == ["docker", "run", "--rm"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="FAILED tests/test_app.py::test_bug\n", stderr="")
        if cmd[:4] == ["docker", "image", "rm", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_workspace_test(
        workspace_path=tmp_path,
        argv=["python", "-m", "pytest", "-q"],
        runtime_requirements=_runtime(),
        timeout_seconds=10,
    )

    assert result.returncode == 1
    assert result.executor == "docker"
    assert calls[0][:3] == ["docker", "build", "-q"]
    assert calls[1] == ["docker", "run", "--rm", "--network", "none", "sha256:test-image"]
    assert calls[2] == ["docker", "image", "rm", "-f", "sha256:test-image"]
    dockerfile = tmp_path / ".validation-task" / "Dockerfile"
    assert "FROM python:3.12-slim" in dockerfile.read_text(encoding="utf-8")
    assert "python -m pip install --no-cache-dir pytest" in dockerfile.read_text(encoding="utf-8")


def test_local_executor_is_explicit_argument(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_workspace_test(
        workspace_path=tmp_path,
        argv=["python", "-m", "pytest", "-q"],
        runtime_requirements=_runtime(),
        timeout_seconds=10,
        executor="local",
    )

    assert result.returncode == 0
    assert result.executor == "host"
    assert calls[0][1:] == ["-m", "pytest", "-q"]
