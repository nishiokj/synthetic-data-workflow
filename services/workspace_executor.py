from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkspaceCommandResult:
    returncode: int
    stdout: str
    stderr: str
    executor: str
    detail: str = ""


APPROVED_PYTHON_IMAGES = {
    "python:3.10-slim",
    "python:3.11-slim",
    "python:3.12-slim",
}


def run_workspace_test(
    *,
    workspace_path: Path,
    argv: list[str],
    runtime_requirements: dict[str, Any] | None,
    timeout_seconds: float,
    executor: str | None = None,
) -> WorkspaceCommandResult:
    selected_executor = (executor or "docker").strip().lower()
    if selected_executor in {"local", "host"}:
        return _run_host(workspace_path=workspace_path, argv=argv, timeout_seconds=timeout_seconds)
    if selected_executor != "docker":
        raise ValueError("workspace validation executor must be 'docker' or 'local'")
    return _run_container(
        workspace_path=workspace_path,
        argv=argv,
        runtime_requirements=runtime_requirements,
        timeout_seconds=timeout_seconds,
    )


def validate_supported_container_runtime(runtime_requirements: dict[str, Any] | None) -> str | None:
    if not isinstance(runtime_requirements, dict) or runtime_requirements.get("kind") != "filesystem_task":
        return "executable workspaces must declare runtime_requirements.kind='filesystem_task'"

    execution = runtime_requirements.get("execution")
    if not isinstance(execution, dict) or execution.get("mode") not in {"task_image", "container"}:
        return "filesystem_task execution.mode must be task_image or container"
    base_image = execution.get("base_image")
    if base_image not in APPROVED_PYTHON_IMAGES:
        return f"filesystem_task base_image must be one of {sorted(APPROVED_PYTHON_IMAGES)}"

    language = runtime_requirements.get("language")
    if not isinstance(language, dict) or str(language.get("name", "")).lower() != "python":
        return "filesystem_task language.name must be python"

    dependencies = runtime_requirements.get("dependencies")
    if isinstance(dependencies, dict):
        policy = dependencies.get("policy")
        packages = dependencies.get("packages")
        if policy not in {None, "none", "stdlib_plus_runner"}:
            return "only dependency policy stdlib_plus_runner is supported for container validation"
        if packages is not None:
            if not isinstance(packages, list) or any(package != "pytest" for package in packages):
                return "only the pytest runner package is supported for container validation"

    network = runtime_requirements.get("network")
    if network not in {None, "disabled_during_eval"}:
        return "only disabled_during_eval network policy is supported for container validation"

    return None


def _run_host(*, workspace_path: Path, argv: list[str], timeout_seconds: float) -> WorkspaceCommandResult:
    host_argv = _host_argv(argv)
    completed = subprocess.run(
        host_argv,
        cwd=workspace_path,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return WorkspaceCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        executor="host",
    )


def _run_container(
    *,
    workspace_path: Path,
    argv: list[str],
    runtime_requirements: dict[str, Any] | None,
    timeout_seconds: float,
) -> WorkspaceCommandResult:
    runtime_error = validate_supported_container_runtime(runtime_requirements)
    if runtime_error is not None:
        raise ValueError(runtime_error)
    assert runtime_requirements is not None
    base_image = runtime_requirements["execution"]["base_image"]
    dockerfile = _write_validation_dockerfile(workspace_path, base_image, runtime_requirements, argv)
    try:
        build = subprocess.run(
            ["docker", "build", "-q", "-f", str(dockerfile.relative_to(workspace_path)), "."],
            cwd=workspace_path,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise
    except OSError as exc:
        raise RuntimeError(f"docker build could not be executed: {exc}") from exc

    if build.returncode != 0:
        return WorkspaceCommandResult(
            returncode=build.returncode,
            stdout=build.stdout,
            stderr=build.stderr,
            executor="docker",
            detail="docker build failed",
        )

    image_id = build.stdout.strip().splitlines()[-1].strip()
    run_cmd = ["docker", "run", "--rm"]
    if runtime_requirements.get("network") == "disabled_during_eval":
        run_cmd.extend(["--network", "none"])
    run_cmd.append(image_id)
    try:
        try:
            completed = subprocess.run(
                run_cmd,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError(f"docker run could not be executed: {exc}") from exc
        return WorkspaceCommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            executor="docker",
        )
    finally:
        subprocess.run(
            ["docker", "image", "rm", "-f", image_id],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )


def _write_validation_dockerfile(
    workspace_path: Path,
    base_image: str,
    runtime_requirements: dict[str, Any],
    argv: list[str],
) -> Path:
    task_dir = workspace_path / ".validation-task"
    task_dir.mkdir(exist_ok=True)
    dockerfile = task_dir / "Dockerfile"
    lines = [
        f"FROM {base_image}",
        "WORKDIR /workspace",
        "COPY . /workspace",
        "RUN rm -rf /workspace/.validation-task /workspace/task",
    ]
    install_command = _dependency_install_command(runtime_requirements)
    if install_command:
        lines.append(f"RUN {install_command}")
    lines.append(f"CMD {json.dumps(argv)}")
    dockerfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dockerfile


def _dependency_install_command(runtime_requirements: dict[str, Any]) -> str:
    dependencies = runtime_requirements.get("dependencies")
    if not isinstance(dependencies, dict):
        return ""
    if dependencies.get("policy") != "stdlib_plus_runner":
        return ""
    packages = dependencies.get("packages")
    if not isinstance(packages, list) or not packages:
        return ""
    return "python -m pip install --no-cache-dir " + " ".join(shlex.quote(str(package)) for package in packages)


def _host_argv(argv: list[str]) -> list[str]:
    if argv and Path(argv[0]).name in {"python", "python3"}:
        return [sys.executable, *argv[1:]]
    return argv
