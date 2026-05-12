from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Any

from config import DomainConfig
from models import CandidateSample, CheckResult, EvidenceRef, RouteCode
from services.virtual_workspace import VirtualWorkspace, VirtualWorkspaceError


def validate_environment_artifact(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if domain.domain_id != "benchmark_code_debug":
        return CheckResult(check_id="environment_artifact", passed=True)

    artifact = candidate.agent_artifact.environment_artifact
    if artifact is None:
        return _failed_environment("missing_workspace", "environment_artifact is required for code benchmarks", "environment_artifact")
    if artifact.kind != "virtual_workspace":
        return _failed_environment("missing_workspace", "code benchmarks require environment_artifact.kind='virtual_workspace'", "environment_artifact.kind")

    try:
        workspace = VirtualWorkspace.from_payload(artifact.payload)
    except VirtualWorkspaceError as exc:
        return _failed_environment(exc.subcode, exc.message, exc.path)

    if not bool(domain.deterministic_rules.get("execute_workspace_tests", False)):
        return CheckResult(check_id="environment_artifact", passed=True)

    command = workspace.commands.get("test")
    argv = _safe_test_command_argv(command)
    if argv is None:
        return _failed_environment(
            "unsupported_workspace_test_command",
            "workspace.commands.test must be a pytest command without shell syntax",
            "environment_artifact.payload.commands.test",
        )

    timeout_seconds = float(domain.deterministic_rules.get("workspace_test_timeout_seconds", 10))
    with workspace.materialize() as materialized:
        try:
            completed = subprocess.run(
                argv,
                cwd=materialized.path,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return _failed_environment(
                "workspace_test_timeout",
                f"workspace test command timed out after {timeout_seconds:g}s",
                "environment_artifact.payload.commands.test",
                output=_short_command_output(exc.stdout, exc.stderr),
            )
        except OSError as exc:
            return _failed_environment(
                "workspace_test_command_failed",
                f"workspace test command could not be executed: {exc}",
                "environment_artifact.payload.commands.test",
            )

    if completed.returncode == 1:
        max_failure_files = int(domain.deterministic_rules.get("max_initial_failure_files", 0))
        failed_files = _pytest_failed_test_files(completed.stdout, completed.stderr)
        if max_failure_files > 0 and len(failed_files) > max_failure_files:
            return _failed_environment(
                "workspace_test_command_failed",
                f"workspace starter tests fail across too many test files: {', '.join(failed_files)}",
                "environment_artifact.payload.commands.test",
                output=_short_command_output(completed.stdout, completed.stderr),
            )
        return CheckResult(
            check_id="environment_artifact",
            passed=True,
            evidence=[
                EvidenceRef(
                    source="workspace_command",
                    path="environment_artifact.payload.commands.test",
                    value=_short_command_output(completed.stdout, completed.stderr),
                )
            ],
        )
    if completed.returncode == 0 and bool(domain.deterministic_rules.get("require_initial_test_failure", False)):
        return _failed_environment(
            "workspace_tests_do_not_reproduce_failure",
            "workspace tests passed on the starter code; the benchmark does not demonstrate a failing behavior before repair",
            "environment_artifact.payload.commands.test",
            output=_short_command_output(completed.stdout, completed.stderr),
        )
    return _failed_environment(
        "workspace_test_command_failed",
        f"workspace test command exited with {completed.returncode}; tests did not run cleanly to assertions",
        "environment_artifact.payload.commands.test",
        output=_short_command_output(completed.stdout, completed.stderr),
    )


def _safe_test_command_argv(command: Any) -> list[str] | None:
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None
    if any(token in command for token in (";", "|", "&", ">", "<", "`", "$(", "\n")):
        return None
    executable = PurePosixPath(parts[0]).name
    if executable == "pytest":
        return parts
    if executable in {"python", "python3"} and len(parts) >= 3 and parts[1:3] == ["-m", "pytest"]:
        return [sys.executable, *parts[1:]]
    return None


def _short_command_output(stdout: str | None, stderr: str | None) -> str:
    output = "\n".join(part for part in [stdout or "", stderr or ""] if part).strip()
    if len(output) > 1600:
        return output[:1600] + "\n...[truncated]"
    return output


def _pytest_failed_test_files(stdout: str | None, stderr: str | None) -> list[str]:
    output = "\n".join(part for part in [stdout or "", stderr or ""] if part)
    failed_files: set[str] = set()
    for line in output.splitlines():
        match = re.match(r"^FAILED\s+([^:\s]+)::", line.strip())
        if match:
            failed_files.add(match.group(1))
    return sorted(failed_files)


def _failed_environment(subcode: str, value: str, path: str, *, output: str = "") -> CheckResult:
    evidence = [EvidenceRef(source="deterministic_rule", path=path, value=value)]
    if output:
        evidence.append(EvidenceRef(source="workspace_command", path=path, value=output))
    return CheckResult(
        check_id="environment_artifact",
        passed=False,
        route_code=RouteCode.REJECT_SCHEMA,
        subcode=subcode,
        evidence=evidence,
    )
