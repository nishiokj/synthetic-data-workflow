from __future__ import annotations

import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


PLACEHOLDER_TEXTS = {"...", "todo", "tbd", "<content>", "<file content>", "# todo", "# tbd"}
PLACEHOLDER_FRAGMENTS = (
    "implementation omitted",
    "content omitted",
    "placeholder file",
    "todo: implement",
    "replace with actual",
)


class VirtualWorkspaceError(ValueError):
    def __init__(self, subcode: str, path: str, message: str) -> None:
        super().__init__(message)
        self.subcode = subcode
        self.path = path
        self.message = message


class MaterializedWorkspace:
    def __init__(self, temp_dir: tempfile.TemporaryDirectory[str]) -> None:
        self._temp_dir = temp_dir
        self.path = Path(temp_dir.name)

    def __enter__(self) -> "MaterializedWorkspace":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._temp_dir.cleanup()


class VirtualWorkspace:
    def __init__(self, files: dict[str, str] | None = None, commands: dict[str, str] | None = None) -> None:
        self._files = dict(files or {})
        self.commands = dict(commands or {})

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VirtualWorkspace":
        if not isinstance(payload, dict):
            raise VirtualWorkspaceError("missing_workspace", "environment_artifact.payload", "virtual workspace payload must be an object")

        raw_files = payload.get("files")
        if not isinstance(raw_files, list) or len(raw_files) < 3:
            raise VirtualWorkspaceError("missing_workspace", "environment_artifact.payload.files", "virtual workspace must contain at least 3 files")

        files: dict[str, str] = {}
        for index, file_entry in enumerate(raw_files):
            if not isinstance(file_entry, dict):
                raise VirtualWorkspaceError("invalid_workspace_file", f"environment_artifact.payload.files.{index}", "workspace file entry must be an object")
            path = file_entry.get("path")
            content = file_entry.get("content")
            path_ref = f"environment_artifact.payload.files.{index}.path"
            content_ref = f"environment_artifact.payload.files.{index}.content"
            normalized_path = normalize_workspace_path(path, path_ref)
            if normalized_path in files:
                raise VirtualWorkspaceError("duplicate_workspace_path", path_ref, "workspace file path is duplicated")
            if not isinstance(content, str) or (not content.strip() and not _allows_empty_file(normalized_path)):
                raise VirtualWorkspaceError("invalid_workspace_file", content_ref, "workspace file content is missing")
            if looks_like_placeholder_file(content):
                raise VirtualWorkspaceError("placeholder_workspace_file", content_ref, "workspace file content is placeholder text, not a materialized artifact")
            files[normalized_path] = content

        commands = payload.get("commands")
        if not isinstance(commands, dict) or not isinstance(commands.get("test"), str) or not commands["test"].strip():
            raise VirtualWorkspaceError("missing_workspace_command", "environment_artifact.payload.commands.test", "workspace.commands.test is required")
        normalized_commands = {str(key): str(value) for key, value in commands.items()}
        if looks_like_placeholder_text(normalized_commands["test"]):
            raise VirtualWorkspaceError("missing_workspace_command", "environment_artifact.payload.commands.test", "workspace.commands.test cannot be placeholder text")

        return cls(files=files, commands=normalized_commands)

    def to_payload(self) -> dict[str, Any]:
        return {
            "files": [{"path": path, "content": self._files[path]} for path in sorted(self._files)],
            "commands": dict(self.commands),
        }

    def list_files(self) -> list[str]:
        return sorted(self._files)

    def read_file(self, path: str) -> str:
        normalized_path = normalize_workspace_path(path, "path")
        if normalized_path not in self._files:
            raise VirtualWorkspaceError("invalid_workspace_path", "path", f"workspace file does not exist: {normalized_path}")
        return self._files[normalized_path]

    def write_file(self, path: str, content: str) -> None:
        normalized_path = normalize_workspace_path(path, "path")
        if not isinstance(content, str) or (not content.strip() and not _allows_empty_file(normalized_path)):
            raise VirtualWorkspaceError("invalid_workspace_file", "content", "workspace file content is missing")
        if looks_like_placeholder_file(content):
            raise VirtualWorkspaceError("placeholder_workspace_file", "content", "workspace file content is placeholder text")
        self._files[normalized_path] = content

    def delete_file(self, path: str) -> None:
        normalized_path = normalize_workspace_path(path, "path")
        self._files.pop(normalized_path, None)

    def materialize(self) -> MaterializedWorkspace:
        temp_dir = tempfile.TemporaryDirectory(prefix="benchmark-workspace-")
        root = Path(temp_dir.name)
        for path, content in self._files.items():
            file_path = root / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        return MaterializedWorkspace(temp_dir)


def normalize_workspace_path(path: Any, ref: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise VirtualWorkspaceError("invalid_workspace_file", ref, "workspace file path is missing")
    normalized_path = path.strip()
    parts = PurePosixPath(normalized_path).parts
    if normalized_path.startswith("/") or "\\" in normalized_path or ".." in parts:
        raise VirtualWorkspaceError("invalid_workspace_path", ref, "workspace file paths must be relative POSIX paths and cannot traverse upward")
    return normalized_path


def looks_like_placeholder_file(content: str) -> bool:
    stripped = content.strip().lower()
    if not stripped:
        return False
    if looks_like_placeholder_text(stripped):
        return True
    return any(fragment in stripped for fragment in PLACEHOLDER_FRAGMENTS)


def looks_like_placeholder_text(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDER_TEXTS


def _allows_empty_file(path: str) -> bool:
    return PurePosixPath(path).name == "__init__.py"
