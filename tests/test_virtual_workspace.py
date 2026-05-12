from __future__ import annotations

import pytest

from services.virtual_workspace import VirtualWorkspace, VirtualWorkspaceError


def _payload(**overrides):
    payload = {
        "files": [
            {"path": "src/app.py", "content": "def main():\n    return 1\n"},
            {"path": "tests/test_app.py", "content": "def test_app():\n    assert True\n"},
            {"path": "README.md", "content": "Run tests."},
        ],
        "commands": {"test": "python -m pytest -q"},
    }
    payload.update(overrides)
    return payload


def test_virtual_workspace_round_trips_and_materializes_files() -> None:
    workspace = VirtualWorkspace.from_payload(_payload())

    assert workspace.list_files() == ["README.md", "src/app.py", "tests/test_app.py"]
    assert workspace.read_file("src/app.py").startswith("def main")
    assert workspace.to_payload()["commands"] == {"test": "python -m pytest -q"}
    with workspace.materialize() as materialized:
        assert (materialized.path / "src/app.py").read_text(encoding="utf-8").startswith("def main")


def test_virtual_workspace_allows_empty_package_marker_files() -> None:
    workspace = VirtualWorkspace.from_payload(
        _payload(
            files=[
                {"path": "src/__init__.py", "content": ""},
                {"path": "src/app.py", "content": "def main():\n    return 1\n"},
                {"path": "tests/test_app.py", "content": "def test_app():\n    assert True\n"},
            ]
        )
    )

    assert workspace.read_file("src/__init__.py") == ""


def test_virtual_workspace_rejects_empty_non_marker_files() -> None:
    with pytest.raises(VirtualWorkspaceError) as exc:
        VirtualWorkspace.from_payload(
            _payload(
                files=[
                    {"path": "src/app.py", "content": ""},
                    {"path": "tests/test_app.py", "content": "def test_app():\n    assert True\n"},
                    {"path": "README.md", "content": "Run tests."},
                ]
            )
        )

    assert exc.value.subcode == "invalid_workspace_file"


def test_virtual_workspace_rejects_unsafe_paths() -> None:
    with pytest.raises(VirtualWorkspaceError) as exc:
        VirtualWorkspace.from_payload(
            _payload(files=[
                {"path": "../escape.py", "content": "x = 1\n"},
                {"path": "tests/test_app.py", "content": "def test_app():\n    assert True\n"},
                {"path": "README.md", "content": "Run tests."},
            ])
        )

    assert exc.value.subcode == "invalid_workspace_path"


def test_virtual_workspace_rejects_duplicate_paths() -> None:
    with pytest.raises(VirtualWorkspaceError) as exc:
        VirtualWorkspace.from_payload(
            _payload(files=[
                {"path": "src/app.py", "content": "x = 1\n"},
                {"path": "src/app.py", "content": "x = 2\n"},
                {"path": "README.md", "content": "Run tests."},
            ])
        )

    assert exc.value.subcode == "duplicate_workspace_path"
