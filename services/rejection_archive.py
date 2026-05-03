from __future__ import annotations

from typing import Any

from observability import StageLogWriter


class RejectionArchive:
    def __init__(self, writer: StageLogWriter) -> None:
        self.writer = writer

    def append(self, artifact: Any, route: Any) -> None:
        value = {
            "artifact": artifact.model_dump(mode="json") if hasattr(artifact, "model_dump") else artifact,
            "route": route.model_dump(mode="json") if hasattr(route, "model_dump") else route,
        }
        self.writer.append_rejection(value)

