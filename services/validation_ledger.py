from __future__ import annotations

from observability import StageLogWriter
from models import SampleVerdict


class ValidationLedger:
    def __init__(self, writer: StageLogWriter) -> None:
        self.writer = writer

    def append(self, verdict: SampleVerdict) -> None:
        self.writer.append_validation(verdict)

