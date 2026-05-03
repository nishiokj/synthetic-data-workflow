from __future__ import annotations

import json
from pathlib import Path

from config import DomainConfig
from models import TaxonomyCell


class CoverageLedger:
    def __init__(self, data_dir: Path, domain: DomainConfig) -> None:
        self.path = data_dir / "coverage" / domain.domain_id / domain.dataset_version / "coverage.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> dict[str, int]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def increment(self, cell: TaxonomyCell) -> None:
        data = self.snapshot()
        data[cell.key()] = int(data.get(cell.key(), 0)) + 1
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)

