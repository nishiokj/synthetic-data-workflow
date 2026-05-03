from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from config import DomainConfig
from models import CandidateSample, CheckResult, CommittedSample, RouteCode, SampleVerdict, TaxonomyCell, Verdict, stable_hash


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> tuple[list[float], dict[str, object]]:
        ...


class CorpusIndex:
    def __init__(self, data_dir: Path, domain: DomainConfig, embedding_client: EmbeddingClient, run_id: str) -> None:
        self.domain = domain
        self.embedding_client = embedding_client
        self.index_path = data_dir / "index" / domain.domain_id / domain.dataset_version / "embeddings.jsonl"
        self.corpus_path = data_dir / "corpus" / "benchmark" / f"{run_id}.jsonl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)

    def curate(
        self,
        *,
        certified_id: str,
        candidate: CandidateSample,
        deterministic_checks: list[CheckResult],
        semantic_checks: list[SampleVerdict],
        run_id: str,
    ) -> tuple[CommittedSample | None, SampleVerdict, dict[str, object]]:
        text = _embedding_text(candidate)
        vector, meta = self.embedding_client.embed(text)
        nearest = self._nearest_distance(vector)
        if nearest is not None and nearest < self.domain.novelty_threshold:
            return (
                None,
                SampleVerdict(
                    candidate_id=candidate.id,
                    check_kind="curation",
                    verdict=Verdict.REJECT,
                    route_code=RouteCode.REJECT_DUPLICATE,
                    subcodes=["near_duplicate"],
                    reason_codes=["near_duplicate"],
                ),
                meta,
            )

        embedding_ref = stable_hash({"candidate_id": candidate.id, "run_id": run_id})
        committed = CommittedSample(
            id=f"{run_id}-committed-{candidate.id}",
            certified_id=certified_id,
            content_hash=stable_hash(candidate.model_dump(mode="json")),
            candidate=candidate,
            deterministic_checks=deterministic_checks,
            semantic_checks=semantic_checks,
            embedding_ref=embedding_ref,
            nn_distance=nearest,
            taxonomy_cell=TaxonomyCell.model_validate(candidate.cell.model_dump()),
        )
        self._append_jsonl(
            self.index_path,
            {"embedding_ref": embedding_ref, "candidate_id": candidate.id, "vector": vector},
        )
        self._append_jsonl(self.corpus_path, committed.model_dump(mode="json"))
        return (
            committed,
            SampleVerdict(
                candidate_id=candidate.id,
                check_kind="curation",
                verdict=Verdict.ACCEPT,
                route_code=RouteCode.ACCEPT,
            ),
            meta,
        )

    def _nearest_distance(self, vector: list[float]) -> float | None:
        nearest: float | None = None
        if not self.index_path.exists():
            return None
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                existing = json.loads(line)["vector"]
                distance = 1.0 - _cosine(vector, existing)
                nearest = distance if nearest is None else min(nearest, distance)
        return nearest

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _embedding_text(candidate: CandidateSample) -> str:
    return "\n".join(
        [
            json.dumps(candidate.benchmark_case, sort_keys=True),
            json.dumps(candidate.ability_z, sort_keys=True),
            json.dumps(candidate.environment_y, sort_keys=True),
            candidate.proxy_claim,
            "\n".join(candidate.diagnostic_pressure),
            json.dumps(candidate.score_x, sort_keys=True),
            "\n".join(candidate.coverage_tags),
        ]
    )
