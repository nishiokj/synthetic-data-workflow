from __future__ import annotations

import json

from config import build_runtime_config
from pipeline import PipelineRunner


class FakeOpenAIClient:
    def __init__(self, config) -> None:
        self.config = config

    def complete_json(self, *, system: str, user: str, temperature: float = 0.4):
        meta = {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
            "prompt_hash": "fake",
        }
        if "Strategist" in system:
            return {
                "seeds": [
                    {
                        "case_type": "proxy_strong",
                        "difficulty": 3,
                        "scenario": "adversarial",
                        "ability": "constrained_poetic_generation",
                        "environment": "single_turn_creative_writing",
                        "diagnostic_pressure": "forbid obvious imagery while preserving emotional intent",
                        "scoring_strategy": "hard_checks_plus_rubric",
                        "leakage_risk": "format-only haiku template",
                        "intent": "Generate a haiku benchmark that pressures metaphor and anti-template behavior.",
                    }
                ]
            }, meta
        if "Plan Auditor" in system:
            return {
                "verdict": "accept",
                "route_code": "accept",
                "subcodes": [],
                "reason_codes": [],
                "evidence": [],
                "rationale": "The seed matches the allowed taxonomy and describes a concrete benchmark pressure.",
            }, meta
        if "Benchmark Case Generator" in system:
            return {
                "benchmark_case": {
                    "prompt": "Write a haiku that evokes a layoff as late autumn without saying work, loss, leaves, cold, or endings.",
                    "setup": "Single-turn creative-writing benchmark.",
                    "inputs": {},
                    "environment": {},
                },
                "score_x": {
                    "score_type": "hard_checks_plus_rubric",
                    "range": [0, 1],
                    "dimensions": [{"name": "constraint_adherence", "weight": 0.4, "high_score_criterion": "Output contains none of the forbidden terms and uses indirect imagery.", "low_score_criterion": "Output uses one or more forbidden terms directly."}],
                },
                "ability_z": {"name": "constrained_poetic_generation", "sub_abilities": ["metaphorical_transfer"]},
                "environment_y": {"name": "single_turn_creative_writing", "assumptions": ["No tools"]},
                "proxy_claim": "A model that succeeds here is showing more than haiku formatting because it must preserve emotional intent while avoiding obvious lexical shortcuts and template seasonal imagery.",
                "diagnostic_pressure": ["forbids obvious imagery", "requires emotional transfer"],
                "scoring_contract": {
                    "credit": ["preserves emotional intent", "obeys forbidden-word constraints"],
                    "penalties": ["generic seasonal template", "mentions the source domain directly"],
                    "uncertainty_policy": "Mark uncertainty when taste and constraint adherence conflict.",
                },
                "leakage_risks": ["A compliant but lifeless template may receive too much credit."],
                "known_limits": ["The case does not prove broad poetic taste."],
                "coverage_tags": ["anti_template", "emotional_transfer"],
                "negative_controls": [{"output": "dead leaves fall at work", "should_fail_because": "uses forbidden imagery and source domain"}],
            }, meta
        return {
            "verdict": "accept",
            "route_code": "accept",
            "subcodes": [],
            "reason_codes": [],
            "evidence": [],
            "rationale": "The candidate is sufficient for this fake smoke test.",
        }, meta

    def embed(self, text: str):
        return [1.0, 0.0, 0.0], {
            "provider": "test",
            "model": "fake-embedding",
            "input_tokens": 1,
            "output_tokens": 0,
            "latency_ms": 1,
            "cost_usd": 0.0,
            "prompt_hash": "fake-embed",
        }


def test_pipeline_smoke_uses_fenced_fake_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.OpenAIClient", FakeOpenAIClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="smoke",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"

    summary = PipelineRunner(config).run()

    assert summary["committed"] == 1
    corpus_path = tmp_path / "data" / "corpus" / "benchmark" / "smoke.jsonl"
    assert corpus_path.exists()
    assert (tmp_path / "logs" / "smoke" / "stage_records.jsonl").exists()

    committed = json.loads(corpus_path.read_text(encoding="utf-8").splitlines()[0])
    assert [check["check_id"] for check in committed["deterministic_checks"]] == [
        "text_hygiene",
        "output_schema",
        "benchmark_case_schema",
        "taxonomy_cell",
        "benchmark_contract",
    ]
    assert committed["candidate"]["output"]["benchmark_case"]["prompt"]
    assert [check["check_kind"] for check in committed["semantic_checks"]] == ["quality", "rubric"]
    assert all(check["rationale"] for check in committed["semantic_checks"])
