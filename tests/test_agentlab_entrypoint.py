from __future__ import annotations

import json
from pathlib import Path

from agentlab_entrypoint import (
    DOMAIN_RUBRIC_CONTEXT_TOKEN,
    _expand_generator_system_prompt_append,
    _generation_envelope_from_trial,
    _metrics_from_run,
    _pipeline_run_id,
    _preflight_result,
    _resolve_domain_path,
    _trial_ids,
)
from config import load_domain


def test_agentlab_trial_input_extracts_generation_envelope() -> None:
    trial_input = {
        "ids": {"run_id": "run", "trial_id": "trial", "variant_id": "blind", "task_id": "seed-1", "repl_idx": 0},
        "task": {
            "id": "seed-1",
            "input": {
                "generation_envelope": {
                    "id": "seed-1",
                    "domain_ref": "domains/benchmark_code_debug.yaml",
                    "design": {
                        "id": "design-1",
                        "target_stage": "benchmark",
                        "cell": {"case_type": "proxy_strong", "difficulty": 4, "scenario": "edge"},
                        "target_ability": "fault_localization",
                        "target_environment": "single_turn_debug_with_test",
                        "design_intent": "Debug a multi-module billing invariant failure.",
                        "failure_mode_family": "upstream normalization creates downstream aggregate error",
                        "content_hash": "hash",
                    },
                    "seed_context": {"suite": "smoke"},
                }
            },
        },
    }

    envelope = _generation_envelope_from_trial(trial_input)
    ids = _trial_ids(trial_input)

    assert envelope.id == "seed-1"
    assert envelope.domain_ref == "domains/benchmark_code_debug.yaml"
    assert envelope.seed_context == {"suite": "smoke"}
    assert _pipeline_run_id(ids) == "run-blind-seed-1-trial-0"


def test_agentlab_domain_ref_overrides_cli_domain_fallback() -> None:
    resolved = _resolve_domain_path(
        "domains/benchmark_haiku.yaml",
        fallback="domains/benchmark_code_debug.yaml",
    )

    assert resolved == Path.cwd() / "domains/benchmark_haiku.yaml"


def test_agentlab_expands_domain_rubric_context_from_resolved_domain() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")
    expanded = _expand_generator_system_prompt_append(
        f"Before.\n{DOMAIN_RUBRIC_CONTEXT_TOKEN}\nAfter.",
        domain=domain,
    )

    assert DOMAIN_RUBRIC_CONTEXT_TOKEN not in expanded
    assert "QUALITY GATE RULES:" in expanded
    assert "RUBRIC GATE RULES:" in expanded
    assert domain.quality_gate_rules[0] in expanded
    assert domain.rubric_gate_rules[0] in expanded


def test_agentlab_metrics_summarize_pipeline_trace(tmp_path) -> None:
    run_dir = tmp_path / "logs" / "run"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "stage_records.jsonl",
        [
            {"role": "generate_candidate_sample", "input_tokens": 10, "output_tokens": 20, "latency_ms": 30, "cost_usd": 0.1},
            {"role": "revise_candidate_from_adversary", "input_tokens": 1, "output_tokens": 2, "latency_ms": 3, "cost_usd": 0.2},
            {"role": "quality_gate_candidate_ensemble", "verdict": "reject"},
            {"role": "rubric_gate_candidate_ensemble", "verdict": "accept"},
        ],
    )
    _write_jsonl(
        run_dir / "validation.jsonl",
        [
            {"check_kind": "quality", "verdict": "reject", "route_code": "accept", "subcodes": ["weak_proxy"]},
            {"check_kind": "rubric", "verdict": "accept", "route_code": "accept", "subcodes": []},
        ],
    )
    _write_jsonl(
        run_dir / "adversary.jsonl",
        [
            {
                "revision_disposition": "revise",
                "attacks": [
                    {"attack_type": "answer_leakage"},
                    {"attack_type": "proxy_overclaim"},
                    {"attack_type": "scoring_ambiguity"},
                ],
                "cheap_pass_strategy": "patch visible symptom",
            }
        ],
    )

    metrics = _metrics_from_run(
        {"committed": 1, "dropped": 0, "final_status": "committed", "candidate_id": "c1", "subcodes": ["quality_gate_rejected"]},
        run_dir,
    )

    assert metrics["committed"] == 1
    assert metrics["adversary_attack_count"] == 3
    assert metrics["adversary_explicitly_covered_attack_count"] == 1
    assert metrics["adversary_indirectly_covered_attack_count"] == 1
    assert metrics["adversary_uncovered_attack_count"] == 1
    assert metrics["adversary_revision_required"] is True
    assert metrics["quality_gate_rejected"] is True
    assert metrics["quality_gate_ensemble_count"] == 1
    assert metrics["quality_gate_ensemble_reject_count"] == 1
    assert metrics["rubric_gate_ensemble_count"] == 1
    assert metrics["rubric_gate_ensemble_reject_count"] == 0
    assert metrics["revision_count"] == 1
    assert metrics["input_tokens_total"] == 11
    assert metrics["cost_usd_total"] == 0.30000000000000004


def test_preflight_result_is_contract_success_without_pipeline_execution() -> None:
    result = _preflight_result(
        {"run_id": "preflight", "trial_id": "trial", "variant_id": "blind", "task_id": "task", "repl_idx": 0}
    )

    assert result["schema_version"] == "agent_result_v1"
    assert result["outcome"] == "success"
    assert result["metrics"]["final_status"] == "preflight_smoke"
    assert result["metrics"]["generation_attempts"] == 0


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
