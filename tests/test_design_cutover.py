from __future__ import annotations

from pathlib import Path


def test_no_obsolete_benchmark_seed_language_remains() -> None:
    roots = [
        Path("agents.py"),
        Path("models.py"),
        Path("pipeline.py"),
        Path("rules.py"),
        Path("router.py"),
        Path("cli_graph.py"),
        Path("run_report.py"),
        Path("tests"),
        Path("domains"),
        Path("README.md"),
        Path("docs"),
    ]
    forbidden = (
        "SeedSpec",
        "PlanVerdict",
        "Strategist",
        "PlanAuditor",
        "environment_seed",
        "validate_seed_plan",
        "audit_seed",
        "select_next_seed",
        "seed_plan",
        "duplicate_seed",
        "StageKind.STRATEGY",
    )
    offenders: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in paths:
            if path.name == "test_design_cutover.py":
                continue
            if path.suffix not in {".py", ".md", ".yaml", ".yml", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            for term in forbidden:
                if term in text:
                    offenders.append(f"{path}:{term}")

    assert offenders == []
