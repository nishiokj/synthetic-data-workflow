from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Show a human-readable pipeline run report.")
    parser.add_argument("run_id", help="Run id to inspect, e.g. demo")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=12, help="Number of accepted/rejected items to show.")
    parser.add_argument("--timeline", action="store_true", help="Show the stage timeline.")
    parser.add_argument("--flat", action="store_true", help="Show flat accepted/rejected candidate lists.")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir) / args.run_id
    corpus_path = Path(args.data_dir) / "corpus" / "benchmark" / f"{args.run_id}.jsonl"
    stages = _read_jsonl(logs_dir / "stage_records.jsonl")
    rejections = _read_jsonl(logs_dir / "rejections.jsonl")
    accepted = _read_jsonl(corpus_path)

    if not stages and not rejections and not accepted:
        print(f"No artifacts found for run_id={args.run_id}")
        print(f"Expected logs under {logs_dir}")
        return 1

    print(f"Run {args.run_id}")
    print("=" * (4 + len(args.run_id)))
    print(f"stages={len(stages)} accepted={len(accepted)} rejected={len(rejections)}")
    if stages:
        first = stages[0].get("wallclock_ts", "?")
        last = stages[-1].get("wallclock_ts", "?")
        print(f"time={first} -> {last}")
    print()

    _print_schema_warning(stages, rejections, accepted)
    _print_counts("Routes", Counter(stage.get("route_code") for stage in stages))
    _print_counts("Validation rejection codes", _code_counts(stage for stage in stages if stage.get("verdict") == "reject"))

    stories = _design_stories(stages, rejections, accepted)
    if stories:
        print("Design Stories")
        print("------------")
        for story in stories[-args.limit :]:
            print(_story_summary(story))
        print()

    if args.flat and accepted:
        print("Accepted")
        print("--------")
        for item in accepted[-args.limit :]:
            candidate = item.get("candidate", {})
            print(_candidate_summary(candidate, prefix="+"))
        print()

    if args.flat and rejections:
        print("Rejected")
        print("--------")
        for item in rejections[-args.limit :]:
            artifact = item.get("artifact", {})
            route = item.get("route", {})
            codes = route.get("reason_codes") or route.get("subcodes") or []
            print(_candidate_summary(artifact, prefix="-", route=route.get("route_code"), codes=codes))
        print()

    if args.timeline and stages:
        print("Timeline")
        print("--------")
        for stage in stages:
            ts = str(stage.get("wallclock_ts", "?")).replace("T", " ").split("+")[0]
            codes = ",".join(stage.get("reason_codes") or stage.get("subcodes") or [])
            suffix = f" codes={codes}" if codes else ""
            print(
                f"{ts} {stage.get('stage_kind')} {stage.get('role')} "
                f"{stage.get('verdict')} route={stage.get('route_code')} "
                f"attempt={int(stage.get('retry_index', 0)) + 1}{suffix}"
            )

    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _print_counts(title: str, counts: Counter[Any]) -> None:
    if not counts:
        return
    print(title)
    print("-" * len(title))
    for key, count in counts.most_common():
        print(f"{count:>4}  {key}")
    print()


def _code_counts(stages: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for stage in stages:
        for code in stage.get("reason_codes") or stage.get("subcodes") or []:
            counts[str(code)] += 1
    return counts


def _print_schema_warning(stages: list[dict[str, Any]], rejections: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> None:
    versions = set()
    for item in accepted:
        versions.add(_artifact_schema(item.get("candidate", {})))
    for item in rejections:
        versions.add(_artifact_schema(item.get("artifact", {})))
    versions.discard("unknown")
    if len(versions) > 1:
        print("Warning")
        print("-------")
        print(f"Run contains mixed artifact schemas: {', '.join(sorted(versions))}")
        print("This usually means the same run id was reused across code versions.")
        print()
    elif stages and accepted and not any(_stage_is_current(stage) for stage in stages):
        print("Warning")
        print("-------")
        print("Run appears to contain only legacy stage records.")
        print()


def _design_stories(
    stages: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stories: dict[str, dict[str, Any]] = {}

    for rejection in rejections:
        artifact = rejection.get("artifact", {})
        if not isinstance(artifact, dict):
            continue
        design_id = str(artifact.get("design_id") or _design_from_candidate_id(str(artifact.get("id", ""))) or "unknown-design")
        story = stories.setdefault(design_id, {"design_id": design_id, "attempts": [], "accepted": []})
        story["attempts"].append({"candidate": artifact, "route": rejection.get("route", {}), "accepted": False})

    for item in accepted:
        candidate = item.get("candidate", {})
        if not isinstance(candidate, dict):
            continue
        design_id = str(candidate.get("design_id") or _design_from_candidate_id(str(candidate.get("id", ""))) or "unknown-design")
        story = stories.setdefault(design_id, {"design_id": design_id, "attempts": [], "accepted": []})
        story["accepted"].append(candidate)
        story["attempts"].append(
            {
                "candidate": candidate,
                "route": {"route_code": "accept", "reason_codes": []},
                "accepted": True,
                "semantic_checks": item.get("semantic_checks", []),
            }
        )

    ordered = list(stories.values())
    ordered.sort(key=lambda story: _story_sort_key(story, stages))
    return ordered


def _story_summary(story: dict[str, Any]) -> str:
    attempts = sorted(story.get("attempts", []), key=_attempt_sort_key)
    accepted = [attempt for attempt in attempts if attempt.get("accepted")]
    final = attempts[-1] if attempts else {}
    final_route = (final.get("route") or {}).get("route_code", "unknown")
    final_status = "accepted" if accepted else "failed"
    first_candidate = (attempts[0].get("candidate") if attempts else {}) or {}
    header = [
        f"Design {story.get('design_id')}",
        f"final={final_status}",
        f"attempts={len(attempts)}",
        f"case={_case_type(first_candidate)}",
        f"ability={_name(first_candidate.get('ability_z')) or '?'}",
    ]
    lines = [" ".join(header)]
    prompt = _prompt(first_candidate)
    if prompt:
        lines.append(f'  prompt="{_clip(prompt, 220)}"')
    proxy = str(first_candidate.get("proxy_claim") or "")
    if proxy:
        lines.append(f'  proxy="{_clip(proxy, 220)}"')
    for index, attempt in enumerate(attempts, start=1):
        route = attempt.get("route") or {}
        candidate = attempt.get("candidate") or {}
        codes = route.get("reason_codes") or route.get("subcodes") or []
        code_text = ",".join(str(code) for code in codes[:7]) or "-"
        attempt_no = _attempt_number(candidate, fallback=index)
        verdict = "accepted" if attempt.get("accepted") else "rejected"
        lines.append(
            f"  attempt {attempt_no}: {verdict} route={route.get('route_code', final_route)} codes={code_text}"
        )
        for gate_line in _semantic_check_lines(attempt.get("semantic_checks", [])):
            lines.append(f"    {gate_line}")
    if final_route and str(final_route).startswith("drop_"):
        lines.append(f"  final drop: {final_route}")
    return "\n".join(lines)


def _story_sort_key(story: dict[str, Any], stages: list[dict[str, Any]]) -> tuple[str, str]:
    design_id = str(story.get("design_id", ""))
    timestamps = []
    for attempt in story.get("attempts", []):
        candidate_id = ((attempt.get("candidate") or {}).get("id") or "")
        for stage in stages:
            if stage.get("parent_artifact_id") == candidate_id or stage.get("artifact_id") == candidate_id:
                timestamps.append(str(stage.get("wallclock_ts", "")))
    return (min(timestamps) if timestamps else "", design_id)


def _attempt_sort_key(attempt: dict[str, Any]) -> tuple[str, int]:
    candidate = attempt.get("candidate") or {}
    return (str(candidate.get("id", "")), _attempt_number(candidate, fallback=0))


def _attempt_number(candidate: dict[str, Any], fallback: int) -> int:
    candidate_id = str(candidate.get("id", ""))
    tail = candidate_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else fallback


def _design_from_candidate_id(candidate_id: str) -> str | None:
    marker = "-candidate-"
    if marker not in candidate_id:
        return None
    rest = candidate_id.split(marker, 1)[1]
    pieces = rest.rsplit("-", 1)
    return pieces[0] if pieces else None


def _candidate_summary(
    candidate: dict[str, Any],
    *,
    prefix: str,
    route: str | None = None,
    codes: list[str] | None = None,
) -> str:
    candidate_id = candidate.get("id", "?")
    case_type = _case_type(candidate)
    ability = _name(candidate.get("ability_z")) or "?"
    prompt = _prompt(candidate)
    proxy = str(candidate.get("proxy_claim") or "")
    parts = [
        f"{prefix} {candidate_id}",
        f"case={case_type}",
        f"ability={ability}",
    ]
    if route:
        parts.append(f"route={route}")
    if codes:
        parts.append(f"codes={','.join(codes[:6])}")
    parts.append(f'prompt="{_clip(prompt, 180)}"')
    if proxy:
        parts.append(f'proxy="{_clip(proxy, 180)}"')
    return "\n  ".join(parts)


def _semantic_check_lines(checks: Any) -> list[str]:
    if not isinstance(checks, list):
        return []
    lines: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        kind = str(check.get("check_kind") or "gate")
        verdict = str(check.get("verdict") or "?")
        codes = ",".join(str(code) for code in (check.get("reason_codes") or check.get("subcodes") or [])[:5]) or "-"
        rationale = _clip(str(check.get("rationale") or ""), 240)
        if rationale:
            lines.append(f"{kind}: {verdict} codes={codes} rationale=\"{rationale}\"")
        else:
            lines.append(f"{kind}: {verdict} codes={codes}")
    return lines


def _name(value: Any) -> str | None:
    return value.get("name") if isinstance(value, dict) else None


def _prompt(candidate: dict[str, Any]) -> str:
    case = candidate.get("benchmark_case")
    if isinstance(case, dict):
        return str(case.get("prompt") or "")
    inner = candidate.get("inner_input")
    if isinstance(inner, dict):
        return str(inner.get("question") or "")
    return ""


def _case_type(candidate: dict[str, Any]) -> str:
    cell = candidate.get("cell") if isinstance(candidate.get("cell"), dict) else {}
    return str(candidate.get("case_type") or cell.get("case_type") or cell.get("failure_mode") or "?")


def _artifact_schema(candidate: dict[str, Any]) -> str:
    if not isinstance(candidate, dict):
        return "unknown"
    if "benchmark_case" in candidate:
        return "benchmark"
    if "inner_input" in candidate:
        return "validator_legacy"
    return "unknown"


def _stage_is_current(stage: dict[str, Any]) -> bool:
    current_roles = {
        "design_batch",
        "validate_design_batch_deterministically",
        "audit_design",
        "generate_candidate_sample",
        "adversary_attack_report",
        "quality_gate_candidate",
        "rubric_gate_candidate",
        "curate_committed_sample",
    }
    current_agents = {
        "designer",
        "design_auditor",
        "sample_generator",
        "adversary",
        "quality_gate",
        "rubric_gate",
    }
    return str(stage.get("role")) in current_roles or str(stage.get("agent_role")) in current_agents


def _clip(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
