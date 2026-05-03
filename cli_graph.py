"""Runtime CLI graph for the synthetic data pipeline LangGraph run."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from models import AgentRole, RouteCode, Verdict


NODE_ORDER = [
    "strategy",
    "validate_seed_plan_det",
    "select_next_seed",
    "audit_seed_plan",
    "generate",
    "validate_det",
    "quality_gate",
    "rubric_gate",
    "curate",
]

NODE_LABELS_FULL = {
    "strategy": "strategy",
    "validate_seed_plan_det": "plan_det",
    "select_next_seed": "select_seed",
    "audit_seed_plan": "plan_audit",
    "generate": "generate",
    "validate_det": "validate_det",
    "quality_gate": "quality_gate",
    "rubric_gate": "rubric_gate",
    "curate": "curate",
}

NODE_LABELS_COMPACT = {
    "strategy": "ST",
    "validate_seed_plan_det": "PD",
    "select_next_seed": "SS",
    "audit_seed_plan": "PA",
    "generate": "GN",
    "validate_det": "VD",
    "quality_gate": "QG",
    "rubric_gate": "RG",
    "curate": "CU",
}

STAGE_TO_NODE = {
    "run": None,
    "strategy": "strategy",
    "plan_det": "validate_seed_plan_det",
    "seed": "select_next_seed",
    "plan_audit": "audit_seed_plan",
    "generation": "generate",
    "validation_det": "validate_det",
    "quality_gate": "quality_gate",
    "rubric_gate": "rubric_gate",
    "curation": "curate",
}

ROLE_TO_NODE = {
    "plan_strategy_batch": "strategy",
    "validate_seed_plan_deterministically": "validate_seed_plan_det",
    "audit_seed_plan": "audit_seed_plan",
    "generate_candidate_sample": "generate",
    "validate_candidate_deterministically": "validate_det",
    "quality_gate_candidate": "quality_gate",
    "rubric_gate_candidate": "rubric_gate",
    "curate_committed_sample": "curate",
}

NODE_AGENT_NAMES = {
    "strategy": "Strategist",
    "audit_seed_plan": "PlanAuditor",
    "generate": "SampleGenerator",
    "quality_gate": "QualityGate",
    "rubric_gate": "RubricGate",
}

AGENT_ROLE_NAMES = {
    AgentRole.STRATEGIST.value: "Strategist",
    AgentRole.PLAN_AUDITOR.value: "PlanAuditor",
    AgentRole.SAMPLE_GENERATOR.value: "SampleGenerator",
    AgentRole.SEMANTIC_VALIDATOR.value: "SemanticValidator",
    AgentRole.QUALITY_GATE.value: "QualityGate",
    AgentRole.RUBRIC_GATE.value: "RubricGate",
}

COLORS = {
    "pending": "\033[90m",
    "running": "\033[94m",
    "accept": "\033[92m",
    "reject": "\033[91m",
    "local": "\033[92m",
}
RESET = "\033[0m"
DIM = "\033[90m"

MIN_FULL_WIDTH = 92

_CX_FULL = {
    "strategy": 46,
    "validate_seed_plan_det": 46,
    "select_next_seed": 46,
    "audit_seed_plan": 46,
    "generate": 46,
    "validate_det": 46,
    "quality_gate": 46,
    "rubric_gate": 46,
    "curate": 46,
}

_CX_COMPACT = {
    "strategy": 24,
    "validate_seed_plan_det": 24,
    "select_next_seed": 24,
    "audit_seed_plan": 24,
    "generate": 24,
    "validate_det": 24,
    "quality_gate": 24,
    "rubric_gate": 24,
    "curate": 24,
}

_BY = {
    "strategy": 0,
    "validate_seed_plan_det": 4,
    "select_next_seed": 8,
    "audit_seed_plan": 12,
    "generate": 16,
    "validate_det": 20,
    "quality_gate": 24,
    "rubric_gate": 28,
    "curate": 32,
}

_CANVAS_H = 38


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 120


def _is_compact() -> bool:
    return _term_width() < MIN_FULL_WIDTH


class _Canvas:
    __slots__ = ("w", "h", "chars", "colors")

    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h
        self.chars = [[" "] * w for _ in range(h)]
        self.colors: list[list[str | None]] = [[None] * w for _ in range(h)]

    def put(self, x: int, y: int, text: str, color: str | None = None) -> None:
        for i, ch in enumerate(text):
            cx = x + i
            if 0 <= cx < self.w and 0 <= y < self.h:
                self.chars[y][cx] = ch
                self.colors[y][cx] = color

    def render(self) -> str:
        lines: list[str] = []
        for y in range(self.h):
            parts: list[str] = []
            current: str | None = None
            for x in range(self.w):
                color = self.colors[y][x]
                if color != current:
                    if current is not None:
                        parts.append(RESET)
                    if color is not None:
                        parts.append(color)
                    current = color
                parts.append(self.chars[y][x])
            if current is not None:
                parts.append(RESET)
            lines.append("".join(parts).rstrip())
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)


def _draw_box(c: _Canvas, cx: int, y: int, label: str, color: str | None) -> None:
    width = len(label) + 4
    x = cx - width // 2
    c.put(x, y, "┌" + "─" * (width - 2) + "┐", color)
    c.put(x, y + 1, f"│ {label} │", color)
    c.put(x, y + 2, "└" + "─" * (width - 2) + "┘", color)


def _vline(c: _Canvas, x: int, y1: int, y2: int) -> None:
    for y in range(y1, y2 + 1):
        c.put(x, y, "│")


def _hline(c: _Canvas, y: int, x1: int, x2: int) -> None:
    for x in range(min(x1, x2), max(x1, x2) + 1):
        c.put(x, y, "─")


def _status_color(status: str) -> str | None:
    return COLORS.get(status)


def _render_connectors(c: _Canvas, cx: dict[str, int], compact: bool, labels: dict[str, str]) -> None:
    center = cx["strategy"]
    for upper, lower in zip(NODE_ORDER, NODE_ORDER[1:]):
        x = cx[upper]
        c.put(x, _BY[upper] + 2, "┬")
        _vline(c, x, _BY[upper] + 3, _BY[lower] - 1)
        c.put(cx[lower], _BY[lower], "┴")

    left = 6 if compact else 16
    right = 42 if compact else 78

    def _roff(node: str) -> int:
        return (len(labels[node]) + 4) // 2

    # Plan retry: deterministic seed-plan reject routes back to strategy.
    strategy_l = center - _roff("strategy")
    c.put(left, 5, "┌")
    _vline(c, left, 2, 4)
    c.put(left, 1, "└")
    _hline(c, 5, left + 1, center)
    _hline(c, 1, left + 1, strategy_l)
    c.put(strategy_l - 1, 1, "→")
    c.put(left, 6, "retry plan", DIM)

    # Generation retry: det/quality/rubric rejects can route back to generation.
    generate_l = center - _roff("generate")
    c.put(left, 22, "┌")
    _vline(c, left, 19, 21)
    c.put(left, 18, "└")
    _hline(c, 22, left + 1, center)
    _hline(c, 18, left + 1, generate_l)
    c.put(generate_l - 1, 18, "→")
    c.put(left, 23, "retry sample", DIM)

    # Curation loops to the next seed, or back to strategy when no seeds remain.
    curate_r = center + _roff("curate")
    select_seed_r = center + _roff("select_next_seed")
    strategy_r = center + _roff("strategy")

    _hline(c, 34, curate_r, right)
    c.put(right, 34, "┘")
    _vline(c, right, 2, 34)
    c.put(right, 10, "┤")
    _hline(c, 10, select_seed_r, right)
    c.put(select_seed_r + 1, 10, "←")
    c.put(right - 11, 9, "next seed", DIM)
    c.put(right, 2, "┤")
    _hline(c, 2, strategy_r, right)
    c.put(strategy_r + 1, 2, "←")
    c.put(right - 10, 1, "new plan", DIM)

    c.put(center - 1, 36, "END", DIM)


def render_graph(
    node_status: dict[str, str],
    stats: dict[str, Any],
    recent: list[str] | None = None,
) -> str:
    compact = _is_compact()
    labels = NODE_LABELS_COMPACT if compact else NODE_LABELS_FULL
    cx = _CX_COMPACT if compact else _CX_FULL
    width = 50 if compact else MIN_FULL_WIDTH

    canvas = _Canvas(width, _CANVAS_H)
    _render_connectors(canvas, cx, compact, labels)
    for node in NODE_ORDER:
        _draw_box(canvas, cx[node], _BY[node], labels[node], _status_color(node_status[node]))

    running_agents = [
        NODE_AGENT_NAMES[node]
        for node in NODE_ORDER
        if node_status[node] == "running" and node in NODE_AGENT_NAMES
    ]
    agent_list = "  ".join(f"{COLORS['running']}{name}{RESET}" for name in running_agents) or "—"

    lines = [
        canvas.render(),
        "",
        f"Active Agents: {agent_list}",
        (
            "Run: "
            f"{stats.get('run_id', '-')}"
            f"  target={stats.get('target', '-')}"
            f"  committed={stats.get('committed', 0)}"
            f"  dropped={stats.get('dropped', 0)}"
            f"  seed={stats.get('seed', '-')}"
        ),
        "Legend: blue=active  green=accepted/last-ok  red=rejected  gray=pending",
    ]
    if compact:
        lines.extend([
            "",
            " ".join(f"{NODE_LABELS_COMPACT[n]}={NODE_LABELS_FULL[n]}" for n in NODE_ORDER),
        ])
    if recent:
        lines.extend(["", "─" * min(width, 80), *recent[-5:]])
    return "\n".join(lines)


def create_graph_callback(run_id: str) -> Callable[[str, dict[str, Any]], None]:
    node_status = {node: "pending" for node in NODE_ORDER}
    stats: dict[str, Any] = {"run_id": run_id, "committed": 0, "dropped": 0, "seed": "-"}
    recent: list[str] = []
    previous_lines = 0
    printed_recent = 0
    is_tty = sys.stdout.isatty()

    def redraw() -> None:
        nonlocal previous_lines, printed_recent
        if not is_tty:
            for line in recent[printed_recent:]:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()
            printed_recent = len(recent)
            return

        output = render_graph(node_status, stats, recent)
        lines = output.split("\n")
        if is_tty:
            try:
                term_h = os.get_terminal_size().lines
            except (ValueError, OSError):
                term_h = 40
            lines = lines[: max(term_h - 1, 1)]
            if previous_lines:
                sys.stdout.write(f"\033[{previous_lines}A\033[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        previous_lines = len(lines)

    redraw()

    def callback(event: str, data: dict[str, Any]) -> None:
        if event == "stage_progress":
            _handle_progress(data, node_status, stats, recent)
            redraw()
            return
        if event == "stage_result":
            _handle_result(data, node_status, stats, recent)
            redraw()

    return callback


def _handle_progress(
    data: dict[str, Any],
    node_status: dict[str, str],
    stats: dict[str, Any],
    recent: list[str],
) -> None:
    stage = str(data.get("stage", ""))
    progress_event = str(data.get("event", ""))
    if stage == "run" and progress_event == "start":
        stats["target"] = data.get("target", "-")
        recent.append(f"run start target={stats['target']} model={data.get('model', '-')}")
        return
    if stage == "candidate":
        recent.append(_candidate_line(progress_event, data))
        return

    node = STAGE_TO_NODE.get(stage)
    if node is None:
        return
    if node == "select_next_seed" and data.get("id"):
        stats["seed"] = data["id"]
    if progress_event in {"start", "select"}:
        for existing, status in list(node_status.items()):
            if status == "running":
                node_status[existing] = "local" if existing in {"validate_seed_plan_det", "validate_det", "curate"} else "pending"
        node_status[node] = "running"
        recent.append(_progress_line(stage, progress_event, data))


def _handle_result(
    data: dict[str, Any],
    node_status: dict[str, str],
    stats: dict[str, Any],
    recent: list[str],
) -> None:
    node = ROLE_TO_NODE.get(str(data.get("role", "")))
    if node is None:
        return

    verdict = _enum_value(data.get("verdict"))
    route = _enum_value(data.get("route_code"))
    status = "accept" if verdict == Verdict.ACCEPT.value else "reject"
    if str(data.get("provider", "local")) == "local" and status == "accept":
        status = "local"
    node_status[node] = status

    if node == "curate" and route == RouteCode.ACCEPT.value:
        stats["committed"] = int(stats.get("committed", 0)) + 1
    elif status == "reject" and (
        node in {"audit_seed_plan", "curate"} or route.startswith("drop_")
    ):
        stats["dropped"] = int(stats.get("dropped", 0)) + 1

    agent = AGENT_ROLE_NAMES.get(str(data.get("agent_role") or ""), "local")
    artifact = _short_id(str(data.get("artifact_id", "-")))
    codes = ",".join(str(code) for code in data.get("reason_codes", [])[:4])
    suffix = f" codes={codes}" if codes else ""
    recent.append(f"{NODE_LABELS_FULL[node]} {verdict} route={route} agent={agent} artifact={artifact}{suffix}")


def _progress_line(stage: str, event: str, data: dict[str, Any]) -> str:
    bits = [event, stage]
    for key in ("round", "seed", "id", "candidate", "attempt", "remaining", "retry"):
        value = data.get(key)
        if value is not None:
            bits.append(f"{key}={_short_id(str(_enum_value(value)))}")
    return " ".join(bits)


def _candidate_line(event: str, data: dict[str, Any]) -> str:
    route = data.get("route")
    codes = data.get("codes")
    bits = [event, "candidate"]
    for key in ("id", "case_type", "ability"):
        value = data.get(key)
        if value:
            bits.append(f"{key}={_short_id(str(value))}")
    if route:
        bits.append(f"route={_enum_value(route)}")
    if codes:
        bits.append(f"codes={_short_id(','.join(str(code) for code in codes))}")
    prompt = str(data.get("prompt") or "").replace("\n", " ")
    proxy = str(data.get("proxy") or "").replace("\n", " ")
    if prompt:
        bits.append(f"prompt=\"{_clip(prompt, 120)}\"")
    if proxy:
        bits.append(f"proxy=\"{_clip(proxy, 120)}\"")
    return " ".join(bits)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _short_id(value: str) -> str:
    if len(value) <= 36:
        return value
    return f"{value[:16]}...{value[-16:]}"
