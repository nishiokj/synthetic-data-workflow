# LangGraph Migration

This document is the patch plan for porting `pipeline.py` from a hand-rolled
`while`/`for` orchestration to a LangGraph `StateGraph`. It is meant to be
executed end-to-end by one agent in one pass.

## Why

`router.route_after()` already returns a `RoutingDecision` with `next_stage`,
`context_policy`, and `terminal`. Today `pipeline.py` ignores `next_stage`
entirely — control flow lives in nested `for`/`while` loops, and the router is
called only to populate log records. This violates PLAN.md §3 principle 9
("the router, not the agent, decides what happens after a verdict") and
silently drops PLAN.md §7 (`context_policy` is computed but never threaded into
producer prompts).

The migration makes `RoutingDecision` load-bearing: graph edges read
`decision.next_stage`, and producer nodes read `decision.context_policy` to
shape their prompts.

## Design intent

1. `router.route_after()` is the single source of state-transition truth. Retry
   budgets live there only.
2. Each pipeline stage is one graph node. Nodes are pure
   `(state) -> state_update` functions. They emit a `StageRecord` and (for
   judged stages) call the router to set `state.last_decision`.
3. One conditional-edge function `route_from_decision(state)` reads
   `state.last_decision.next_stage` and dispatches to the corresponding node or
   `END`. It is reused on every routed edge.
4. Producer prompts (`Designer`, `SampleGenerator`) accept a `retry_route_code`
   /`retry_subcodes` pair. When set, the system prompt frames the call as a
   retry and includes route code + descriptive subcodes only — never judge
   prose, never the prior failed artifact (PLAN.md §7
   `CRITERIA_PLUS_ROUTE_CODE`).

## DELETE

`pipeline.py`:

- `PipelineRunner.run()` body (the outer `while plan_round <= max_design_retries`
  loop)
- `_design_batch()` — re-emerges as `node_design`
- `_audit_design()` — re-emerges as `node_audit_design`
- `_run_seed()` — the `for attempt in range(...)` retry loop disappears
  entirely; retries become graph cycles driven by `route_from_decision`
- `_validate_and_curate()` — splits into `node_validate_det`,
  `node_validate_semantic`, `node_curate`
- `_local_plan_verdict()` — moves to a top-level helper or inlines into
  `node_validate_design_batch_det`
- All `if decision.terminal: break` branches — replaced by the conditional
  edge mapping `decision.terminal` to `END`
- All `for attempt in range(...)` retry counters — replaced by the router's
  `retry_index < max_*_retries` check, which is already correct

`agents.py`, `models.py`, `router.py`, `rules.py`, `services/*`,
`observability.py`, `config.py`, `analyze.py`, `main.py`, `domains/qa_item.yaml`:
no deletions.

## ADD

### `requirements.txt`

```
langgraph
```

Pin during install. The migration uses only `StateGraph`, `START`, `END`, and
`add_conditional_edges` — no `Send`, no checkpointer, no advanced features.

### `pipeline.py` — new shape

#### State

```python
from typing import TypedDict
from models import (
    CandidateSample, CertifiedSample, CheckResult, RouteCode, RoutingDecision,
    SampleVerdict, DesignBrief,
)

class PipelineState(TypedDict, total=False):
    # static for the run
    run_id: str
    target_n: int

    # plan-round state (router consults this for design retries)
    plan_round: int                    # router's retry_index for plan stage
    plan_retry_route_code: RouteCode | None
    plan_retry_subcodes: list[str]

    # batch state
    designs_queue: list[DesignBrief]

    # per-design state
    design: DesignBrief | None
    gen_attempt: int                   # router's retry_index for gen/validation
    gen_retry_route_code: RouteCode | None
    gen_retry_subcodes: list[str]
    candidate: CandidateSample | None
    det_checks: list[CheckResult]
    sem_verdict: SampleVerdict | None

    # routing
    last_decision: RoutingDecision | None

    # bookkeeping
    committed_count: int
    dropped_count: int
```

State is owned by the runner instance only for the duration of one `invoke()`
call. Resources (clients, ledgers, writers) are attached to the runner and
closed over by node functions — they do not live in state.

#### Nodes

| Node name | Replaces | Stage logged | Calls router on |
|---|---|---|---|
| `design` | `_design_batch` | DESIGN | empty design list (terminal/retry) |
| `validate_design_batch_det` | `validate_design_batch` batch call | PLAN_AUDIT (det) | every call |
| `select_next_design` | `for design in designs:` head | (none, orchestration) | never |
| `audit_design` | `_audit_design` | PLAN_AUDIT (LLM) | every call (per-design reject is treated as drop, not replan — see Behavior changes #1) |
| `generate` | `generator.generate(...)` block | GENERATION | every call |
| `validate_det` | `deterministic_sample_verdict` block | VALIDATION (det) | every call |
| `validate_semantic` | `semantic_validator.validate(...)` block | VALIDATION (sem) | every call |
| `curate` | `corpus.curate(...)` block | CURATION | every call |

Each node:

1. Reads what it needs from `state`.
2. Does its work (LLM call, det check, ledger write, etc.).
3. Writes one `StageRecord` via `self.writer.write_stage_record(...)`.
4. For judged stages, calls `route_after(...)` and stores the result in
   `state["last_decision"]`.
5. Returns the partial state update.

#### Conditional edges

One reusable router-driven edge:

```python
def route_from_decision(state: PipelineState) -> str:
    decision = state["last_decision"]
    if decision.terminal:
        return END
    return {
        StageKind.DESIGN: "design",
        StageKind.PLAN_AUDIT: "audit_design",
        StageKind.GENERATION: "generate",
        StageKind.VALIDATION: "validate_det",
        StageKind.CURATION: "curate",
    }[decision.next_stage]
```

Two orchestration-only edges (do not consult the router):

```python
def after_curate(state: PipelineState) -> str:
    if state["committed_count"] >= state["target_n"]:
        return END
    if state["designs_queue"]:
        return "select_next_design"
    return "design"   # next plan round; retry budget enforced by router
                        # the first time the new round produces a reject

def after_select_next_design(state: PipelineState) -> str:
    return "audit_design" if state["design"] else "design"

def after_audit_design(state: PipelineState) -> str:
    # Per-design plan reject is a drop, not a replan trigger. Always advance.
    decision = state["last_decision"]
    if decision.verdict == Verdict.ACCEPT:
        return "generate"
    return "select_next_design" if state["designs_queue"] else "design"
```

#### Graph wiring

```
START                       -> design
design                      -> validate_design_batch_det          (always)
validate_design_batch_det      -> route_from_decision
                                 PLAN_AUDIT -> select_next_design
                                 DESIGN   -> design
                                 END        -> END
select_next_design            -> after_select_next_design
                                 audit_design / design
audit_design             -> after_audit_design
                                 generate / select_next_design / design
generate                    -> route_from_decision
                                 VALIDATION -> validate_det
                                 GENERATION -> generate
                                 END        -> END
validate_det                -> route_from_decision
                                 VALIDATION -> validate_semantic
                                 GENERATION -> generate
                                 END        -> END
validate_semantic           -> route_from_decision
                                 CURATION   -> curate
                                 GENERATION -> generate
                                 END        -> END
curate                      -> after_curate
                                 select_next_design / design / END
```

Note: `validate_det` accepting promotes to `validate_semantic`. The router
returns `next_stage=VALIDATION` for that case; the conditional edge maps that to
`validate_semantic` for det-stage callers and to `validate_det` for the
generate→validate edge. Two ways to handle this:

- **Recommended:** in `node_validate_det`, on ACCEPT, set
  `state["last_decision"].next_stage` to a synthetic value handled by a
  per-edge mapping (e.g., a slightly different conditional function on this
  edge). Or simpler: have `node_validate_det` on ACCEPT directly transition to
  `validate_semantic` without going through `route_from_decision`. This is a
  fixed promotion within VALIDATION, not a routing decision.
- Alternative: split `StageKind.VALIDATION` into `VALIDATION_DET` and
  `VALIDATION_SEM`. Cleaner but bigger contract change. Defer for now.

Pick the simple fix: `node_validate_det` returns
`{"next": "validate_semantic"}` on ACCEPT and uses `route_from_decision` only
on REJECT. Wire the conditional edge as:

```python
def after_validate_det(state):
    if state["det_accepted"]:
        return "validate_semantic"
    return route_from_decision(state)
```

#### Compiled graph

```python
class PipelineRunner:
    def __init__(self, config: RuntimeConfig) -> None:
        # ... existing field initialization ...
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(PipelineState)
        g.add_node("design", self.node_design)
        g.add_node("validate_design_batch_det", self.node_validate_design_batch_det)
        g.add_node("select_next_design", self.node_select_next_design)
        g.add_node("audit_design", self.node_audit_design)
        g.add_node("generate", self.node_generate)
        g.add_node("validate_det", self.node_validate_det)
        g.add_node("validate_semantic", self.node_validate_semantic)
        g.add_node("curate", self.node_curate)
        g.add_edge(START, "design")
        g.add_edge("design", "validate_design_batch_det")
        g.add_conditional_edges("validate_design_batch_det", route_from_decision)
        g.add_conditional_edges("select_next_design", after_select_next_design)
        g.add_conditional_edges("audit_design", after_audit_design)
        g.add_conditional_edges("generate", route_from_decision)
        g.add_conditional_edges("validate_det", after_validate_det)
        g.add_conditional_edges("validate_semantic", route_from_decision)
        g.add_conditional_edges("curate", after_curate)
        return g

    def run(self) -> dict[str, Any]:
        initial: PipelineState = {
            "run_id": self.config.run_id,
            "target_n": self.config.target_n,
            "plan_round": 0,
            "designs_queue": [],
            "gen_attempt": 0,
            "committed_count": 0,
            "dropped_count": 0,
        }
        final = self.graph.invoke(initial)
        return {
            "run_id": self.config.run_id,
            "committed": final["committed_count"],
            "dropped": final["dropped_count"],
        }
```

`PipelineRunner.__init__` and `run()` external signatures are preserved so
`main.py` and `tests/test_pipeline_smoke.py` do not change.

## CHANGE (small surgical edits)

### `agents.py`

`Designer.plan(...)` adds two optional parameters:

```python
def plan(
    self,
    *,
    run_id: str,
    target_n: int,
    coverage_snapshot: dict[str, int],
    retry_route_code: RouteCode | None = None,
    retry_subcodes: list[str] | None = None,
) -> tuple[list[DesignBrief], dict[str, Any]]:
```

When `retry_route_code` is set, prepend a single line to the user payload:

```
"prior_plan_rejection": {
    "route_code": "<route_code value>",
    "subcodes": [<descriptive labels>],
}
```

Keep the system prompt as-is (must continue to start with `"You are the
Designer..."` so the smoke-test fake routes correctly via substring match).

`SampleGenerator.generate(...)` mirrors this change with `retry_route_code` /
`retry_subcodes`. System prompt unchanged.

`DesignAuditor.audit(...)` and `SemanticValidator.validate(...)`: no change.
Judges are stateless and use `CRITERIA_ONLY`.

### `router.py`

No code change required. The existing branches are correct once `next_stage`
is consumed.

Optional cleanup: `from_stage=DESIGN` branch with `verdict=REJECT` returns
`terminal=True, next_stage=None`. After migration, `node_design` calls
`route_after` only when designs are empty — and the router currently treats that
as a hard terminal. We want the empty-designs case to retry within the plan
budget. Two options:

- **Recommended:** in `node_design`, when designs are empty, do not call
  `route_after(from_stage=DESIGN)`. Instead increment `plan_round` and let
  the graph re-enter `design` via the orchestration edge. The plan retry
  budget is already enforced by `node_validate_design_batch_det` calling
  `route_after(from_stage=PLAN_AUDIT, retry_index=plan_round)` on the next
  pass. This means an empty Designer response counts as one consumed plan
  round.
- Alternative: extend the `DESIGN` branch in `route_after` to support
  `RETRY_PROVIDER_EMPTY` as a retryable reject. Larger router change. Skip.

## Behavior changes (intentional)

These should be called out in the PR description.

1. **Per-design plan reject** changes from "drop and continue with next design in
   batch" to ... the same thing. The new orchestration edge
   `after_audit_design` preserves the existing behavior. Calling out
   explicitly because a naive port of `route_from_decision` to this edge would
   trigger a full replan on any per-design reject — wrong.
2. **Validation reject → Generation resample** today: pure resample with
   `FRESH` context (Generator prompt has no failure context). After migration:
   the `node_generate` reads `state["gen_retry_route_code"]` /
   `state["gen_retry_subcodes"]` and frames the call as
   `CRITERIA_PLUS_ROUTE_CODE`. **This closes the PLAN.md §7 / §8 gap**, where
   the router was computing `CRITERIA_PLUS_ROUTE_CODE` and the producer was
   ignoring it.
3. **Empty Designer output** today: silently spins the outer while loop.
   After migration: counts as one consumed plan round and the run terminates
   cleanly with `DROP_RETRY_EXHAUSTED` if every plan round comes back empty.
4. **`decision.terminal`** changes from "consulted in 1 place" to "consulted
   wherever a routed edge fires." Run-level termination is no longer dependent
   on the redundant outer `while` condition.

## What stays untouched

- `PipelineRunner` external signature (`__init__`, `run` return dict)
- All Pydantic models in `models.py`
- `router.route_after()` body (modulo the optional cleanup above)
- `rules.py`, `services/*`, `observability.py`, `config.py`, `analyze.py`,
  `main.py`, `domains/qa_item.yaml`
- The CLI surface
- The on-disk artifact paths and JSONL formats

## Test impact

- `tests/test_router.py` — no change. Router contract unchanged.
- `tests/test_rules.py` — no change.
- `tests/test_schemas.py` — no change.
- `tests/test_pipeline_smoke.py` — passes unchanged. The fake monkeypatches
  `pipeline.OpenAIClient`, the new pipeline keeps that import, and
  `PipelineRunner(config).run()` returns the same dict shape with the same
  side-effect file paths. **Constraint:** keep `"Designer"`,
  `"Design Auditor"`, `"Sample Generator"`, `"Semantic Validator"` as substrings
  in the respective system prompts so the fake's substring routing still
  works.
- New `tests/test_pipeline_graph.py`:
  - The compiled graph contains exactly the eight nodes named above.
  - `route_from_decision` returns the correct node name for each
    `(next_stage, terminal)` combination. Build a synthetic `PipelineState`
    with a `RoutingDecision` and assert.
  - `after_curate`, `after_select_next_design`, `after_audit_design` each
    cover their branches with one-line state stubs.

## Order of operations

Each step ends with a passing `pytest`. Do not stack steps.

1. Add `langgraph` to `requirements.txt`. Install. Confirm
   `from langgraph.graph import StateGraph, START, END` imports.
2. Add `PipelineState`, `route_from_decision`, `after_curate`,
   `after_select_next_design`, `after_audit_design`, `after_validate_det` to
   `pipeline.py` as module-level definitions. Do not wire them yet. Run tests.
3. Add `node_*` methods on `PipelineRunner` as no-ops that only call
   `self._record(...)` with placeholder data. Add `_build_graph()` and
   `compile()`. Do not switch `run()` over yet. Run tests.
4. Port logic into nodes one at a time, in this order:
   `node_design` → `node_validate_design_batch_det` →
   `node_select_next_design` → `node_audit_design` → `node_generate` →
   `node_validate_det` → `node_validate_semantic` → `node_curate`. Each port
   is a separate commit. The smoke test stays passing throughout because
   `run()` still uses the old hand-rolled body.
5. Switch `run()` to call `self.graph.invoke(initial)`. Run tests.
6. Delete the old `_design_batch`, `_audit_design`, `_run_seed`,
   `_validate_and_curate`, `_local_plan_verdict`. Run tests.
7. Update `Designer.plan` and `SampleGenerator.generate` signatures to accept
   retry context. Update the two corresponding nodes to pass the retry context
   through from state. Run tests.
8. Add `tests/test_pipeline_graph.py`. Run tests.

If any step fails the smoke test, do not advance. Diagnose the actual node;
do not paper over with a try/except.

## Out of scope for this migration

These are flagged in the audit but are not part of this patch:

- Per-stage `criteria_hash` (currently the whole-domain hash on every record)
- Splitting `subcodes` and `reason_codes` in `rules.py:60-61`
- Cost/latency tracking (always `0.0`)
- Splitting the conflated retry budget for content vs. infra/parse
- Missing length-bound and conditional context-non-empty deterministic rules
- README pipeline-diagram naming drift
- Designer not receiving aggregate route-code summaries

These should land in separate commits after the migration is in.
