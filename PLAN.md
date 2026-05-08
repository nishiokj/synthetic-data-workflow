# Synthetic Data Pipeline - POC Implementation Plan

> Superseded note: this document captures the original Validator-training-data
> POC. The implemented direction has pivoted to benchmark proxy case generation.
> See `docs/BENCHMARK_PROXY_PATCH_PLAN.md` and `README.md` for the current
> benchmark-oriented framing and command examples.

A staged LangGraph pipeline that uses stateless, role-scoped agents to generate
training data for one pipeline target per run. POC 1 target: **Validator
training data**.

---

## 1. Goal

Run a pipeline end-to-end that emits:

1. A **committed dataset** of Validator training examples (one JSONL file).
2. A **structured run log** (per-stage JSONL) that doubles as a meta-dataset
   for future stage-level RL.
3. **Diversity and quality-proxy metrics** computed offline over 1 + 2.

The user eyeballs the committed dataset. No human-in-the-loop, no production
concerns.

The deeper goal is to test a workflow discipline: agents do bounded jobs from
bounded context, judges do not create or repair, and a deterministic router owns
state transitions.

## 2. Non-goals (POC)

- Any RL training (that is POC-2, on a rented node).
- Human review queue or retries beyond the documented bounds.
- Swapping domains mid-run. One domain YAML per run.
- Cross-run parallelism. Single-process, single-run.

## 3. Core principles

1. **Engineered diversity**, not emergent. Taxonomy drives sampling; embedding
   novelty gates ingest.
2. **Role separation.** No stage judges its own output. Judges are stateless
   and receive artifact + criteria only — never producer reasoning.
3. **Deterministic wherever possible.** LLM calls are reserved for semantic
   checks, subjective criteria, and generation. Curator is fully deterministic;
   Validator and Design Auditor may combine deterministic checks with LLM semantic
   checks.
4. **No modification.** Every judge verdict is `ACCEPT | REJECT` plus fixed
   route codes, subcodes, reason codes, and evidence references. A judge never
   rewrites an upstream artifact.
5. **Retry vs. repair is stage-dependent, not universal.** Policy table in §8.
6. **Plan diversity ≠ generation diversity.** Both are enforced, at different
   stages, with different mechanisms.
7. **Structured observability is a product, not a side effect.** The Stage Run
   Log is the meta-dataset.
8. **Domain-agnostic roles, domain-specific contracts.** A domain YAML carries
   taxonomy, schemas, and reason codes. Roles themselves do not change.
9. **Agents do not manage pipeline state.** A stage is a pipeline state with a
   context contract. An agent is an executor role used by that stage. The router,
   not the agent, decides what happens after a verdict.
10. **Judges classify, they do not bridge.** Judge outputs are fixed verdicts,
    fixed route codes, descriptive subcodes, and evidence references. They do
    not provide repair instructions, rewritten artifacts, or "how to fix this"
    suggestions.
11. **Post-rejection producers rediscover the failure.** A producer invoked
    after rejection receives criteria and a route hint, not the judge's proposed
    solution. This avoids treating judge feedback as ground truth.

## 4. Pitfalls addressed (and by which mechanism)

| Pitfall | Defense |
|---|---|
| Model/mode collapse | Curator embedding-novelty gate + pairwise-distance monitoring |
| Shallow diversity | Coverage ledger drives Designer; Design Auditor rejects near-duplicates at plan time |
| Coverage gaps | Taxonomy cells + bias toward undercovered cells |
| Reward hacking | Role separation; stateless judges with independent criteria |
| Label/solution leakage | Deterministic regex + schema checks in Validator |
| Subjective/semantic correctness drift | LLM semantic checks with fixed criteria, route codes, subcodes, and evidence |
| Stylistic homogeneity | Style fingerprint dispersion tracked in metrics (POC: monitor; reject: POC-2) |
| Trivially-easy bias | Difficulty axis in taxonomy; coverage ledger counts cells |
| Factuality drift | Out of POC scope (eyeball) |

## 5. Concepts

### Agents vs. stages

An **agent** is a stateless role executor. It receives a compact prompt contract,
one bounded task, and a schema for its own output. It does not need to understand
the full pipeline, the identity of other agents, or the fact that downstream
completion is desired.

A **stage** is a pipeline state. It defines:

- which artifact type is consumed and produced
- which criteria apply
- which agent role, if any, may be invoked
- what context policy is used
- which route codes it may emit or receive

This separation prevents prompts like "fix what the validator found" from
turning a neutral producer into a pipeline-completion optimizer.

### Judge output boundary

Judges may emit:

- `verdict`
- fixed `route_code`
- fixed descriptive `subcodes`
- fixed `reason_codes`
- evidence references, such as check ids or short spans from the artifact

Judges must not emit:

- rewritten artifacts
- enriched artifacts
- suggested patches
- natural-language repair plans
- subcodes that imply a specific cure or implementation path
- hidden chain-of-thought or producer reasoning

## 6. Pipeline shape

```
  ┌────────────┐      ┌──────────────────┐      ┌───────────┐
  │ Designer │ ───► │   Design Auditor     │ ───► │ Generator │ ──┐
  └────────────┘      │  (det + LLM)     │      └───────────┘   │
       ▲              └──────────────────┘                      │
       │                      │                                 ▼
       │                      │                      ┌──────────────────┐
       │                      │                      │    Validator     │
       │                      │                      │   (det + LLM)    │
       │                      │                      └──────────────────┘
       │                      │                                 │
       │                      │                                 ▼
       │                      │                           ┌──────────┐
       │                      │                           │ Curator  │
       │                      │                           │ (det)    │
       │                      │                           └──────────┘
       │                      │                                 │
       │            (coverage-gap feedback)                     │
       └──────────────────────┴─────────────────────────────────┘
```

### Stage contracts

| Stage | Det / LLM | Input | Output | Post-hoc outcome metric |
|---|---|---|---|---|
| Designer | LLM | `DomainSpec`, `CoverageLedger` snapshot, aggregate route-code summaries | `list[DesignBrief]` | fraction of designs approved by Design Auditor |
| Design Auditor (det) | Det | `list[DesignBrief]`, corpus index | accepted designs + `DesignVerdict` records | n/a (filter) |
| Design Auditor (LLM) | LLM | novelty-filtered designs, taxonomy, plan criteria | `DesignVerdict` records only | downstream commit rate per design |
| Generator | LLM | one approved design, inner-candidate schema, reason-code taxonomy | `Candidate` (a Validator training example) | fraction passing Validator |
| Validator (det) | Det | `Candidate`, schema, leakage rules | deterministic `SampleVerdict` | fraction passing Curator |
| Validator (LLM) | LLM | `Candidate`, semantic criteria, deterministic verdict trail | semantic `SampleVerdict` | fraction passing Curator |
| Curator | Det | `CertifiedSample`, corpus index, coverage ledger | `CommittedSample`, ledger + index updates | diversity metrics over corpus |

Logged outcome metrics are **not** computed in-loop and are never shown to the
agent performing the work. They fall out of the Stage Run Log via `analyze.py`
post-hoc.

`Design Auditor (LLM)` is a judge, not an enricher. If design enrichment becomes
useful, it must be introduced as a separate producer stage, then judged by a
Design Auditor.

## 7. Routing and outcome contract

Routing is first-class state management. Every stage invocation terminates in a
small, fixed outcome vocabulary. Free-form explanation can be logged for humans,
but it is not used for automated routing and is not passed as repair guidance.
The router is binary in POC 1: a stage accepts or rejects. A run either commits
approved data or terminates because a documented retry, timeout, or policy
ceiling was hit.

Initial route-code vocabulary:

```python
class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"

class RouteCode(str, Enum):
    ACCEPT = "accept"

    # Content or criteria failures.
    REJECT_CRITERIA_MISMATCH = "reject_criteria_mismatch"
    REJECT_SCHEMA = "reject_schema"
    REJECT_LEAKAGE = "reject_leakage"
    REJECT_DUPLICATE = "reject_duplicate"
    REJECT_COVERAGE_MISMATCH = "reject_coverage_mismatch"
    REJECT_SEMANTIC_MISMATCH = "reject_semantic_mismatch"
    REJECT_UPSTREAM_INVARIANT = "reject_upstream_invariant"

    # Non-semantic execution failures.
    RETRY_INFRA = "retry_infra"
    RETRY_PARSE = "retry_parse"
    RETRY_PROVIDER_EMPTY = "retry_provider_empty"

    # Terminal outcomes.
    DROP_RETRY_EXHAUSTED = "drop_retry_exhausted"
    DROP_TIMEOUT = "drop_timeout"
    DROP_POLICY_CEILING = "drop_policy_ceiling"

class SubCode(str, Enum):
    # Domain YAML owns the concrete values. Examples:
    AMBIGUOUS_QUESTION = "ambiguous_question"
    OFF_TOPIC_ANSWER = "off_topic_answer"
    TRIVIALLY_EASY = "trivially_easy"
    EMPTY_CONTEXT_CITATION = "empty_context_citation"

class ContextPolicy(str, Enum):
    FRESH = "fresh"
    SAME_INPUT_RETRY = "same_input_retry"
    CRITERIA_ONLY = "criteria_only"
    ROUTE_CODE_ONLY = "route_code_only"
    CRITERIA_PLUS_ROUTE_CODE = "criteria_plus_route_code"

class RoutingDecision(BaseModel):
    run_id: str
    from_stage: StageKind
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[SubCode]
    reason_codes: list[str]
    next_stage: StageKind | None
    context_policy: ContextPolicy
    retry_index: int
    attempt_of: str | None
    terminal: bool
```

Route codes are domain-configurable but closed per run. A stage may only emit
route codes declared in the domain YAML and allowed by the stage contract.

Subcodes are descriptive, not prescriptive. They may identify what criterion was
not met, such as `ambiguous_question`, but must not encode a cure, such as
`rewrite_question_to_be_specific`.

`REJECT_UPSTREAM_INVARIANT` is a watchlist code, not a back-routing mechanism in
POC 1. The first pass assumes that reaching stage C implies stage A and stage B
accepted their artifacts correctly. If a later stage repeatedly rejects an
artifact that should have failed earlier, the run logs the invariant break and
eventually hits a retry or policy ceiling rather than sending the artifact back
to an earlier semantic stage.

### Context policy rules

- Infra, provider, and parse failures use `SAME_INPUT_RETRY`.
- Pure resampling uses `FRESH`; the producer sees the same criteria, no failure
  context.
- Reconciliation-style stages, if added, use `CRITERIA_PLUS_ROUTE_CODE`. They
  may receive route code + descriptive subcodes. They do not receive
  natural-language fix suggestions.
- Judges use `CRITERIA_ONLY`; they do not receive producer reasoning or prior
  failed attempts unless that history is explicitly part of the artifact under
  judgment.
- Downstream agents are never told that the pipeline needs completion.

## 8. Retry / repair policy

| Boundary | On REJECT | Bound | Rationale |
|---|---|---|---|
| Designer → Design Auditor | **Re-plan**: produce a fresh plan from criteria + route-code summary | N=2 then drop batch | Plan failure may indicate a bad cell choice or duplicate plan. The producer should rediscover the issue, not implement judge advice. |
| Generator → Validator (det or semantic) | **Pure resample** for content failures; **same-input retry** only for infra/parse failures | N=2 then drop design | Content feedback biases surface fixes. Infra failures are not semantic signal. |
| Validator → Curator (novelty) | **Discard; log coverage-gap to ledger** | No retry | Sample is genuinely redundant; retry cannot help |

POC 1 avoids true reconciliation stages unless necessary. If a reconciliation
stage is introduced later, it must be specified as its own stage with a context
contract, route-code inputs, and independent downstream judgment.

The expected happy path is always an approved piece of data. Terminal drops are
not alternate success states; they indicate retry exhaustion, timeout, policy
ceiling, insufficient model performance, bad criteria, or an upstream invariant
break that the POC router cannot safely repair.

## 9. Services (beyond blob store)

| Service | Purpose | Backing (POC) |
|---|---|---|
| Corpus Index | Embeddings of committed samples + approved designs. k-NN novelty queries. | OpenAI `text-embedding-3-small` + NumPy flat array on disk |
| Coverage Ledger | Counts per taxonomy cell. Drives Designer bias. | JSON file |
| Validation Ledger | Per-sample verdict trail with route codes and reason codes. Feeds deterministic routing and aggregate coverage-gap feedback. | JSONL file |
| Rejection Archive | Full rejected artifacts, verdict trail, route codes, subcodes, and evidence for future analysis. Not used for novelty memory in POC 1. | JSONL file |
| Stage Run Log | One record per stage invocation. **This is the meta-dataset.** | JSONL file per run |

No SQLite in POC. Flat files are enough and trivially inspectable.

State scope:

- `run_id` identifies one execution.
- `domain_id` identifies the domain YAML and reason-code vocabulary.
- `dataset_version` identifies the committed corpus namespace.
- Corpus index and coverage ledger are domain-scoped and dataset-version-scoped,
  not merely run-scoped, so collapse can be measured across runs.
- Rejected artifacts are retained in the Rejection Archive, but only committed
  samples and approved designs enter the POC novelty index.

## 10. Data models (sketches)

Pydantic. Every artifact carries a stable `id`, a content hash, and provenance
to the upstream artifact.

```python
class StageKind(str, Enum):
    DESIGN = "design"
    PLAN_AUDIT = "plan_audit"
    GENERATION = "generation"
    VALIDATION = "validation"
    CURATION = "curation"

class AgentRole(str, Enum):
    STRATEGIST = "designer"
    PLAN_JUDGE = "plan_judge"
    GENERATOR = "generator"
    SEMANTIC_VALIDATOR = "semantic_validator"

class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"

class DesignBrief(BaseModel):
    id: str
    target_stage: Literal["validator"]           # POC 1
    cell: TaxonomyCell                           # failure_mode × difficulty × scenario
    intent: str                                  # natural-language brief for Generator
    parent_plan_id: str | None

class ApprovedPlan(BaseModel):
    id: str
    designs: list[DesignBrief]                        # accepted exactly as submitted
    rejections: list[RejectionRecord]            # designs dropped + route/evidence

class DesignVerdict(BaseModel):
    design_id: str
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[SubCode]
    reason_codes: list[str]
    evidence: list[EvidenceRef]

class SampleVerdict(BaseModel):
    candidate_id: str
    check_kind: Literal["deterministic", "semantic"]
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[SubCode]
    reason_codes: list[str]
    evidence: list[EvidenceRef]

class Candidate(BaseModel):
    """A Validator training example. This IS what the pipeline emits."""
    id: str
    design_id: str
    inner_input: InnerCandidate                  # the artifact the inner Validator would see
    inner_criteria: InnerCriteria                # schema + rules the inner Validator checks
    inner_verdict: Verdict                       # the ground-truth label the 4B must learn
    inner_reason_codes: list[str]                # ground-truth reasons (may be empty if ACCEPT)
    difficulty: int                              # 1..5
    failure_mode: str | None                     # None if inner_verdict == ACCEPT

class CertifiedSample(BaseModel):
    id: str; candidate_id: str
    deterministic_checks: list[CheckResult]      # schema, leakage regex, length bounds
    semantic_checks: list[SampleVerdict]         # LLM checks for semantic rules/principles

class CommittedSample(BaseModel):
    id: str; certified_id: str
    embedding_ref: str                           # index key
    nn_distance: float                           # distance to nearest existing sample
    taxonomy_cell: TaxonomyCell

class StageRecord(BaseModel):
    run_id: str; stage_id: str; role: str
    stage_kind: StageKind; agent_role: AgentRole | None
    parent_artifact_id: str | None; artifact_id: str
    model: str; provider: str
    prompt_hash: str; input_tokens: int; output_tokens: int
    latency_ms: int; cost_usd: float
    verdict: Verdict; route_code: RouteCode
    subcodes: list[SubCode]; reason_codes: list[str]
    criteria_hash: str; context_policy: ContextPolicy
    retry_index: int; attempt_of: str | None
    wallclock_ts: str
```

`InnerCandidate` / `InnerCriteria` schemas are loaded from the domain YAML —
roles don't change when the domain does; contracts do.

## 11. First target: Validator training data

A committed sample is a tuple the 4B Validator will learn from:

```
input  =  (inner_input, inner_criteria)
output =  (inner_verdict, inner_reason_codes)
```

The pipeline's job is to generate these tuples with:

- **Breadth** across failure modes (not all ACCEPTs; not all the same reject)
- **Balanced difficulty**
- **Non-redundancy** (novelty gate)
- **Correctness** of `inner_verdict` given `inner_input` + `inner_criteria`
  (enforced through deterministic checks where possible and LLM semantic checks
  for rules that require judgment)

### Inner-candidate domain (POC 1 choice)

`InnerCandidate` = a short structured Q&A item: `question`, `claimed_answer`,
`context` (optional). `InnerCriteria` = schema + a short list of rule
predicates (e.g., "answer must not appear verbatim in question", "context
must be non-empty if claimed_answer cites it").

Chosen because:
- Universally in pretraining distribution.
- Failure modes are concrete and deterministically detectable for a subset:
  `label_leakage`, `empty_context_citation`, `schema_violation`,
  `trivially_easy`, `ambiguous_question`, `off_topic_answer`.
- Swappable via domain YAML without touching role code.

Subjective labels are allowed in POC 1, but they must be explicitly supported by
LLM semantic checks with written criteria, fixed route codes, descriptive
subcodes, and evidence. They are not "eyeball only" labels.

### Failure-mode taxonomy (initial)

```
failure_mode ∈ {
  accept_clean,
  label_leakage,
  schema_violation,
  empty_context_citation,
  trivially_easy,
  ambiguous_question,
  off_topic_answer,
}
difficulty ∈ {1, 2, 3, 4, 5}
scenario  ∈ {nominal, edge, adversarial}
```

Taxonomy cell = `(failure_mode, difficulty, scenario)`. Coverage ledger counts
these. Designer samples underrepresented cells.

## 12. Measurement

Computed by `analyze.py` over the Stage Run Log + committed dataset. None
in-loop.

**Diversity**
- Mean and distribution of pairwise cosine distance across committed samples.
- Coverage entropy `H(cell)` across taxonomy cells.
- Near-duplicate rate at k=5 NN under threshold τ (config).
- Style fingerprint dispersion: length dist, type-token ratio, punctuation
  profile per failure_mode.

**Quality proxy**
- Deterministic-Validator pass rate, broken down by route code, subcode, and
  reason code.
- LLM semantic-Validator pass rate, broken down by route code, subcode, and
  reason code.
- Route-code distribution, including terminal drop reasons.
- Curator accept rate.
- First-try vs. post-retry commit ratio.
- Per-stage LLM cost and latency (sanity / budgeting).

Primary signal for POC: **eyeball the committed dataset**. Metrics exist to
surface obvious failures early.

## 13. Repo layout

```
main.py                  # CLI entrypoint
pipeline.py              # LangGraph wiring: nodes, edges, retry policy
agents.py                # LLM roles: designer, plan_judge, generator, semantic_validator
router.py                # RoutingDecision table + context policies
rules.py                 # Deterministic: schema, leakage regex, length, novelty, coverage
models.py                # Pydantic schemas (§10)
config.py                # Per-stage model assignment, thresholds, domain YAML loader
observability.py         # StageRecord writer, cost/latency tracking
analyze.py               # Diversity + quality-proxy metrics

services/
  corpus_index.py        # Embed + NN index (OpenAI embeddings + NumPy)
  coverage_ledger.py     # Taxonomy-cell counts (JSON)
  validation_ledger.py   # Verdict trail (JSONL)
  rejection_archive.py   # Rejected artifacts retained outside novelty memory

domains/
  qa_item.yaml           # POC 1: inner-candidate schema, criteria, failure modes

logs/                    # run_id/stage_records.jsonl, run_id/validation.jsonl
                         # run_id/rejections.jsonl
data/
  corpus/                # committed samples (JSONL)
  index/                 # embeddings + id map

tests/
  test_router.py         # route-code transition table
  test_rules.py          # deterministic checks
  test_schemas.py        # pydantic roundtrip
  test_pipeline_smoke.py # one-pass end-to-end with tiny target_n
```

## 14. CLI surface

```
python main.py \
  --domain domains/qa_item.yaml \
  --target-stage validator \
  --target-n 200 \
  --model gpt-5-mini \
  --provider openai \
  --per-stage-model designer=gpt-5-mini,plan_judge=gpt-5-mini,generator=gpt-5-mini \
  --seed 42 \
  --run-id auto
```

Defaults: `gpt-5-mini`, `openai`, per-stage inherits `--model`. `--seed`
controls sampling determinism (not model determinism).

## 15. Config shape

`config.py` resolves, in order: CLI flags → env vars → domain YAML defaults →
hardcoded fallbacks. Domain YAML owns: inner-candidate schema, failure-mode
taxonomy, reason codes, route-code allowlist, subcode vocabulary,
deterministic rule set, semantic rule set, novelty threshold.

## 16. Build order (milestones)

1. **M1 — schemas + logging.** `models.py`, `observability.py`,
   `router.py`, `test_schemas.py`, `test_router.py`. Nothing runs yet, but
   every artifact shape exists, `RoutingDecision` is testable, and
   `StageRecord` emits JSONL.
2. **M2 — deterministic rules + services.** `rules.py` (schema, leakage,
   length), `services/*`. Unit-tested in isolation. Curator and Rejection
   Archive work.
3. **M3 — single-design generator slice.** Hard-code one `DesignBrief`, run
   Generator → Validator → Curator with an LLM call. One sample commits. Log
   records emitted.
4. **M4 — Designer + PlanJudge + semantic Validator + LangGraph wiring.**
   Full pipeline one-shot, `--target-n 5`. Routing table and retry policy live.
5. **M5 — coverage-driven sampling + feedback loop.** Designer reads
   ledger; aggregate route-code summaries feed back.
6. **M6 — `analyze.py`.** Diversity + quality-proxy over a 200-sample run.
   Eyeball the dataset. Iterate on taxonomy / thresholds.

Each milestone is runnable on its own. No milestone is "just refactoring."

## 17. Open items (resolve during build, not before)

- Exact novelty threshold τ — pick by looking at the distance distribution on
  a small warmup run.
- Reason-code vocabulary for the inner Validator — start with the seven in
  §11, expand only if eyeballing shows genuine ambiguity.
- Exact semantic subcode vocabulary for subjective checks. Subcodes must be
  descriptive labels, not disguised fix instructions.
- Whether rejected artifacts should enter any future anti-collapse memory index
  after POC 1. Default: retain them in the archive only.
