# Pipeline Reference

A single-page guide to runtime nodes, route codes, context policies, and retry bounds.

---

## Runtime Graph

```
                         ╔══════════════════════════════════╗
                         ║         PLANNING  BATCH          ║
                         ║                                  ║
              ┌──────────╢  ┌─────────────────────────────┐ ║
              │          ║  │  strategy                   │ ║
              │          ║  │  Strategist · LLM           │ ║
              │          ║  └──────────────┬──────────────┘ ║
              │          ║                 │ seed batch      ║
              │          ║  ┌──────────────▼──────────────┐ ║
              │          ║  │  validate_seed_plan_det     │ ║
              │          ║  │  Batch plan check · Det     │ ║
              │          ║  └──────────────┬──────────────┘ ║
              │          ╚═════════════════╪════════════════╝
              │                           │
              │   ┌── reject_coverage_mismatch
              │   │   reject_duplicate ◄──┘  (retry < max_plan_retries)
              │   │
              └───┘   drop_retry_exhausted ──────────────────────► END
                                │
                                │ accept
                                ▼
                      ╔═════════════════════════════════════════════╗
                      ║            PER-SEED  EXECUTION             ║
                      ║                                            ║
                      ║   ┌────────────────────────────────────┐   ║
                      ║   │  select_next_seed                  │   ║
                      ║   │  Queue cursor · Det                │◄──╫──────────────────┐
                      ║   └──────────────────┬─────────────────┘   ║                  │
                      ║          seed        │         queue        ║                  │
                      ║       available      │         empty    ────╫──► strategy      │
                      ║                      ▼                      ║                  │
                      ║   ┌────────────────────────────────────┐   ║                  │
                      ║   │  audit_seed_plan                   │   ║                  │
                      ║   │  PlanAuditor · LLM                 │   ║                  │
                      ║   └──────────────────┬─────────────────┘   ║                  │
                      ║     reject_* ────────┼──► drop archive     ║                  │
                      ║                      │ accept               ║                  │
                      ║                      ▼                      ║                  │
                      ║   ┌────────────────────────────────────┐   ║                  │
                      ║   │  generate                          │◄──╫──┐               │
                      ║   │  SampleGenerator · LLM             │   ║  │ retry_infra   │
                      ║   └──────────────────┬─────────────────┘   ║  │ retry_parse   │
                      ║                      │                      ║  └─ retry_provider_empty
                      ║     drop_retry_exhausted ─────────────────►╫──► drop archive  │
                      ║                      │ accept               ║                  │
                      ║                      ▼                      ║                  │
                      ║   ┌────────────────────────────────────┐   ║                  │
                      ║   │  validate_det                      │   ║                  │
                      ║   │  Schema · leakage · taxonomy · Det │   ║                  │
                      ║   └──────────────────┬─────────────────┘   ║                  │
                      ║    reject_schema ─────┤                     ║                  │
                      ║    reject_leakage ────┼──► generate (FRESH) ║                  │
                      ║    reject_coverage ───┘                     ║                  │
                      ║     drop_retry_exhausted ─────────────────►╫──► drop archive  │
                      ║                      │ accept               ║                  │
                      ║                      ▼                      ║                  │
                      ║   ┌────────────────────────────────────┐   ║                  │
                      ║   │  validate_semantic                 │   ║                  │
                      ║   │  SemanticValidator · LLM           │   ║                  │
                      ║   └──────────────────┬─────────────────┘   ║                  │
                      ║  reject_criteria ─────┤                     ║                  │
                      ║  reject_semantic ─────┼──► generate (FRESH) ║                  │
                      ║     drop_retry_exhausted ─────────────────►╫──► drop archive  │
                      ║                      │ accept               ║                  │
                      ║                      ▼                      ║                  │
                      ║   ┌────────────────────────────────────┐   ║                  │
                      ║   │  curate                            │   ║                  │
                      ║   │  Novelty + commit · Det            │   ║                  │
                      ║   └──────────────────┬─────────────────┘   ║                  │
                      ║   reject_duplicate ───┴──► drop archive     ║                  │
                      ╚══════════════════════╪════════════════════════                  │
                                             │ accept                                   │
                                             ▼                                          │
                                   ┌──────────────────┐                                │
                                   │  COMMITTED SAMPLE │                                │
                                   │  data/corpus/     │                                │
                                   └────────┬─────────┘                                │
                            target_n        │       more seeds      queue empty /       │
                            reached         │       queued          target not met      │
                               ▼            │           └──────────────────────────────┘
                              END           └──────────► select_next_seed
```

Every stage writes one `StageRecord` to `logs/<run_id>/stage_records.jsonl`.

---

## Node Quick Reference

```
  Node                        fn name                    Role              Det/LLM
  ──────────────────────────────────────────────────────────────────────────────────
  Strategy                    strategy                   Strategist        LLM
  Batch plan check            validate_seed_plan_det     —                 Det
  Select next seed            select_next_seed           —                 Det (queue)
  Seed plan audit             audit_seed_plan            PlanAuditor       LLM
  Generate sample             generate                   SampleGenerator   LLM
  Deterministic validation    validate_det               —                 Det
  Semantic validation         validate_semantic          SemanticValidator LLM
  Curate corpus               curate                     —                 Det
```

---

## Route Codes

### Accept

```
  accept                      Artifact passes all criteria; advance to next stage.
```

### Content / criteria failures  ← these trigger FRESH resample

```
  reject_criteria_mismatch    Failed a stated evaluation criterion       semantic validator
  reject_schema               Artifact violates JSON/Pydantic schema     det validator
  reject_leakage              Label/answer appears verbatim in inputs    det validator
  reject_duplicate            Embedding distance below novelty τ         batch plan check · curator
  reject_coverage_mismatch    Taxonomy cell saturated or mismatched      batch plan check · det validator
  reject_semantic_mismatch    Semantic rule violated (LLM judgment)      semantic validator
  reject_upstream_invariant   Should have failed an earlier stage        any  (watchlist only, POC 1)
```

### Infra / execution failures  ← these trigger SAME_INPUT_RETRY

```
  retry_infra                 Network error · timeout · provider failure
  retry_parse                 Response didn't parse into expected schema
  retry_provider_empty        Valid but empty provider response
```

### Terminal outcomes

```
  drop_retry_exhausted        Retry ceiling hit; artifact discarded
  drop_timeout                Wall-clock timeout exceeded
  drop_policy_ceiling         Run-level drop ceiling exceeded
```

---

## Context Policies

```
  Policy                  Used when                               What the producer sees
  ─────────────────────────────────────────────────────────────────────────────────────
  FRESH                   Pure resample after content rejection   Criteria only — no failure context
  SAME_INPUT_RETRY        Infra / parse / provider-empty          Identical input, no failure context
  CRITERIA_ONLY           Any judge invocation                    Artifact + criteria only
  ROUTE_CODE_ONLY         (Reserved)                              Artifact + route code
  CRITERIA_PLUS_ROUTE_CODE Reconciliation stage (post-POC 1)     Criteria + route code + subcodes
```

---

## Routing Transitions

### Planning batch

```
  strategy
    ├─ (batch emitted) ──────────────────────────────► validate_seed_plan_det
    └─ retry_provider_empty (still routed through batch check)

  validate_seed_plan_det
    ├─ accept ───────────────────────────────────────► select_next_seed
    ├─ reject_coverage_mismatch  }
    │  reject_duplicate          }  retry < max_plan_retries  ──► strategy  [FRESH]
    └─ drop_retry_exhausted ─────────────────────────► END  (drop batch)
```

### Seed loop

```
  select_next_seed
    ├─ seed available ───────────────────────────────► audit_seed_plan
    └─ queue empty · target not met ────────────────► strategy

  audit_seed_plan
    ├─ accept ───────────────────────────────────────► generate
    └─ reject_* ─────── archive ─────────────────────► select_next_seed  (or strategy)

  generate
    ├─ accept ───────────────────────────────────────► validate_det
    ├─ retry_infra / retry_parse / retry_provider_empty
    │    retry < max_generation_retries  ────────────► generate  [SAME_INPUT_RETRY]
    └─ drop_retry_exhausted ─────── archive ─────────► select_next_seed

  validate_det
    ├─ accept ───────────────────────────────────────► validate_semantic
    ├─ reject_schema / reject_leakage / reject_coverage_mismatch
    │    retry < max_generation_retries  ────────────► generate  [FRESH]
    └─ drop_retry_exhausted ─────── archive ─────────► select_next_seed

  validate_semantic
    ├─ accept ───────────────────────────────────────► curate
    ├─ reject_criteria_mismatch / reject_semantic_mismatch
    │    retry < max_generation_retries  ────────────► generate  [FRESH]
    └─ drop_retry_exhausted ─────── archive ─────────► select_next_seed

  curate
    ├─ accept ───────────────────────────────────────► commit
    └─ reject_duplicate ──── archive ────────────────► select_next_seed  (no retry useful)

  commit
    ├─ target_n reached ─────────────────────────────► END
    ├─ more seeds queued ────────────────────────────► select_next_seed
    └─ queue empty · target not met ────────────────► strategy
```

---

## Retry Bounds

```
  Boundary                                 Failure type       Limit              On exhaustion
  ──────────────────────────────────────────────────────────────────────────────────────────────
  strategy → validate_seed_plan_det        Content/coverage   max_plan_retries   Drop batch → END
  generate  (infra/parse)                  Infra · parse      max_gen_retries    Drop seed
  generate  (content from det/sem)         Schema · semantic  max_gen_retries    Drop seed
  curate    (novelty)                       Duplicate          No retry           Log gap; discard
```

---

## Invariants

```
  1. Agents do not pick routes.
     Every stage emits (verdict, route_code); router.route_after() owns the transition.

  2. Judges never rewrite.
     A verdict contains verdict · route_code · subcodes · reason_codes · evidence — nothing else.

  3. No repair guidance in context.
     Producers retrying after rejection receive criteria (+ optionally route code),
     never judge prose or a suggested fix.

  4. No stage judges its own output.
     Role separation is enforced by stage contract, not by convention.

  5. Metrics are post-hoc.
     analyze.py reads Stage Run Logs after the run.
     No metric is fed back to in-loop agents during the same run.
```
