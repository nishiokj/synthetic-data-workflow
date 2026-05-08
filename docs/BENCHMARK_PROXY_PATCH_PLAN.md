# Benchmark Proxy Patch Plan

## Status

This document is a patch plan, not an implementation spec. It captures the
current product/design pivot and identifies the code surfaces that likely need
to change. It intentionally leaves unresolved questions open instead of
pretending the answers are known.

The current repository is still oriented around generating Validator training
examples for a Q&A-style domain. That domain made the pipeline difficult to
reason about because it was meta: the pipeline generated examples for a future
validator, and each generated example was itself a validation task.

The proposed pivot is to make the POC generate benchmark cases.

The central claim becomes:

```text
Score X on benchmark B should be a strong proxy for an agent or entity's
ability to do Z in environment Y.
```

The pipeline should generate benchmark cases and enough supporting metadata to
make that proxy claim inspectable, attackable, and measurable at corpus level.

## Product Frame

The product is not a universal benchmark generator from no context. It cannot
read the user's mind. The honest product is a benchmark construction framework
that can use progressively richer user-provided context.

Minimum user input may be natural language:

```text
Generate a hard benchmark suite for gauging an agent's ability to write haikus.
```

Richer future input may include:

```text
- Target agent description
- System prompt
- Tool definitions
- Environment schemas
- Environment snapshots
- Production transcripts
- Business policies
- Known failures
- Human grading rubrics
- Risk taxonomy
- Existing benchmark items
- Historical success/failure examples
```

The system output is not just a list of prompts. It should be a benchmark suite
with explicit proxy reasoning:

```text
- benchmark cases
- score definitions
- target abilities
- environment assumptions
- diagnostic pressure
- scoring contracts
- leakage risks
- known limits
- corpus coverage metrics
```

The system should be honest about shortcomings. The benchmark suite is a proxy,
not proof of production performance.

## Core Conceptual Objects

### Benchmark Case

A benchmark case is an item that can be presented to an agent or model.

For a haiku benchmark, the case might be a prompt plus constraints. For a future
enterprise-agent benchmark, the case might include a user request, environment
state, available tools, discoverable artifacts, and policy constraints.

Open question: should the first patch represent environment as free-form JSON,
typed domain-specific fields, or both?

### Score X

`score_x` is the observable score or measurement collected from the tested
agent's output on a benchmark case.

Important: `score_x` does not have to be perfectly objective. It can combine
hard checks, judge rubrics, pairwise preferences, human review, or environment
state checks. The important requirement is that the scoring method is explicit.

Examples:

```text
- scalar score from 0.0 to 1.0
- rubric dimension scores
- pass/fail with reason codes
- pairwise preference against a control output
- policy compliance score
- task-completion score
```

Open question: should the POC allow only one score shape, or should it support a
small union of score shapes from the start?

### Ability Z

`ability_z` is the capability the benchmark item is intended to probe.

Examples:

```text
- tasteful constrained haiku writing
- preserving poetic intent under lexical constraints
- safe escalation under ambiguous procurement requests
- using environment records without hallucinating state
- maintaining user intent across multi-step tool use
```

The benchmark is only valuable if success on the case plausibly implies
something about this ability.

### Environment Y

`environment_y` is the setting where the ability matters.

For a haiku POC, this may be a natural-language creative-writing interaction.
For enterprise agents, it may be a tool-rich business environment such as ERP,
CRM, HRIS, Zendesk, claims processing, procurement, or finance ops.

Environment can include:

```text
- user interaction mode
- available tools
- records or documents
- policies
- system instructions
- memory or historical context
- hidden or discoverable state
```

Open question: should the first POC intentionally avoid tool environments, or
should the schema be future-proofed with optional `environment_spec` fields?

### Proxy Claim

The proxy claim states why `score_x` on this benchmark case should correlate
with ability `z` in environment `y`.

Example:

```text
An agent that succeeds on this case likely has some ability to preserve
emotional intent and poetic compression while obeying multiple lexical
constraints. A shallow haiku template is likely to fail because the obvious
imagery is forbidden.
```

The proxy claim is not proof. It is an argument that downstream stages should
attack.

### Diagnostic Pressure

Diagnostic pressure is the part of the benchmark case that forces the tested
agent to demonstrate the target ability rather than coast on shallow competence.

Examples:

```text
- multiple constraints that interact
- tempting but forbidden shortcuts
- ambiguity that must be handled intentionally
- need to preserve style while satisfying content constraints
- distractors in environment state
- policies that conflict with user pressure
```

The Quality Gate should reject cases that are merely plausible but apply weak
diagnostic pressure.

### Scoring Contract

The scoring contract defines how `score_x` is determined.

It should specify:

```text
- what observable behavior earns credit
- what behavior loses credit
- what shallow passes should be penalized
- what partial credit means
- what is explicitly out of scope
- when a judge should mark uncertainty
```

This is not the same thing as hard ground truth. For complex agent domains,
ground truth may be a defensible scoring contract rather than one exact final
answer.

### Score Reliability

Score reliability asks:

```text
Can score X be determined consistently enough to support the benchmark's claim?
```

This should be a first-class validation target. The pipeline should be
adversarial toward scoring methods, not merely toward generated cases.

Score reliability can be supported by:

```text
- clear scoring contract
- positive controls
- negative controls
- pairwise discrimination tests
- calibrated judge comparisons
- adversarial scorer tests
- explicit uncertainty
```

Open question: which score reliability mechanisms belong in the first POC, and
which should only be represented in schema for later work?

### Proxy Validity

Proxy validity asks:

```text
Does score X actually tell us something meaningful about ability Z in
environment Y?
```

This is separate from score reliability. A score can be reliably computed but
still be a bad proxy.

Example:

```text
Counting three haiku lines is reliable, but it is a weak proxy for poetic taste.
```

### Leakage Risk

Leakage risk names ways the case or scoring contract can be passed without the
target ability.

Examples:

```text
- prompt leaks the desired answer
- score rewards formatting instead of capability
- task can be solved by keyword matching
- judge over-rewards verbosity
- model can safely over-escalate every time
- environment permits shortcut actions
- benchmark rewards compliance but ignores quality
```

The system should not pretend leakage can be eliminated. It should surface and
score the risk.

### Corpus-Level Claim

A single benchmark case supports a narrow proxy claim. A benchmark corpus should
signal something broader because it covers different sub-abilities, contexts,
difficulties, failure modes, scoring contracts, and leakage patterns.

The corpus-level claim is:

```text
Corpus A containing benchmark cases B1..Bn is a stronger proxy for ability Z in
environment Y because the cases cover meaningful regions of the capability and
environment space.
```

Open question: what coverage dimensions should be required in the first POC
versus domain-defined?

## Important Non-Goals

The patch should not claim:

```text
- perfect objective ground truth
- universal benchmark generation from zero context
- guaranteed production performance prediction
- fully automated enterprise environment modeling
- fully solved subjective scoring
```

The patch should also avoid centering table-stakes validation as the product.
Schema validity, grammar, punctuation, line counts, JSON shape, compilation, and
linting may be useful checks, but they are not the core value.

The core value is diagnostic benchmark design:

```text
Does this benchmark case strain the claimed capability in a way that makes
success meaningfully predictive?
```

## Proposed Pipeline Interpretation

The top-level pipeline shape can remain mostly intact. The artifact ontology and
agent prompts should change.

### Designer

Current role:

```text
Creates design briefs for Validator training data.
```

Proposed role:

```text
Creates benchmark design briefs that target capability coverage, environment
coverage, diagnostic pressure, scoring reliability, and leakage exploration.
```

The Designer should output design briefs, not benchmark cases.

Design briefs should include at least:

```text
- ability_z or sub-ability
- environment_y or environment slice
- case_type
- difficulty
- scenario
- intended diagnostic pressure
- intended scoring strategy
- known or hypothesized leakage risk
- generation intent
```

Open question: should `case_type` replace `failure_mode` entirely in the first
patch, or should we keep `failure_mode` internally for compatibility and rename
later?

### Design Auditor

Current role:

```text
Checks whether a design matches the domain criteria.
```

Proposed role:

```text
Rejects vague, non-diagnostic, unscorable, irrelevant, or underspecified
benchmark design briefs before generation.
```

The Design Auditor should not rewrite designs. It should emit verdict metadata only.

Potential rejection reasons:

```text
- weak_proxy_claim
- vague_ability
- vague_environment
- weak_diagnostic_pressure
- unscorable_plan
- scoring_contract_missing
- leakage_unaddressed
- coverage_mismatch
- duplicate_design
```

### Generator

Current role:

```text
Generates one Validator training example.
```

Proposed role:

```text
Generates one benchmark case plus the initial proxy/scoring metadata required to
evaluate that case as a benchmark candidate.
```

The Generator should not judge its own work. It can create a proposed scoring
contract and proxy claim, but those proposals must be validated downstream.

Candidate output should include:

```text
- benchmark_case
- score_x
- ability_z
- environment_y
- proxy_claim
- diagnostic_pressure
- scoring_contract
- leakage_risks
- known_limits
- coverage_tags
- proposed_quality_label or expected_admission_verdict
```

Open question: should the Generator propose an expected admission verdict, or
should all generated candidates be treated as "candidate only" with no
self-label?

### Deterministic Validator

Current role:

```text
Checks schema, taxonomy, label consistency, leakage, and triviality for Q&A
validator examples.
```

Proposed role:

```text
Checks table-stakes structural validity for benchmark candidates.
```

It should not be sold as the quality gate. It should only reject things that are
mechanically invalid.

Potential deterministic checks:

```text
- required fields exist
- benchmark_case matches domain schema
- score_x shape is valid
- scoring_contract is present
- scoring dimensions have weights or clear descriptions when required
- ability_z and environment_y are non-empty
- diagnostic_pressure is non-empty
- leakage_risks is non-empty for nontrivial cases
- coverage tags are allowed
- difficulty and scenario are allowed
```

Open question: should deterministic validation enforce that leakage risks are
always present, or only for certain case types/difficulties?

### Semantic Validator / Proxy Critic

Current role:

```text
Judges whether the generated training example label is semantically correct.
```

Proposed role:

```text
Adversarially judges whether the benchmark candidate has defensible score
reliability and proxy validity.
```

This is the important quality gate. It should be borderline hostile to weak
benchmark cases.

It should ask:

```text
- Is score X reliably measurable from the tested agent's output?
- Does score X plausibly proxy ability Z in environment Y?
- Does the case create diagnostic pressure?
- Can a weak agent pass via shallow behavior?
- Is the scoring contract too vague?
- Is the case fake-hard but not actually diagnostic?
- Is success evidence of the intended ability?
- Is failure likely to reflect inability, or irrelevant confusion?
- Are leakage risks honestly surfaced?
- Are known limits stated?
```

Potential rejection reasons:

```text
- weak_proxy_validity
- unreliable_score
- weak_diagnostic_pressure
- shortcut_leakage
- vague_scoring_contract
- fake_difficulty
- irrelevant_environment
- ambiguous_success_criteria
- overbroad_proxy_claim
- missing_known_limits
```

### Curator

Current role:

```text
Commits samples after novelty and duplicate checks.
```

Proposed role:

```text
Commits benchmark cases that improve the corpus-level proxy claim through
coverage, novelty, and risk-balanced suite composition.
```

Potential curation criteria:

```text
- novelty relative to existing cases
- coverage over abilities
- coverage over environments
- coverage over diagnostic pressure types
- coverage over scoring methods
- coverage over leakage risks
- difficulty distribution
- avoidance of near-duplicates
```

Open question: should curation remain deterministic in the first patch, or does
suite-level proxy composition require an LLM judge sooner than expected?

### Router

The Router should remain conceptually unchanged. It owns state transitions and
retry/drop behavior. Agents should not choose routes.

The route code vocabulary may need new subcodes, but the main `accept/reject`
and retry/drop shape can probably remain.

## Proposed Schema Direction

This section sketches possible schema names. It is not final.

### Existing Shape

Current `CandidateSample` is effectively:

```text
- inner_input
- inner_criteria
- inner_verdict
- inner_reason_codes
- difficulty
- failure_mode
- taxonomy cell
```

This is too specific to Validator training examples.

### Proposed Candidate Shape

Possible replacement:

```json
{
  "id": "candidate-1",
  "design_id": "design-1",
  "content_hash": "...",
  "cell": {
    "case_type": "shortcut_leakage",
    "difficulty": 4,
    "scenario": "adversarial"
  },
  "benchmark_case": {
    "prompt": "Write a haiku that makes a layoff feel like late autumn without mentioning work, loss, leaves, cold, or endings.",
    "inputs": {},
    "environment": {}
  },
  "score_x": {
    "score_type": "rubric",
    "range": [0, 1],
    "dimensions": [
      {
        "name": "constraint_adherence",
        "weight": 0.3,
        "description": "Obeys explicit lexical and form constraints."
      },
      {
        "name": "poetic_compression",
        "weight": 0.25,
        "description": "Uses concise imagery rather than explanation."
      }
    ]
  },
  "ability_z": {
    "name": "tasteful constrained poetic generation",
    "sub_abilities": [
      "constraint_following",
      "metaphorical_transfer",
      "non_cliche_imagery"
    ]
  },
  "environment_y": {
    "name": "natural language creative writing interaction",
    "assumptions": [
      "No external tools",
      "Single-turn prompt"
    ]
  },
  "proxy_claim": "Success suggests the model can preserve emotional intent while avoiding obvious lexical shortcuts.",
  "diagnostic_pressure": [
    "forbids obvious imagery",
    "requires emotional transfer",
    "requires compact poetic form"
  ],
  "scoring_contract": {
    "credit": [
      "Preserves the emotional valence without naming the source domain.",
      "Avoids forbidden terms and obvious substitutes."
    ],
    "penalties": [
      "Generic seasonal imagery.",
      "Technically compliant but emotionally flat output."
    ],
    "uncertainty_policy": "Mark uncertainty when aesthetic quality and instruction adherence trade off."
  },
  "leakage_risks": [
    "A model may produce a compliant but lifeless haiku and still receive too much credit."
  ],
  "known_limits": [
    "Aesthetic quality remains judge-dependent."
  ],
  "coverage_tags": [
    "constraint_interaction",
    "anti_cliche",
    "emotional_transfer"
  ],
  "expected_admission": {
    "verdict": "accept",
    "reason_codes": ["proxy_strong"]
  }
}
```

Open question: should `expected_admission` exist? If it remains, it risks
reintroducing the confusing validator-training framing. If it is removed, tests
and validation ledgers need a cleaner way to record ground-truth labels for the
pipeline's own checks.

### Proposed Taxonomy Cell

Current:

```text
failure_mode | difficulty | scenario
```

Possible replacement:

```text
case_type | capability | difficulty | scenario
```

Potential `case_type` values:

```text
- proxy_strong
- score_unreliable
- weak_diagnostic_pressure
- shortcut_leakage
- fake_difficulty
- ambiguous_success_criteria
- low_environment_relevance
- redundant_coverage
```

Open question: should `proxy_strong` be included as a case type, or should case
types describe only targeted risks/failures while accept/reject remains a
separate verdict?

## Domain YAML Direction

The current `domains/qa_item.yaml` is not a good first domain for this product.
The new POC domain should be legible and expressive without requiring external
tools or private enterprise data.

Possible first domain:

```text
benchmark_haiku
```

Reason:

```text
- easy to explain
- not purely objective
- exposes why line counts are table stakes
- supports real diagnostic pressure around taste, constraints, cliche, voice,
  revision, and semantic transfer
- can produce visible benchmark cases without integrations
```

Risks:

```text
- may look unserious to enterprise buyers
- subjective scoring may distract from the general benchmark framework
- could over-index on creative-writing taste
```

Alternative first domains:

```text
- executive email rewriting under constraints
- customer escalation response under policy constraints
- contract clause risk review with evidence/rubric scoring
- procurement decision scenarios with simulated records
```

Open question: should the first POC optimize for conceptual clarity
(`benchmark_haiku`) or enterprise relevance (for example, procurement or support
escalation)?

Potential domain YAML fields:

```yaml
domain_id: benchmark_haiku
dataset_version: poc-2

case_types:
  - proxy_strong
  - score_unreliable
  - weak_diagnostic_pressure
  - shortcut_leakage
  - fake_difficulty
  - ambiguous_success_criteria
  - low_environment_relevance
  - redundant_coverage

difficulties: [1, 2, 3, 4, 5]
scenarios: [nominal, edge, adversarial]

abilities:
  - constrained_poetic_generation
  - metaphorical_transfer
  - non_cliche_imagery
  - tone_control
  - revision_sensitivity
  - semantic_precision

environments:
  - single_turn_creative_writing
  - revision_interaction

diagnostic_pressure_types:
  - forbidden_obvious_imagery
  - interacting_constraints
  - emotional_indirection
  - style_without_explicit_style_words
  - anti_template

scoring_methods:
  - rubric
  - pairwise_preference
  - hard_checks_plus_rubric

reason_codes:
  - proxy_strong
  - weak_proxy_validity
  - unreliable_score
  - weak_diagnostic_pressure
  - shortcut_leakage
  - vague_scoring_contract
  - fake_difficulty
  - irrelevant_environment
  - ambiguous_success_criteria
  - overbroad_proxy_claim
  - missing_known_limits

benchmark_case_schema:
  ...
```

Open question: should `DomainConfig` remain generic enough to load arbitrary
domain YAML fields, or should it explicitly model benchmark-specific fields?

## Score Reliability Strategy

The POC should not assert that scores are perfectly accurate. It should assert
something narrower and more defensible:

```text
The score is meaningful under an explicit scoring contract, and the scoring
contract has been stress-tested against known controls and leakage risks.
```

Possible reliability mechanisms:

### Negative Controls

Known-bad outputs that should score poorly.

Examples:

```text
- technically compliant but generic output
- output that mirrors keywords but misses intent
- output that violates an explicit constraint
- output that is verbose and judge-pleasing but capability-empty
```

Why useful:

```text
It is often easier to assert that known-bad outputs should fail than to prove
that a single output is excellent.
```

Open question: should each generated benchmark case include negative controls in
the first POC?

### Positive Controls

Known-strong outputs that should score well.

Why useful:

```text
They calibrate scoring expectations, but they may be expensive or subjective.
```

Open question: should positive controls be required, optional, or omitted until
there is human review?

### Pairwise Discrimination

Instead of relying only on absolute scores, ask whether the scorer prefers a
stronger output over a weaker controlled output.

Potential metric:

```text
Across controlled pairs where one output contains a known failure and the other
does not, the scorer selects the stronger output N% of the time.
```

Open question: should pairwise controls be part of this POC or only the future
scoring-evaluator layer?

### Adversarial Scorer Testing

Generate outputs designed to fool the scorer:

```text
- rubric-mirroring
- keyword stuffing
- confident unsupported claims
- format-perfect but empty output
- excessive caution
- evasive safety-sounding response
```

Open question: is adversarial scorer testing in scope before the benchmark
generator itself is stable?

## Corpus-Level Metrics

The existing `analyze.py` reports route counts, stage counts, validation pass
rates, and taxonomy coverage entropy. The new benchmark corpus should report
metrics closer to the proxy thesis.

Potential metrics:

```text
- committed_count
- rejection_reason_distribution
- ability_coverage
- environment_coverage
- case_type_distribution
- difficulty_distribution
- diagnostic_pressure_distribution
- scoring_method_distribution
- score_reliability_distribution
- proxy_validity_distribution
- leakage_risk_distribution
- coverage_entropy
- near_duplicate_rate
- average_novelty_distance
```

Important: these metrics are themselves proxies. The report should avoid saying
they prove production performance.

Possible report language:

```text
This suite covers 6 target sub-abilities across 4 diagnostic pressure types.
It has high coverage of constraint-interaction and anti-template probes, but low
coverage of revision sensitivity. Scoring reliability is weakest for aesthetic
originality and strongest for explicit constraint adherence.
```

Open question: where should confidence labels come from: generator proposal,
semantic validator verdict, deterministic scoring rules, or separate scorer
calibration?

## Code Patch Surfaces

### `models.py`

Likely changes:

```text
- Introduce benchmark-specific models, or generalize existing models.
- Replace or supplement TaxonomyCell.failure_mode.
- Replace CandidateSample.inner_input/inner_criteria/inner_verdict fields.
- Add score/proxy/scoring/leakage fields.
- Keep StageRecord, RoutingDecision, Verdict, RouteCode mostly intact.
```

Open question: should this be a breaking schema migration, or a compatibility
layer that supports both old QA validator samples and new benchmark cases?

### `config.py`

Likely changes:

```text
- DomainConfig needs benchmark-specific fields.
- Existing required `failure_modes` may need to become `case_types`.
- Existing required `inner_input_schema` may need to become
  `benchmark_case_schema`.
```

Open question: should `DomainConfig` allow additional fields so domains can
experiment faster?

### `domains/*.yaml`

Likely changes:

```text
- Add new benchmark domain YAML.
- Either retire `qa_item.yaml` from the default demo or keep it as legacy.
```

Open question: should the CLI default still point to `domains/qa_item.yaml`, or
should docs and commands switch immediately?

### `agents.py`

Likely changes:

```text
- Designer prompt: benchmark design briefs for X/Z/Y proxy coverage.
- DesignAuditor prompt: reject non-diagnostic or unscorable designs.
- SampleGenerator prompt: benchmark cases plus proxy/scoring metadata.
- SemanticValidator prompt: adversarial proxy validity and score reliability
  critic.
- Parsing logic: read new payload fields instead of question/claimed_answer.
```

Open question: should SemanticValidator be renamed in code to ProxyCritic, or
should names remain stable until after the POC runs?

### `rules.py`

Likely changes:

```text
- Remove QA-specific deterministic label checks.
- Add schema/taxonomy/contract checks for benchmark candidates.
- Keep deterministic validation intentionally humble.
```

Open question: which checks should remain deterministic versus LLM-backed?

### `pipeline.py`

Likely changes:

```text
- Mostly preserve graph shape.
- Update progress labels that print `mode=design.cell.failure_mode`.
- Update commit/coverage calls if TaxonomyCell changes.
- Possibly rename stage roles in logs for readability.
```

Open question: should we introduce a distinct Scoring Contract Builder stage
now, or keep scoring-contract generation inside Generator and validation inside
SemanticValidator for the first patch?

### `services/coverage_ledger.py`

Likely changes:

```text
- Coverage key may need to use case_type/capability/scenario instead of
  failure_mode/difficulty/scenario.
```

Open question: should coverage use one canonical taxonomy cell, or multiple
coverage projections?

### `services/corpus_index.py`

Likely changes:

```text
- Embedding text should be built from benchmark case/proxy fields instead of
  inner_input/criteria.
- Novelty checks should probably consider benchmark prompt plus diagnostic
  pressure and ability tags.
```

Open question: what text should represent a benchmark candidate for novelty?

### `analyze.py`

Likely changes:

```text
- Read benchmark corpus path.
- Report coverage over abilities, environments, diagnostic pressure, leakage
  risk, scoring methods, and difficulty.
```

Open question: should metrics remain one JSON file, or should there be both
machine metrics and a human-readable report?

### Tests

Likely changes:

```text
- Update schema tests for benchmark candidate round trips.
- Update deterministic rule tests.
- Update smoke pipeline tests to expect benchmark cases.
- Add tests for rejecting weak or missing scoring contracts.
```

Open question: should tests keep legacy QA examples around to prove backwards
compatibility?

### Docs

Likely changes:

```text
- README.md should no longer describe the POC as Validator training data.
- PLAN.md should be updated or superseded.
- PIPELINE_STATE_MACHINE.md can mostly remain but stage descriptions should
  change.
- Add an artifact reference doc for benchmark cases.
```

Open question: should this patch update existing docs directly or add the new
benchmark plan first, then rewrite docs after code lands?

## Suggested Implementation Phases

### Phase 1: Add Benchmark Schema Beside Current Schema

Goal:

```text
Make benchmark candidate objects representable without breaking the current
pipeline immediately.
```

Work:

```text
- Add benchmark models in `models.py`.
- Add new domain YAML.
- Add tests for benchmark model round trip.
- Do not change pipeline behavior yet.
```

Reason:

```text
This creates a concrete artifact target and flushes out schema ambiguity before
prompt changes.
```

### Phase 2: Switch Generation To Benchmark Candidates

Goal:

```text
Have the live pipeline generate benchmark candidates instead of validator QA
examples.
```

Work:

```text
- Update Designer prompt.
- Update DesignAuditor prompt.
- Update SampleGenerator prompt and parsing.
- Update deterministic validation.
- Update SemanticValidator prompt.
- Update tests.
```

Risk:

```text
This is the main breaking change.
```

### Phase 3: Update Curation And Analysis

Goal:

```text
Make the committed corpus and metrics explain benchmark-suite coverage.
```

Work:

```text
- Update coverage keying.
- Update corpus embedding text.
- Update analyze.py metrics.
- Add human-readable summary fields where useful.
```

Risk:

```text
Metrics may become superficial if confidence fields are not well-defined.
```

### Phase 4: Documentation And Demo

Goal:

```text
Make the repo legible to someone inspecting the output.
```

Work:

```text
- Update README.
- Update PLAN or replace it with a new POC plan.
- Update state-machine docs.
- Add sample JSONL excerpt and explanation.
```

Risk:

```text
Docs can overclaim. Keep language honest and proxy-based.
```

## Decision Points Before Coding

These questions should be answered before making large code changes:

1. What should the first demo domain be?

   Options:

   ```text
   - haiku benchmark
   - executive writing benchmark
   - support escalation benchmark
   - procurement/policy benchmark
   - other
   ```

2. Should we preserve backwards compatibility with `qa_item`, or make a clean
   breaking pivot?

3. Should `failure_mode` be renamed in the first patch, or shimmed as
   `case_type` later?

4. Should a benchmark candidate include `expected_admission`, or should only
   validators assign admission verdicts?

5. Should scoring controls be required in POC 1 of the benchmark pivot?

   Possible choices:

   ```text
   - no controls yet
   - negative controls only
   - positive and negative controls
   - pairwise controls
   ```

6. Should score reliability and proxy validity be scalar scores, categorical
   labels, reason-code verdicts, or all of the above?

7. Should the semantic validation stage be split into separate score reliability
   and proxy validity critics, or kept as one adversarial quality gate for now?

8. Should curation remain deterministic, or does suite-level proxy composition
   require an LLM-backed curator?

9. What is the minimum corpus-level report that would make a generated suite
   feel inspectable rather than hand-wavy?

10. What is the success criterion for the POC run?

    Possible examples:

    ```text
    - Generates N committed benchmark cases.
    - Emits a coverage report over ability/environment/diagnostic-pressure axes.
    - Shows rejected cases with meaningful adversarial rejection reasons.
    - Produces output that a human can inspect and agree is not generic filler.
    ```

## Proposed Near-Term Default Choices

These are not final decisions. They are suggested defaults if we want to move
quickly while avoiding large assumptions.

```text
Demo domain:
  benchmark_haiku

Compatibility:
  breaking pivot is acceptable if docs are updated clearly

Taxonomy:
  rename failure_mode to case_type in model/domain concepts

Validation stages:
  keep current graph shape
  make SemanticValidator the adversarial proxy/score critic

Scoring controls:
  require at least one negative control per case if feasible
  make positive controls optional

Score reliability:
  categorical label plus reason codes first
  scalar confidence later

Corpus metrics:
  ability coverage, diagnostic pressure distribution, leakage risk distribution,
  difficulty distribution, rejection reason distribution
```

These defaults should be revisited before implementation.

## Summary

The patch should pivot the POC from:

```text
Generate Validator training examples.
```

to:

```text
Generate benchmark cases where score X is an inspectable proxy for ability Z in
environment Y.
```

The value is not objectivity for its own sake. The value is diagnostic force.

The system should generate benchmark cases, propose scoring contracts, state
proxy claims, identify leakage risks, and then adversarially validate whether
those cases deserve to enter a benchmark corpus.

The most important quality gate is not punctuation, grammar, schema validity, or
format compliance. Those are table stakes. The important gate is whether the
case and scoring contract would expose real ability rather than reward shallow
success.
