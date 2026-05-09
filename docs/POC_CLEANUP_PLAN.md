# POC Cleanup Plan

This repo is a POC, not a published library. Prefer one readable path over
compatibility layers, versioned artifacts, fallback schemas, and unused future
surface area. Delete scaffolding for users, workflows, and data shapes that do
not exist yet.

## 1. Collapse `reason_codes` and `subcodes` - Done

Keep `subcodes`; delete `reason_codes`.

Completed edits:
- Remove `reason_codes` from `DesignVerdict`, `SampleVerdict`,
  `RoutingDecision`, and `StageRecord` in `models.py`.
- Remove `reason_codes` from both domain YAML files.
- Update `router.route_after()` to accept and pass only `subcodes`.
- Update pipeline stage recording in `pipeline.py` to write only `subcodes`.
- Replace fallback reads such as `reason_codes or subcodes` in `run_report.py`,
  `pipeline.py`, and `agents.py` with direct `subcodes`.
- Update semantic gate parsing in `agents.py` so LLM gate output only asks for
  and consumes `subcodes`.
- Update tests that assert or construct `reason_codes`.

Files to inspect:
- `models.py`
- `router.py`
- `pipeline.py`
- `agents.py`
- `run_report.py`
- `domains/benchmark_haiku.yaml`
- `domains/benchmark_code_debug.yaml`
- `tests/`

Verification run:
- `rg -n "reason_codes" models.py config.py router.py pipeline.py agents.py rules.py run_report.py cli_graph.py services tests domains`
- `.venv/bin/python -m pytest`

## 2. Delete legacy validator/reporting paths - Done

`run_report.py` still supports old validator-shaped artifacts. This creates
mixed-schema detection and fallback rendering that are not part of the current
benchmark pipeline.

Completed edits:
- Delete `_print_schema_warning()`.
- Delete `_artifact_schema()`.
- Delete `_stage_is_current()`.
- Delete `_design_from_candidate_id()`.
- In `_prompt()`, remove the `inner_input.question` fallback. Only read
  `benchmark_case.prompt`.
- In `_case_type()`, remove `cell.failure_mode` fallback. Only use
  `candidate.case_type` or `candidate.cell.case_type`.
- Rewrite `tests/test_run_report.py` around current benchmark-only report
  behavior.

Files to inspect:
- `run_report.py`
- `tests/test_run_report.py`

Verification run:
- `rg -n "legacy|inner_input|failure_mode|validator_legacy|_stage_is_current|_artifact_schema" run_report.py tests`
- `.venv/bin/python -m pytest`

## 3. Remove dataset versioning - Done

`dataset_version` is unnecessary namespacing for this POC. It creates divergent
paths and implies migrations/releases that do not exist.

Completed edits:
- Remove `dataset_version` from `DomainConfig` in `config.py`.
- Remove `dataset_version: poc-2` from both domain YAML files.
- Change corpus index path from:
  `data/index/<domain_id>/<dataset_version>/embeddings.jsonl`
  to:
  `data/index/<domain_id>/embeddings.jsonl`
- Change coverage path from:
  `data/coverage/<domain_id>/<dataset_version>/coverage.json`
  to:
  `data/coverage/<domain_id>/coverage.json`
- No tests or fixtures expected the old path shape.

Files to inspect:
- `config.py`
- `services/corpus_index.py`
- `services/coverage_ledger.py`
- `domains/benchmark_haiku.yaml`
- `domains/benchmark_code_debug.yaml`
- `tests/`

Verification run:
- `rg -n "dataset_version|poc-[0-9]" config.py services models.py cli_graph.py domains agents.py tests`
- `.venv/bin/python -m pytest`

## 4. Delete unused model and enum surface - Done

Several types and enum values are old design leftovers or future-facing
scaffolding. If they are not exercised by the current graph, remove them.

Completed edits:
- Delete `InnerInput`.
- Delete `InnerCriteria`.
- Delete `StageResult`.
- Delete `AgentRole.SEMANTIC_VALIDATOR`.
- Delete `RouteCode.DROP_TIMEOUT`.
- Delete `RouteCode.DROP_POLICY_CEILING`.
- Delete `ContextPolicy.ROUTE_CODE_ONLY`.
- Delete matching entries from the domain YAML `route_codes`, if present.
- Update `cli_graph.py` labels if they reference removed roles.

Files to inspect:
- `models.py`
- `cli_graph.py`
- `domains/benchmark_haiku.yaml`
- `domains/benchmark_code_debug.yaml`
- `tests/`

Verification run:
- `rg -n "InnerInput|InnerCriteria|StageResult|SEMANTIC_VALIDATOR|DROP_TIMEOUT|DROP_POLICY_CEILING|ROUTE_CODE_ONLY" config.py services models.py cli_graph.py domains agents.py tests`
- `.venv/bin/python -m pytest`

## 5. Replace fallback JSON shapes with one schema contract - Done

`fallback_required_json_shape` duplicates schema information and now has
domain-specific mutation. That makes generation prompts harder to read and easy
to drift.

Completed edits:
- In `SampleGenerator.generate()`, remove `fallback_required_json_shape` or
  rename it to a small `example_output` that is explicitly an example, not a
  fallback contract.
- `SampleGenerator.revise_from_attack()` already uses `required_revision_patch_shape`
  and did not have a complete-candidate fallback shape.
- If examples are retained, build them with one helper function:
  `_example_output_for_domain(domain)`.
- Keep code-debug-specific `environment_artifact` in that helper only.
- Avoid mutating payloads after construction with
  `pop("environment_artifact", None)`.

Files to inspect:
- `agents.py`
- `tests/test_agents.py`
- `tests/test_pipeline_smoke.py`

Verification run:
- `rg -n "fallback_required_json_shape|pop\\(\"environment_artifact\"" agents.py tests`
- `.venv/bin/python -m pytest`

## 6. Simplify schema loading

`output_schema_path` plus hydrated `output_schema` is an indirection that makes
sense for a configurable product, not necessarily this POC.

Choose one direction:
- Either inline `output_schema` directly in each domain YAML.
- Or hardcode the shared schema path in `load_domain()` and remove
  `output_schema_path` from the domain contract.

Preferred POC choice:
- Hardcode `domains/schemas/benchmark_output.schema.json` as the benchmark
  output schema.
- Remove `output_schema_path` from `DomainConfig`.
- Remove `output_schema_path` from both domain YAML files.
- Keep `benchmark_case_schema` in the domain YAML because that differs by
  domain.

Files to inspect:
- `config.py`
- `domains/benchmark_haiku.yaml`
- `domains/benchmark_code_debug.yaml`
- `tests/test_config.py`

Verification:
- `rg -n "output_schema_path" . --glob '!data/**' --glob '!logs/**'`
- `.venv/bin/python -m pytest`

## 7. Re-evaluate provider/base-url abstraction

The code has `provider`, `base_url`, and `OpenAIClient`, but no real provider
interface beyond OpenAI-compatible HTTP. This is potentially useful for local
OpenAI-compatible servers, but the naming suggests more abstraction than exists.

Specific options:
- Keep `base_url` if OpenAI-compatible endpoints matter.
- Remove or demote `provider` to metadata if it does not change behavior.
- Consider renaming `OpenAIClient` to `OpenAICompatibleClient` if `base_url`
  remains.
- Remove CLI/env overrides that are not used in the actual demo workflow.

Files to inspect:
- `config.py`
- `agents.py`
- `main.py`
- `sample_outputs.py`
- `.env.example`
- `README.md`

Verification:
- `rg -n "provider|base_url|OPENAI_PROVIDER|OPENAI_BASE_URL" . --glob '!data/**' --glob '!logs/**'`
- `.venv/bin/python -m pytest`

## 8. Fix or delete virtual workspace editing methods

`VirtualWorkspace.write_file()` and `VirtualWorkspace.delete_file()` look like
tool implementations, but there is no tool registry or tool-call execution path.
Keeping them as methods without exposing them makes the environment artifact
story misleading.

Choose one direction:
- If tool use is in scope now, add an explicit tool interface that calls these
  methods and tests that an evaluated agent can use it.
- If tool use is not in scope now, delete the methods and keep
  `VirtualWorkspace` as a materialization/read helper only.

Files to inspect:
- `services/virtual_workspace.py`
- `sample_outputs.py`
- `agents.py`
- any future tool execution module, if added

Verification if deleting:
- `rg -n "write_file|delete_file" . --glob '!data/**' --glob '!logs/**'`
- `.venv/bin/python -m pytest`

Verification if implementing:
- Add tests that exercise the actual tool path, not just direct method calls.
- Confirm generated sample outputs can invoke the tools instead of receiving a
  flattened prompt only.

## 9. Audit Python version assumptions

The model import now works under Python 3.9, but the bundle script still enforces
Python 3.10+ and the repo uses PEP 604 annotations in non-model code.

Specific edits:
- Decide whether Python 3.9 is supported for the whole repo or just model import.
- If supporting Python 3.9, remove or rewrite runtime-evaluated PEP 604
  annotations as needed.
- Audit `requirements.txt`; it currently includes `eval_type_backport`, which
  may be unnecessary after removing runtime-sensitive Pydantic union syntax.
- Update `scripts/build_agent_bundle.sh` if 3.10 is no longer required by pinned
  dependencies.
- Update `README.md` to state the actual supported Python version.

Files to inspect:
- `requirements.txt`
- `scripts/build_agent_bundle.sh`
- `README.md`
- `models.py`
- `pipeline.py`
- `agents.py`
- `services/`

Verification:
- `python3 - <<'PY'\nimport models\nprint("models ok")\nPY`
- `pytest tests/test_rules.py tests/test_corpus_index.py tests/test_sample_outputs.py tests/test_virtual_workspace.py tests/test_schemas.py`
- `.venv/bin/python -m pytest`

## 10. Tighten data-shape assumptions in scripts

Analysis/report/output scripts currently use many defensive `.get(..., {})` and
`or []` paths. Some of that is ordinary script resilience, but too much of it
hides bad artifacts.

Specific edits:
- In `sample_outputs.py`, require committed corpus rows to contain a dict
  `candidate`.
- Require `candidate.agent_artifact.benchmark_case.prompt`; fail loudly if missing.
- In `analyze.py`, decide whether missing committed fields should be counted as
  `unknown` or treated as bad corpus data. Prefer failing loudly for the POC.
- In `run_report.py`, stop rendering partial reports for malformed current
  artifacts unless that is truly useful during debugging.

Files to inspect:
- `sample_outputs.py`
- `analyze.py`
- `run_report.py`

Verification:
- Add one negative test per script for malformed current artifacts.
- `.venv/bin/python -m pytest`
