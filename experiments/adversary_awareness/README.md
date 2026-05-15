# Generator Adversary Awareness Experiment

This is the AgentLab scaffold for the v1 experiment:

- 10 `GenerationEnvelope` task rows
- 4 variants:
  - `generator_minimal`
  - `generator_minimal_adversary`
  - `generator_minimal_rubric`
  - `generator_minimal_rubric_adversary`
- 3 replications per seed
- synth pipeline runs as the agent runtime

`tasks.jsonl` currently contains 5 manually-authored Python/pytest code-debug seeds and 5
haiku/text seeds imported from `logs/e2e-seed-42-20260514-codex`.
Each task row carries `generation_envelope.domain_ref`, so a single AgentLab experiment can mix
seed domains; the CLI `--domain` value in `experiment.yaml` is only a default for older rows or
preflight.
Rubric-aware variants use `{{DOMAIN_RUBRIC_CONTEXT}}`; the AgentLab entrypoint expands that token
from the resolved domain YAML's `quality_gate_rules` and `rubric_gate_rules` for each task row.
For this POC, each task row uses `synth-pipeline-agent:local` as its image so the task-carried runtime has Python, pytest, and the synth pipeline dependencies available.

## Build the Agent Image

```bash
docker build \
  -f /Users/jevinnishioka/Desktop/synth-data-pipeline-agents/experiments/adversary_awareness/Dockerfile.agent \
  -t synth-pipeline-agent:local \
  /Users/jevinnishioka/Desktop/synth-data-pipeline-agents
```

## Build and Run

You can run this from any directory. The `lab` alias is just a shell convenience:

```bash
alias lab=/Users/jevinnishioka/Desktop/Experiments/rust/target/release/lab-cli

BUILD_DIR=/private/tmp/synth-generator-adversary-awareness-5x3

lab build-run \
  /Users/jevinnishioka/Desktop/synth-data-pipeline-agents/experiments/adversary_awareness/experiment.yaml \
  --out "$BUILD_DIR" \
  --secret-file codex_oauth="$HOME/.codex/auth.json" \
  --materialize full \
  --json
```

Use a fresh `BUILD_DIR` if that directory already exists; AgentLab build output directories must be empty.

For debugging a new package, split the same flow into `lab build`, `lab preflight`, and `lab run`.

The agent runtime writes full synth traces under the `synth-pipeline` output mount.
