# Probe Principles

This pipeline treats benchmarks as engineered encounters, not mirrors of ability.
A benchmark case is useful when performance on it leaves defensible evidence about
an entity's ability in a target environment.

Generated cases start as plausible candidates, not accepted benchmarks. The
validation gates should critique them enough to decide whether the evidence
promotes them into useful benchmark proxies under stated assumptions and limits.

The core claim remains:

```text
Score X on benchmark B should be a strong proxy for ability Z in environment Y.
```

Hard checks are allowed, but they are usually floor checks. They can show that a
system failed table stakes. They should not be mistaken for evidence of high
ability.

## General Probe Principles

These principles are domain-agnostic lenses. They are not a complete theory of
excellence and should not become a checklist that benchmarks overfit.

### Meaningful Constraint Pressure

The case should force choices where optimizing one desirable behavior can damage
another desirable behavior. Success should reveal judgment or capability rather
than checklist compliance.

Test question:

```text
What must the model balance, and what shallow single-axis strategy would fail?
```

### Informative Success

A strong performance should increase our belief that the entity has the target
ability by a nontrivial amount.

Test question:

```text
If a weak but instruction-following model passed this case, would we be surprised?
```

### Shallow Strategy Resistance

The case should make obvious shortcuts fail or score poorly. Common shortcuts
include keyword stuffing, format-only compliance, forbidden-token avoidance
without positive evidence, cliche substitution, generic safe output, and
checklist overfitting.

### Critical Common Sense

The written criteria are not an exhaustive loophole list. A competent judge
should reject obvious fake compliance, bad-faith literalism, and shallow gaming
even when the exact trick is not named, while staying fair to genuinely strong
unanticipated solutions.

Test question:

```text
Could a weak model get a high score by exploiting wording, swapping a few tokens,
adding decorative mentions, or satisfying the checklist while plainly missing
the intended capability?
```

Specificity is still required, but it cannot replace common sense. If the rubric
would reward an output that obviously games the benchmark intent, the rubric is
wrong.

### Multiple Strong Paths

The case should allow more than one valid high-quality strategy. It should not
require the tested entity to guess one hidden preferred answer style.

Test question:

```text
Could two very different excellent outputs both score well for defensible reasons?
```

### Process Revelation

Where possible, the task or final artifact should expose evidence about how the
entity balances constraints, chooses, revises, escalates, or reasons. In creative
domains, process may only be inferred from the artifact.

### Ceiling Signal

The case should include at least one scoring signal that can distinguish
competent from excellent performance, not only bad from acceptable performance.

Test question:

```text
What would an excellent output do that an adequate output would not?
```

### Explicit Limits

The case must state what success does not prove and which capability regions
remain uncovered.

## Anti-Overfit Policy

Domain traits are lenses, not complete definitions of excellence.

The pipeline should avoid turning benchmarks into games where the model wins by
satisfying a visible checklist. That matters most in subjective and creative
domains, where excellence can be emergent and hard to reduce to fixed traits.

The active anti-overfit policy is:

```text
- Written criteria are not exhaustive; reject obvious gaming, fake compliance,
  and bad-faith literalism.
- Do not reward checklist satisfaction alone.
- Separate table-stakes disqualifiers from ceiling-level ability signals.
- Prefer probes that allow unexpected excellence outside named traits.
- Do not turn creative domains into mechanical compliance games unless the
  claimed ability is compliance under creative constraints.
- Hard checks may invalidate outputs, but should not be the main evidence of
  high ability.
- Report known limits and uncovered capability regions.
- Reject rubrics that would unfairly punish a clearly excellent but
  unanticipated solution.
```

## Future Work

We have not yet implemented score auditing against real or synthetic outputs.

That future stage would take:

```text
benchmark case
scoring contract
real or synthetic output
assigned score
```

and ask whether the score is defensible. This is useful because rubrics often
sound good abstractly but fail when applied to real outputs.
