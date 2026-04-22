---
name: mo-rl-pipeline
description: Use when modifying src/integration handoff logic, benchmark scenarios, layout ownership rules, or routing evaluation flows that connect MO outputs to RL inputs.
compatibility: opencode
metadata:
  module: integration
  scope: project
---

# MO RL Pipeline

## Overview

Keep MO -> RL orchestration inside `src/integration/` so layout handoff, benchmark scenarios, and routing evaluation remain explicit, testable, and decoupled from the internals of `mo_module` and `rl_module`.

## When to Use

- Editing `src/integration/` orchestration or benchmark scenarios.
- Touching the `initial_layout` handoff contract.
- Comparing `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL` scenarios.

## Quick Reference

- `src/integration/` owns MO -> RL orchestration.
- `mo_module` produces layouts; it does not call `rl_module` directly.
- `rl_module` may consume `initial_layout`, but it does not own the handoff protocol.
- Keep the shared layout convention explicit in docs and code.

## Implementation

- Accept layouts through caller-controlled interfaces such as `env.reset(options={"initial_layout": layout})`.
- Preserve scenario naming and comparability across `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL`.
- Document whether a layout is `logical_qubit -> physical_qubit` or the inverse; do not leave the mapping ambiguous.

## Common Mistakes

- Moving orchestration logic into `mo_module` or `rl_module`.
- Referring to direct MO -> RL imports or ownership across module boundaries.
- Treating benchmark scenarios as interchangeable when they test different ownership paths.

## Project References

- `src/integration/docs/internal_documentation.md`
- `src/integration/README.md`
- `src/rl_module/docs/synthesis_mode_status.md`
- `src/mo_module/docs/internal_documentation.md`
