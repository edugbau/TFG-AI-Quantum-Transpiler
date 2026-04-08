# Integration v1

`src/integration/` owns the routing evaluation scenarios that connect the project modules at integration level.

## Current v1 scope

Integration v1 covers these routing evaluation scenarios:

- `Baseline`
- `MO_Only`
- `RL_Only`
- `MO+RL`

The current implementation focuses on scenario orchestration and shared contracts for routing evaluation.

## Known RL limitation

In this v1 scope, RL-based scenarios return episode summaries, not final circuits.

That means `RL_Only` and `MO+RL` currently report routing-level outcomes such as layouts, rewards, step counts, and swap counts, but they do not reconstruct or export a final routed circuit artifact yet.

Reconstructing or exporting final circuits from RL is left for a future iteration.

## Deferred work

- Final circuit reconstruction/export for RL-based runs is deferred to a future iteration.
- QASM input is also deferred to a future iteration.
