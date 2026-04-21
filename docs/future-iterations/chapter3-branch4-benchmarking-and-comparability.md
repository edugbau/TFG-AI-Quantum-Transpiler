# Chapter 3 Branch 4: Benchmarking And Comparability

## Deferred focus

This branch is reserved for making scenario comparison more uniform across `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL`.

## Why it comes after branch 1

The current branch-1 implementation keeps the public v1 limitation intact: RL scenarios return episode summaries, not final circuits. That means comparability between Qiskit-facing scenarios and RL-facing scenarios is still intentionally incomplete. A benchmarking-focused branch should come later, after the contract and RL episode surface are stable enough to compare consistently.

## Expected direction

- tighten reporting conventions across scenario families;
- document which metrics are directly comparable and which remain scenario-local;
- prepare the ground for a future iteration that may reconstruct final circuits for RL-based runs.

## Why it is deferred here

Branch 1 should not overstate comparability that does not yet exist. The current worktree keeps the contract honest: routing-only v1 orchestration, episode summaries for RL scenarios, and explicit notes when legacy metadata fallback is used.
