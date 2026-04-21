# Chapter 3 Branch 3: Generalization And Curriculum

## Deferred focus

This branch is reserved for training breadth: curriculum structure, distribution shifts across routing tasks, and evaluation setups that measure whether saved policies generalize beyond the narrowest training slice.

## Why it comes after branch 2

Generalization work is easier to reason about after the routing episode interface is more mature. Branch 2 is the natural place to settle richer observations and feasible-action handling first; branch 3 can then evaluate training strategy on top of that more stable RL surface.

## Expected direction

- define curriculum stages for routing difficulty;
- document train/eval splits that avoid over-reading a single benchmark slice;
- compare policy behavior across backend or circuit variations while still consuming checkpoints through `integration`.

## Why it is deferred here

Branch 1 is only about persisted metadata and evaluation-contract reuse. Mixing curriculum or generalization changes into the current worktree would broaden scope beyond the implemented routing-only v1 integration contract.
