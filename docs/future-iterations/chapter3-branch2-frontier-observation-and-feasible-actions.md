# Chapter 3 Branch 2: Frontier Observation And Feasible Actions

## Deferred focus

This branch is reserved for improving the routing episode interface inside `rl_module`, especially richer frontier-aware observations and action filtering that better matches feasible routing choices.

## Why it comes after branch 1

Branch 1 first stabilizes how `integration` evaluates saved routing checkpoints through persisted metadata and legacy fallback behavior. Only after that contract is stable does it make sense to widen the observation or action surface used during routing episodes.

## Expected direction

- expand the routing observation state with more frontier context;
- reduce invalid or clearly infeasible routing choices through feasible-action guidance;
- keep the public scenario ownership unchanged, with `integration` still owning `RL_Only` and `MO+RL` orchestration.

## Why it is deferred here

This work would change how routing episodes are represented and consumed during evaluation, so it should not be mixed into the branch-1 contract work. Branch 1 already defines the persisted train/eval contract; branch 2 can build on that without reopening sidecar semantics or fallback wording.
