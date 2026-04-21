# Chapter 3 Branch 5: Target-Aware Cost Signals

## Deferred focus

This branch is reserved for reward or cost shaping that reflects backend-aware routing targets more directly during RL training and evaluation.

## Why it comes last

Target-aware cost signals should build on top of stable checkpoint metadata, a settled routing episode interface, and clearer benchmarking conventions. Branches 1 through 4 establish those prerequisites in order.

## Expected direction

- introduce richer cost signals tied to backend-aware routing objectives;
- evaluate whether those signals improve routing outcomes without obscuring the public evaluation contract;
- keep orchestration ownership in `src/integration/` rather than creating direct cross-module coupling.

## Why it is deferred here

Branch 1 is intentionally narrower: persisted RL metadata, contract reuse in `integration`, and fallback behavior for older checkpoints. Reward redesign belongs later, once the evaluation and comparison layers are less volatile.
