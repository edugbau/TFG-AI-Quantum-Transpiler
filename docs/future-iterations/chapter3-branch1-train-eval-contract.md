# Chapter 3 Branch 1: Train/Eval Contract

## Focus

Branch 1 defines the routing-evaluation contract shared between persisted RL checkpoints and `src/integration/`.

The implemented direction in this worktree is:

- `integration` remains routing-only v1 orchestration for `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL`.
- RL scenarios still return episode summaries, not final circuits.
- RL checkpoints may carry a neighboring `run_metadata.json` sidecar that persists the routing evaluation contract.
- `integration` reuses that persisted contract when present and falls back to legacy defaults when the sidecar is missing.
- The fallback is surfaced publicly through `ScenarioResult.notes` so old checkpoints remain evaluable without hiding the compatibility path.

## Why this branch comes first

Later Chapter 3 work depends on a stable answer to a simple question: how does `integration` evaluate a saved routing model without coupling directly to RL internals?

Branch 1 answers that by keeping the contract persisted next to the checkpoint and consumed only from `src/integration/`. That preserves the module boundary:

- `rl_module` owns training artifacts and emits metadata describing the routing contract.
- `integration` owns MO-to-RL orchestration and decides how `RL_Only` and `MO+RL` consume saved checkpoints.
- No direct MO/RL communication is introduced outside `integration`.

## Implemented scope

This branch is about contract reuse, not new RL capabilities. The implemented scope is limited to:

- persisted routing metadata sidecar support;
- evaluation-time contract resolution in `integration`;
- explicit fallback behavior for legacy checkpoints;
- documentation and tests that keep the public v1 scope clear.

It does not add:

- RL training orchestration inside `integration`;
- final circuit reconstruction for RL scenarios;
- direct MO/RL coupling outside `src/integration/`;
- new benchmarking layers beyond the current routing-evaluation v1 surface.

## Completion criteria

Branch 1 is complete when all of the following are true:

1. A saved routing checkpoint can be evaluated by `integration` using persisted metadata from `run_metadata.json` when the sidecar is available.
2. `RL_Only` and `MO+RL` keep returning routing episode summaries rather than final circuits.
3. Missing sidecars trigger legacy defaults instead of hard failure.
4. That fallback is reported in `ScenarioResult.notes` with public, test-covered wording.
5. Repository docs consistently describe `integration` as the owner of MO-to-RL orchestration and the only cross-module consumer of the persisted routing contract.

## Relationship to later branches

Branch 1 intentionally leaves richer observations, broader training strategies, stronger benchmarking comparability, and target-aware reward work for later branches. Those later branches can now build on a stable persisted contract instead of redefining checkpoint evaluation each time.
