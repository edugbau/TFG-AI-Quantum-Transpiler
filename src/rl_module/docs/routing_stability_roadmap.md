# Routing Stability Roadmap

## Current Limitation Summary

The routing environment currently exposes only short-horizon transition stability signals. Task 2 adds lightweight detection for repeated layouts, immediate undo-swaps, and a routing progress delta derived from the existing lookahead routing-distance observation. This helps the agent distinguish obviously unproductive local oscillations, but it does not model longer temporal motifs, policy memory, or hard action feasibility constraints.

## Why Task 2 Is Intermediate Only

This level is intentionally intermediate because it stays inside the current environment-plus-reward architecture. The new signals are local shaping hints, not a full anti-oscillation control system. They can discourage short cycles, but they cannot reliably prevent deeper loops, credit assignment drift across long routing horizons, or exploration waste caused by valid-yet-hopeless swaps.

## When To Escalate To Level 3

Escalate to level 3 when one or more of the following persists after reward tuning:

- repeated A->B->A or short-cycle behavior still dominates trajectories,
- training remains unstable across seeds,
- the agent spends substantial rollout budget on obviously dominated swaps,
- reward shaping starts compensating for missing policy memory or missing feasibility constraints,
- benchmark circuits require longer-horizon recovery than the current local signals can express.

## Level 3 Options

- MaskablePPO to support the new masked routing regime for new checkpoints only.
- Explicit action masks generated from frontier-aware heuristics.
- Recurrent policy variants when short-term observation augmentation is no longer enough to encode trajectory context.
- Optional longer transition history features if recurrence is too costly or unavailable.

## Current masked-routing interpretation

Task 5 clarifies the intended public contract of the masked-routing regime:

- routing still uses a fixed action space over coupling-map edges;
- `action_masks()` applies a deterministic frontier-aware hard mask over that fixed index set;
- this behaves as a SABRE-style candidate restriction layer rather than as a dynamic action-space redesign;
- `MaskablePPO` is the standard trainer for new masked-routing checkpoints only;
- legacy PPO/DQN checkpoints remain supported through legacy/default or otherwise unmasked evaluation contracts.

## Migration Risks

- Action-mask semantics can drift from environment validity semantics if they are maintained separately.
- MaskablePPO introduces tighter trainer-policy coupling and may reduce algorithm portability.
- Recurrent policy support changes rollout collection, batching, and evaluation protocols.
- Longer memory features can overfit to specific topologies if not benchmarked carefully.
- Mixed shaping plus masking can hide regressions if ablations are not kept in the evaluation loop.
