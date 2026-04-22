---
name: rl-quantum-synthesis
description: Use when modifying src/rl_module environments, rewards, action or observation design, SB3 training flows, or checkpointed evaluation for routing or synthesis work.
compatibility: opencode
metadata:
  module: rl_module
  scope: project
---

# RL Quantum Synthesis

## Overview

Guide work inside `src/rl_module/` so Gymnasium environment semantics, reward shaping, training callbacks, and checkpointed evaluation stay aligned with the current routing-first project scope.

## When to Use

- Editing `src/rl_module/environment.py` or related environment helpers.
- Changing reward logic, action selection, or observation encoding.
- Updating Stable-Baselines3 training, evaluation, checkpoints, or TensorBoard logging.

## Quick Reference

- Implement environments with `gymnasium.Env`, not legacy `gym`.
- Keep `reset(seed=..., options=...)` compatible with external `initial_layout` injection.
- Return Gymnasium-standard tuples from `step()` and `reset()`.
- Use SB3 callbacks for evaluation, checkpoints, and training telemetry.

## Core Pattern

- Define clear observation and action spaces.
- Penalize invalid or redundant actions explicitly.
- Separate routing-capable behavior from limited synthesis-capable behavior in docs and code.
- Keep MO -> RL ownership in `src/integration/` even when `initial_layout` is provided.

## Common Mistakes

- Reintroducing legacy `gym` patterns or wrong `step()`/`reset()` signatures.
- Assuming `initial_layout` is owned by `mo_module` instead of by the caller or `src/integration/`.
- Expanding docs to imply full synthesis support when only limited flows are implemented.

## Project References

- `src/rl_module/docs/internal_documentation.md`
- `src/rl_module/docs/synthesis_mode_status.md`
- `src/rl_module/environment.py`
- `src/rl_module/training.py`
