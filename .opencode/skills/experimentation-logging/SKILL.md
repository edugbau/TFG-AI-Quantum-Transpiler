---
name: experimentation-logging
description: Use when running benchmarks, exporting experiment tables, setting deterministic seeds, or coordinating TensorBoard and plotting across MO, RL, or integration workflows.
compatibility: opencode
metadata:
  module: cross-cutting
  scope: project
---

# Experimentation Logging

## Overview

Standardize reproducibility and comparison across experiments by keeping seeds, training telemetry, tabular exports, and plots consistent across project modules.

## When to Use

- Running benchmark suites or comparative experiments.
- Logging Stable-Baselines3 training or evaluation.
- Exporting experiment outputs for analysis, reporting, or reproducibility.

## Quick Reference

- Seed `random`, `numpy`, and `torch` before reproducible runs.
- Use `env.reset(seed=...)` for Gymnasium environments.
- Keep tabular exports consistent across runs.
- Use TensorBoard and checkpoints for long-running RL experiments.

## Implementation

- Set global seeds before training, optimization, or benchmarking starts.
- Export tables with stable columns such as `Circuit`, `Method`, `Depth`, `CNOTs`, `Fidelity/Error`, `Execution_Time`, and `Seed`.
- Use clear labels, legends, and units for Pareto plots and learning curves.

## Common Mistakes

- Setting seeds after work has already started.
- Exporting tables with drifting column names between modules.
- Logging RL metrics without checkpoints or reproducible metadata.

## Project References

- `docs/ENVIRONMENT.md`
- `src/rl_module/training.py`
- `src/mo_module/benchmark/runner.py`
- `experiments/`
