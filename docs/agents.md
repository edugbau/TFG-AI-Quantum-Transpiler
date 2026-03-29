# Project Module Contracts

## Architecture Overview

The repository is organized into four modules:

| Module | Responsibility | Not Responsible For |
| --- | --- | --- |
| `src/qiskit_interface/` | Backends, transpilation, metrics, and baselines | MO -> RL orchestration |
| `src/rl_module/` | Gymnasium environment, rewards, agent training, and generic `initial_layout` ingestion | Producing layouts or orchestrating experiments |
| `src/mo_module/` | Multi-objective layout search, Pareto fronts, and layout evaluation | Driving RL directly |
| `src/integration/` | Orchestration of `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL` | Re-implementing module internals |

## Shared Layout Convention

The shared layout format is:

```python
layout[i] = physical_qubit_for_logical_qubit_i
```

- `qiskit_interface` may evaluate this layout through transpilation helpers.
- `rl_module` may ingest it through `env.reset(options={"initial_layout": layout})`.
- `integration` owns the process that connects producer and consumer.

## Current Status

- `src/integration/` is currently a stub.
- `src/rl_module/` supports routing; `synthesis` remains placeholder work.
- `mo_module` and `rl_module` should remain independently testable.

## Scenario Ownership

- `Baseline`: Qiskit default transpilation only.
- `MO_Only`: Qiskit with an MO-selected layout only.
- `RL_Only`: RL starting from a non-MO layout.
- `MO+RL`: layout produced by MO and injected into RL by `src/integration/`.
