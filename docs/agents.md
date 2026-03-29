# Project Module Contracts

## Architecture Overview

| Module | Responsibilities | Not responsible for |
| --- | --- | --- |
| `src/qiskit_interface/` | Qiskit backend abstraction, circuit metrics, and transpilation helpers used to evaluate candidate layouts. | Owning MO search, RL policy training, or end-to-end orchestration across modules. |
| `src/rl_module/` | Gymnasium-based RL environment and agent logic for routing and synthesis workflows that consume agreed inputs. | Generating MO layouts, wrapping Qiskit backend concerns, or coordinating full benchmark scenarios alone. |
| `src/mo_module/` | Multi-objective optimization that produces candidate initial layouts and related Pareto analysis. | Running RL training, owning Qiskit transpilation wrappers, or orchestrating the full pipeline. |
| `src/integration/` | Process ownership for connecting module outputs and inputs across Baseline, MO_Only, RL_Only, and MO+RL scenarios. | Replacing the internal responsibilities of qiskit_interface, rl_module, or mo_module. |

## Shared Layout Convention

```text
layout[i] = physical_qubit_for_logical_qubit_i
```

- `qiskit_interface` may evaluate this layout through transpilation helpers.
- `rl_module` may ingest it through `env.reset(options={"initial_layout": layout})`.
- `integration` owns the process that connects producer and consumer.
- `src/integration/` owns MO->RL orchestration.

## Current Status

- `integration` is currently a stub.
- `rl_module` supports routing and synthesis remains placeholder work.
- `mo_module` and `rl_module` should remain independently testable.

## Scenario Ownership

- `Baseline`: transpilation baseline without MO-produced layouts or RL handoff orchestration.
- `MO_Only`: `mo_module` produces layouts and `qiskit_interface` evaluates them without RL synthesis.
- `RL_Only`: `rl_module` runs without MO-produced layouts, using the shared layout contract only when provided by integration.
- `MO+RL`: `integration` orchestrates the handoff where MO produces layouts and RL consumes them under the shared module contract.
