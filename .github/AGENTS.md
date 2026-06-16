# Workspace Instructions

Hybrid Quantum Transpilation: Multi-Objective Layout Optimization and Reinforcement Learning Synthesis.

## Architecture

The project has 4 interconnected modules:

1. **`src/qiskit_interface/`** owns Qiskit backend abstraction, circuit helpers, metrics, and transpilation baselines.
2. **`src/mo_module/`** owns multi-objective layout optimization, Pareto analysis, benchmark helpers, and tuning utilities.
3. **`src/rl_module/`** owns the Gymnasium environment, reward strategies, model metadata, RL agents, training, and GUI helpers.
4. **`src/integration/`** owns orchestration across modules, MO-to-RL handoff, scenarios, campaigns, reporting, and compatibility contracts.

## Build And Test

- Use Python 3.10 or newer.
- Install dependencies with `pip install -r requirements.txt`.
- Run tests from the repository root with `pytest tests/`.

## Boundaries

Module boundaries are respected:

- `qiskit_interface` does not orchestrate MO or RL workflows.
- `mo_module` produces and evaluates layout candidates, but does not train RL models.
- `rl_module` consumes circuits and optional initial layouts, but does not generate MO layouts.
- `integration owns` scenario orchestration, campaign execution, and cross-module contracts.

