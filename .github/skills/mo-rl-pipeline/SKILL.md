---
name: mo-rl-pipeline
description: 'Define the MO->RL integration flow: handoff of Pareto-optimal layouts to Gymnasium, logical/physical mapping conventions, and structured benchmarking across Baseline, MO_Only, RL_Only, and MO+RL.'
argument-hint: 'Describe your current handoff point and which benchmark scenario is failing or missing.'
user-invocable: true
---

# MO to RL Pipeline

## When to Use
- Working on the `integration` module.
- Passing layouts from the multi-objective module into the RL environment.
- Running MO+RL comparisons against baselines.

## Objective
Standardize the transfer of genotypic information from the optimization module (MO) to the reinforcement learning agent (RL), plus experimental orchestration.

## Procedure
1. Finalize MO and collect valid Pareto-optimal layouts.
2. Inject the layout in RL environment `reset`.
3. Run benchmarking with comparable scenarios.
4. Verify and document the logical/physical mapping convention.

## Implementation Rules

### 1) State Handoff
- MO output: set of Pareto-optimal individuals (initial layouts, for example `[1, 0, 3, 2]`).
- RL ingestion: accept layout via `env.reset(options={"initial_layout": layout})`.

### 2) Structured Benchmarking
- Execute sequentially:
  - `Baseline`: Qiskit (SABRE or default transpilation level 3).
  - `MO_Only`: Qiskit with MO layout only, no additional RL.
  - `RL_Only`: Qiskit with random/trivial layout plus RL synthesis.
  - `MO+RL`: MO layout used as RL agent input.

### 3) Physical vs Logical Qubit Handling
- Ensure the layout array convention is explicit (`logical_qubit -> physical_qubit` or inverse).
- Document exactly what each index represents.
- Initialize RL `state` consistently with initial assignment and routing/synthesis actions.

## Project References
- `docs/agents.md`
- `src/rl_module/docs/synthesis_mode_status.md`
- `src/mo_module/docs/internal_documentation.md`
