# Skill: MO -> RL Pipeline (Integration)
**Context for Module 4 (`integration`).**

## Objective
Transfer genotypic information from the Optimization Module (MO) to the Reinforcement Learning Agent (RL) and orchestrate the full flow.

## Implementation Rules

1. **The Handoff (State Transfer)**
   - **MO Output:** The `pymoo` process finishes and returns a set of Pareto-optimal individuals (the Pareto Front). Each individual is a valid initial layout (e.g. `[1, 0, 3, 2]`).
   - **RL Ingestion:** The Gymnasium environment should be created (or reset) by accepting one of these optimal layouts directly as initial state or `initial_layout`. The method `env.reset(options={"initial_layout": layout})` is the recommended entry point.

2. **Structured Benchmarking**
   - The pipeline should run multiple experiments sequentially:
     - `Baseline`: Qiskit (SABRE or default transpilation level 3).
     - `MO_Only`: Qiskit using only the layout obtained from MO (no extra RL).
     - `RL_Only`: Qiskit with a random (or trivial) layout + RL synthesis.
     - `MO+RL`: Layout obtained from MO is passed to RL agent. Final comparison.

3. **Qubit Handling (Physical vs Logical)**
   - During handoff, ensure the MO-returned array follows the correct mapping (`logical_qubit -> physical_qubit` or the inverse, depending on project convention). Always document what each index means.
   - RL should initialize its `state` (observation) assuming abstract-circuit gates are already mapped to that initial assignment, and actions either insert SWAPs to resolve remaining routing constraints or synthesize directly on top of that mapping.
