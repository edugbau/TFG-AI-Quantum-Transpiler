---
name: qiskit-2x-compliance
description: 'Enforce Qiskit >=2.0 compatibility: valid imports, backend.run or V2 primitives, non-deprecated circuit APIs, and FakeBackends for local tests.'
argument-hint: 'Paste the Qiskit code section you want checked for 2.x compatibility issues.'
user-invocable: true
---

# Qiskit 2.x Compliance

## When to Use
- Any project change that touches Qiskit code.
- Reviewing imports, simulation, transpilation, or circuit APIs.
- Detecting legacy patterns from pre-1.0/2.x Qiskit.

## Objective
Guarantee compatibility with Qiskit >= 2.0 (project currently at 2.3.0) while avoiding legacy APIs and breaking-change pitfalls.

## Non-Negotiable Rules

### 1) Imports
- Forbidden:
  - `qiskit.terra`
  - `qiskit.aer`
  - `qiskit.ignis`
  - `qiskit.aqua`
  - `qiskit.providers.ibmq`
- Correct:
  - `import qiskit`
  - `from qiskit import QuantumCircuit`
  - Simulator as a standalone package: `import qiskit_aer`
  - Fake backends: `from qiskit_ibm_runtime.fake_provider import FakeTorino`

### 2) Execution and Simulation
- Forbidden: `qiskit.execute()`, `QuantumInstance`.
- Correct:
  - `backend.run(circuit, **kwargs)`
  - V2 primitives (`SamplerV2`, `EstimatorV2`) when appropriate.

### 3) Circuit Operations
- Forbidden: `circuit.qasm()`.
- Correct:
  - Export/import with `qiskit.qasm2` or `qiskit.qasm3`.
  - Use `assign_parameters()` instead of `bind_parameters()` in deprecated contexts.

### 4) Transpilation
- Prefer pipelines with `PassManager` (`from qiskit.transpiler import PassManager`).
- Avoid iterative `transpile` calls when passes can be grouped.
- Extract physical properties from backend `Target` or `CouplingMap`.

### 5) Test Policy
- Do not send circuits to real hardware.
- Use FakeBackend for topology, native gate set, and coupling map.

## Project References
- `docs/agents.md`
- `docs/ENVIRONMENT.md`
- `src/qiskit_interface/README.md`
- `src/qiskit_interface/docs/internal_documentation.md`
