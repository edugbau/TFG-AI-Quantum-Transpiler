---
name: qiskit-2x-compliance
description: Use when modifying Qiskit code, reviewing imports, transpilation flows, backend execution, or test scaffolding that must stay compatible with Qiskit 2.x APIs.
compatibility: opencode
metadata:
  module: qiskit_interface
  scope: project
---

# Qiskit 2.x Compliance

## Overview

Keep all Qiskit-facing code aligned with Qiskit >= 2.0 and avoid legacy APIs that no longer match the project's environment or testing model.

## When to Use

- Editing `src/qiskit_interface/`.
- Touching Qiskit imports, backend execution, transpilation, or QASM serialization.
- Reviewing code for legacy APIs such as `qiskit.execute()` or `QuantumInstance`.

## Quick Reference

- Prefer top-level imports such as `from qiskit import QuantumCircuit`.
- Use `backend.run(...)` or V2 primitives when appropriate.
- Use `qiskit_aer` as a separate package instead of `qiskit.aer`.
- Use fake backends for local tests; do not target real hardware.
- Use `qiskit.qasm2` or `qiskit.qasm3` instead of `circuit.qasm()`.

## Implementation

### Allowed Patterns

- `backend.run(circuit, **kwargs)`
- `SamplerV2` and `EstimatorV2` when a primitive-based flow is appropriate
- `assign_parameters()` in modern parameter binding flows
- `PassManager`-based transpilation pipelines when composing passes explicitly

### Forbidden Patterns

- `qiskit.terra`
- `qiskit.aer`
- `qiskit.execute()`
- `QuantumInstance`
- `circuit.qasm()`

## Common Mistakes

- Reintroducing pre-1.0 import paths because old tutorials still use them.
- Mixing fake-backend local evaluation with code that assumes real hardware access.
- Using deprecated serialization helpers instead of `qasm2` or `qasm3` modules.

## Project References

- `docs/ENVIRONMENT.md`
- `src/qiskit_interface/README.md`
- `src/qiskit_interface/docs/internal_documentation.md`
