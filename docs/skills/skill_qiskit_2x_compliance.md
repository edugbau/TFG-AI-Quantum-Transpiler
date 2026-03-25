# Skill: Qiskit 2.x Compliance
**Cross-cutting context for the entire project.**

## Objective
Guarantee compatibility of generated code with **Qiskit >= 2.0** (currently 2.3.0). This major version introduced extensive API breaking changes compared to the 0.x series.

## Non-Negotiable Rules (The Monolith is Gone)

1. **Imports (Packages):**
   - ❌ **FORBIDDEN:** `qiskit.terra`, `qiskit.aer`, `qiskit.ignis`, `qiskit.aqua`, `qiskit.providers.ibmq`.
   - ✅ **CORRECT:** `import qiskit`, `from qiskit import QuantumCircuit`.
   - ✅ **Simulator:** `import qiskit_aer` (standalone package).
   - ✅ **Hardware / Fake Backends:** `from qiskit_ibm_runtime.fake_provider import FakeTorino` (or similar).

2. **Execution and Simulation:**
   - ❌ **FORBIDDEN:** `qiskit.execute()`, `QuantumInstance`.
   - ✅ **CORRECT:** `backend.run(circuit, **kwargs)`.
   - ✅ **CORRECT:** V2 primitives (`SamplerV2`, `EstimatorV2`) for expectation-value extraction.

3. **Circuit Operations:**
   - ❌ **FORBIDDEN:** `circuit.qasm()` (legacy string method).
   - ✅ **CORRECT:** Use `qiskit.qasm2` or `qiskit.qasm3` for export/import.
   - ✅ **Parameter binding:** Use `assign_parameters()` instead of `bind_parameters()` in deprecated contexts.

4. **Transpilation:**
   - ✅ **CORRECT:** Build pipelines with `PassManager` (`from qiskit.transpiler import PassManager`). Avoid iterative `transpile` calls when passes can be grouped.
   - ✅ Extract physical properties from backend `Target` or `CouplingMap`.

5. **Testing Environment:**
   - Do not send circuits to real hardware. Always use `FakeBackend` to simulate topology, native gate set, and coupling map.
