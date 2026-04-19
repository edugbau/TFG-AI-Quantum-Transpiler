# Chapter 1 Base Transpilation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** implementar soporte visible para entradas `library` y `qasm_file`, artefactos estructurados de transpilación y baselines explícitos sobre los fake backends actuales.

**Architecture:** `src/qiskit_interface/` seguirá siendo dueño del parsing QASM, de las métricas y del catálogo de baselines. `src/integration/` ampliará únicamente sus contratos y runners para consumir el nuevo cargador de circuitos y devolver el nuevo artefacto estructurado sin alterar la semántica de los escenarios RL.

**Tech Stack:** Python 3.10, pytest, Qiskit 2.x, qiskit-ibm-runtime fake backends, dataclasses.

---

## File Map

- Create: `docs/superpowers/specs/2026-04-19-chapter1-base-transpilation-design.md` - diseño validado del capítulo.
- Create: `docs/superpowers/plans/2026-04-19-chapter1-base-transpilation.md` - este plan.
- Modify: `src/qiskit_interface/circuit_utils.py` - cargador normalizado `library`/`qasm_file` y metadata de entrada.
- Modify: `src/qiskit_interface/backend_info.py` - resumen hardware-aware estable derivado de `BackendInfo`.
- Modify: `src/qiskit_interface/transpiler.py` - `baseline_name`, `to_artifact_dict()` y catálogo de baselines explícitos.
- Modify: `src/qiskit_interface/__init__.py` - reexportación de nuevas APIs públicas.
- Modify: `src/qiskit_interface/README.md` - documentación de inputs y baselines.
- Modify: `src/integration/contracts.py` - nuevos campos de `ScenarioRequest` y `transpilation_artifact`.
- Modify: `src/integration/scenarios.py` - uso del cargador normalizado y retorno del artefacto estructurado.
- Modify: `src/integration/runner.py` - flags CLI para `qasm_file`.
- Modify: `src/integration/README.md` - documentar soporte QASM para escenarios Qiskit.
- Modify: `src/integration/docs/internal_documentation.md` - actualizar alcance y pipelines.
- Modify: `tests/test_qiskit_interface/test_qiskit_interface.py` - tests TDD de carga normalizada, artefactos y baselines.
- Modify: `tests/test_integration/test_contracts.py` - tests TDD de contratos de entrada/salida.
- Modify: `tests/test_integration/test_scenarios.py` - tests TDD de escenarios con `qasm_file`.
- Modify: `tests/test_integration/test_runner.py` - tests TDD de CLI.
- Modify: `tests/test_integration/test_docs.py` - tests de docs tras el cambio de alcance.

### Task 1: Input Normalization In qiskit_interface

**Files:**
- Modify: `src/qiskit_interface/circuit_utils.py`
- Modify: `src/qiskit_interface/__init__.py`
- Test: `tests/test_qiskit_interface/test_qiskit_interface.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_circuit_from_library_attaches_input_metadata():
    circuit = qiskit_interface.load_circuit(
        "library",
        circuit_name="ghz",
        num_qubits=3,
        seed=42,
    )

    assert circuit.metadata["source_kind"] == "library"
    assert circuit.metadata["source_format"] == "library"
    assert circuit.metadata["resolved_circuit_name"] == "ghz"


def test_load_circuit_from_qasm2_file_attaches_metadata(simple_circuit, tmp_path):
    path = tmp_path / "simple.qasm"
    qiskit_interface.save_circuit_to_qasm2(simple_circuit, path)

    circuit = qiskit_interface.load_circuit(
        "qasm_file",
        circuit_path=path,
        circuit_format="auto",
    )

    assert circuit.metadata["source_kind"] == "qasm_file"
    assert circuit.metadata["source_format"] == "qasm2"
    assert circuit.metadata["source_path"] == str(path)
    assert circuit.metadata["resolved_circuit_name"] == "simple"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_qiskit_interface/test_qiskit_interface.py -k "load_circuit_from_library_attaches_input_metadata or load_circuit_from_qasm2_file_attaches_metadata" -q`

Expected: FAIL because `load_circuit` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def load_circuit(
    source_kind: str,
    *,
    circuit_name: str | None = None,
    num_qubits: int | None = None,
    seed: int = 42,
    circuit_path: str | Path | None = None,
    circuit_format: str = "auto",
) -> QuantumCircuit:
    ...
```

Implement:

- validation for `library` and `qasm_file`;
- autodetection of `qasm2`/`qasm3` from file contents when `auto`;
- metadata attachment on `QuantumCircuit.metadata`.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_qiskit_interface/test_qiskit_interface.py -k "load_circuit_from_library_attaches_input_metadata or load_circuit_from_qasm2_file_attaches_metadata" -q`

Expected: PASS.

### Task 2: Structured Artifacts And Named Baselines

**Files:**
- Modify: `src/qiskit_interface/backend_info.py`
- Modify: `src/qiskit_interface/transpiler.py`
- Modify: `src/qiskit_interface/__init__.py`
- Test: `tests/test_qiskit_interface/test_qiskit_interface.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_transpilation_result_exposes_structured_artifact(simple_circuit, backend_torino):
    circuit = qiskit_interface.load_circuit(
        "library",
        circuit_name="ghz",
        num_qubits=3,
        seed=42,
    )
    result = qiskit_interface.transpile_circuit(circuit, backend=backend_torino, seed=42)

    artifact = result.to_artifact_dict()

    assert artifact["artifact_version"] == "transpilation_result.v1"
    assert artifact["circuit"]["source_kind"] == "library"
    assert artifact["backend"]["backend_name"] == "fake_torino"
    assert "original" in artifact["metrics"]
    assert "transpiled" in artifact["metrics"]


def test_run_named_baseline_tags_result_with_baseline_name(simple_circuit, backend_torino):
    result = qiskit_interface.run_named_baseline(
        simple_circuit,
        baseline_name="qiskit_level_2",
        backend=backend_torino,
        seed=42,
    )

    assert result.baseline_name == "qiskit_level_2"
    assert result.to_dict()["baseline_name"] == "qiskit_level_2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_qiskit_interface/test_qiskit_interface.py -k "structured_artifact or run_named_baseline_tags_result" -q`

Expected: FAIL because `to_artifact_dict()` and `run_named_baseline()` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
BASELINE_PRESETS = {
    "qiskit_level_0": {"optimization_level": 0, "requires_layout": False},
    "qiskit_level_1": {"optimization_level": 1, "requires_layout": False},
    "qiskit_level_2": {"optimization_level": 2, "requires_layout": False},
    "qiskit_level_3": {"optimization_level": 3, "requires_layout": False},
    "custom_layout_level_1": {"optimization_level": 1, "requires_layout": True},
}
```

Implement:

- backend summary helper from `BackendInfo`;
- `baseline_name` and `hardware_summary` in `TranspilationResult`;
- `to_artifact_dict()`;
- `list_available_baselines()`;
- `run_named_baseline(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_qiskit_interface/test_qiskit_interface.py -k "structured_artifact or run_named_baseline_tags_result" -q`

Expected: PASS.

### Task 3: integration Contracts And QASM Entry Points

**Files:**
- Modify: `src/integration/contracts.py`
- Modify: `src/integration/scenarios.py`
- Modify: `src/integration/runner.py`
- Test: `tests/test_integration/test_contracts.py`
- Test: `tests/test_integration/test_scenarios.py`
- Test: `tests/test_integration/test_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_qasm_file_request_requires_circuit_path():
    with pytest.raises(ValueError, match="circuit_path"):
        ScenarioRequest(
            scenario_name="Baseline",
            circuit_source="qasm_file",
            backend_name="fake_torino",
        )


def test_run_baseline_scenario_returns_structured_artifact_for_qasm_input(monkeypatch):
    request = _make_request(
        "Baseline",
        circuit_source="qasm_file",
        circuit_path="circuits/sample.qasm",
        circuit_format="qasm2",
        circuit_name=None,
        num_qubits=None,
    )
    ...
    assert result.transpilation_artifact is not None
    assert result.circuit_name == "sample"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_contracts.py tests/test_integration/test_scenarios.py tests/test_integration/test_runner.py -k "qasm_file or transpilation_artifact" -q`

Expected: FAIL because the new fields and outputs do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement:

- `CircuitSource` and `CircuitFormat` enums in `contracts.py`;
- new `ScenarioRequest` fields and validations;
- `transpilation_artifact` in `ScenarioResult`;
- `runner.py` flags `--circuit-source`, `--circuit-path`, `--circuit-format`;
- scenario loading via `qiskit_interface.load_circuit(...)`;
- `Baseline` using `run_named_baseline(..., "qiskit_level_1")`;
- `MO_Only` using `run_named_baseline(..., "custom_layout_level_1")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_contracts.py tests/test_integration/test_scenarios.py tests/test_integration/test_runner.py -k "qasm_file or transpilation_artifact" -q`

Expected: PASS.

### Task 4: Docs And Full Verification

**Files:**
- Modify: `src/qiskit_interface/README.md`
- Modify: `src/integration/README.md`
- Modify: `src/integration/docs/internal_documentation.md`
- Test: `tests/test_integration/test_docs.py`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing docs assertions**

```python
def test_integration_docs_describe_qasm_support_for_qiskit_facing_scenarios() -> None:
    readme_text = read_text("README.md")

    assert "QASM input is supported for Baseline and MO_Only" in readme_text
    assert "episode summaries" in readme_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_docs.py tests/test_module_contracts.py -k "QASM or integration" -q`

Expected: FAIL because the docs still say QASM is deferred.

- [ ] **Step 3: Write minimal documentation updates**

Update docs to state:

- QASM input is available for Qiskit-facing scenarios in `integration`;
- RL-based scenarios still output episode summaries, not final circuits;
- the backend catalog is intentionally limited to the current fake backends.

- [ ] **Step 4: Run the full verification suites**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_qiskit_interface tests/test_integration tests/test_module_contracts.py -q`

Expected: PASS.

## Self-Review

- Spec coverage: el plan cubre inputs, artefactos, baselines, integración y documentación.
- Placeholder scan: no hay `TODO`, `TBD` ni referencias vagas a “lo de arriba”.
- Type consistency: `circuit_source`, `circuit_format`, `baseline_name` y `transpilation_artifact` se usan con el mismo nombre en todas las tareas.

Plan complete and saved to `docs/superpowers/plans/2026-04-19-chapter1-base-transpilation.md`.

Dos opciones de ejecución:

1. Subagent-Driven (recommended) - despachar una sub-tarea por bloque.
2. Inline Execution - ejecutar las tareas en esta misma sesión.

La petición actual del usuario ya fija ejecución inline en esta misma sesión.
