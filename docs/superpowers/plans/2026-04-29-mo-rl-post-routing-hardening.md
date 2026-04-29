# MO+RL Post-Routing Hardening Plan

> **For agentic workers:** Use a test-first workflow when touching the scenario contract, reconstruction logic, or transpilation metrics. Run commands with `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe`.

**Goal:** cerrar las issues detectadas tras habilitar `MO+RL -> routed circuit -> Qiskit post-routing`, manteniendo el pipeline funcional y haciendo fiables tanto el `ScenarioResult` como las métricas comparativas.

**Architecture:** `src/integration/` seguirá siendo dueño de la reconstrucción del circuito ruteado y de la consistencia entre episodio RL, circuito reconstruido y resultado final. `src/qiskit_interface/` seguirá siendo dueño de las métricas y de la transpilación post-routing, pero se endurecerá para distinguir entre anchura física materializada y anchura lógica/activa usada en benchmarking.

**Tech Stack:** Python 3.10, pytest, Qiskit 2.x, dataclasses, fake backends de Qiskit.

---

## Problem Summary

Se han detectado cuatro problemas a resolver:

1. `run_mo_rl_scenario()` falla con `ValueError` si el episodio RL no completa el routing, en vez de devolver un resultado controlado.
2. No se valida que `swap_trace`, `final_layout` reconstruido y `routing_summary.final_layout` sean consistentes entre sí.
3. `routing_summary.total_swaps` y el circuito reconstruido pueden divergir cuando el entorno cuenta swaps inválidos/no-op.
4. Las métricas `trans_num_qubits` y `trans_width` reflejan la anchura física materializada por Qiskit, no el número de qubits activos/lógicos usados por el circuito, lo que puede inducir a comparaciones erróneas en benchmarks y reporting.

---

## File Map

- Modify: `src/integration/contracts.py` - reforzar el contrato del resumen RL y, si hace falta, distinguir contadores de swap intentados vs materializados.
- Modify: `src/integration/routing_evaluator.py` - validar consistencia entre `swap_trace`, `final_layout` reconstruido y resumen del episodio.
- Modify: `src/integration/scenarios.py` - manejar episodios incompletos sin romper `ScenarioResult`.
- Modify: `src/qiskit_interface/circuit_utils.py` - añadir métricas derivadas de qubits activos/materializados si se decide centralizarlo ahí.
- Modify: `src/qiskit_interface/transpiler.py` - exponer en `TranspilationResult` métricas de qubits activos/anchura lógica o etiquetar explícitamente las métricas físicas.
- Modify: `src/integration/README.md` or `README.md` - actualizar la documentación del comportamiento RL v2 y de la interpretación de métricas.
- Modify: `tests/test_integration/test_contracts.py`
- Modify: `tests/test_integration/test_routing_evaluator.py`
- Modify: `tests/test_integration/test_scenarios.py`
- Modify: `tests/test_qiskit_interface/test_qiskit_interface.py`
- Modify: `tests/test_integration/test_docs.py` if public/internal docs are updated.

---

## Task 1: Graceful Handling For Incomplete RL Episodes

**Files:**
- Modify: `src/integration/scenarios.py`
- Test: `tests/test_integration/test_scenarios.py`

- [ ] **Step 1: Write failing tests for incomplete MO+RL episodes**

Add scenario tests that cover:

```python
def test_run_mo_rl_scenario_returns_result_without_transpilation_when_episode_is_truncated(...):
    summary = RoutingEpisodeSummary(
        initial_layout=[2, 0, 1],
        final_layout=[1, 0, 2],
        steps_executed=10,
        total_reward=-1.0,
        completed=False,
        truncated=True,
        total_swaps=4,
        gates_executed_count=1,
        swap_trace=[(0, 1)],
    )
    ...
    assert result.success is False
    assert result.routing_summary is summary
    assert result.transpilation_metrics is None
    assert any("incomplete routing episode" in note.lower() for note in result.notes)
```

and:

```python
def test_run_mo_rl_scenario_returns_result_without_transpilation_when_episode_is_not_completed(...):
    ...
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_scenarios.py -k "mo_rl and (truncated or not_completed)" -q
```

- [ ] **Step 3: Implement graceful fallback in `run_mo_rl_scenario()`**

Implementation requirements:

- If `routing_summary.completed` is `False` or `routing_summary.truncated` is `True`, do not call `build_routed_circuit()`.
- Return a valid `ScenarioResult` with:
  - `success=False`
  - `routing_summary` preserved
  - `transpilation_metrics=None`
  - `transpilation_artifact=None`
  - an explicit note or error explaining that post-routing transpilation was skipped because the RL episode did not finish.
- Keep `RL_Only` semantics unchanged.

- [ ] **Step 4: Re-run tests and confirm pass**

---

## Task 2: Enforce Reconstruction Consistency

**Files:**
- Modify: `src/integration/routing_evaluator.py`
- Modify: `src/integration/scenarios.py`
- Test: `tests/test_integration/test_routing_evaluator.py`
- Test: `tests/test_integration/test_scenarios.py`

- [ ] **Step 1: Write failing consistency tests**

Add tests for:

```python
def test_build_routed_circuit_reconstructed_layout_matches_swap_trace(...):
    ...

def test_run_mo_rl_scenario_rejects_mismatched_routing_summary_final_layout(...):
    summary = RoutingEpisodeSummary(
        initial_layout=[2, 0, 1],
        final_layout=[2, 1, 0],  # intentionally wrong
        ...,
        swap_trace=[(0, 1)],
    )
    ...
    with pytest.raises(ValueError, match="final_layout"):
        scenarios.run_mo_rl_scenario(request)
```

- [ ] **Step 2: Run targeted tests to verify failure**

- [ ] **Step 3: Implement consistency checks**

Implementation requirements:

- After `build_routed_circuit()` returns, compare reconstructed `final_layout` with `routing_summary.final_layout` when that field is present.
- If they differ, fail loudly with a message that identifies the inconsistency.
- Add a small helper so the consistency rule is centralized and reusable.
- Optionally also validate `initial_layout == routing_summary.initial_layout` when both are present.

- [ ] **Step 4: Re-run targeted tests**

---

## Task 3: Clarify And Fix Swap Counters

**Files:**
- Modify: `src/integration/contracts.py`
- Modify: `src/integration/routing_evaluator.py`
- Modify: `src/integration/scenarios.py` if needed for artifact/reporting notes
- Test: `tests/test_integration/test_contracts.py`
- Test: `tests/test_integration/test_routing_evaluator.py`

- [ ] **Step 1: Decide the contract explicitly**

Recommended option:

- keep `total_swaps` as the count of swaps actually materialized in the reconstructed circuit,
- add a second field such as `swap_attempts` if preserving the old raw env counter is useful.

Alternative acceptable option:

- keep `total_swaps` as attempted swaps,
- add `applied_swaps` derived from `len(swap_trace)`,
- document that reconstruction uses `applied_swaps`/`swap_trace`, not raw attempts.

- [ ] **Step 2: Write failing tests for the chosen contract**

Example for the recommended option:

```python
def test_evaluate_routing_episode_reports_materialized_swaps_separately_from_attempts(...):
    ...
    assert summary.total_swaps == 1
    assert summary.swap_attempts == 3
    assert summary.swap_trace == [(1, 2)]
```

- [ ] **Step 3: Implement the contract**

Implementation requirements:

- Remove ambiguity between raw env swap count and replayed swap count.
- Make artifact generation and any user-facing reporting use the materialized count when comparing against final circuit metrics.
- Preserve backward compatibility only if there is a concrete consumer of the old meaning; otherwise prefer the cleaner contract.

- [ ] **Step 4: Re-run targeted tests**

---

## Task 4: Fix Metric Semantics For Active vs Materialized Width

**Files:**
- Modify: `src/qiskit_interface/circuit_utils.py`
- Modify: `src/qiskit_interface/transpiler.py`
- Modify: `README.md` and/or `src/integration/README.md`
- Test: `tests/test_qiskit_interface/test_qiskit_interface.py`
- Test: `tests/test_integration/test_scenarios.py` if scenario rows are extended

- [ ] **Step 1: Write failing tests that expose the ambiguity**

Suggested test shape:

```python
def test_transpilation_metrics_distinguish_active_qubits_from_materialized_backend_width(backend_torino):
    qc = create_ghz_circuit(3)
    result = transpile_with_custom_layout(qc, [10, 11, 12], backend=backend_torino, optimization_level=1)

    assert result.transpiled_metrics.num_qubits == 133  # physical materialization remains true
    assert result.transpiled_metrics.active_qubits == 3
```

and optionally:

```python
def test_transpilation_result_to_dict_exposes_active_qubit_metrics(...):
    assert row["trans_active_qubits"] == 3
```

- [ ] **Step 2: Choose the metric contract**

Recommended contract:

- keep existing `num_qubits` and `width` fields unchanged because they reflect the literal `QuantumCircuit`,
- add new derived metrics:
  - `active_qubits`
  - `active_width` or `used_qubit_span`
- document clearly that benchmarking should prefer `active_qubits` over `num_qubits` when sparse physical layouts are used.

- [ ] **Step 3: Implement active-qubit metrics**

Implementation ideas:

- Derive active physical qubits by scanning instruction operands rather than using register size.
- Add the new fields to `CircuitMetrics`, `to_dict()`, artifact generation, and any relevant summaries.
- Do not silently redefine `num_qubits`; that would be a breaking semantic change.

- [ ] **Step 4: Update docs**

Update the project docs so they no longer state or imply that `trans_num_qubits` equals logical program size for routed/layout-aware baselines.

- [ ] **Step 5: Re-run targeted tests**

---

## Task 5: Documentation And Review Closure

**Files:**
- Modify: `README.md`
- Modify: `docs/minimum_gap_analysis.md` or add a short follow-up note under `docs/future-iterations/` if preferred
- Modify: `src/integration/docs/internal_documentation.md` if it documents scenario semantics
- Test: `tests/test_integration/test_docs.py` if needed

- [ ] **Step 1: Update RL scenario semantics in docs**

Document:

- `RL_Only` still returns episode summaries only.
- `MO+RL` now attempts full post-routing transpilation.
- `MO+RL` may skip transpilation and return a controlled failure/result when the RL episode is incomplete.
- `swap_trace` is now the canonical reconstruction payload.

- [ ] **Step 2: Update metric interpretation docs**

Document the distinction between:

- materialized physical register size,
- active used qubits,
- logical circuit size.

- [ ] **Step 3: Run docs-related tests if present**

---

## Acceptance Criteria

- `MO+RL` no longer crashes uncontrolled on incomplete routing episodes.
- Reconstructed `final_layout` is validated against the episode summary before post-routing transpilation proceeds.
- Swap counters are semantically unambiguous and consistent with the reconstructed circuit.
- Benchmark rows and artifacts expose a metric that reflects active qubits, so sparse-layout runs do not mislead consumers.
- Tests cover happy path and failure path for MO+RL reconstruction.
- Public/internal docs describe the updated RL and metric semantics.

---

## Suggested Verification Commands

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_contracts.py tests/test_integration/test_routing_evaluator.py tests/test_integration/test_scenarios.py tests/test_qiskit_interface/test_qiskit_interface.py -q
```

If docs are updated:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_docs.py -q
```
