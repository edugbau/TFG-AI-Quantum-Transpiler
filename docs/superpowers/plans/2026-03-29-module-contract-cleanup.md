# Module Contract Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align docs, public narratives, and local APIs with the intended four-module architecture without implementing the integration module itself.

**Architecture:** Preserve the existing `logical_qubit -> physical_qubit` layout convention and the current `initial_layout` API, but narrow responsibility claims inside each module. Add one canonical architecture doc in `docs/agents.md`, update module-local narratives to route MO -> RL orchestration through `src/integration/`, and protect the cleanup with one small repo-contract test module plus the existing module suites.

**Tech Stack:** Python 3.10, pytest, Markdown docs, Qiskit 2.3, Gymnasium.

---

## File Map

- Create: `.worktrees/module-contract-cleanup/docs/agents.md` - canonical architecture and shared layout conventions.
- Create: `.worktrees/module-contract-cleanup/tests/test_module_contracts.py` - regression tests for architecture docs, public contract wording, and import boundaries.
- Modify: `.worktrees/module-contract-cleanup/README.md` - point readers to the canonical architecture doc and describe `integration` as orchestration owner.
- Modify: `.worktrees/module-contract-cleanup/src/qiskit_interface/transpiler.py` - remove bridge language from module/function docstrings and keep custom-layout semantics local.
- Modify: `.worktrees/module-contract-cleanup/src/qiskit_interface/README.md` - describe custom-layout transpilation as local evaluation, not MO -> RL handoff.
- Modify: `.worktrees/module-contract-cleanup/src/mo_module/docs/internal_documentation.md` - route future RL consumption through `integration` instead of direct MO -> RL ownership.
- Modify: `.worktrees/module-contract-cleanup/src/rl_module/environment.py` - make `initial_layout` source-agnostic in public docstrings/comments.
- Modify: `.worktrees/module-contract-cleanup/src/rl_module/docs/lookahead_frontier.md` - clarify that `initial_layout` is generic external input.
- Modify: `.worktrees/module-contract-cleanup/src/integration/__init__.py` - declare orchestration ownership and current stub status.

### Task 1: Canonical Architecture Doc and Root Contract

**Files:**
- Create: `.worktrees/module-contract-cleanup/tests/test_module_contracts.py`
- Create: `.worktrees/module-contract-cleanup/docs/agents.md`
- Modify: `.worktrees/module-contract-cleanup/README.md`

- [ ] **Step 1: Write the failing architecture-doc tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_agents_exists_and_describes_four_modules():
    agents_doc = ROOT / "docs" / "agents.md"
    assert agents_doc.exists(), "README y .github/AGENTS.md apuntan a docs/agents.md"

    text = agents_doc.read_text(encoding="utf-8")
    for token in (
        "src/qiskit_interface/",
        "src/rl_module/",
        "src/mo_module/",
        "src/integration/",
    ):
        assert token in text


def test_readme_architecture_reference_points_to_real_doc():
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[agents.md](docs/agents.md)" in readme_text
    assert (ROOT / "docs" / "agents.md").exists()
```

- [ ] **Step 2: Run the new tests and verify they fail first**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py -q`

In: `.worktrees/module-contract-cleanup`

Expected: FAIL because `docs/agents.md` does not exist yet.

- [ ] **Step 3: Add the canonical architecture doc and tighten the README wording**

Create `.worktrees/module-contract-cleanup/docs/agents.md` with:

```md
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
```

Update `.worktrees/module-contract-cleanup/README.md` so the architecture block becomes:

```md
src/
├── qiskit_interface/   # Módulo 1: Interfaz con Qiskit
├── rl_module/          # Módulo 2: Aprendizaje por Refuerzo
├── mo_module/          # Módulo 3: Optimización Multiobjetivo
└── integration/        # Módulo 4: Orquestación MO->RL y experimentación (stub actual)

Para más detalles sobre la arquitectura y los contratos entre módulos, ver [agents.md](docs/agents.md).
```

- [ ] **Step 4: Re-run the architecture-doc tests and verify they pass**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py -q`

In: `.worktrees/module-contract-cleanup`

Expected: PASS (`2 passed`).

- [ ] **Step 5: Commit the architecture-doc task**

```bash
git add README.md docs/agents.md tests/test_module_contracts.py
git commit -m "docs: add canonical module contract guide"
```

### Task 2: Qiskit Interface Public Contract

**Files:**
- Modify: `.worktrees/module-contract-cleanup/tests/test_module_contracts.py`
- Modify: `.worktrees/module-contract-cleanup/src/qiskit_interface/transpiler.py`
- Modify: `.worktrees/module-contract-cleanup/src/qiskit_interface/README.md`

- [ ] **Step 1: Add failing tests for the Qiskit-interface wording contract**

Append to `.worktrees/module-contract-cleanup/tests/test_module_contracts.py`:

```python
def test_qiskit_interface_docstring_frames_custom_layout_as_local_evaluation():
    text = (ROOT / "src" / "qiskit_interface" / "transpiler.py").read_text(encoding="utf-8")
    assert "evaluación local de layouts suministrados por el llamador" in text
    assert "No implementa la integración MO -> RL" in text
    assert "puente principal" not in text
    assert "híbrido MO+RL" not in text


def test_qiskit_interface_readme_avoids_bridge_language():
    text = (ROOT / "src" / "qiskit_interface" / "README.md").read_text(encoding="utf-8")
    assert "helper de evaluación local" in text
    assert "Función puente para el Módulo MO" not in text
```

- [ ] **Step 2: Run the contract tests and verify the new assertions fail**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py -q`

In: `.worktrees/module-contract-cleanup`

Expected: FAIL on the new Qiskit-interface wording assertions.

- [ ] **Step 3: Narrow the Qiskit-interface public contract to local evaluation**

Update the header docstring in `.worktrees/module-contract-cleanup/src/qiskit_interface/transpiler.py` to:

```python
Este fichero implementa la transpilación estándar de Qiskit como
línea base (baseline) y como helper de evaluación local de layouts
suministrados por el llamador.

Funcionalidades principales:
  - Transpilación estándar a los 4 niveles de optimización de Qiskit (0-3).
  - Transpilación con layout inicial personalizado.
  - Comparación pre/post transpilación con métricas detalladas.
  - Transpilación batch de múltiples circuitos.
  - Generación de resultados de benchmark tabulados.
```

Update the "Soporte de layout inicial" design note in the same file to:

```python
4. **Soporte de layout inicial** - Se permite pasar un ``initial_layout``
   (lista de qubits físicos) para evaluar layouts externos con el
   mismo pipeline de transpilación. La orquestación entre módulos se
   delega a ``src/integration/``.
```

Update the `transpile_with_custom_layout(...)` docstring in the same file to:

```python
"""Transpila un circuito usando un layout inicial específico.

Este helper se usa para evaluar cómo se comporta un layout
suministrado por el llamador bajo las restricciones del backend y
del pipeline de transpilación de Qiskit. No implementa la integración
MO -> RL ni asume quién produjo el layout.

Decisión: la función delega en ``transpile_circuit(...)`` y conserva
el mismo contrato de ``initial_layout`` (mapeo lógico -> físico).
"""
```

Update `.worktrees/module-contract-cleanup/src/qiskit_interface/README.md` so the public API section reads:

```md
#### `transpile_circuit(...) -> TranspilationResult`
Transpila un circuito individual con control total de parámetros.
- **Entrada**:
    - `circuit`: Circuito cuántico.
    - `backend` / `backend_name`: Objetivo de hardware.
    - `optimization_level`: 0 (nada) a 3 (máximo).
    - `initial_layout`: Lista opcional de qubits físicos (para evaluar layouts externos con un contrato común).
    - `seed`: Semilla para reproducibilidad.
- **Salida**: `TranspilationResult` con métricas pre/post transpilación, tiempos y reducción de profundidad.

#### `transpile_with_custom_layout(circuit, layout, ...)`
Helper de evaluación local para transpilación con layout inicial explícito.
Acepta un layout en formato lógico -> físico y delega la orquestación entre módulos a `src/integration/`.
```

- [ ] **Step 4: Re-run the contract tests plus the Qiskit-interface suite**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py tests/test_qiskit_interface -q`

In: `.worktrees/module-contract-cleanup`

Expected: PASS with the new contract tests and the existing Qiskit-interface tests green.

- [ ] **Step 5: Commit the Qiskit-interface task**

```bash
git add tests/test_module_contracts.py src/qiskit_interface/transpiler.py src/qiskit_interface/README.md
git commit -m "docs: narrow qiskit interface layout contract"
```

### Task 3: MO and RL Generic Handoff Contract

**Files:**
- Modify: `.worktrees/module-contract-cleanup/tests/test_module_contracts.py`
- Modify: `.worktrees/module-contract-cleanup/src/mo_module/docs/internal_documentation.md`
- Modify: `.worktrees/module-contract-cleanup/src/rl_module/environment.py`
- Modify: `.worktrees/module-contract-cleanup/src/rl_module/docs/lookahead_frontier.md`
- Modify: `.worktrees/module-contract-cleanup/src/integration/__init__.py`

- [ ] **Step 1: Add failing tests for MO/RL wording and dependency boundaries**

Append to `.worktrees/module-contract-cleanup/tests/test_module_contracts.py`:

```python
def test_mo_docs_route_future_rl_consumption_through_integration():
    text = (ROOT / "src" / "mo_module" / "docs" / "internal_documentation.md").read_text(encoding="utf-8")
    assert "Salida hacia el módulo `rl_module`" not in text
    assert "consumibles por el módulo `integration`" in text


def test_rl_environment_reset_docstring_is_source_agnostic():
    text = (ROOT / "src" / "rl_module" / "environment.py").read_text(encoding="utf-8")
    assert "layout inicial externo" in text
    assert "desde el Módulo MO" not in text


def test_rl_frontier_docs_keep_initial_layout_generic():
    text = (ROOT / "src" / "rl_module" / "docs" / "lookahead_frontier.md").read_text(encoding="utf-8")
    assert "productor del `initial_layout` es externo al módulo" in text
    assert "MO -> RL pertenecerá a `src/integration/`" in text


def test_integration_stub_declares_handoff_ownership():
    text = (ROOT / "src" / "integration" / "__init__.py").read_text(encoding="utf-8")
    assert "único dueño del handoff MO -> RL" in text
    assert "stub" in text.lower()


def test_mo_module_has_no_direct_rl_imports():
    mo_py_files = sorted((ROOT / "src" / "mo_module").rglob("*.py"))
    offenders = []
    for path in mo_py_files:
        text = path.read_text(encoding="utf-8")
        if (
            "from src.rl_module" in text
            or "import src.rl_module" in text
            or "from ..rl_module" in text
        ):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
```

- [ ] **Step 2: Run the contract tests and verify the new assertions fail**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py -q`

In: `.worktrees/module-contract-cleanup`

Expected: FAIL on the MO/RL wording assertions while the direct-import guard stays green.

- [ ] **Step 3: Route future handoff ownership through `src/integration/`**

Replace the MO integration section in `.worktrees/module-contract-cleanup/src/mo_module/docs/internal_documentation.md` with:

```md
### Salida hacia el módulo `integration` (Módulo 4):
- Los layouts del frente de Pareto (`OptimizationResult.pareto_layouts`) son datos de salida consumibles por el módulo `integration`.
- `get_compromise_layout()` y `get_best_layout()` proporcionan un layout único para escenarios `MO_Only` y futuros flujos `MO+RL`.
- `OptimizationResult.to_dict()` genera datos tabulares para análisis.
- `compare_layouts()` permite comparar MO vs baselines (SABRE, trivial).
- `ParetoMetrics` proporciona indicadores de calidad para reportes.
```

Update the reset docstring and local comment in `.worktrees/module-contract-cleanup/src/rl_module/environment.py` to:

```python
"""
Reinicia el entorno.
Permite inyectar un `initial_layout` externo a través de `options`.
"""

# Ingesta genérica de layout inicial desde el llamador
if options and "initial_layout" in options:
```

Extend `.worktrees/module-contract-cleanup/src/rl_module/docs/lookahead_frontier.md` so the reset section includes:

```md
- Si se inyecta `initial_layout`, el entorno lo respeta exactamente.
- El productor del `initial_layout` es externo al módulo; el handoff MO -> RL pertenecerá a `src/integration/`.
- Si no se inyecta, el layout por defecto es determinista: `[0, 1, 2, ...]`.
```

Replace `.worktrees/module-contract-cleanup/src/integration/__init__.py` with:

```python
"""Módulo 4: Integración y experimentación.

Este paquete será el único dueño del handoff MO -> RL y de la
orquestación de escenarios `Baseline`, `MO_Only`, `RL_Only` y `MO+RL`.

Estado actual: stub. La implementación del pipeline todavía no forma
parte de este cambio.
"""

__all__: list[str] = []
```

- [ ] **Step 4: Re-run the contract tests plus the MO and RL suites**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py tests/test_mo_module tests/test_rl_module -q`

In: `.worktrees/module-contract-cleanup`

Expected: PASS with the contract tests, MO suite, and RL suite green.

- [ ] **Step 5: Commit the MO/RL contract task**

```bash
git add tests/test_module_contracts.py src/mo_module/docs/internal_documentation.md src/rl_module/environment.py src/rl_module/docs/lookahead_frontier.md src/integration/__init__.py
git commit -m "docs: route module handoff through integration"
```

### Task 4: Full Verification and Cleanup Check

**Files:**
- Verify only: `.worktrees/module-contract-cleanup/README.md`
- Verify only: `.worktrees/module-contract-cleanup/docs/agents.md`
- Verify only: `.worktrees/module-contract-cleanup/src/qiskit_interface/transpiler.py`
- Verify only: `.worktrees/module-contract-cleanup/src/qiskit_interface/README.md`
- Verify only: `.worktrees/module-contract-cleanup/src/mo_module/docs/internal_documentation.md`
- Verify only: `.worktrees/module-contract-cleanup/src/rl_module/environment.py`
- Verify only: `.worktrees/module-contract-cleanup/src/rl_module/docs/lookahead_frontier.md`
- Verify only: `.worktrees/module-contract-cleanup/src/integration/__init__.py`
- Verify only: `.worktrees/module-contract-cleanup/tests/test_module_contracts.py`

- [ ] **Step 1: Run the focused verification command for the entire cleanup surface**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py tests/test_qiskit_interface tests/test_mo_module tests/test_rl_module -q`

In: `.worktrees/module-contract-cleanup`

Expected: PASS with all four test areas green.

- [ ] **Step 2: Run the full repository test suite**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests -q`

In: `.worktrees/module-contract-cleanup`

Expected: PASS with the full test suite green.

- [ ] **Step 3: Inspect the worktree and confirm only intended files changed**

Run: `git status --short`

In: `.worktrees/module-contract-cleanup`

Expected: only the contract-cleanup files appear; no unrelated source edits and no generated artifacts left for commit.
