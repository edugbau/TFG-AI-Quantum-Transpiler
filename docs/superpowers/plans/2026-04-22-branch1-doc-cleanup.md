# Branch 1 Doc Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `docs/agents.md`, make `README.md` the canonical architecture reference, and close Chapter 3 Branch 1 if the remaining active documentation matches the implemented routing-contract scope.

**Architecture:** Keep all production behavior unchanged. This plan only updates canonical documentation, active repository instructions, and the tests that encode the old documentation location or stale public wording. Branch 1 closes only if the active docs consistently describe integration v1 as the owner of routing scenario orchestration with RL episode summaries and metadata-sidecar fallback.

**Tech Stack:** Markdown documentation, Python `pytest`, repository doc-contract tests

---

## File Map

- Modify: `README.md`
  - become the single canonical architecture and module-contract reference.
- Modify: `.github/AGENTS.md`
  - stop pointing to `docs/agents.md`; point to `README.md` for live architecture context.
- Modify: `src/rl_module/docs/lookahead_frontier.md`
  - remove stale placeholder-only wording for current synthesis status if it is still treated as an active user-facing RL doc.
- Modify: `.github/skills/mo-rl-pipeline/SKILL.md`
  - stop referencing removed `docs/agents.md`.
- Modify: `.github/skills/rl-quantum-synthesis/SKILL.md`
  - stop referencing removed `docs/agents.md`.
- Modify: `.github/skills/qiskit-2x-compliance/SKILL.md`
  - stop referencing removed `docs/agents.md`.
- Modify: `.github/skills/experimentation-logging/SKILL.md`
  - stop referencing removed `docs/agents.md`.
- Modify: `.github/skills/mo-optimization/SKILL.md`
  - stop referencing removed `docs/agents.md`.
- Delete: `docs/agents.md`
  - remove the obsolete duplicate architecture document.
- Modify: `tests/test_module_contracts.py`
  - change the canonical-doc assertions from `docs/agents.md` to `README.md` and keep the four-module ownership checks alive.
- Test: `tests/test_module_contracts.py`
- Test: `tests/test_integration/test_docs.py`

### Task 1: Update canonical architecture docs

**Files:**
- Modify: `README.md`
- Modify: `.github/AGENTS.md`
- Modify: `src/rl_module/docs/lookahead_frontier.md`

- [ ] **Step 1: Write the failing test for canonical doc location and branch-1 wording**

Edit `tests/test_module_contracts.py` so `test_docs_and_workspace_metadata_define_four_module_ownership()` asserts the canonical content directly in `README.md` instead of reading `docs/agents.md`, and so the test no longer expects `[agents.md](docs/agents.md)`.

Use this target shape inside the test:

```python
def test_docs_and_workspace_metadata_define_four_module_ownership() -> None:
    workspace_agents_text = read_text(".github/AGENTS.md")
    readme_text = read_text("README.md")

    assert_contains_all(
        readme_text,
        (
            "src/qiskit_interface/",
            "src/rl_module/",
            "src/mo_module/",
            "src/integration/",
            "layout[i] = physical_qubit_for_logical_qubit_i",
            "MO+RL",
        ),
    )
    assert "integration" in readme_text
    assert_any_contains(
        readme_text,
        (
            "owns the process that connects producer and consumer",
            "owns the process",
        ),
    )
    assert_contains_all(
        workspace_agents_text,
        (
            "The project has 4 interconnected modules:",
            "src/integration/",
            "benchmark scenarios",
        ),
    )
    assert_contains_all(workspace_agents_text, ("Module boundaries are respected", "integration owns"))
    assert_contains_all(workspace_agents_text, ("orchestration", "handoff scenarios"))
    assert_contains_all(
        readme_text,
        (
            "Proyecto de transpilación cuántica organizado en cuatro módulos",
            "MO y RL evolucionan como módulos separados",
            "MO+RL",
        ),
    )
    assert_contains_all(readme_text, ("integration`", "handoff", "orquestación"))
```

- [ ] **Step 2: Run the doc-contract test and verify it fails**

Run: `pytest tests/test_module_contracts.py::test_docs_and_workspace_metadata_define_four_module_ownership -v`

Expected: FAIL because `README.md` does not yet contain the full canonical contract content and/or because the test still depends on `docs/agents.md`.

- [ ] **Step 3: Add the canonical architecture section to `README.md`**

Update `README.md` to contain the four-module ownership table and shared layout convention directly, keeping the current integration-v1 wording already present.

Insert content equivalent to this near the architecture overview section:

```md
## Contratos entre módulos

| Módulo | Responsabilidad | No es responsable de |
| --- | --- | --- |
| `src/qiskit_interface/` | Backends, transpilación, métricas y baselines | Orquestación MO -> RL |
| `src/rl_module/` | Entorno Gymnasium, recompensas, entrenamiento e ingesta genérica de `initial_layout` | Producir layouts o poseer la orquestación experimental |
| `src/mo_module/` | Búsqueda multiobjetivo de layouts, frentes de Pareto y evaluación de layouts | Invocar RL directamente |
| `src/integration/` | Orquestación de `Baseline`, `MO_Only`, `RL_Only` y `MO+RL` | Reimplementar internals de los módulos |

### Convención compartida de layout

```python
layout[i] = physical_qubit_for_logical_qubit_i
```

- `qiskit_interface` puede evaluar este layout mediante helpers de transpilación.
- `rl_module` puede consumirlo mediante `env.reset(options={"initial_layout": layout})`.
- `integration` posee el proceso que conecta productor y consumidor.

### Estado actual resumido

- `src/integration/` ya no es un stub: posee la orquestación v1 de routing para `Baseline`, `MO_Only`, `RL_Only` y `MO+RL`.
- `src/rl_module/` soporta routing y también un primer alcance funcional de `synthesis` limitado a Clifford con layout fijo.
- Los escenarios basados en RL siguen devolviendo `episode summaries`, no `final circuits`.
```

- [ ] **Step 4: Update `.github/AGENTS.md` to point at `README.md`**

Replace the two `docs/agents.md` references with `README.md` and keep the instruction meaning intact.

Use edits equivalent to:

```md
- **Qiskit Compatibility**: Enforce Qiskit >= 2.0. Load the `qiskit-2x-compliance` skill for any Qiskit code changes. See [README.md](../README.md) for the live module contracts and current project architecture.
...
See [README.md](../README.md) for detailed module descriptions and current module boundaries.
```

- [ ] **Step 5: Align `src/rl_module/docs/lookahead_frontier.md` with the real synthesis status**

Replace the stale placeholder-only limitation block with wording that matches the current docs in `src/rl_module/docs/synthesis_mode_status.md`.

Use replacement text equivalent to:

```md
## Limitación actual

El modo `synthesis` ya no es un placeholder puro, pero sigue siendo una primera fase acotada.
Actualmente la infraestructura de observación/frontier se comparte con routing, mientras que `synthesis` se apoya en un flujo residual-céntrico limitado a circuitos Clifford con layout fijo.

La expansión a síntesis no-Clifford, `swap` dinámico dentro del episodio y criterios de equivalencia más generales queda para iteraciones futuras.
```

- [ ] **Step 6: Run the updated doc-contract test and verify it passes**

Run: `pytest tests/test_module_contracts.py::test_docs_and_workspace_metadata_define_four_module_ownership -v`

Expected: PASS

- [ ] **Step 7: Run the RL doc-contract test to verify the active RL docs remain aligned**

Run: `pytest tests/test_module_contracts.py::test_rl_docs_and_reset_contract_keep_initial_layout_generic -v`

Expected: PASS

### Task 2: Remove `docs/agents.md` and repair live references

**Files:**
- Delete: `docs/agents.md`
- Modify: `.github/skills/mo-rl-pipeline/SKILL.md`
- Modify: `.github/skills/rl-quantum-synthesis/SKILL.md`
- Modify: `.github/skills/qiskit-2x-compliance/SKILL.md`
- Modify: `.github/skills/experimentation-logging/SKILL.md`
- Modify: `.github/skills/mo-optimization/SKILL.md`
- Modify: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing cleanup assertions**

Extend `tests/test_module_contracts.py` so the canonical-doc test explicitly fails if `docs/agents.md` still exists and so no assertion mentions `[agents.md](docs/agents.md)`.

Use this target shape in the existing test:

```python
def test_docs_and_workspace_metadata_define_four_module_ownership() -> None:
    assert not (ROOT / "docs" / "agents.md").exists()
    workspace_agents_text = read_text(".github/AGENTS.md")
    readme_text = read_text("README.md")

    assert "docs/agents.md" not in workspace_agents_text
    assert "docs/agents.md" not in readme_text
```

- [ ] **Step 2: Run the focused test and verify it fails before deletion**

Run: `pytest tests/test_module_contracts.py::test_docs_and_workspace_metadata_define_four_module_ownership -v`

Expected: FAIL because `docs/agents.md` still exists and live references still mention it.

- [ ] **Step 3: Update active skill docs that still point to `docs/agents.md`**

Replace the `docs/agents.md` project reference in these files with `README.md`:

```md
- `README.md`
```

Apply that replacement in:

- `.github/skills/mo-rl-pipeline/SKILL.md`
- `.github/skills/rl-quantum-synthesis/SKILL.md`
- `.github/skills/qiskit-2x-compliance/SKILL.md`
- `.github/skills/experimentation-logging/SKILL.md`
- `.github/skills/mo-optimization/SKILL.md`

- [ ] **Step 4: Delete `docs/agents.md`**

Remove the obsolete file entirely.

Expected git diff shape:

```diff
- docs/agents.md
```

- [ ] **Step 5: Run the focused cleanup test and verify it passes**

Run: `pytest tests/test_module_contracts.py::test_docs_and_workspace_metadata_define_four_module_ownership -v`

Expected: PASS

### Task 3: Verify Branch 1 closure against active docs

**Files:**
- Modify: `README.md` if any final branch-1 wording still needs tightening
- Modify: `src/integration/README.md` only if an active inconsistency is found
- Modify: `tests/test_module_contracts.py` only if a precise assertion is needed for branch-1 closure
- Test: `tests/test_module_contracts.py`
- Test: `tests/test_integration/test_docs.py`

- [ ] **Step 1: Add one focused assertion for branch-1 closure if it is missing**

If `tests/test_module_contracts.py` does not already force the canonical README to state the integration-v1 limitation clearly, extend the canonical-doc test with these checks:

```python
assert_contains_all(
    readme_text,
    (
        "run_metadata.json",
        "episode summaries",
        "not final circuits",
        "integration v1",
    ),
)
assert "stub" not in readme_text.lower()
```

- [ ] **Step 2: Run the branch-1 documentation tests**

Run: `pytest tests/test_module_contracts.py tests/test_integration/test_docs.py -v`

Expected: all selected tests PASS

- [ ] **Step 3: Do one final repository search for stale live references**

Run: `rg "docs/agents\.md|integration.*stub|synthesis.*placeholder" README.md .github src tests`

Expected:
- no live reference to `docs/agents.md` in active docs/tests
- no active canonical document claiming `src/integration/` is a stub
- no active canonical document claiming current `synthesis` is pure placeholder work

- [ ] **Step 4: Commit the cleanup**

Run:

```bash
git add README.md .github/AGENTS.md .github/skills/mo-rl-pipeline/SKILL.md .github/skills/rl-quantum-synthesis/SKILL.md .github/skills/qiskit-2x-compliance/SKILL.md .github/skills/experimentation-logging/SKILL.md .github/skills/mo-optimization/SKILL.md src/rl_module/docs/lookahead_frontier.md tests/test_module_contracts.py docs/superpowers/specs/2026-04-22-branch1-close-and-branch2-design.md docs/superpowers/plans/2026-04-22-branch1-doc-cleanup.md
git add -u docs/agents.md
git commit -m "docs: close branch 1 canonical contract cleanup"
```

Expected: commit succeeds with only documentation/test cleanup changes.

## Self-Review Checklist

- The plan updates the canonical architecture location from `docs/agents.md` to `README.md`.
- The plan removes the obsolete file instead of leaving a redirect stub.
- The plan accounts for active skill references, not just root docs.
- The plan includes the stale RL frontier doc because it still presents current behavior and would otherwise block branch-1 closure.
- The verification section distinguishes active docs from historical specs and plans.
