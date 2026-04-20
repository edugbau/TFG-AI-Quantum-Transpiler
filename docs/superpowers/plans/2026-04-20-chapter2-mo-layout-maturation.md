# Chapter 2 MO Layout Maturation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase 3 layout-campaign presets and adversarial reference layouts while reinforcing the MO-to-RL boundary through tests and MO-local documentation.

**Architecture:** Extend the existing `layout_campaigns.py` helper flow instead of introducing a new abstraction layer. Preset selection stays local to the benchmark tooling, while documentation and contract tests make explicit that `mo_module` only emits data consumable by `src/integration/` and never talks directly to `rl_module`.

**Tech Stack:** Python, pytest, PowerShell, existing MO benchmark helpers

---

### Task 1: Add failing benchmark tests for Phase 3 presets and references

**Files:**
- Modify: `tests/test_mo_module/test_benchmark.py`
- Test: `tests/test_mo_module/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_reference_layouts_exposes_phase3_reference_set(self, monkeypatch):
    refs = campaign_module.build_reference_layouts(3, 7)
    assert refs == {
        "trivial": [0, 1, 2],
        "reverse_trivial": [2, 1, 0],
        "high_index_block": [4, 5, 6],
        "heaviest_hex": [6, 5, 4],
    }


def test_build_layout_campaign_spec_supports_phase3_presets(self):
    quick = campaign_module.build_layout_campaign_spec(preset="quick")
    balanced = campaign_module.build_layout_campaign_spec(preset="balanced")
    thorough = campaign_module.build_layout_campaign_spec(preset="thorough")
    assert quick["reference_names"] == ["trivial", "heaviest_hex"]
    assert quick["mo_candidate_names"] == ["compromise"]
    assert balanced["reference_names"] == ["trivial", "heaviest_hex"]
    assert balanced["include_knee"] is True
    assert balanced["include_best_per_objective"] is True
    assert thorough["reference_names"] == [
        "trivial",
        "heaviest_hex",
        "reverse_trivial",
        "high_index_block",
    ]
    assert thorough["mo_candidate_names"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_mo_module/test_benchmark.py -q`
Expected: FAIL because the new preset/reference helpers are missing or do not match the requested contract.

- [ ] **Step 3: Write minimal implementation**

```python
def build_reference_layouts(num_qubits: int, backend_num_qubits: int) -> dict[str, list[int]]:
    ...


def build_layout_campaign_spec(*, preset: str = "balanced") -> dict[str, object]:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_mo_module/test_benchmark.py -q`
Expected: PASS for the new helper coverage and existing benchmark tests.

### Task 2: Add failing campaign preset-selection tests

**Files:**
- Modify: `tests/test_mo_module/test_benchmark.py`
- Test: `tests/test_mo_module/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_layout_selection_campaign_quick_uses_compromise_and_two_references(...):
    rows = campaign_fn(..., preset="quick")
    assert compare_calls == [{
        "compromise": [3, 4, 5],
        "trivial": [0, 1, 2],
        "heaviest_hex": [6, 5, 4],
    }]


def test_run_layout_selection_campaign_thorough_uses_full_candidate_and_reference_set(...):
    rows = campaign_fn(..., preset="thorough")
    assert compare_calls == [{
        "compromise": [3, 4, 5],
        "knee": [1, 2, 3],
        "best_depth": [2, 3, 4],
        "best_cnot_count": [0, 1, 2],
        "trivial": [0, 1, 2],
        "heaviest_hex": [6, 5, 4],
        "reverse_trivial": [2, 1, 0],
        "high_index_block": [4, 5, 6],
    }]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_mo_module/test_benchmark.py -q`
Expected: FAIL because `run_layout_selection_campaign(..., preset=...)` does not yet filter layouts by preset.

- [ ] **Step 3: Write minimal implementation**

```python
def _select_candidate_layouts(candidates: dict, spec: dict[str, object]) -> dict[str, list[int]]:
    ...


def run_layout_selection_campaign(..., preset: str = "balanced") -> list[dict]:
    spec = build_layout_campaign_spec(preset=preset)
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_mo_module/test_benchmark.py -q`
Expected: PASS with quick, balanced, and thorough preset behavior all green.

### Task 3: Add failing contract tests for MO/RL boundary wording

**Files:**
- Modify: `tests/test_module_contracts.py`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mo_docs_keep_mo_to_rl_handoff_exclusive_to_integration() -> None:
    internal_text = read_text("src/mo_module/docs/internal_documentation.md")
    benchmark_text = read_text("src/mo_module/docs/benchmark_documentation.md")
    assert_contains_all(
        internal_text,
        (
            "consumibles por el módulo `integration`",
            "no debe comunicarse directamente con `rl_module`",
            "handoff MO -> RL pertenece exclusivamente a `src/integration/`",
        ),
    )
    assert_contains_all(
        benchmark_text,
        (
            "tooling experimental local al módulo MO",
            "no actúan como puente de orquestación hacia `rl_module`",
            "src/integration/",
        ),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_module_contracts.py -q`
Expected: FAIL because the stronger boundary wording is not present yet.

- [ ] **Step 3: Write minimal implementation**

```markdown
- Los layouts y métricas MO son consumibles por `src/integration/`.
- `mo_module` no debe comunicarse directamente con `rl_module`.
- El handoff MO -> RL pertenece exclusivamente a `src/integration/`.
- Las layout campaigns son tooling experimental local al módulo MO, no un puente de orquestación.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_module_contracts.py -q`
Expected: PASS with import-boundary tests still green.

### Task 4: Focused verification and single commit

**Files:**
- Modify: `src/mo_module/benchmark/layout_campaigns.py`
- Modify: `tests/test_mo_module/test_benchmark.py`
- Modify: `tests/test_module_contracts.py`
- Modify: `src/mo_module/docs/internal_documentation.md`
- Modify: `src/mo_module/docs/benchmark_documentation.md`
- Modify: `docs/superpowers/plans/2026-04-20-chapter2-mo-layout-maturation.md`

- [ ] **Step 1: Run focused verification**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_mo_module/test_benchmark.py -q`
Expected: PASS

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_module_contracts.py -q`
Expected: PASS

- [ ] **Step 2: Self-review for minimality and contract safety**

Check that:
- only benchmark preset/reference behavior changed in `layout_campaigns.py`
- no new fitness objectives were added
- `OptimizationResult`, `get_compromise_layout()`, and `get_best_layout()` remain untouched
- no files under `src/integration/` or `src/rl_module/` changed
- docs keep MO -> RL handoff exclusive to `src/integration/`

- [ ] **Step 3: Check git status before commit**

Run: `git status --short`
Expected: source/docs/tests changed, `.pyc` files may appear but must remain unstaged.

- [ ] **Step 4: Create one commit for Phase 3**

Run: `git add docs/superpowers/plans/2026-04-20-chapter2-mo-layout-maturation.md src/mo_module/benchmark/layout_campaigns.py src/mo_module/docs/internal_documentation.md src/mo_module/docs/benchmark_documentation.md tests/test_mo_module/test_benchmark.py tests/test_module_contracts.py`

Run: `git commit -m "feat: add phase 3 layout campaign presets"`

Expected: one commit containing the full Phase 3 scope and no staged `.pyc` files.

## Self-Review Notes

- Spec coverage: the tasks cover preset helpers, preset-driven campaign filtering, boundary-doc wording, focused verification, and the single final commit.
- Placeholder scan: all tasks include concrete files, behaviors, and exact pytest invocations.
- Type consistency: the plan uses `build_reference_layouts`, `build_layout_campaign_spec`, and `run_layout_selection_campaign(..., preset=...)` consistently across tests and implementation.
