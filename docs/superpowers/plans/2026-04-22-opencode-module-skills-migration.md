# OpenCode Module Skills Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the five active project skills from `.github/skills/` to `.opencode/skills/`, rewrite them in an OpenCode-compatible superpowers format, and align live repository routing to the new canonical location.

**Architecture:** The migration keeps the five existing skill names and rewrites each `SKILL.md` under `.opencode/skills/<name>/`. `.github/AGENTS.md` remains the routing entry point by module and task type, while `tests/test_module_contracts.py` becomes the guardrail that enforces the new canonical skill location and rejects stale references to `.github/skills/`.

**Tech Stack:** Markdown skills, OpenCode skill discovery, YAML frontmatter, pytest documentation contract tests.

---

### Task 1: Lock The New Canonical Skill Location In Tests

**Files:**
- Modify: `tests/test_module_contracts.py`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

Add this test near the other workspace metadata checks in `tests/test_module_contracts.py`:

```python
def test_workspace_agents_reference_opencode_skills_as_canonical_location() -> None:
    workspace_agents_text = read_text(".github/AGENTS.md")

    assert_contains_all(
        workspace_agents_text,
        (
            ".opencode/skills/",
            "qiskit-2x-compliance",
            "mo-optimization",
            "rl-quantum-synthesis",
            "mo-rl-pipeline",
            "experimentation-logging",
        ),
    )
    assert_excludes_all(
        workspace_agents_text,
        (
            "[`./skills/`](./skills/)",
            ".github/skills/",
        ),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_module_contracts.py::test_workspace_agents_reference_opencode_skills_as_canonical_location -v`
Expected: FAIL because `.github/AGENTS.md` still points readers to `./skills/` and does not mention `.opencode/skills/`.

- [ ] **Step 3: Extend the same file-system contract with a second failing test**

Add this second test in `tests/test_module_contracts.py`:

```python
def test_opencode_skill_files_exist_with_opencode_frontmatter_contract() -> None:
    skill_roots = (
        ROOT / ".opencode/skills/mo-optimization/SKILL.md",
        ROOT / ".opencode/skills/rl-quantum-synthesis/SKILL.md",
        ROOT / ".opencode/skills/mo-rl-pipeline/SKILL.md",
        ROOT / ".opencode/skills/qiskit-2x-compliance/SKILL.md",
        ROOT / ".opencode/skills/experimentation-logging/SKILL.md",
    )

    for skill_file in skill_roots:
        text = skill_file.read_text(encoding="utf-8")
        assert text.startswith("---\nname: ")
        assert "description: Use when" in text
        assert "compatibility: opencode" in text
        assert "argument-hint:" not in text
        assert "user-invocable:" not in text
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_module_contracts.py::test_opencode_skill_files_exist_with_opencode_frontmatter_contract -v`
Expected: FAIL because `.opencode/skills/.../SKILL.md` does not exist yet.

- [ ] **Step 5: Commit**

```bash
git add tests/test_module_contracts.py
git commit -m "test: add OpenCode skill migration contracts"
```

### Task 2: Rewrite `qiskit-2x-compliance` Under `.opencode/skills/`

**Files:**
- Create: `.opencode/skills/qiskit-2x-compliance/SKILL.md`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

The failing test from Task 1 already covers this file. Keep it red and implement only this skill first.

- [ ] **Step 2: Write minimal implementation**

Create `.opencode/skills/qiskit-2x-compliance/SKILL.md` with this content:

```md
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

- Prefer `from qiskit import QuantumCircuit` and other top-level imports.
- Use `backend.run(...)` or V2 primitives when appropriate.
- Use `qiskit_aer` as a separate package instead of `qiskit.aer`.
- Use fake backends for local tests; do not target real hardware.
- Use `qiskit.qasm2` or `qiskit.qasm3` instead of `circuit.qasm()`.

## Implementation

### Allowed patterns

- `backend.run(circuit, **kwargs)`
- `SamplerV2` and `EstimatorV2` when a primitive-based flow is appropriate
- `assign_parameters()` in modern parameter binding flows
- `PassManager`-based transpilation pipelines when composing passes explicitly

### Forbidden patterns

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
```

- [ ] **Step 3: Run targeted test to verify partial progress**

Run: `pytest tests/test_module_contracts.py::test_opencode_skill_files_exist_with_opencode_frontmatter_contract -v`
Expected: FAIL, but only because the other four `.opencode/skills/.../SKILL.md` files do not exist yet.

- [ ] **Step 4: Commit**

```bash
git add .opencode/skills/qiskit-2x-compliance/SKILL.md
git commit -m "docs: migrate qiskit compatibility skill to OpenCode"
```

### Task 3: Rewrite `mo-optimization` Under `.opencode/skills/`

**Files:**
- Create: `.opencode/skills/mo-optimization/SKILL.md`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

Reuse the still-failing OpenCode frontmatter contract from Task 1.

- [ ] **Step 2: Write minimal implementation**

Create `.opencode/skills/mo-optimization/SKILL.md` with this content:

```md
---
name: mo-optimization
description: Use when modifying src/mo_module fitness, encoding, operators, Pareto analysis, or tuning workflows for quantum layout optimization.
compatibility: opencode
metadata:
  module: mo_module
  scope: project
---

# MO Optimization

## Overview

Guide changes inside `src/mo_module/` so layout optimization stays focused on the project's multi-objective contract: valid layout encodings, deterministic evolutionary search, and Pareto analysis over the active objectives.

## When to Use

- Editing `src/mo_module/` problem definitions, fitness evaluation, or operators.
- Adjusting Pareto metrics, Hypervolume analysis, or selection helpers.
- Touching Optuna tuning logic for MO search configuration.

## Quick Reference

- Keep the layout representation valid for the backend and circuit width.
- Treat `depth` and `cnot_equivalent` as the active optimization contract.
- Use `pymoo` primitives for problem modeling and search execution.
- Keep MO output consumable by `src/integration/`, not directly by `src/rl_module/`.

## Core Pattern

- Encode layouts as stable logical-to-physical assignments.
- Evaluate candidates through the shared transpilation-backed cost path.
- Extract and analyze non-dominated solutions with Pareto-aware helpers.
- Tune search parameters without changing module boundaries.

## Common Mistakes

- Treating `two_qubit_gates` as the optimization contract instead of `cnot_equivalent`.
- Letting `mo_module` own MO -> RL orchestration instead of `src/integration/`.
- Adding tuning knobs before the base encoding or operator semantics are stable.

## Project References

- `src/mo_module/docs/internal_documentation.md`
- `src/mo_module/docs/tuning.md`
- `src/mo_module/optimizer.py`
- `src/mo_module/pareto.py`
```

- [ ] **Step 3: Run targeted test to verify partial progress**

Run: `pytest tests/test_module_contracts.py::test_opencode_skill_files_exist_with_opencode_frontmatter_contract -v`
Expected: FAIL, but only because three `.opencode/skills/.../SKILL.md` files are still missing.

- [ ] **Step 4: Commit**

```bash
git add .opencode/skills/mo-optimization/SKILL.md
git commit -m "docs: migrate mo optimization skill to OpenCode"
```

### Task 4: Rewrite `rl-quantum-synthesis` Under `.opencode/skills/`

**Files:**
- Create: `.opencode/skills/rl-quantum-synthesis/SKILL.md`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

Reuse the still-failing OpenCode frontmatter contract from Task 1.

- [ ] **Step 2: Write minimal implementation**

Create `.opencode/skills/rl-quantum-synthesis/SKILL.md` with this content:

```md
---
name: rl-quantum-synthesis
description: Use when modifying src/rl_module environments, rewards, action or observation design, SB3 training flows, or checkpointed evaluation for routing or synthesis work.
compatibility: opencode
metadata:
  module: rl_module
  scope: project
---

# RL Quantum Synthesis

## Overview

Guide work inside `src/rl_module/` so Gymnasium environment semantics, reward shaping, training callbacks, and checkpointed evaluation stay aligned with the current routing-first project scope.

## When to Use

- Editing `src/rl_module/environment.py` or related environment helpers.
- Changing reward logic, action selection, or observation encoding.
- Updating Stable-Baselines3 training, evaluation, checkpoints, or TensorBoard logging.

## Quick Reference

- Implement environments with `gymnasium.Env`, not legacy `gym`.
- Keep `reset(seed=..., options=...)` compatible with external `initial_layout` injection.
- Return Gymnasium-standard tuples from `step()` and `reset()`.
- Use SB3 callbacks for evaluation, checkpoints, and training telemetry.

## Core Pattern

- Define clear observation and action spaces.
- Penalize invalid or redundant actions explicitly.
- Separate routing-capable behavior from limited synthesis-capable behavior in docs and code.
- Keep MO -> RL ownership in `src/integration/` even when `initial_layout` is provided.

## Common Mistakes

- Reintroducing legacy `gym` patterns or wrong step/reset signatures.
- Assuming `initial_layout` is owned by `mo_module` instead of by the caller or `src/integration/`.
- Expanding docs to imply full synthesis support when only limited flows are implemented.

## Project References

- `src/rl_module/docs/internal_documentation.md`
- `src/rl_module/docs/synthesis_mode_status.md`
- `src/rl_module/environment.py`
- `src/rl_module/training.py`
```

- [ ] **Step 3: Run targeted test to verify partial progress**

Run: `pytest tests/test_module_contracts.py::test_opencode_skill_files_exist_with_opencode_frontmatter_contract -v`
Expected: FAIL, but only because two `.opencode/skills/.../SKILL.md` files are still missing.

- [ ] **Step 4: Commit**

```bash
git add .opencode/skills/rl-quantum-synthesis/SKILL.md
git commit -m "docs: migrate rl synthesis skill to OpenCode"
```

### Task 5: Rewrite `mo-rl-pipeline` Under `.opencode/skills/`

**Files:**
- Create: `.opencode/skills/mo-rl-pipeline/SKILL.md`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

Reuse the still-failing OpenCode frontmatter contract from Task 1.

- [ ] **Step 2: Write minimal implementation**

Create `.opencode/skills/mo-rl-pipeline/SKILL.md` with this content:

```md
---
name: mo-rl-pipeline
description: Use when modifying src/integration handoff logic, benchmark scenarios, layout ownership rules, or routing evaluation flows that connect MO outputs to RL inputs.
compatibility: opencode
metadata:
  module: integration
  scope: project
---

# MO RL Pipeline

## Overview

Keep MO -> RL orchestration inside `src/integration/` so layout handoff, benchmark scenarios, and routing evaluation remain explicit, testable, and decoupled from the internals of `mo_module` and `rl_module`.

## When to Use

- Editing `src/integration/` orchestration or benchmark scenarios.
- Touching the `initial_layout` handoff contract.
- Comparing `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL` scenarios.

## Quick Reference

- `src/integration/` owns MO -> RL orchestration.
- `mo_module` produces layouts; it does not call `rl_module` directly.
- `rl_module` may consume `initial_layout`, but it does not own the handoff protocol.
- Keep the shared layout convention explicit in docs and code.

## Implementation

- Accept layouts through caller-controlled interfaces such as `env.reset(options={"initial_layout": layout})`.
- Preserve scenario naming and comparability across `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL`.
- Document whether a layout is logical-to-physical or the inverse; do not leave the mapping ambiguous.

## Common Mistakes

- Moving orchestration logic into `mo_module` or `rl_module`.
- Referring to direct MO -> RL imports or ownership across module boundaries.
- Treating benchmark scenarios as interchangeable when they test different ownership paths.

## Project References

- `src/integration/docs/internal_documentation.md`
- `src/integration/README.md`
- `src/rl_module/docs/synthesis_mode_status.md`
- `src/mo_module/docs/internal_documentation.md`
```

- [ ] **Step 3: Run targeted test to verify partial progress**

Run: `pytest tests/test_module_contracts.py::test_opencode_skill_files_exist_with_opencode_frontmatter_contract -v`
Expected: FAIL, but only because one `.opencode/skills/.../SKILL.md` file is still missing.

- [ ] **Step 4: Commit**

```bash
git add .opencode/skills/mo-rl-pipeline/SKILL.md
git commit -m "docs: migrate mo rl pipeline skill to OpenCode"
```

### Task 6: Rewrite `experimentation-logging` Under `.opencode/skills/`

**Files:**
- Create: `.opencode/skills/experimentation-logging/SKILL.md`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

Reuse the still-failing OpenCode frontmatter contract from Task 1.

- [ ] **Step 2: Write minimal implementation**

Create `.opencode/skills/experimentation-logging/SKILL.md` with this content:

```md
---
name: experimentation-logging
description: Use when running benchmarks, exporting experiment tables, setting deterministic seeds, or coordinating TensorBoard and plotting across MO, RL, or integration workflows.
compatibility: opencode
metadata:
  module: cross-cutting
  scope: project
---

# Experimentation Logging

## Overview

Standardize reproducibility and comparison across experiments by keeping seeds, training telemetry, tabular exports, and plots consistent across project modules.

## When to Use

- Running benchmark suites or comparative experiments.
- Logging Stable-Baselines3 training or evaluation.
- Exporting experiment outputs for analysis, reporting, or reproducibility.

## Quick Reference

- Seed `random`, `numpy`, and `torch` before reproducible runs.
- Use `env.reset(seed=...)` for Gymnasium environments.
- Keep tabular exports consistent across runs.
- Use TensorBoard and checkpoints for long-running RL experiments.

## Implementation

- Set global seeds before training, optimization, or benchmarking starts.
- Export tables with stable columns such as `Circuit`, `Method`, `Depth`, `CNOTs`, `Fidelity/Error`, `Execution_Time`, and `Seed`.
- Use clear labels, legends, and units for Pareto plots and learning curves.

## Common Mistakes

- Setting seeds after work has already started.
- Exporting tables with drifting column names between modules.
- Logging RL metrics without checkpoints or reproducible metadata.

## Project References

- `docs/ENVIRONMENT.md`
- `src/rl_module/training.py`
- `src/mo_module/benchmark/runner.py`
- `experiments/`
```

- [ ] **Step 3: Run targeted test to verify it passes**

Run: `pytest tests/test_module_contracts.py::test_opencode_skill_files_exist_with_opencode_frontmatter_contract -v`
Expected: PASS because all five OpenCode skill files now exist with allowed frontmatter and without legacy fields.

- [ ] **Step 4: Commit**

```bash
git add .opencode/skills/experimentation-logging/SKILL.md
git commit -m "docs: migrate experimentation logging skill to OpenCode"
```

### Task 7: Align `.github/AGENTS.md` With The OpenCode Skill Location

**Files:**
- Modify: `.github/AGENTS.md`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

The first failing test from Task 1 already covers this file.

- [ ] **Step 2: Write minimal implementation**

Change the final line in `.github/AGENTS.md` from:

```md
For in-depth guidance, see the project skills in [`./skills/`](./skills/).
```

to:

```md
For in-depth guidance, see the project skills in [`.opencode/skills/`](../.opencode/skills/).
```

Do not rename any of the five skill references in the routing bullets above.

- [ ] **Step 3: Run targeted test to verify it passes**

Run: `pytest tests/test_module_contracts.py::test_workspace_agents_reference_opencode_skills_as_canonical_location -v`
Expected: PASS because `.github/AGENTS.md` now points to `.opencode/skills/` and no longer presents the old location as canonical.

- [ ] **Step 4: Commit**

```bash
git add .github/AGENTS.md
git commit -m "docs: point workspace routing to OpenCode skills"
```

### Task 8: Remove The Old `.github/skills/` Copies

**Files:**
- Delete: `.github/skills/mo-optimization/SKILL.md`
- Delete: `.github/skills/rl-quantum-synthesis/SKILL.md`
- Delete: `.github/skills/mo-rl-pipeline/SKILL.md`
- Delete: `.github/skills/qiskit-2x-compliance/SKILL.md`
- Delete: `.github/skills/experimentation-logging/SKILL.md`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_module_contracts.py`:

```python
def test_legacy_github_skill_copies_are_not_kept_as_active_project_skills() -> None:
    legacy_skill_files = (
        ROOT / ".github/skills/mo-optimization/SKILL.md",
        ROOT / ".github/skills/rl-quantum-synthesis/SKILL.md",
        ROOT / ".github/skills/mo-rl-pipeline/SKILL.md",
        ROOT / ".github/skills/qiskit-2x-compliance/SKILL.md",
        ROOT / ".github/skills/experimentation-logging/SKILL.md",
    )

    for legacy_file in legacy_skill_files:
        assert not legacy_file.exists(), f"legacy skill copy still exists: {legacy_file}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_module_contracts.py::test_legacy_github_skill_copies_are_not_kept_as_active_project_skills -v`
Expected: FAIL because the `.github/skills/.../SKILL.md` files still exist.

- [ ] **Step 3: Write minimal implementation**

Delete these files:

```text
.github/skills/mo-optimization/SKILL.md
.github/skills/rl-quantum-synthesis/SKILL.md
.github/skills/mo-rl-pipeline/SKILL.md
.github/skills/qiskit-2x-compliance/SKILL.md
.github/skills/experimentation-logging/SKILL.md
```

- [ ] **Step 4: Run targeted test to verify it passes**

Run: `pytest tests/test_module_contracts.py::test_legacy_github_skill_copies_are_not_kept_as_active_project_skills -v`
Expected: PASS because no legacy skill copies remain under `.github/skills/`.

- [ ] **Step 5: Commit**

```bash
git add .github/skills .
git commit -m "docs: remove legacy github skill copies"
```

### Task 9: Run Full Migration Verification

**Files:**
- Modify: none unless a precise doc-contract assertion needs correction
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Run the focused contract suite**

Run: `pytest tests/test_module_contracts.py -v`
Expected: PASS, including the four-module ownership checks plus the new OpenCode skill migration checks.

- [ ] **Step 2: If a precise assertion fails, apply the minimal fix**

Allowed fixes are limited to:

```text
tests/test_module_contracts.py
.github/AGENTS.md
.opencode/skills/*/SKILL.md
```

Do not broaden scope beyond the migration contract.

- [ ] **Step 3: Re-run the full contract suite**

Run: `pytest tests/test_module_contracts.py -v`
Expected: PASS with no skipped migration checks.

- [ ] **Step 4: Commit**

```bash
git add tests/test_module_contracts.py .github/AGENTS.md .opencode/skills
git commit -m "test: verify OpenCode skill migration contracts"
```

### Task 10: Final Repository Snapshot

**Files:**
- Modify: `docs/superpowers/specs/2026-04-22-opencode-module-skills-design.md` only if a one-line note is required during execution
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Verify the final file map matches the approved design**

The final repository state should include exactly these active skill files:

```text
.opencode/skills/mo-optimization/SKILL.md
.opencode/skills/rl-quantum-synthesis/SKILL.md
.opencode/skills/mo-rl-pipeline/SKILL.md
.opencode/skills/qiskit-2x-compliance/SKILL.md
.opencode/skills/experimentation-logging/SKILL.md
```

and should no longer include the legacy `.github/skills/*/SKILL.md` copies.

- [ ] **Step 2: Run the final verification command**

Run: `pytest tests/test_module_contracts.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add .opencode/skills .github/AGENTS.md tests/test_module_contracts.py docs/superpowers/specs/2026-04-22-opencode-module-skills-design.md docs/superpowers/plans/2026-04-22-opencode-module-skills-migration.md
git commit -m "docs: migrate project skills to OpenCode"
```
