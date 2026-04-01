# RL GUI Episode Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the RL GUI into one shell with routing/synthesis subviews and replace text-only evaluation with a structured episode inspector that makes actions and executed gates visually understandable.

**Architecture:** Keep `RLBenchmarkGUI` as the shared application shell for training, evaluation, plots, and threading. Add mode-specific subviews for `routing` and `synthesis`, then build a shared episode inspector around structured step records populated from new environment/frontier evaluation traces.

**Tech Stack:** Python 3.10, CustomTkinter, `tkinter.ttk`, matplotlib, pytest, Gymnasium, Stable-Baselines3, Qiskit 2.3.0.

---

## File Map

- Create: `src/rl_module/gui/routing_view.py` - routing-specific controls and evaluation-detail rendering.
- Create: `src/rl_module/gui/synthesis_view.py` - synthesis-specific controls and evaluation-detail rendering.
- Create: `src/rl_module/gui/evaluation_panel.py` - structured episode table, summary widgets, detail panel, and step-record dataclasses.
- Create: `tests/test_rl_module/test_rl_gui.py` - GUI tests focused on mode views and the structured episode inspector.
- Modify: `src/rl_module/gui/rl_gui.py` - shared shell, subview switching, structured evaluation capture, and inspector integration.
- Modify: `src/rl_module/environment.py` - expose GUI-facing evaluation trace fields and visible-frontier snapshots.
- Modify: `src/rl_module/frontier.py` - optionally capture executed gate traces during cascade execution.
- Modify: `tests/test_rl_module/test_frontier.py` - coverage for executed gate traces.
- Modify: `tests/test_rl_module/test_rl_module.py` - preserve and extend evaluation-thread coverage for the new structured model.
- Modify: `src/rl_module/docs/internal_documentation.md` - document the shared shell, subviews, and inspector.
- Modify: `src/rl_module/docs/synthesis_mode_status.md` - explain residual-centric synthesis visualization.

### Task 1: Evaluation Trace Plumbing

**Files:**
- Modify: `src/rl_module/frontier.py`
- Modify: `src/rl_module/environment.py`
- Modify: `tests/test_rl_module/test_frontier.py`
- Modify: `tests/test_rl_module/test_rl_module.py`

- [ ] **Step 1: Write the failing trace-plumbing tests**

Add tests that assert:

- `SequentialFrontier.execute_ready_cascade(...)` can optionally append executed `GateTuple` values to an external list;
- `DagFrontier.execute_ready_cascade(...)` can optionally append executed `GateTuple` values to an external list;
- `QuantumTranspilationEnv.step(...)` in `routing` exposes `swap_edge` and `executed_gates` in `info`;
- `QuantumTranspilationEnv.step(...)` in `synthesis` exposes `primitive_name`, `primitive_physical_qargs`, `primitive_cost`, and residual-distance fields in `info`.

- [ ] **Step 2: Run the focused tests and verify they fail first**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_frontier.py tests/test_rl_module/test_rl_module.py -v`

Expected: FAIL because executed gate traces and the new `info` fields do not exist yet.

- [ ] **Step 3: Implement the minimal trace plumbing**

Implement these changes:

- add an optional `executed_trace` output list parameter to both frontier implementations without changing their integer return value;
- thread that optional trace through `environment.py` when routing executes a cascade;
- populate `info["swap_edge"]`, `info["executed_gates"]`, and `info["primitive_physical_qargs"]`;
- add a read-oriented helper on the environment that returns the current visible frontier entries for GUI rendering.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_frontier.py tests/test_rl_module/test_rl_module.py -v`

Expected: PASS for the new tests and no regressions in existing RL module coverage.

### Task 2: Split The GUI Into Shared Shell Plus Mode Views

**Files:**
- Modify: `src/rl_module/gui/rl_gui.py`
- Create: `src/rl_module/gui/routing_view.py`
- Create: `src/rl_module/gui/synthesis_view.py`
- Create: `tests/test_rl_module/test_rl_gui.py`

- [ ] **Step 1: Write the failing GUI-mode tests**

Add tests that assert:

- routing-only controls are shown in `routing` mode;
- synthesis-only controls are shown in `synthesis` mode;
- `_get_config()` keeps using `basis_gates` for synthesis while preserving shared training fields;
- synthesis does not keep routing-only wording as visible primary controls.

- [ ] **Step 2: Run the GUI-mode tests and verify they fail first**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_gui.py -v`

Expected: FAIL because the subviews and new configuration boundaries do not exist yet.

- [ ] **Step 3: Implement the shared shell and the two subviews**

Implement these changes:

- keep `RLBenchmarkGUI` as the window shell and the owner of training/evaluation threads;
- move routing-specific widgets into `RoutingView`;
- move synthesis-specific widgets into `SynthesisView`;
- switch the active view from the selected mode while keeping one app window and one entrypoint.

- [ ] **Step 4: Re-run the GUI-mode tests and verify they pass**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_gui.py -v`

Expected: PASS.

### Task 3: Replace Text-Only Evaluation With Structured Step Records

**Files:**
- Modify: `src/rl_module/gui/rl_gui.py`
- Create: `src/rl_module/gui/evaluation_panel.py`
- Modify: `tests/test_rl_module/test_rl_gui.py`
- Modify: `tests/test_rl_module/test_rl_module.py`

- [ ] **Step 1: Write the failing structured-evaluation tests**

Add tests that assert `_evaluation_thread()` now builds structured step records containing:

- `step`, `reward`, `action_type`, `is_valid_action`;
- `layout_before`, `layout_after`;
- `visible_frontier_before` for routing;
- `executed_gates`, `swap_edge`, `routing_progress_delta`, `repeated_layout`, `undo_swap` for routing;
- `primitive_name`, `primitive_physical_qargs`, `primitive_cost`, `residual_distance_before`, `residual_distance_after`, `residual_distance_delta` for synthesis.

- [ ] **Step 2: Run the structured-evaluation tests and verify they fail first**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_gui.py tests/test_rl_module/test_rl_module.py -v`

Expected: FAIL because evaluation still writes primarily to a text box and does not populate a rich structured model.

- [ ] **Step 3: Implement the structured episode model and panel integration**

Implement these changes:

- add step-record dataclasses in `evaluation_panel.py`;
- keep `self._eval_log` as a list of structured records instead of loose dict placeholders;
- capture before/after snapshots inside `_evaluation_thread()`;
- keep text output only for headings, summaries, and errors;
- render the episode as a selectable table with a detail pane.

- [ ] **Step 4: Re-run the structured-evaluation tests and verify they pass**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_gui.py tests/test_rl_module/test_rl_module.py -v`

Expected: PASS and preservation of deterministic-policy and model-selection behavior.

### Task 4: Add Mode-Specific Detail Rendering

**Files:**
- Modify: `src/rl_module/gui/evaluation_panel.py`
- Modify: `src/rl_module/gui/routing_view.py`
- Modify: `src/rl_module/gui/synthesis_view.py`
- Modify: `tests/test_rl_module/test_rl_gui.py`

- [ ] **Step 1: Write the failing detail-render tests**

Add tests that assert:

- routing details show `swap_edge`, `executed_gates`, visible frontier entries, and oscillation flags;
- synthesis details show primitive name, physical qargs, primitive cost, and residual progression;
- synthesis summaries avoid routing-only labels such as `SWAPs insertados` as the primary metric.

- [ ] **Step 2: Run the detail-render tests and verify they fail first**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_gui.py -v`

Expected: FAIL because the current evaluation view is not mode-aware beyond textual branching.

- [ ] **Step 3: Implement the mode-specific renderers**

Implement these changes:

- render routing rows and detail summaries around SWAP effect and executed gates;
- render synthesis rows and detail summaries around primitive application and residual change;
- keep the first version lightweight and table-oriented rather than building a full DAG viewer.

- [ ] **Step 4: Re-run the detail-render tests and verify they pass**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_rl_gui.py -v`

Expected: PASS.

### Task 5: Documentation And RL Verification

**Files:**
- Modify: `src/rl_module/docs/internal_documentation.md`
- Modify: `src/rl_module/docs/synthesis_mode_status.md`
- Modify: `tests/test_rl_module/test_frontier.py`
- Modify: `tests/test_rl_module/test_rl_gui.py`
- Modify: `tests/test_rl_module/test_rl_module.py`

- [ ] **Step 1: Update the RL module documentation**

Document:

- one app with two specialized mode views;
- routing inspector semantics around frontier, layout, SWAPs, and executed gates;
- synthesis inspector semantics around primitives and residual progress.

- [ ] **Step 2: Run the focused RL verification suite**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module/test_frontier.py tests/test_rl_module/test_rl_gui.py tests/test_rl_module/test_rl_module.py -v`

Expected: PASS.

- [ ] **Step 3: Run a broader RL smoke pass**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m pytest tests/test_rl_module -v`

Expected: PASS.

- [ ] **Step 4: Run the GUI entrypoint manually**

Run: `& "C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe" -m src.rl_module.gui.rl_gui`

Verify manually:

- the mode switch changes the active view;
- routing shows frontier-related controls and detail labels;
- synthesis shows basis-profile controls and residual-centric labels;
- episode evaluation renders a selectable inspector rather than only a text log.

## Execution Rules

- Do not split the app into two standalone GUI entrypoints.
- Do not change the RL observation contract unless strictly required for GUI rendering.
- Preserve deterministic evaluation and current model-loading behavior.
- Prefer the smallest correct UI split that removes cross-mode confusion.
- Use TDD for each task: write the focused failing tests first, verify the red state, then implement the minimum code to turn the suite green.
