# Campaign MO-Conditioned Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make Campaign `MO+RL` train RL from the exact MO-selected layout for each Campaign Case, while leaving `RL_Only`, the unitary scenario CLI, and non-Campaign training behavior unchanged.

**Architecture:** keep the MO -> RL ownership in `src/integration/` by extracting the layout from `MO_Only` inside the Campaign runner and forwarding it through the Campaign training bridge into `src/rl_module.training.setup_training_pipeline(...)` as an optional `initial_layout`. The new behavior is mandatory for Campaign hybrid cases but remains opt-in at the RL training API level so existing call sites keep their current semantics.

**Tech Stack:** Python 3.10, pytest, dataclasses, pathlib, Stable-Baselines3, Gymnasium, Qiskit 2.x.

---

## File Map

- Modify: `src/rl_module/training.py` - accept optional `initial_layout` and apply it when resetting train/eval environments.
- Modify: `src/integration/training_bridge.py` - accept and forward Campaign-provided `initial_layout` to the RL training pipeline.
- Modify: `src/integration/campaign_runner.py` - require the `MO_Only` layout for Campaign hybrid training and pass it into the bridge.
- Modify: `src/integration/README.md` - document that Campaign `MO+RL` now uses MO-conditioned training.
- Modify: `src/integration/docs/internal_documentation.md` - update ownership and pipeline narrative for Campaign hybrid training.
- Modify: `README.md` - keep top-level integration description aligned with the new Campaign semantics.
- Test: `tests/test_integration/test_training_bridge.py`
- Test: `tests/test_integration/test_campaign_runner.py`
- Test: `tests/test_rl_module/test_rl_module.py`
- Test: `tests/test_module_contracts.py`

---

### Task 1: Add Optional `initial_layout` to the RL Training Pipeline

**Files:**
- Modify: `src/rl_module/training.py`
- Test: `tests/test_rl_module/test_rl_module.py`

- [ ] **Step 1: Write the failing training-pipeline test for layout-conditioned resets**

Add a focused test alongside the existing training pipeline tests:

```python
def test_setup_training_pipeline_resets_train_and_eval_envs_with_initial_layout(monkeypatch, tmp_path):
    from qiskit import QuantumCircuit
    from src.rl_module import training

    reset_calls = []

    class FakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def reset(self, *, seed=None, options=None):
            reset_calls.append((seed, options))
            return {}, {}

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self, total_timesteps, callbacks):
            return None

        def save(self, path):
            return None

    monkeypatch.setattr(training, "QuantumTranspilationEnv", FakeEnv)
    monkeypatch.setattr(training, "Monitor", lambda env: env)
    monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
    monkeypatch.setattr(training, "EvalCallback", lambda *args, **kwargs: object())
    monkeypatch.setattr(training, "QuantumRLAgent", FakeAgent)
    monkeypatch.setattr(training, "set_global_seeds", lambda seed: None)
    monkeypatch.setattr(training, "save_run_metadata", lambda *args, **kwargs: None)

    training.setup_training_pipeline(
        target_circuit=QuantumCircuit(5),
        coupling_map=[(0, 1), (1, 2)],
        algorithm="PPO",
        total_timesteps=10,
        seed=42,
        log_dir=str(tmp_path / "logs"),
        model_save_dir=str(tmp_path / "models"),
        initial_layout=[4, 3, 2, 1, 0],
    )

    assert reset_calls == [
        (42, {"initial_layout": [4, 3, 2, 1, 0]}),
        (42, {"initial_layout": [4, 3, 2, 1, 0]}),
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_rl_module/test_rl_module.py -k "resets_train_and_eval_envs_with_initial_layout" -q
```

Expected: FAIL because `setup_training_pipeline(...)` does not yet accept or use `initial_layout`.

- [ ] **Step 3: Implement the minimal optional parameter in `training.py`**

Update the function signature and reset behavior:

```python
def setup_training_pipeline(
    target_circuit: QuantumCircuit,
    coupling_map: List[Tuple[int, int]],
    mode: str = "routing",
    frontier_mode: str = "sequential",
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    seed: int = 42,
    log_dir: str = "./experiments/logs/rl_logs",
    model_save_dir: str = "./experiments/models/rl_models",
    lookahead_window: int = 10,
    max_steps: int = 1000,
    hyperparams: Optional[dict] = None,
    basis_gates: Optional[List[str]] = None,
    initial_layout: Optional[List[int]] = None,
) -> QuantumRLAgent:
    ...
    reset_options = None
    if initial_layout is not None:
        reset_options = {"initial_layout": list(initial_layout)}
    raw_env.reset(seed=seed, options=reset_options)
    ...
    eval_raw_env.reset(seed=seed, options=reset_options)
```

- [ ] **Step 4: Add the no-layout regression test**

Add a second test to keep the default behavior stable:

```python
def test_setup_training_pipeline_keeps_default_reset_behavior_without_initial_layout(monkeypatch, tmp_path):
    from qiskit import QuantumCircuit
    from src.rl_module import training

    reset_calls = []

    class FakeEnv:
        def __init__(self, **kwargs):
            pass

        def reset(self, *, seed=None, options=None):
            reset_calls.append((seed, options))
            return {}, {}

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def train(self, total_timesteps, callbacks):
            return None

        def save(self, path):
            return None

    monkeypatch.setattr(training, "QuantumTranspilationEnv", FakeEnv)
    monkeypatch.setattr(training, "Monitor", lambda env: env)
    monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
    monkeypatch.setattr(training, "EvalCallback", lambda *args, **kwargs: object())
    monkeypatch.setattr(training, "QuantumRLAgent", FakeAgent)
    monkeypatch.setattr(training, "set_global_seeds", lambda seed: None)
    monkeypatch.setattr(training, "save_run_metadata", lambda *args, **kwargs: None)

    training.setup_training_pipeline(
        target_circuit=QuantumCircuit(5),
        coupling_map=[(0, 1), (1, 2)],
        algorithm="PPO",
        total_timesteps=10,
        seed=42,
        log_dir=str(tmp_path / "logs"),
        model_save_dir=str(tmp_path / "models"),
    )

    assert reset_calls == [(42, None), (42, None)]
```

- [ ] **Step 5: Run the targeted RL tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_rl_module/test_rl_module.py -k "initial_layout or setup_training_pipeline" -q
```

Expected: PASS.

---

### Task 2: Forward the MO Layout Through the Campaign Training Bridge

**Files:**
- Modify: `src/integration/training_bridge.py`
- Test: `tests/test_integration/test_training_bridge.py`

- [ ] **Step 1: Write the failing bridge-forwarding test**

Add a test like this:

```python
def test_train_case_forwards_initial_layout_to_setup_training_pipeline(tmp_path, monkeypatch):
    from src.integration.training_bridge import train_case

    captured = {}

    class FakeAgent:
        run_model_dir = tmp_path / "case" / "training" / "models" / "run-001"
        run_log_dir = tmp_path / "case" / "training" / "logs" / "run-001"
        best_model_path = run_model_dir / "best_model.zip"
        last_model_path = run_model_dir / "final_model.zip"

    FakeAgent.run_model_dir.mkdir(parents=True)
    FakeAgent.run_log_dir.mkdir(parents=True)
    FakeAgent.best_model_path.write_text("best", encoding="utf-8")
    FakeAgent.last_model_path.write_text("final", encoding="utf-8")

    def fake_setup_training_pipeline(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=..., 
        campaign_config=..., 
        target_circuit=..., 
        coupling_map=[(0, 1)],
        case_output_dir=tmp_path / "case",
        initial_layout=[4, 3, 2, 1, 0],
    )

    assert captured["initial_layout"] == [4, 3, 2, 1, 0]
    assert result.status == "completed"
```

- [ ] **Step 2: Run the bridge test to verify it fails**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_training_bridge.py -k "forwards_initial_layout" -q
```

Expected: FAIL because `train_case(...)` does not yet accept `initial_layout`.

- [ ] **Step 3: Implement the minimal bridge signature change**

Update the bridge signature and forward the new parameter:

```python
def train_case(
    *,
    campaign_case: CampaignCase,
    campaign_config: CampaignConfig,
    target_circuit: QuantumCircuit,
    coupling_map: Sequence[tuple[int, int]],
    case_output_dir: Path | str,
    initial_layout: Sequence[int] | None = None,
) -> TrainingBridgeResult:
    ...
    agent = setup_training_pipeline(
        ...,
        initial_layout=list(initial_layout) if initial_layout is not None else None,
    )
```

- [ ] **Step 4: Run the targeted bridge tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_training_bridge.py -q
```

Expected: PASS.

---

### Task 3: Make Campaign Hybrid Training Mandatory on the MO Layout

**Files:**
- Modify: `src/integration/campaign_runner.py`
- Test: `tests/test_integration/test_campaign_runner.py`

- [ ] **Step 1: Write the failing Campaign runner test**

Add a test like this:

```python
def test_run_campaign_trains_mo_rl_cases_from_mo_only_selected_layout(tmp_path):
    from src.integration.campaign_runner import run_campaign

    captured_training_layouts = []

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        captured_training_layouts.append(list(initial_layout) if initial_layout is not None else None)
        return _build_training_result(campaign_case)

    report = run_campaign(
        _build_campaign(),
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: object(),
        run_baseline=lambda request, *, circuit: _build_result("Baseline", _build_campaign().build_cases()[0], metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: ScenarioResult(
            scenario_name="MO_Only",
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=42,
            success=True,
            selected_layout=[2, 1, 0],
            transpilation_metrics=_build_metrics(90),
        ),
        train_case_fn=fake_train_case,
        run_mo_rl=lambda request, *, circuit: _build_result("MO+RL", _build_campaign().build_cases()[0], metrics=_build_metrics(80)),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(backend_name=backend_name, coupling_edges=[(0, 1), (1, 2)]),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert captured_training_layouts == [[2, 1, 0]]
    assert report.case_reports[0].status == "completed"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_runner.py -k "trains_mo_rl_cases_from_mo_only_selected_layout" -q
```

Expected: FAIL because the runner does not yet pass the MO-selected layout into training.

- [ ] **Step 3: Implement the mandatory Campaign-only layout handoff**

In `run_campaign(...)`, after `MO_Only` succeeds:

```python
selected_layout = case_report.mo_only_result.selected_layout
if selected_layout is None:
    case_report.status = "failed"
    case_report.incidents.append("MO_Only did not produce a selected layout for Campaign MO+RL training.")
else:
    case_report.training_result = train_case_fn(
        campaign_case=campaign_case,
        campaign_config=campaign.config,
        target_circuit=circuit,
        coupling_map=list(getattr(backend_bundle, "coupling_edges")),
        case_output_dir=case_output_dir,
        initial_layout=selected_layout,
    )
```

Keep all existing failure and persistence behavior intact.

- [ ] **Step 4: Add the missing-layout regression test**

Add a test for the explicit failure path:

```python
def test_run_campaign_fails_case_when_mo_only_result_has_no_selected_layout(tmp_path):
    ...
    assert report.case_reports[0].status == "failed"
    assert "selected layout" in report.case_reports[0].incidents[0]
```

- [ ] **Step 5: Run the targeted Campaign runner tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_runner.py -q
```

Expected: PASS.

---

### Task 4: Update Campaign Documentation for MO-Conditioned Training

**Files:**
- Modify: `src/integration/README.md`
- Modify: `src/integration/docs/internal_documentation.md`
- Modify: `README.md`
- Test: `tests/test_module_contracts.py`

- [ ] **Step 1: Update public and internal docs**

Adjust the docs so they state that Campaign `MO+RL` now trains from the exact MO-selected layout for each Campaign Case.

Required wording to cover:

- `MO_Only` selects the layout;
- Campaign training for `MO+RL` starts from that exact layout;
- `MO+RL` evaluation uses the same layout and resulting Training Artifact;
- `RL_Only` remains unchanged and outside the guided Campaign comparison flow.

- [ ] **Step 2: Run the affected docs/contracts tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py -q
```

Expected: PASS.

---

### Task 5: Full Verification Sweep

**Files:**
- Modify or create the files above

- [ ] **Step 1: Run the focused integration suites**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_training_bridge.py tests/test_integration/test_campaign_runner.py -q
```

- [ ] **Step 2: Run the adjacent RL tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_rl_module/test_rl_module.py -k "initial_layout or setup_training_pipeline or model_metadata" -q
```

- [ ] **Step 3: Run the full integration suite**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration -q
```

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Confirm that only the intended source, test, and documentation files for the MO-conditioned Campaign change are modified.

---

## Sequencing Notes

- Keep the new training capability opt-in in `src/rl_module.training`, but mandatory from the Campaign runner.
- Do not change the public request surface of `RL_Only`, `MO+RL`, or `src/integration/runner.py`.
- Do not move MO ownership into `rl_module`; the runner extracts the layout and forwards it through the bridge.
- Prefer the smallest contract extension that preserves all existing call sites.

## Self-Review Notes

- The plan keeps `integration` as owner of the MO -> RL handoff.
- The plan preserves the existing single-scenario workflow unchanged.
- The plan updates only the Campaign hybrid semantics, not the whole project training model.
- The plan explicitly covers both the new behavior and the no-layout regression path.
