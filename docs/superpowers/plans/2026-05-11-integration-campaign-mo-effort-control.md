# Integration Campaign MO Effort Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add explicit `auto/custom` MO effort control to the guided Campaign CLI, make the configured MO knobs actually reach `integration` scenario execution, and scale MO effort automatically for larger Campaign Cases without moving MO ownership out of `src/integration/`.

**Architecture:** keep the adaptive policy in `src/integration/` by introducing a small MO-effort resolver that maps `num_qubits` to effective MO settings, then let `campaign_runner.py` resolve per-Case MO effort before building `ScenarioRequest`. Extend `CampaignConfig` with `mo_effort_mode`, keep `ScenarioRequest` focused on effective MO knobs, and update `campaign_cli.py` plus `campaign_reporting.py` so both the guided CLI and the Summary Document describe whether the Campaign is using `auto` or `custom` MO effort truthfully.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `src/integration` Campaign contracts/runner/reporting, existing `src/mo_module` public API (`OptimizerConfig`, `optimize_layout`, `optimize_layout_quick`).

---

## File Map

- Create: `src/integration/mo_effort.py` - central auto/custom MO effort resolver and preview helpers owned by `integration`.
- Modify: `src/integration/campaign_contracts.py` - add `CampaignConfig.mo_effort_mode` validation while keeping current config shape stable for non-CLI callers.
- Modify: `src/integration/contracts.py` - add effective `ScenarioRequest` MO sizing fields and scenario-level validation.
- Modify: `src/integration/campaign_runner.py` - resolve effective MO effort per Campaign Case and pass it into `ScenarioRequest`.
- Modify: `src/integration/scenarios.py` - forward effective MO knobs to `mo_module.optimize_layout_quick(...)` or `mo_module.optimize_layout(..., config=OptimizerConfig(...))`.
- Modify: `src/integration/campaign_cli.py` - expose `MO effort mode (auto/custom)` in advanced mode, make default mode use `auto`, and print truthful confirmation summaries.
- Modify: `src/integration/campaign_reporting.py` - render `MO Effort Mode` and auto-preview lines in the Summary Document.
- Create: `tests/test_integration/test_mo_effort.py` - focused tests for the new resolver.
- Modify: `tests/test_integration/test_contracts.py` - extend `ScenarioRequest` coverage for the new MO sizing fields.
- Modify: `tests/test_integration/test_campaign_contracts.py` - validate `CampaignConfig.mo_effort_mode`.
- Modify: `tests/test_integration/test_campaign_runner.py` - verify per-Case MO effort resolution and propagation into scenario requests.
- Modify: `tests/test_integration/test_scenarios.py` - verify quick/full MO paths receive the effective knobs.
- Modify: `tests/test_integration/test_campaign_cli.py` - cover `auto/custom` prompting, defaults, and confirmation summary behavior.
- Modify: `tests/test_integration/test_campaign_reporting.py` - cover rendered `MO Effort Mode` and auto-preview lines.
- Modify: `tests/test_integration/test_training_bridge.py` - pin helper `CampaignConfig` fixtures to `custom` so they keep describing explicit manual MO settings.

---

### Task 1: Add the Integration-Owned MO Effort Resolver

**Files:**
- Create: `src/integration/mo_effort.py`
- Create: `tests/test_integration/test_mo_effort.py`

- [ ] **Step 1: Write the failing resolver tests**

Create `tests/test_integration/test_mo_effort.py` with this content:

```python
from src.integration.mo_effort import (
    EffectiveMoSettings,
    build_auto_mo_effort_preview,
    resolve_effective_mo_settings,
)


def test_resolve_effective_mo_settings_returns_auto_tiers_by_qubit_count() -> None:
    assert resolve_effective_mo_settings(
        mo_effort_mode="auto",
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        num_qubits=7,
    ) == EffectiveMoSettings(mo_use_quick=True, mo_population_size=30, mo_n_generations=50)
    assert resolve_effective_mo_settings(
        mo_effort_mode="auto",
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        num_qubits=8,
    ) == EffectiveMoSettings(mo_use_quick=False, mo_population_size=60, mo_n_generations=120)
    assert resolve_effective_mo_settings(
        mo_effort_mode="auto",
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        num_qubits=12,
    ) == EffectiveMoSettings(mo_use_quick=False, mo_population_size=80, mo_n_generations=160)
    assert resolve_effective_mo_settings(
        mo_effort_mode="auto",
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        num_qubits=15,
    ) == EffectiveMoSettings(mo_use_quick=False, mo_population_size=100, mo_n_generations=220)


def test_resolve_effective_mo_settings_returns_custom_values_verbatim() -> None:
    assert resolve_effective_mo_settings(
        mo_effort_mode="custom",
        mo_use_quick=False,
        mo_population_size=64,
        mo_n_generations=128,
        num_qubits=11,
    ) == EffectiveMoSettings(mo_use_quick=False, mo_population_size=64, mo_n_generations=128)


def test_build_auto_mo_effort_preview_collapses_duplicate_qubit_sizes_in_order() -> None:
    assert build_auto_mo_effort_preview((3, 8, 3, 12)) == (
        (3, EffectiveMoSettings(mo_use_quick=True, mo_population_size=30, mo_n_generations=50)),
        (8, EffectiveMoSettings(mo_use_quick=False, mo_population_size=60, mo_n_generations=120)),
        (12, EffectiveMoSettings(mo_use_quick=False, mo_population_size=80, mo_n_generations=160)),
    )
```

- [ ] **Step 2: Run the new resolver tests to verify they fail**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_mo_effort.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.integration.mo_effort'`.

- [ ] **Step 3: Implement the resolver module with the agreed qubit tiers**

Create `src/integration/mo_effort.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EffectiveMoSettings:
    mo_use_quick: bool
    mo_population_size: int
    mo_n_generations: int


_AUTO_MO_EFFORT_TIERS: tuple[tuple[int | None, EffectiveMoSettings], ...] = (
    (7, EffectiveMoSettings(mo_use_quick=True, mo_population_size=30, mo_n_generations=50)),
    (10, EffectiveMoSettings(mo_use_quick=False, mo_population_size=60, mo_n_generations=120)),
    (14, EffectiveMoSettings(mo_use_quick=False, mo_population_size=80, mo_n_generations=160)),
    (None, EffectiveMoSettings(mo_use_quick=False, mo_population_size=100, mo_n_generations=220)),
)


def resolve_effective_mo_settings(
    *,
    mo_effort_mode: str,
    mo_use_quick: bool,
    mo_population_size: int,
    mo_n_generations: int,
    num_qubits: int,
) -> EffectiveMoSettings:
    if mo_effort_mode == "custom":
        return EffectiveMoSettings(
            mo_use_quick=mo_use_quick,
            mo_population_size=mo_population_size,
            mo_n_generations=mo_n_generations,
        )

    for max_qubits, settings in _AUTO_MO_EFFORT_TIERS:
        if max_qubits is None or num_qubits <= max_qubits:
            return settings

    raise ValueError(f"Unable to resolve automatic MO effort for num_qubits={num_qubits}")


def build_auto_mo_effort_preview(qubit_sizes: Iterable[int]) -> tuple[tuple[int, EffectiveMoSettings], ...]:
    ordered_unique_sizes = tuple(dict.fromkeys(int(size) for size in qubit_sizes))
    return tuple(
        (
            num_qubits,
            resolve_effective_mo_settings(
                mo_effort_mode="auto",
                mo_use_quick=True,
                mo_population_size=30,
                mo_n_generations=50,
                num_qubits=num_qubits,
            ),
        )
        for num_qubits in ordered_unique_sizes
    )
```

- [ ] **Step 4: Run the resolver tests to verify they pass**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_mo_effort.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the resolver task**

Run:

```bash
git add src/integration/mo_effort.py tests/test_integration/test_mo_effort.py
git commit -m "feat: add integration MO effort resolver"
```

Expected: a new commit that only contains the resolver module and its focused tests.

---

### Task 2: Extend Campaign and Scenario Contracts for MO Effort Control

**Files:**
- Modify: `src/integration/campaign_contracts.py`
- Modify: `src/integration/contracts.py`
- Modify: `tests/test_integration/test_campaign_contracts.py`
- Modify: `tests/test_integration/test_contracts.py`

- [ ] **Step 1: Write the failing contract tests**

Add these tests to `tests/test_integration/test_campaign_contracts.py`:

```python
def test_campaign_config_accepts_auto_and_custom_mo_effort_modes() -> None:
    auto_config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_effort_mode="auto",
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="default",
    )
    custom_config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_effort_mode="custom",
        mo_use_quick=False,
        mo_population_size=64,
        mo_n_generations=128,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="advanced",
    )

    assert auto_config.mo_effort_mode == "auto"
    assert custom_config.mo_effort_mode == "custom"


def test_campaign_config_rejects_unknown_mo_effort_mode() -> None:
    try:
        CampaignConfig(
            circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
            backend_names=["fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="dag",
            rl_lookahead_window=12,
            rl_max_steps=256,
            seed=42,
            mo_effort_mode="burst",
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
        )
    except ValueError as exc:
        assert "mo_effort_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown mo_effort_mode")
```

Add these tests to `tests/test_integration/test_contracts.py`:

```python
def test_scenario_request_defaults_include_effective_mo_sizing_knobs() -> None:
    request = ScenarioRequest(
        scenario_name="MO_Only",
        circuit_name="ghz_5",
        num_qubits=5,
        backend_name="fake_backend",
    )

    assert request.mo_use_quick is True
    assert request.mo_population_size == 30
    assert request.mo_n_generations == 50


def test_baseline_request_rejects_non_default_mo_population_and_generation() -> None:
    for kwargs in ({"mo_population_size": 31}, {"mo_n_generations": 51}):
        try:
            ScenarioRequest(
                scenario_name="Baseline",
                circuit_name="ghz",
                num_qubits=3,
                backend_name="fake_torino",
                **kwargs,
            )
        except ValueError as exc:
            assert "Baseline" in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for Baseline kwargs: {kwargs}")
```

- [ ] **Step 2: Run the targeted contract tests to verify they fail**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_contracts.py tests/test_integration/test_contracts.py -q
```

Expected: FAIL with unexpected keyword arguments for `mo_effort_mode`, missing `ScenarioRequest` fields, or incorrect default field ordering assertions.

- [ ] **Step 3: Extend `CampaignConfig` and `ScenarioRequest` with the new fields and validation**

In `src/integration/campaign_contracts.py`, add the new allowed set near the top:

```python
_ALLOWED_MO_EFFORT_MODES = frozenset({"auto", "custom"})
```

Then replace the `CampaignConfig` field block with:

```python
@dataclass(frozen=True, slots=True)
class CampaignConfig:
    circuit_specs: tuple[CampaignCircuitSpec, ...]
    backend_names: tuple[str, ...]
    rl_algorithm: str
    rl_total_timesteps: int
    rl_frontier_mode: str
    rl_lookahead_window: int
    rl_max_steps: int
    seed: int
    mo_effort_mode: str = "custom"
    mo_use_quick: bool = True
    mo_population_size: int = 30
    mo_n_generations: int = 50
    layout_policy: LayoutSelectionPolicy = LayoutSelectionPolicy.COMPROMISE
    mo_objective_name: str | None = None
    mode: str = "default"
```

Inside `CampaignConfig.__post_init__`, add validation and normalization right before the existing `mo_use_quick` checks:

```python
        normalized_mo_effort_mode = self.mo_effort_mode.strip()
        if normalized_mo_effort_mode not in _ALLOWED_MO_EFFORT_MODES:
            raise ValueError("CampaignConfig mo_effort_mode must be one of auto, custom")
```

And add this normalization at the end of the method with the other `object.__setattr__` calls:

```python
        object.__setattr__(self, "mo_effort_mode", normalized_mo_effort_mode)
```

In `src/integration/contracts.py`, replace the `ScenarioRequest` field block with:

```python
@dataclass(slots=True)
class ScenarioRequest:
    scenario_name: str
    backend_name: str
    circuit_name: str | None = None
    num_qubits: int | None = None
    seed: int = 42
    layout_policy: LayoutSelectionPolicy = LayoutSelectionPolicy.COMPROMISE
    mo_use_quick: bool = True
    mo_population_size: int = 30
    mo_n_generations: int = 50
    initial_layout: list[int] | None = None
    rl_model_path: str | None = None
    mo_objective_index: int = 0
    circuit_source: CircuitSource = CircuitSource.LIBRARY
    circuit_path: str | None = None
    circuit_format: CircuitFormat = CircuitFormat.AUTO
```

Then add these validations in `ScenarioRequest.__post_init__` before `_validate_scenario_specific_constraints()`:

```python
        if self.mo_population_size <= 0:
            raise ValueError("mo_population_size must be greater than zero")
        if self.mo_n_generations <= 0:
            raise ValueError("mo_n_generations must be greater than zero")
```

And extend the scenario-specific restrictions with these exact checks:

```python
        if self.scenario_name == "Baseline":
            if self.mo_population_size != 30:
                raise ValueError("Baseline does not accept non-default mo_population_size")
            if self.mo_n_generations != 50:
                raise ValueError("Baseline does not accept non-default mo_n_generations")
            return
```

```python
        if self.scenario_name == "RL_Only":
            if self.mo_population_size != 30:
                raise ValueError("RL_Only does not accept non-default mo_population_size")
            if self.mo_n_generations != 50:
                raise ValueError("RL_Only does not accept non-default mo_n_generations")
            return
```

Finally, update the `ScenarioRequest` field-order assertion in `tests/test_integration/test_contracts.py` to this exact list:

```python
    assert [field.name for field in fields(ScenarioRequest)] == [
        "scenario_name",
        "backend_name",
        "circuit_name",
        "num_qubits",
        "seed",
        "layout_policy",
        "mo_use_quick",
        "mo_population_size",
        "mo_n_generations",
        "initial_layout",
        "rl_model_path",
        "mo_objective_index",
        "circuit_source",
        "circuit_path",
        "circuit_format",
    ]
```

- [ ] **Step 4: Run the targeted contract tests to verify they pass**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_contracts.py tests/test_integration/test_contracts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the contract task**

Run:

```bash
git add src/integration/campaign_contracts.py src/integration/contracts.py tests/test_integration/test_campaign_contracts.py tests/test_integration/test_contracts.py
git commit -m "feat: extend integration MO effort contracts"
```

Expected: a new commit that adds `mo_effort_mode` and the effective `ScenarioRequest` MO sizing fields.

---

### Task 3: Resolve MO Effort Per Campaign Case and Propagate It Into MO Execution

**Files:**
- Modify: `src/integration/campaign_runner.py`
- Modify: `src/integration/scenarios.py`
- Modify: `tests/test_integration/test_campaign_runner.py`
- Modify: `tests/test_integration/test_scenarios.py`

- [ ] **Step 1: Write the failing runner/scenario tests**

Add this test to `tests/test_integration/test_campaign_runner.py`:

```python
def test_run_campaign_resolves_auto_mo_effort_per_case_qubit_size(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    config = CampaignConfig(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=8),
        ],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_effort_mode="auto",
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="default",
    )
    campaign = Campaign.from_config(campaign_id="campaign-001", config=config)
    captured_requests: list[tuple[str, int, bool, int, int]] = []

    def fake_run_baseline(request, *, circuit):
        case = next(case for case in campaign.build_cases() if case.case_id == _case_id_from_request(request))
        return _build_result("Baseline", case, metrics=_build_metrics(100))

    def fake_run_mo_only(request, *, circuit):
        captured_requests.append(
            (request.circuit_name, request.num_qubits, request.mo_use_quick, request.mo_population_size, request.mo_n_generations)
        )
        case = next(case for case in campaign.build_cases() if case.case_id == _case_id_from_request(request))
        return _build_result("MO_Only", case, metrics=_build_metrics(90))

    def fake_run_mo_rl(request, *, circuit, injected_layout):
        captured_requests.append(
            (request.circuit_name, request.num_qubits, request.mo_use_quick, request.mo_population_size, request.mo_n_generations)
        )
        case = next(case for case in campaign.build_cases() if case.case_id == _case_id_from_request(request))
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=fake_run_baseline,
        run_mo_only=fake_run_mo_only,
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert captured_requests == [
        ("ghz_3", 3, True, 30, 50),
        ("ghz_3", 3, True, 30, 50),
        ("qft_8", 8, False, 60, 120),
        ("qft_8", 8, False, 60, 120),
    ]
```

Update the existing baseline-safe test in `tests/test_integration/test_campaign_runner.py` so the assertion block ends with:

```python
    assert captured_baseline_request.mo_use_quick is True
    assert captured_baseline_request.mo_population_size == 30
    assert captured_baseline_request.mo_n_generations == 50
    assert captured_baseline_request.mo_objective_index == 0
```

Add these tests to `tests/test_integration/test_scenarios.py`:

```python
def test_run_mo_only_scenario_passes_population_and_generation_to_quick_optimizer(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO_Only", mo_population_size=44, mo_n_generations=66)
    bundle = SimpleNamespace(backend_name="fake_backend", backend="backend-object")
    quick_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda **kwargs: quick_calls.append(kwargs) or "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: [2, 0, 1],
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "run_named_baseline",
        lambda baseline_name, circuit, backend_names, seed, layout=None, include_artifact=False: [
            (
                {
                    "backend_name": "fake_backend",
                    "baseline_name": "custom_layout_level_1",
                    "initial_layout": [2, 0, 1],
                },
                {
                    "artifact_version": "transpilation_result.v1",
                    "baseline_name": "custom_layout_level_1",
                    "backend": {"backend_name": "fake_backend"},
                    "transpilation": {"initial_layout": [2, 0, 1]},
                },
            )
        ],
    )

    scenarios.run_mo_only_scenario(request)

    assert quick_calls == [
        {
            "circuit": circuit,
            "backend": "backend-object",
            "seed": 17,
            "population_size": 44,
            "n_generations": 66,
        }
    ]


def test_run_mo_only_scenario_builds_optimizer_config_for_full_mo_path(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO_Only", mo_use_quick=False, mo_population_size=64, mo_n_generations=128)
    bundle = SimpleNamespace(backend_name="fake_backend", backend="backend-object")
    optimize_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout",
        lambda **kwargs: optimize_calls.append(kwargs) or "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: [2, 0, 1],
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "run_named_baseline",
        lambda baseline_name, circuit, backend_names, seed, layout=None, include_artifact=False: [
            (
                {
                    "backend_name": "fake_backend",
                    "baseline_name": "custom_layout_level_1",
                    "initial_layout": [2, 0, 1],
                },
                {
                    "artifact_version": "transpilation_result.v1",
                    "baseline_name": "custom_layout_level_1",
                    "backend": {"backend_name": "fake_backend"},
                    "transpilation": {"initial_layout": [2, 0, 1]},
                },
            )
        ],
    )

    scenarios.run_mo_only_scenario(request)

    assert len(optimize_calls) == 1
    assert optimize_calls[0]["config"].population_size == 64
    assert optimize_calls[0]["config"].n_generations == 128
    assert optimize_calls[0]["config"].seed == 17
    assert optimize_calls[0]["config"].verbose is False
```

- [ ] **Step 2: Run the targeted propagation tests to verify they fail**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_runner.py tests/test_integration/test_scenarios.py -q
```

Expected: FAIL because `campaign_runner.py` does not yet resolve per-Case MO effort and `scenarios._run_mo(...)` does not yet pass `population_size` or `n_generations` into MO.

- [ ] **Step 3: Wire the new resolver into `campaign_runner.py` and `scenarios.py`**

At the top of `src/integration/campaign_runner.py`, add this import:

```python
from src.integration.mo_effort import resolve_effective_mo_settings
```

Then replace `_build_scenario_request(...)` with this implementation:

```python
def _build_scenario_request(
    *,
    campaign: Campaign,
    campaign_case: CampaignCase,
    scenario_name: str,
    rl_model_path: str | None = None,
) -> ScenarioRequest:
    if scenario_name == "Baseline":
        return ScenarioRequest(
            scenario_name=scenario_name,
            circuit_name=_case_library_name(campaign_case),
            num_qubits=campaign_case.num_qubits,
            backend_name=campaign_case.backend_name,
            seed=campaign.config.seed,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            mo_objective_index=0,
            rl_model_path=rl_model_path,
        )

    effective_mo = resolve_effective_mo_settings(
        mo_effort_mode=campaign.config.mo_effort_mode,
        mo_use_quick=campaign.config.mo_use_quick,
        mo_population_size=campaign.config.mo_population_size,
        mo_n_generations=campaign.config.mo_n_generations,
        num_qubits=campaign_case.num_qubits,
    )
    mo_objective_index = _resolve_mo_objective_index(campaign)

    return ScenarioRequest(
        scenario_name=scenario_name,
        circuit_name=_case_library_name(campaign_case),
        num_qubits=campaign_case.num_qubits,
        backend_name=campaign_case.backend_name,
        seed=campaign.config.seed,
        layout_policy=campaign.config.layout_policy,
        mo_use_quick=effective_mo.mo_use_quick,
        mo_population_size=effective_mo.mo_population_size,
        mo_n_generations=effective_mo.mo_n_generations,
        mo_objective_index=mo_objective_index,
        rl_model_path=rl_model_path,
    )
```

In `src/integration/scenarios.py`, replace `_run_mo(...)` with this implementation:

```python
def _run_mo(request: ScenarioRequest, circuit, backend_bundle):
    if request.mo_use_quick:
        return mo_module.optimize_layout_quick(
            circuit=circuit,
            backend=backend_bundle.backend,
            seed=request.seed,
            population_size=request.mo_population_size,
            n_generations=request.mo_n_generations,
        )
    return mo_module.optimize_layout(
        circuit=circuit,
        backend=backend_bundle.backend,
        backend_name=backend_bundle.backend_name,
        config=mo_module.OptimizerConfig(
            population_size=request.mo_population_size,
            n_generations=request.mo_n_generations,
            seed=request.seed,
            verbose=False,
        ),
    )
```

- [ ] **Step 4: Run the targeted propagation tests to verify they pass**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_runner.py tests/test_integration/test_scenarios.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the propagation task**

Run:

```bash
git add src/integration/campaign_runner.py src/integration/scenarios.py tests/test_integration/test_campaign_runner.py tests/test_integration/test_scenarios.py
git commit -m "feat: wire MO effort through campaign execution"
```

Expected: a new commit that resolves per-Case MO effort in `campaign_runner.py` and forwards the effective knobs into MO execution.

---

### Task 4: Add `auto/custom` Campaign CLI UX and Truthful Summary Reporting

**Files:**
- Modify: `src/integration/campaign_cli.py`
- Modify: `src/integration/campaign_reporting.py`
- Modify: `tests/test_integration/test_campaign_cli.py`
- Modify: `tests/test_integration/test_campaign_reporting.py`
- Modify: `tests/test_integration/test_training_bridge.py`

- [ ] **Step 1: Write the failing CLI/reporting tests**

Update `tests/test_integration/test_campaign_cli.py` by changing `test_build_default_campaign_config_uses_canonical_defaults()` so it includes this assertion:

```python
    assert config.mo_effort_mode == "auto"
```

Replace the `responses` list in `test_run_interactive_campaign_cli_allows_multiple_backends_in_advanced_mode()` with:

```python
        [
            "ghz,qft",
            "3,5",
            "advanced",
            "fake_torino,fake_brisbane",
            "MaskablePPO",
            "7000",
            "dag",
            "15",
            "300",
            "123",
            "custom",
            "false",
            "40",
            "60",
            "compromise",
            "y",
        ]
```

And change the MO assertions in that same test to:

```python
    assert captured["campaign"].config.mo_effort_mode == "custom"
    assert captured["campaign"].config.mo_use_quick is False
    assert captured["campaign"].config.mo_population_size == 40
    assert captured["campaign"].config.mo_n_generations == 60
```

Add this new CLI test to the same file:

```python
def test_run_interactive_campaign_cli_auto_effort_skips_manual_mo_knobs_and_prints_preview() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz,qft",
            "3,8",
            "advanced",
            "fake_torino",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "42",
            "auto",
            "compromise",
            "y",
        ]
    )
    captured = {}

    def run_campaign_fn(campaign, *, output_root):
        captured["campaign"] = campaign
        return SimpleNamespace(campaign_status="completed")

    campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-adv-auto-001",
        output_root=Path("campaigns"),
    )

    rendered = "\n".join(outputs)

    assert captured["campaign"].config.mo_effort_mode == "auto"
    assert captured["campaign"].config.mo_use_quick is True
    assert captured["campaign"].config.mo_population_size == 30
    assert captured["campaign"].config.mo_n_generations == 50
    assert "MO population size:" not in rendered
    assert "MO generations:" not in rendered
    assert "MO Effort Mode: auto" in rendered
    assert "MO Auto Preview (3q): quick=True, population_size=30, n_generations=50" in rendered
    assert "MO Auto Preview (8q): quick=False, population_size=60, n_generations=120" in rendered
```

Add this reporting test to `tests/test_integration/test_campaign_reporting.py`:

```python
def test_summary_markdown_renders_auto_mo_effort_preview() -> None:
    config = CampaignConfig(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=8),
        ],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_effort_mode="auto",
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="default",
    )
    completed_case = _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")

    report = build_campaign_report(
        campaign_id="campaign-auto-mo",
        campaign_status="completed",
        campaign_config=config,
        case_reports=[
            CampaignCaseReport(
                case=completed_case,
                status="completed",
                baseline_result=_build_scenario_result("Baseline", completed_case, metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", completed_case, metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=_build_scenario_result("MO+RL", completed_case, metrics=_build_metrics(80, 24, 36.0, 2.0)),
                training_result=_build_training_result(completed_case),
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "MO Effort Mode: `auto`" in markdown
    assert "MO Auto Preview (3q): `quick=True, population_size=30, n_generations=50`" in markdown
    assert "MO Auto Preview (8q): `quick=False, population_size=60, n_generations=120`" in markdown
```

- [ ] **Step 2: Run the targeted CLI/reporting tests to verify they fail**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py tests/test_integration/test_campaign_reporting.py -q
```

Expected: FAIL because `campaign_cli.py` does not yet expose `MO effort mode` and `campaign_reporting.py` does not yet render `MO Effort Mode` or auto-preview lines.

- [ ] **Step 3: Implement the CLI changes, auto preview rendering, and reporting updates**

At the top of `src/integration/campaign_cli.py`, add this import and constant:

```python
from src.integration.mo_effort import build_auto_mo_effort_preview

_MO_EFFORT_MODES = ("auto", "custom")
```

Update `build_default_campaign_config(...)` so it returns:

```python
    return CampaignConfig(
        circuit_specs=tuple(circuit_specs),
        backend_names=(backend_name,),
        rl_algorithm=_DEFAULT_RL_ALGORITHM,
        rl_total_timesteps=_DEFAULT_RL_TIMESTEPS,
        rl_frontier_mode=_DEFAULT_RL_FRONTIER_MODE,
        rl_lookahead_window=_DEFAULT_RL_LOOKAHEAD,
        rl_max_steps=_DEFAULT_RL_MAX_STEPS,
        seed=_DEFAULT_SEED,
        mo_effort_mode="auto",
        mo_use_quick=_DEFAULT_MO_USE_QUICK,
        mo_population_size=_DEFAULT_MO_POPULATION_SIZE,
        mo_n_generations=_DEFAULT_MO_N_GENERATIONS,
        layout_policy=_DEFAULT_LAYOUT_POLICY,
        mode="default",
    )
```

Add this helper to `src/integration/campaign_cli.py` right above `_collect_advanced_config(...)`:

```python
def _collect_mo_effort_config(input_fn, output_fn) -> tuple[str, bool, int, int]:
    mo_effort_mode = _prompt_csv_choices(
        input_fn,
        output_fn,
        prompt=f"Choose MO effort mode ({', '.join(_MO_EFFORT_MODES)}): ",
        valid_values=_MO_EFFORT_MODES,
        allow_multiple=False,
    )[0]
    if mo_effort_mode == "auto":
        return (
            "auto",
            _DEFAULT_MO_USE_QUICK,
            _DEFAULT_MO_POPULATION_SIZE,
            _DEFAULT_MO_N_GENERATIONS,
        )

    mo_use_quick = _prompt_bool(input_fn, output_fn, prompt="MO quick (true/false): ")
    mo_population_size = _prompt_int(input_fn, output_fn, prompt="MO population size: ", minimum=1)
    mo_n_generations = _prompt_int(input_fn, output_fn, prompt="MO generations: ", minimum=1)
    return ("custom", mo_use_quick, mo_population_size, mo_n_generations)
```

Then replace the MO section inside `_collect_advanced_config(...)` with:

```python
    mo_effort_mode, mo_use_quick, mo_population_size, mo_n_generations = _collect_mo_effort_config(
        input_fn,
        output_fn,
    )
```

And build the returned `CampaignConfig` with:

```python
        mo_effort_mode=mo_effort_mode,
        mo_use_quick=mo_use_quick,
        mo_population_size=mo_population_size,
        mo_n_generations=mo_n_generations,
```

Replace `_print_confirmation_summary(...)` with this implementation:

```python
def _print_confirmation_summary(output_fn, *, campaign: Campaign) -> None:
    config = campaign.config
    output_fn("Confirmation Summary")
    output_fn(f"Campaign ID: {campaign.campaign_id}")
    output_fn(f"Mode: {config.mode}")
    output_fn("Circuits: " + ", ".join(f"{spec.family} ({spec.num_qubits}q)" for spec in config.circuit_specs))
    output_fn("Backends: " + ", ".join(config.backend_names))
    output_fn(
        "RL: "
        f"algorithm={config.rl_algorithm}, timesteps={config.rl_total_timesteps}, "
        f"frontier_mode={config.rl_frontier_mode}, lookahead={config.rl_lookahead_window}, "
        f"max_steps={config.rl_max_steps}, seed={config.seed}"
    )
    output_fn(f"MO Effort Mode: {config.mo_effort_mode}")
    if config.mo_effort_mode == "auto":
        for num_qubits, settings in build_auto_mo_effort_preview(
            spec.num_qubits for spec in config.circuit_specs
        ):
            output_fn(
                f"MO Auto Preview ({num_qubits}q): quick={settings.mo_use_quick}, "
                f"population_size={settings.mo_population_size}, n_generations={settings.mo_n_generations}"
            )
    else:
        output_fn(
            "MO: "
            f"quick={config.mo_use_quick}, population_size={config.mo_population_size}, "
            f"n_generations={config.mo_n_generations}, layout_policy={config.layout_policy.value}"
        )
    output_fn(f"Layout Policy: {config.layout_policy.value}")
    if config.mo_objective_name is not None:
        output_fn(f"MO Objective: {config.mo_objective_name}")
```

In `src/integration/campaign_reporting.py`, add this import near the top:

```python
from src.integration.mo_effort import build_auto_mo_effort_preview
```

Then replace `_render_config(...)` with:

```python
def _render_config(report: CampaignReport) -> list[str]:
    config = report.campaign_config
    status_label = "Final Campaign Status" if report.campaign_status != "running" else "Campaign Status"
    lines = [
        "## Campaign Metadata",
        f"Campaign ID: `{report.campaign_id}`",
        f"Campaign Mode: `{config.mode}`",
        f"{status_label}: `{report.campaign_status}`",
        "",
        "## Global Configuration",
        f"RL Algorithm: `{config.rl_algorithm}`",
        f"RL Timesteps: `{config.rl_total_timesteps}`",
        f"RL Frontier Mode: `{config.rl_frontier_mode}`",
        f"RL Lookahead Window: `{config.rl_lookahead_window}`",
        f"RL Max Steps: `{config.rl_max_steps}`",
        f"Seed: `{config.seed}`",
        f"MO Effort Mode: `{config.mo_effort_mode}`",
        f"Layout Policy: `{config.layout_policy.value}`",
    ]
    if config.mo_effort_mode == "auto":
        for num_qubits, settings in build_auto_mo_effort_preview(
            spec.num_qubits for spec in config.circuit_specs
        ):
            lines.append(
                f"MO Auto Preview ({num_qubits}q): `quick={settings.mo_use_quick}, "
                f"population_size={settings.mo_population_size}, n_generations={settings.mo_n_generations}`"
            )
    else:
        lines.extend(
            [
                f"MO Quick: `{config.mo_use_quick}`",
                f"MO Population Size: `{config.mo_population_size}`",
                f"MO Generations: `{config.mo_n_generations}`",
            ]
        )
    if config.layout_policy.value == "best_on_objective":
        lines.append(f"MO Objective: `{config.mo_objective_name}`")
    return lines
```

Finally, pin the helper `CampaignConfig` fixtures in `tests/test_integration/test_training_bridge.py` and `tests/test_integration/test_campaign_reporting.py` to explicit custom mode by adding `mo_effort_mode="custom",` immediately before `mo_use_quick=...` in each shared `_build_campaign_config()` helper. That keeps the existing tests describing manual MO configs instead of inheriting the new auto/default CLI path accidentally.

- [ ] **Step 4: Run the targeted CLI/reporting tests to verify they pass**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py tests/test_integration/test_campaign_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the CLI/reporting task**

Run:

```bash
git add src/integration/campaign_cli.py src/integration/campaign_reporting.py tests/test_integration/test_campaign_cli.py tests/test_integration/test_campaign_reporting.py tests/test_integration/test_training_bridge.py
git commit -m "feat: add auto and custom MO effort campaign UX"
```

Expected: a new commit that exposes `auto/custom` Campaign CLI control and keeps the Summary Document aligned with actual MO effort behavior.

---

### Task 5: Run the Full Focused Verification Suite

**Files:**
- Modify: none
- Test: `tests/test_integration/test_mo_effort.py`
- Test: `tests/test_integration/test_contracts.py`
- Test: `tests/test_integration/test_campaign_contracts.py`
- Test: `tests/test_integration/test_campaign_cli.py`
- Test: `tests/test_integration/test_campaign_runner.py`
- Test: `tests/test_integration/test_scenarios.py`
- Test: `tests/test_integration/test_campaign_reporting.py`
- Test: `tests/test_integration/test_training_bridge.py`
- Test: `tests/test_integration/test_runner.py`

- [ ] **Step 1: Run the full focused integration suite for the changed surface area**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_mo_effort.py tests/test_integration/test_contracts.py tests/test_integration/test_campaign_contracts.py tests/test_integration/test_campaign_cli.py tests/test_integration/test_campaign_runner.py tests/test_integration/test_scenarios.py tests/test_integration/test_campaign_reporting.py tests/test_integration/test_training_bridge.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the adjacent single-scenario runner tests to confirm no CLI regression leaked into `runner.py`**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_runner.py -q
```

Expected: PASS.

- [ ] **Step 3: Check the worktree before any follow-up commit or PR work**

Run:

```bash
git status --short
```

Expected: only the intended `integration` source files, tests, and the saved plan file remain modified, plus any unrelated pre-existing user changes.

---

## Self-Review

- Spec coverage: this plan covers both requested behaviors from the approved design: real manual MO control in the Campaign CLI and automatic MO effort scaling for larger Campaign Cases, while keeping MO ownership in `src/integration/` and preserving the current `mo_module`/`rl_module` boundaries.
- Placeholder scan: no `TODO`, `TBD`, or deferred “write tests later” steps remain; every code step includes concrete code or exact replacement snippets.
- Type consistency: the plan consistently uses `CampaignConfig.mo_effort_mode`, `ScenarioRequest.mo_population_size`, `ScenarioRequest.mo_n_generations`, `EffectiveMoSettings`, and `resolve_effective_mo_settings(...)` across CLI, runner, scenarios, and reporting.
