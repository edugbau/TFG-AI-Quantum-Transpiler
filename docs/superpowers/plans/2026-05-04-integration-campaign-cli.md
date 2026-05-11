# Integration Campaign CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a guided terminal CLI that builds a `CampaignConfig`, runs a Campaign through `src/integration/campaign_runner.py`, and prints stable final output paths without disturbing the existing single-scenario `runner.py` entrypoint.

**Architecture:** create a dedicated `src/integration/campaign_cli.py` module with three focused seams: prompt/reprompt helpers, pure config-building helpers, and a thin execution wrapper that builds a `Campaign` and calls `run_campaign(...)`. Keep `src/integration/runner.py` unchanged and test the new module primarily through injectable input/output callables rather than full stdin integration.

**Tech Stack:** Python 3.10, pytest, argparse, pathlib, dataclasses, existing `src/integration` Campaign contracts/runner/reporting.

---

## File Map

- Create: `src/integration/campaign_cli.py` - guided Campaign CLI, prompt helpers, config builders, entrypoint.
- Create: `tests/test_integration/test_campaign_cli.py` - focused tests for default/advanced config building, reprompt behavior, confirmation, and final execution summary.
- Keep unchanged: `src/integration/runner.py` - stable unitary-scenario CLI.
- Reuse: `src/integration/campaign_contracts.py` - `CampaignConfig`, `CampaignCircuitSpec`, `Campaign`.
- Reuse: `src/integration/campaign_runner.py` - Campaign execution.

---

### Task 1: Add Default Campaign Config Builder

**Files:**
- Create: `src/integration/campaign_cli.py`
- Test: `tests/test_integration/test_campaign_cli.py`

- [ ] **Step 1: Write the failing default-config test**

```python
from src.integration.contracts import LayoutSelectionPolicy


def test_default_campaign_cli_builds_canonical_defaults() -> None:
    from src.integration.campaign_cli import build_campaign_config_interactively

    answers = iter([
        "ghz,qft",
        "3,5",
        "default",
    ])

    config = build_campaign_config_interactively(
        input_fn=lambda prompt: next(answers),
        output_fn=lambda message: None,
    )

    assert [(spec.family, spec.num_qubits) for spec in config.circuit_specs] == [
        ("ghz", 3),
        ("ghz", 5),
        ("qft", 3),
        ("qft", 5),
    ]
    assert config.backend_names == ("fake_torino",)
    assert config.mode == "default"
    assert config.rl_algorithm == "MaskablePPO"
    assert config.rl_total_timesteps == 5000
    assert config.rl_frontier_mode == "dag"
    assert config.rl_lookahead_window == 10
    assert config.rl_max_steps == 200
    assert config.seed == 42
    assert config.mo_use_quick is True
    assert config.mo_population_size == 30
    assert config.mo_n_generations == 50
    assert config.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert config.mo_objective_name is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py::test_default_campaign_cli_builds_canonical_defaults -q
```

Expected: FAIL with `ModuleNotFoundError` for `src.integration.campaign_cli` or missing `build_campaign_config_interactively`.

- [ ] **Step 3: Write the minimal default builder and parsing helpers**

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.integration.campaign_contracts import Campaign, CampaignCircuitSpec, CampaignConfig
from src.integration.contracts import LayoutSelectionPolicy
from src.integration.campaign_runner import run_campaign


_AVAILABLE_CIRCUIT_FAMILIES = ("ghz", "qft", "qft_inv", "random_shallow", "random_deep", "clifford")
_AVAILABLE_BACKENDS = ("fake_torino", "fake_brisbane")
_AVAILABLE_ALGORITHMS = ("PPO", "MaskablePPO")
_AVAILABLE_LAYOUT_POLICIES = {
    "compromise": LayoutSelectionPolicy.COMPROMISE,
    "best_on_objective": LayoutSelectionPolicy.BEST_ON_OBJECTIVE,
}
_AVAILABLE_OBJECTIVE_NAMES = ("depth", "cnot_count")


def _emit(output_fn: Callable[[str], None], message: str) -> None:
    output_fn(message)


def _parse_csv_tokens(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Please enter at least one value.")
    return values


def _prompt_choice_list(
    *,
    label: str,
    options: tuple[str, ...],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> list[str]:
    while True:
        _emit(output_fn, f"{label}: {', '.join(options)}")
        try:
            values = _parse_csv_tokens(input_fn(f"{label}: "))
        except ValueError as exc:
            _emit(output_fn, str(exc))
            continue
        invalid = [value for value in values if value not in options]
        if invalid:
            _emit(output_fn, f"Invalid selection: {', '.join(invalid)}")
            continue
        return values


def _prompt_int_list(
    *,
    label: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> list[int]:
    while True:
        try:
            values = [int(item) for item in _parse_csv_tokens(input_fn(f"{label}: "))]
        except ValueError:
            _emit(output_fn, "Please enter one or more integers separated by commas.")
            continue
        if any(value <= 0 for value in values):
            _emit(output_fn, "Qubit sizes must be greater than zero.")
            continue
        return values


def _prompt_mode(input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> str:
    while True:
        mode = input_fn("Campaign mode (default/advanced): ").strip().lower()
        if mode in {"default", "advanced"}:
            return mode
        _emit(output_fn, "Invalid mode. Choose default or advanced.")


def _build_circuit_specs(families: list[str], qubit_sizes: list[int]) -> list[CampaignCircuitSpec]:
    return [CampaignCircuitSpec(family=family, num_qubits=num_qubits) for family in families for num_qubits in qubit_sizes]


def build_campaign_config_interactively(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> CampaignConfig:
    families = _prompt_choice_list(
        label="Circuit families",
        options=_AVAILABLE_CIRCUIT_FAMILIES,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    qubit_sizes = _prompt_int_list(label="Qubit sizes", input_fn=input_fn, output_fn=output_fn)
    mode = _prompt_mode(input_fn, output_fn)
    circuit_specs = _build_circuit_specs(families, qubit_sizes)
    if mode == "default":
        return CampaignConfig(
            circuit_specs=circuit_specs,
            backend_names=["fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="dag",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            mode="default",
        )
    raise NotImplementedError("advanced mode will be added in the next task")
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py::test_default_campaign_cli_builds_canonical_defaults -q
```

Expected: PASS.

---

### Task 2: Add Advanced Mode Config Collection and Reprompt Behavior

**Files:**
- Modify: `src/integration/campaign_cli.py`
- Test: `tests/test_integration/test_campaign_cli.py`

- [ ] **Step 1: Write failing advanced-mode tests**

```python
from src.integration.contracts import LayoutSelectionPolicy


def test_advanced_campaign_cli_accepts_multiple_backends() -> None:
    from src.integration.campaign_cli import build_campaign_config_interactively

    answers = iter([
        "ghz",
        "3",
        "advanced",
        "fake_torino,fake_brisbane",
        "MaskablePPO",
        "7000",
        "dag",
        "12",
        "256",
        "99",
        "false",
        "40",
        "60",
        "compromise",
    ])

    config = build_campaign_config_interactively(
        input_fn=lambda prompt: next(answers),
        output_fn=lambda message: None,
    )

    assert config.mode == "advanced"
    assert config.backend_names == ("fake_torino", "fake_brisbane")
    assert config.rl_algorithm == "MaskablePPO"
    assert config.rl_total_timesteps == 7000
    assert config.rl_frontier_mode == "dag"
    assert config.rl_lookahead_window == 12
    assert config.rl_max_steps == 256
    assert config.seed == 99
    assert config.mo_use_quick is False
    assert config.mo_population_size == 40
    assert config.mo_n_generations == 60
    assert config.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert config.mo_objective_name is None


def test_advanced_campaign_cli_collects_mo_policy_and_objective() -> None:
    from src.integration.campaign_cli import build_campaign_config_interactively

    answers = iter([
        "qft_inv",
        "5",
        "advanced",
        "fake_torino",
        "PPO",
        "9000",
        "dag",
        "14",
        "300",
        "17",
        "true",
        "50",
        "80",
        "best_on_objective",
        "depth",
    ])

    config = build_campaign_config_interactively(
        input_fn=lambda prompt: next(answers),
        output_fn=lambda message: None,
    )

    assert config.layout_policy is LayoutSelectionPolicy.BEST_ON_OBJECTIVE
    assert config.mo_objective_name == "depth"


def test_campaign_cli_reprompts_on_invalid_backend_selection() -> None:
    from src.integration.campaign_cli import build_campaign_config_interactively

    outputs: list[str] = []
    answers = iter([
        "ghz",
        "3",
        "advanced",
        "not_a_backend",
        "fake_torino",
        "MaskablePPO",
        "5000",
        "dag",
        "12",
        "256",
        "42",
        "true",
        "30",
        "50",
        "compromise",
    ])

    config = build_campaign_config_interactively(
        input_fn=lambda prompt: next(answers),
        output_fn=outputs.append,
    )

    assert config.backend_names == ("fake_torino",)
    assert any("Invalid selection" in message for message in outputs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py::test_advanced_campaign_cli_accepts_multiple_backends tests/test_integration/test_campaign_cli.py::test_advanced_campaign_cli_collects_mo_policy_and_objective tests/test_integration/test_campaign_cli.py::test_campaign_cli_reprompts_on_invalid_backend_selection -q
```

Expected: FAIL because advanced mode is not implemented yet.

- [ ] **Step 3: Write the minimal advanced-mode prompt helpers**

```python
def _prompt_choice(
    *,
    label: str,
    options: tuple[str, ...],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    while True:
        _emit(output_fn, f"{label}: {', '.join(options)}")
        value = input_fn(f"{label}: ").strip()
        if value in options:
            return value
        _emit(output_fn, f"Invalid selection: {value}")


def _prompt_int(*, label: str, input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> int:
    while True:
        raw_value = input_fn(f"{label}: ").strip()
        try:
            value = int(raw_value)
        except ValueError:
            _emit(output_fn, f"{label} must be an integer.")
            continue
        if value <= 0:
            _emit(output_fn, f"{label} must be greater than zero.")
            continue
        return value


def _prompt_bool(*, label: str, input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> bool:
    while True:
        raw_value = input_fn(f"{label} (true/false): ").strip().lower()
        if raw_value in {"true", "false"}:
            return raw_value == "true"
        _emit(output_fn, "Please answer true or false.")


def _build_advanced_campaign_config(
    *,
    circuit_specs: list[CampaignCircuitSpec],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> CampaignConfig:
    backend_names = _prompt_choice_list(
        label="Backends",
        options=_AVAILABLE_BACKENDS,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    rl_algorithm = _prompt_choice(
        label="RL algorithm",
        options=_AVAILABLE_ALGORITHMS,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    rl_total_timesteps = _prompt_int(label="RL timesteps", input_fn=input_fn, output_fn=output_fn)
    rl_frontier_mode = input_fn("RL frontier mode: ").strip()
    rl_lookahead_window = _prompt_int(label="RL lookahead window", input_fn=input_fn, output_fn=output_fn)
    rl_max_steps = _prompt_int(label="RL max steps", input_fn=input_fn, output_fn=output_fn)
    seed = _prompt_int(label="Seed", input_fn=input_fn, output_fn=output_fn)
    mo_use_quick = _prompt_bool(label="Use MO quick", input_fn=input_fn, output_fn=output_fn)
    mo_population_size = _prompt_int(label="MO population size", input_fn=input_fn, output_fn=output_fn)
    mo_n_generations = _prompt_int(label="MO generations", input_fn=input_fn, output_fn=output_fn)
    layout_policy_key = _prompt_choice(
        label="Layout policy",
        options=tuple(_AVAILABLE_LAYOUT_POLICIES.keys()),
        input_fn=input_fn,
        output_fn=output_fn,
    )
    layout_policy = _AVAILABLE_LAYOUT_POLICIES[layout_policy_key]
    mo_objective_name = None
    if layout_policy is LayoutSelectionPolicy.BEST_ON_OBJECTIVE:
        mo_objective_name = _prompt_choice(
            label="MO objective",
            options=_AVAILABLE_OBJECTIVE_NAMES,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    return CampaignConfig(
        circuit_specs=circuit_specs,
        backend_names=backend_names,
        rl_algorithm=rl_algorithm,
        rl_total_timesteps=rl_total_timesteps,
        rl_frontier_mode=rl_frontier_mode,
        rl_lookahead_window=rl_lookahead_window,
        rl_max_steps=rl_max_steps,
        seed=seed,
        mo_use_quick=mo_use_quick,
        mo_population_size=mo_population_size,
        mo_n_generations=mo_n_generations,
        layout_policy=layout_policy,
        mo_objective_name=mo_objective_name,
        mode="advanced",
    )
```

Then change the end of `build_campaign_config_interactively(...)` to:

```python
    if mode == "default":
        return CampaignConfig(
            circuit_specs=circuit_specs,
            backend_names=["fake_torino"],
            rl_algorithm="MaskablePPO",
            rl_total_timesteps=5000,
            rl_frontier_mode="dag",
            rl_lookahead_window=10,
            rl_max_steps=200,
            seed=42,
            mo_use_quick=True,
            mo_population_size=30,
            mo_n_generations=50,
            layout_policy=LayoutSelectionPolicy.COMPROMISE,
            mode="default",
        )
    return _build_advanced_campaign_config(
        circuit_specs=circuit_specs,
        input_fn=input_fn,
        output_fn=output_fn,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py::test_advanced_campaign_cli_accepts_multiple_backends tests/test_integration/test_campaign_cli.py::test_advanced_campaign_cli_collects_mo_policy_and_objective tests/test_integration/test_campaign_cli.py::test_campaign_cli_reprompts_on_invalid_backend_selection -q
```

Expected: PASS.

---

### Task 3: Add Confirmation, Campaign Execution, and Final Output Printing

**Files:**
- Modify: `src/integration/campaign_cli.py`
- Test: `tests/test_integration/test_campaign_cli.py`

- [ ] **Step 1: Write failing execution-wrapper tests**

```python
def test_run_campaign_cli_builds_campaign_confirms_and_prints_final_paths(tmp_path) -> None:
    from src.integration.campaign_cli import run_campaign_cli

    outputs: list[str] = []
    run_calls: list[tuple[str, tuple[str, ...]]] = []
    answers = iter([
        "ghz",
        "3",
        "default",
        "yes",
    ])

    def fake_run_campaign(campaign, *, output_root):
        run_calls.append((campaign.campaign_id, campaign.config.backend_names))
        return type(
            "Report",
            (),
            {
                "campaign_id": campaign.campaign_id,
                "campaign_status": "completed",
                "summary": campaign.summary,
            },
        )()

    exit_code = run_campaign_cli(
        input_fn=lambda prompt: next(answers),
        output_fn=outputs.append,
        run_campaign_fn=fake_run_campaign,
        output_root=tmp_path / "campaigns",
        campaign_id_factory=lambda: "campaign-test",
    )

    assert exit_code == 0
    assert run_calls == [("campaign-test", ("fake_torino",))]
    assert any("Campaign ID: campaign-test" in message for message in outputs)
    assert any("Final Status: completed" in message for message in outputs)
    assert any("summary.md" in message for message in outputs)


def test_run_campaign_cli_aborts_when_confirmation_is_rejected() -> None:
    from src.integration.campaign_cli import run_campaign_cli

    outputs: list[str] = []
    answers = iter([
        "ghz",
        "3",
        "default",
        "no",
    ])
    run_calls: list[str] = []

    exit_code = run_campaign_cli(
        input_fn=lambda prompt: next(answers),
        output_fn=outputs.append,
        run_campaign_fn=lambda campaign, *, output_root: run_calls.append(campaign.campaign_id),
        campaign_id_factory=lambda: "campaign-test",
    )

    assert exit_code == 1
    assert run_calls == []
    assert any("Campaign cancelled before execution." in message for message in outputs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py::test_run_campaign_cli_builds_campaign_confirms_and_prints_final_paths tests/test_integration/test_campaign_cli.py::test_run_campaign_cli_aborts_when_confirmation_is_rejected -q
```

Expected: FAIL because `run_campaign_cli` and confirmation flow do not exist yet.

- [ ] **Step 3: Write the minimal execution wrapper**

```python
def _prompt_yes_no(*, label: str, input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> bool:
    while True:
        value = input_fn(f"{label} (yes/no): ").strip().lower()
        if value in {"yes", "no"}:
            return value == "yes"
        _emit(output_fn, "Please answer yes or no.")


def _render_campaign_config_summary(config: CampaignConfig) -> list[str]:
    lines = [
        "Campaign Configuration Summary",
        f"Mode: {config.mode}",
        f"Circuits: {[f'{spec.family}_{spec.num_qubits}' for spec in config.circuit_specs]}",
        f"Backends: {list(config.backend_names)}",
        f"RL Algorithm: {config.rl_algorithm}",
        f"RL Timesteps: {config.rl_total_timesteps}",
        f"RL Frontier Mode: {config.rl_frontier_mode}",
        f"RL Lookahead Window: {config.rl_lookahead_window}",
        f"RL Max Steps: {config.rl_max_steps}",
        f"Seed: {config.seed}",
        f"MO Quick: {config.mo_use_quick}",
        f"MO Population Size: {config.mo_population_size}",
        f"MO Generations: {config.mo_n_generations}",
        f"Layout Policy: {config.layout_policy.value}",
    ]
    if config.mo_objective_name is not None:
        lines.append(f"MO Objective: {config.mo_objective_name}")
    return lines


def run_campaign_cli(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    run_campaign_fn: Callable[..., object] = run_campaign,
    output_root: Path | str = "campaigns",
    campaign_id_factory: Callable[[], str] | None = None,
) -> int:
    config = build_campaign_config_interactively(input_fn=input_fn, output_fn=output_fn)
    for line in _render_campaign_config_summary(config):
        _emit(output_fn, line)
    confirmed = _prompt_yes_no(label="Execute campaign", input_fn=input_fn, output_fn=output_fn)
    if not confirmed:
        _emit(output_fn, "Campaign cancelled before execution.")
        return 1

    if campaign_id_factory is None:
        from uuid import uuid4

        campaign_id_factory = lambda: f"campaign-{uuid4().hex[:8]}"

    campaign = Campaign.from_config(campaign_id=campaign_id_factory(), config=config)
    report = run_campaign_fn(campaign, output_root=output_root)
    campaign_output_dir = Path(output_root) / campaign.campaign_id
    _emit(output_fn, f"Campaign ID: {campaign.campaign_id}")
    _emit(output_fn, f"Final Status: {report.campaign_status}")
    _emit(output_fn, f"Summary Path: {campaign_output_dir / 'summary.md'}")
    _emit(output_fn, f"Structured Output Path: {campaign_output_dir / 'campaign.json'}")
    return 0


def main() -> int:
    return run_campaign_cli()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py::test_run_campaign_cli_builds_campaign_confirms_and_prints_final_paths tests/test_integration/test_campaign_cli.py::test_run_campaign_cli_aborts_when_confirmation_is_rejected -q
```

Expected: PASS.

---

### Task 4: Run the Focused CLI Test Suite

**Files:**
- Modify: `src/integration/campaign_cli.py`
- Test: `tests/test_integration/test_campaign_cli.py`

- [ ] **Step 1: Run the full new CLI test file**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent runner tests to confirm no regression in the existing scenario CLI**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_runner.py tests/test_integration/test_campaign_cli.py -q
```

Expected: PASS.

- [ ] **Step 3: Check worktree state before moving to later tasks**

Run:

```bash
git status --short
```

Expected: only intended `campaign_cli.py`, `test_campaign_cli.py`, and pre-existing unrelated changes remain.

---

## Self-Review

- Spec coverage check: this plan covers the separate `campaign_cli.py` entrypoint, reprompt behavior, default and advanced config building, confirmation summary, execution wrapper, and final path printing.
- Placeholder scan: no `TODO`, `TBD`, or deferred “write tests later” steps remain.
- Type consistency check: the plan consistently uses `CampaignConfig`, `CampaignCircuitSpec`, `Campaign.from_config(...)`, and `run_campaign(...)` from the existing integration layer.
