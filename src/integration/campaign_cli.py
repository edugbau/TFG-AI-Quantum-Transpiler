from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.integration.campaign_contracts import Campaign, CampaignCircuitSpec, CampaignConfig
from src.integration.mo_effort import MIN_CUSTOM_MO_POPULATION_SIZE, build_auto_mo_effort_preview
from src.integration.campaign_runner import run_campaign
from src.integration.contracts import LayoutSelectionPolicy
from src.integration.synthetic_topology import SYNTHETIC_TOPOLOGY_SHAPES, SyntheticTopologySpec
from src.mo_module.fitness import get_preset_objectives


_CIRCUIT_FAMILIES = (
    "ghz",
    "qft",
    "qft_inv",
    "random_shallow",
    "random_deep",
    "clifford",
)
_BACKENDS = ("fake_torino", "fake_brisbane")
_RL_ALGORITHMS = ("MaskablePPO", "PPO", "DQN")
_FRONTIER_MODES = ("dag", "sequential")
_LAYOUT_POLICIES = tuple(policy.value for policy in LayoutSelectionPolicy)

_DEFAULT_RL_ALGORITHM = "MaskablePPO"
_DEFAULT_RL_TIMESTEPS = 5000
_DEFAULT_RL_FRONTIER_MODE = "dag"
_DEFAULT_RL_LOOKAHEAD = 10
_DEFAULT_RL_MAX_STEPS = 200
_DEFAULT_SEED = 42
_DEFAULT_MO_USE_QUICK = True
_DEFAULT_MO_POPULATION_SIZE = 30
_DEFAULT_MO_N_GENERATIONS = 50
_MO_EFFORT_MODES = ("auto", "custom")
_TOPOLOGY_SOURCES = ("backend", "synthetic")
_DEFAULT_LAYOUT_POLICY = LayoutSelectionPolicy.COMPROMISE
_DEFAULT_BACKEND = "fake_torino"


def _default_campaign_id() -> str:
    return f"campaign-{uuid4().hex[:8]}"


def _available_objective_names() -> tuple[str, ...]:
    return tuple(get_preset_objectives("default"))


def build_default_campaign_config(
    *,
    circuit_specs: list[CampaignCircuitSpec] | tuple[CampaignCircuitSpec, ...],
    backend_name: str = _DEFAULT_BACKEND,
) -> CampaignConfig:
    return CampaignConfig(
        circuit_specs=tuple(circuit_specs),
        backend_names=(backend_name,),
        rl_algorithm=_DEFAULT_RL_ALGORITHM,
        rl_total_timesteps=_DEFAULT_RL_TIMESTEPS,
        rl_frontier_mode=_DEFAULT_RL_FRONTIER_MODE,
        rl_lookahead_window=_DEFAULT_RL_LOOKAHEAD,
        rl_max_steps=_DEFAULT_RL_MAX_STEPS,
        seed=_DEFAULT_SEED,
        mo_use_quick=_DEFAULT_MO_USE_QUICK,
        mo_population_size=_DEFAULT_MO_POPULATION_SIZE,
        mo_n_generations=_DEFAULT_MO_N_GENERATIONS,
        mo_effort_mode="auto",
        layout_policy=_DEFAULT_LAYOUT_POLICY,
        mode="default",
        topology_source="backend",
    )


def _prompt_non_empty(input_fn, output_fn, *, prompt: str, error_message: str) -> str:
    while True:
        value = input_fn(prompt).strip()
        if value:
            return value
        output_fn(error_message)


def _prompt_csv_choices(
    input_fn,
    output_fn,
    *,
    prompt: str,
    valid_values: tuple[str, ...],
    allow_multiple: bool,
    default_values: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    valid_lookup = {value.lower(): value for value in valid_values}
    default_values = tuple(default_values or ())
    while True:
        raw = input_fn(prompt).strip()
        if not raw and default_values:
            return default_values
        parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
        if not parts:
            output_fn(f"Invalid selection. Choose from: {', '.join(valid_values)}")
            continue
        normalized: list[str] = []
        invalid = [part for part in parts if part not in valid_lookup]
        if invalid:
            output_fn(f"Invalid selection. Choose from: {', '.join(valid_values)}")
            continue
        for part in parts:
            value = valid_lookup[part]
            if value not in normalized:
                normalized.append(value)
        if not allow_multiple and len(normalized) != 1:
            output_fn(f"Invalid selection. Choose one of: {', '.join(valid_values)}")
            continue
        return tuple(normalized)


def _prompt_qubit_sizes(input_fn, output_fn) -> tuple[int, ...]:
    while True:
        raw = input_fn("Choose qubit sizes (comma-separated positive integers): ").strip()
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            output_fn("Invalid selection. Enter at least one positive integer.")
            continue
        try:
            values = [int(part) for part in parts]
        except ValueError:
            output_fn("Invalid selection. Enter at least one positive integer.")
            continue
        if any(value <= 0 for value in values):
            output_fn("Invalid selection. Enter at least one positive integer.")
            continue
        return tuple(dict.fromkeys(values))


def _prompt_mode(input_fn, output_fn) -> str:
    return _prompt_csv_choices(
        input_fn,
        output_fn,
        prompt="Choose campaign mode (default/advanced): ",
        valid_values=("default", "advanced"),
        allow_multiple=False,
    )[0]


def _prompt_int(input_fn, output_fn, *, prompt: str, minimum: int = 0) -> int:
    while True:
        raw = input_fn(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            output_fn(f"Invalid selection. Enter an integer >= {minimum}.")
            continue
        if value < minimum:
            output_fn(f"Invalid selection. Enter an integer >= {minimum}.")
            continue
        return value


def _prompt_mo_effort_mode(input_fn, output_fn) -> str:
    return _prompt_csv_choices(
        input_fn,
        output_fn,
        prompt=f"Choose MO effort mode ({', '.join(_MO_EFFORT_MODES)}): ",
        valid_values=_MO_EFFORT_MODES,
        allow_multiple=False,
    )[0]


def _prompt_topology_source(input_fn, output_fn) -> str:
    return _prompt_csv_choices(
        input_fn,
        output_fn,
        prompt=f"Choose topology source ({', '.join(_TOPOLOGY_SOURCES)}): ",
        valid_values=_TOPOLOGY_SOURCES,
        allow_multiple=False,
    )[0]


def _prompt_synthetic_topology(input_fn, output_fn, *, required_qubits: int) -> SyntheticTopologySpec:
    while True:
        shape = _prompt_csv_choices(
            input_fn,
            output_fn,
            prompt=f"Choose synthetic topology ({', '.join(SYNTHETIC_TOPOLOGY_SHAPES)}): ",
            valid_values=SYNTHETIC_TOPOLOGY_SHAPES,
            allow_multiple=False,
        )[0]
        try:
            if shape in {"full", "line", "ring"}:
                spec = SyntheticTopologySpec(
                    shape=shape,
                    num_qubits=_prompt_int(input_fn, output_fn, prompt="Synthetic num qubits: ", minimum=1),
                )
            elif shape in {"grid", "hexagonal_lattice"}:
                spec = SyntheticTopologySpec(
                    shape=shape,
                    rows=_prompt_int(input_fn, output_fn, prompt="Synthetic rows: ", minimum=1),
                    cols=_prompt_int(input_fn, output_fn, prompt="Synthetic columns: ", minimum=1),
                )
            else:
                spec = SyntheticTopologySpec(
                    shape=shape,
                    distance=_prompt_int(input_fn, output_fn, prompt="Synthetic distance: ", minimum=1),
                )
        except ValueError as exc:
            output_fn(f"Invalid synthetic topology. {exc}")
            continue
        try:
            physical_qubits = spec.physical_qubits
        except Exception as exc:
            output_fn(f"Invalid synthetic topology. {exc}")
            continue
        if physical_qubits < required_qubits:
            output_fn(
                "Invalid synthetic topology. "
                f"Requires at least {required_qubits} physical qubits, got {physical_qubits}."
            )
            continue
        return spec


def _prompt_bool(input_fn, output_fn, *, prompt: str) -> bool:
    while True:
        raw = input_fn(prompt).strip().lower()
        if raw in {"true", "t", "yes", "y", "1"}:
            return True
        if raw in {"false", "f", "no", "n", "0"}:
            return False
        output_fn("Invalid selection. Enter true/false.")


def _prompt_confirmation(input_fn, output_fn) -> bool:
    while True:
        raw = input_fn("Proceed with campaign execution? [y/n]: ").strip().lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        output_fn("Invalid selection. Enter y or n.")


def _build_circuit_specs(families: tuple[str, ...], qubit_sizes: tuple[int, ...]) -> tuple[CampaignCircuitSpec, ...]:
    return tuple(CampaignCircuitSpec(family=family, num_qubits=num_qubits) for family in families for num_qubits in qubit_sizes)


def _collect_advanced_config(input_fn, output_fn, *, circuit_specs: tuple[CampaignCircuitSpec, ...]) -> CampaignConfig:
    topology_source = _prompt_topology_source(input_fn, output_fn)
    synthetic_topology = None
    if topology_source == "backend":
        backend_names = _prompt_csv_choices(
            input_fn,
            output_fn,
            prompt=f"Choose backend(s) ({', '.join(_BACKENDS)}): ",
            valid_values=_BACKENDS,
            allow_multiple=True,
        )
    else:
        synthetic_topology = _prompt_synthetic_topology(
            input_fn,
            output_fn,
            required_qubits=max(spec.num_qubits for spec in circuit_specs),
        )
        backend_names = (synthetic_topology.backend_name,)
    rl_algorithm = _prompt_csv_choices(
        input_fn,
        output_fn,
        prompt=f"Choose RL algorithm ({', '.join(_RL_ALGORITHMS)}): ",
        valid_values=_RL_ALGORITHMS,
        allow_multiple=False,
    )[0]
    rl_timesteps = _prompt_int(input_fn, output_fn, prompt="RL timesteps: ", minimum=1)
    rl_frontier_mode = _prompt_csv_choices(
        input_fn,
        output_fn,
        prompt=f"Choose RL frontier mode ({', '.join(_FRONTIER_MODES)}): ",
        valid_values=_FRONTIER_MODES,
        allow_multiple=False,
    )[0]
    rl_lookahead = _prompt_int(input_fn, output_fn, prompt="RL lookahead window: ", minimum=1)
    rl_max_steps = _prompt_int(input_fn, output_fn, prompt="RL max steps: ", minimum=1)
    seed = _prompt_int(input_fn, output_fn, prompt="Seed: ", minimum=0)
    mo_effort_mode = _prompt_mo_effort_mode(input_fn, output_fn)
    mo_use_quick = _DEFAULT_MO_USE_QUICK
    mo_population_size = _DEFAULT_MO_POPULATION_SIZE
    mo_n_generations = _DEFAULT_MO_N_GENERATIONS
    if mo_effort_mode == "custom":
        mo_use_quick = _prompt_bool(input_fn, output_fn, prompt="MO quick: ")
        mo_population_size = _prompt_int(
            input_fn,
            output_fn,
            prompt="MO population size: ",
            minimum=MIN_CUSTOM_MO_POPULATION_SIZE,
        )
        mo_n_generations = _prompt_int(input_fn, output_fn, prompt="MO generations: ", minimum=1)
    layout_policy_name = _prompt_csv_choices(
        input_fn,
        output_fn,
        prompt=f"Choose layout policy ({', '.join(_LAYOUT_POLICIES)}): ",
        valid_values=_LAYOUT_POLICIES,
        allow_multiple=False,
    )[0]
    layout_policy = LayoutSelectionPolicy(layout_policy_name)
    mo_objective_name = None
    if layout_policy is LayoutSelectionPolicy.BEST_ON_OBJECTIVE:
        objective_names = _available_objective_names()
        mo_objective_name = _prompt_csv_choices(
            input_fn,
            output_fn,
            prompt=f"Choose MO objective ({', '.join(objective_names)}): ",
            valid_values=objective_names,
            allow_multiple=False,
        )[0]

    return CampaignConfig(
        circuit_specs=circuit_specs,
        backend_names=backend_names,
        rl_algorithm=rl_algorithm,
        rl_total_timesteps=rl_timesteps,
        rl_frontier_mode=rl_frontier_mode,
        rl_lookahead_window=rl_lookahead,
        rl_max_steps=rl_max_steps,
        seed=seed,
        mo_use_quick=mo_use_quick,
        mo_population_size=mo_population_size,
        mo_n_generations=mo_n_generations,
        mo_effort_mode=mo_effort_mode,
        layout_policy=layout_policy,
        mo_objective_name=mo_objective_name,
        mode="advanced",
        topology_source=topology_source,
        synthetic_topology=synthetic_topology,
    )


def _collect_campaign_config(input_fn, output_fn) -> CampaignConfig:
    families = _prompt_csv_choices(
        input_fn,
        output_fn,
        prompt=f"Choose circuit family/families ({', '.join(_CIRCUIT_FAMILIES)}): ",
        valid_values=_CIRCUIT_FAMILIES,
        allow_multiple=True,
    )
    qubit_sizes = _prompt_qubit_sizes(input_fn, output_fn)
    mode = _prompt_mode(input_fn, output_fn)
    circuit_specs = _build_circuit_specs(families, qubit_sizes)

    if mode == "default":
        return build_default_campaign_config(circuit_specs=circuit_specs)

    return _collect_advanced_config(input_fn, output_fn, circuit_specs=circuit_specs)


def _print_confirmation_summary(output_fn, *, campaign: Campaign) -> None:
    config = campaign.config
    output_fn("Confirmation Summary")
    output_fn(f"Campaign ID: {campaign.campaign_id}")
    output_fn(f"Mode: {config.mode}")
    output_fn("Circuits: " + ", ".join(f"{spec.family} ({spec.num_qubits}q)" for spec in config.circuit_specs))
    output_fn(f"Topology Source: {config.topology_source}")
    if config.synthetic_topology is None:
        output_fn("Backends: " + ", ".join(config.backend_names))
    else:
        output_fn(f"Synthetic Topology: {config.synthetic_topology.backend_name}")
        output_fn(f"Synthetic Physical Qubits: {config.synthetic_topology.physical_qubits}")
        output_fn("Synthetic Basis Gates: " + ", ".join(config.synthetic_topology.basis_gates))
    output_fn(
        "RL: "
        f"algorithm={config.rl_algorithm}, timesteps={config.rl_total_timesteps}, "
        f"frontier_mode={config.rl_frontier_mode}, lookahead={config.rl_lookahead_window}, "
        f"max_steps={config.rl_max_steps}, seed={config.seed}"
    )
    output_fn(f"MO Effort Mode: {config.mo_effort_mode}")
    if config.mo_effort_mode == "auto":
        for qubit_count, settings in build_auto_mo_effort_preview(
            spec.num_qubits for spec in config.circuit_specs
        ):
            output_fn(
                f"MO Auto Preview ({qubit_count}q): "
                f"quick={settings.mo_use_quick}, population_size={settings.mo_population_size}, "
                f"n_generations={settings.mo_n_generations}"
            )
    else:
        output_fn(f"MO Quick: {config.mo_use_quick}")
        output_fn(f"MO Population Size: {config.mo_population_size}")
        output_fn(f"MO Generations: {config.mo_n_generations}")
    output_fn(f"Layout Policy: {config.layout_policy.value}")
    if config.mo_objective_name is not None:
        output_fn(f"MO Objective: {config.mo_objective_name}")


def run_interactive_campaign_cli(
    *,
    input_fn=input,
    output_fn=print,
    run_campaign_fn=run_campaign,
    campaign_id_factory=_default_campaign_id,
    output_root: Path | str = "campaigns",
) -> int:
    config = _collect_campaign_config(input_fn, output_fn)
    campaign = Campaign.from_config(campaign_id=campaign_id_factory(), config=config)
    _print_confirmation_summary(output_fn, campaign=campaign)

    if not _prompt_confirmation(input_fn, output_fn):
        output_fn("Campaign execution aborted.")
        return 0

    output_fn("Executing campaign...")
    report = run_campaign_fn(campaign, output_root=output_root)
    campaign_output_dir = Path(output_root) / campaign.campaign_id
    output_fn(f"Campaign ID: {campaign.campaign_id}")
    output_fn(f"Final Status: {report.campaign_status}")
    output_fn(f"Summary Document: {campaign_output_dir / 'summary.md'}")
    output_fn(f"Structured Campaign Output: {campaign_output_dir / 'campaign.json'}")
    return 0


def main() -> int:
    return run_interactive_campaign_cli()


if __name__ == "__main__":
    raise SystemExit(main())
