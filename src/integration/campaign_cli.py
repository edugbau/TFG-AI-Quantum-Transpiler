from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from src.integration.campaign_contracts import Campaign, CampaignCircuitSpec, CampaignConfig
from src.integration.campaign_matrix import (
    ALL_MO_SELECTION_MODES,
    is_matrix_campaign_config,
    run_campaign_matrix,
)
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
_MO_SELECTION_MODE_CHOICES = ("all", "compromise", "best_depth", "best_cnot_count", "best_on_objective")

_DEFAULT_RL_ALGORITHM = "MaskablePPO"
_DEFAULT_RL_TIMESTEPS = 5000
_DEFAULT_RL_FRONTIER_MODE = "dag"
_DEFAULT_RL_LOOKAHEAD = 10
_DEFAULT_RL_MAX_STEPS = 200
_DEFAULT_RL_LEARNING_RATE = 1e-4
_DEFAULT_RL_CLIP_RANGE = 0.1
_DEFAULT_RL_TARGET_KL = 0.03
_DEFAULT_RL_N_EVAL_EPISODES = 5
_DEFAULT_SEED = 42
_DEFAULT_MO_USE_QUICK = True
_DEFAULT_MO_POPULATION_SIZE = 30
_DEFAULT_MO_N_GENERATIONS = 50
_MO_EFFORT_MODES = ("auto", "custom")
_TOPOLOGY_SOURCES = ("backend", "synthetic")
_DEFAULT_LAYOUT_POLICY = LayoutSelectionPolicy.COMPROMISE
_DEFAULT_BACKEND = "fake_torino"


@dataclass(frozen=True, slots=True)
class BatchCampaignResult:
    campaign_id: str
    status: str
    summary_document: str
    structured_output: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BatchReport:
    status: str
    dry_run: bool
    total_campaigns: int
    completed_campaigns: int
    failed_campaigns: int
    results: tuple[BatchCampaignResult, ...]
    summary_path: str


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
        rl_learning_rate=_DEFAULT_RL_LEARNING_RATE,
        rl_clip_range=_DEFAULT_RL_CLIP_RANGE,
        rl_target_kl=_DEFAULT_RL_TARGET_KL,
        rl_n_eval_episodes=_DEFAULT_RL_N_EVAL_EPISODES,
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


def _prompt_seed_values(input_fn, output_fn) -> tuple[int, ...]:
    while True:
        raw = input_fn("Seeds (comma-separated non-negative integers): ").strip()
        try:
            return _normalize_seed_values(raw, field_name="seeds")
        except ValueError as exc:
            output_fn(f"Invalid selection. {exc}")


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


def _prompt_mo_selection_modes(input_fn, output_fn) -> tuple[str, ...]:
    while True:
        values = _prompt_csv_choices(
            input_fn,
            output_fn,
            prompt=f"Choose MO selection mode(s) ({', '.join(_MO_SELECTION_MODE_CHOICES)}): ",
            valid_values=_MO_SELECTION_MODE_CHOICES,
            allow_multiple=True,
        )
        if "all" in values:
            if len(values) != 1:
                output_fn("Invalid selection. Use all by itself.")
                continue
            return ALL_MO_SELECTION_MODES
        if "best_on_objective" in values:
            if len(values) != 1:
                output_fn("Invalid selection. best_on_objective must be selected by itself.")
                continue
            objective_names = _available_objective_names()
            objective_name = _prompt_csv_choices(
                input_fn,
                output_fn,
                prompt=f"Choose MO objective ({', '.join(objective_names)}): ",
                valid_values=objective_names,
                allow_multiple=False,
            )[0]
            return (_selection_mode_from_objective(objective_name),)
        return tuple(values)


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
    seeds = _prompt_seed_values(input_fn, output_fn)
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
    mo_selection_modes = _prompt_mo_selection_modes(input_fn, output_fn)
    layout_policy, mo_objective_name = _policy_for_selection_mode(mo_selection_modes[0])
    parallel_workers = 1
    if len(seeds) * len(mo_selection_modes) > 1:
        parallel_workers = _prompt_int(input_fn, output_fn, prompt="Parallel workers: ", minimum=1)

    return CampaignConfig(
        circuit_specs=circuit_specs,
        backend_names=backend_names,
        rl_algorithm=rl_algorithm,
        rl_total_timesteps=rl_timesteps,
        rl_frontier_mode=rl_frontier_mode,
        rl_lookahead_window=rl_lookahead,
        rl_max_steps=rl_max_steps,
        rl_learning_rate=_DEFAULT_RL_LEARNING_RATE,
        rl_clip_range=_DEFAULT_RL_CLIP_RANGE,
        rl_target_kl=_DEFAULT_RL_TARGET_KL,
        rl_n_eval_episodes=_DEFAULT_RL_N_EVAL_EPISODES,
        seed=seeds[0],
        seeds=seeds,
        mo_use_quick=mo_use_quick,
        mo_population_size=mo_population_size,
        mo_n_generations=mo_n_generations,
        mo_effort_mode=mo_effort_mode,
        layout_policy=layout_policy,
        mo_objective_name=mo_objective_name,
        mo_selection_modes=mo_selection_modes,
        parallel_workers=parallel_workers,
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
        f"max_steps={config.rl_max_steps}, learning_rate={config.rl_learning_rate}, "
        f"clip_range={config.rl_clip_range}, target_kl={config.rl_target_kl}, "
        f"n_eval_episodes={config.rl_n_eval_episodes}, seed={config.seed}"
    )
    if len(config.seeds) > 1:
        output_fn("Seeds: " + ", ".join(str(seed) for seed in config.seeds))
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
    output_fn("MO Selection Modes: " + ", ".join(config.mo_selection_modes))
    if is_matrix_campaign_config(config):
        output_fn(f"Parallel Workers: {config.parallel_workers}")


def _require_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_int(mapping: dict[str, Any], key: str, default: int, *, minimum: int = 0) -> int:
    if key not in mapping:
        return default
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _optional_float(mapping: dict[str, Any], key: str, default: float, *, minimum_exclusive: float = 0.0) -> float:
    if key not in mapping:
        return default
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    normalized = float(value)
    if normalized <= minimum_exclusive:
        raise ValueError(f"{key} must be greater than {minimum_exclusive}")
    return normalized


def _optional_bool(mapping: dict[str, Any], key: str, default: bool) -> bool:
    if key not in mapping:
        return default
    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _normalize_choice(
    value: object,
    *,
    field_name: str,
    valid_values: tuple[str, ...],
) -> str:
    raw = _require_string(value, field_name=field_name).lower()
    valid_lookup = {valid_value.lower(): valid_value for valid_value in valid_values}
    if raw not in valid_lookup:
        raise ValueError(
            f"{field_name} has invalid value {raw!r}; choose one of: {', '.join(valid_values)}"
        )
    return valid_lookup[raw]


def _normalize_backend_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("backend_names must be a non-empty array")
    backends = tuple(
        _normalize_choice(backend_name, field_name="backend_names", valid_values=_BACKENDS)
        for backend_name in value
    )
    if len(set(backends)) != len(backends):
        raise ValueError("backend_names cannot contain duplicates")
    return backends


def _normalize_seed_values(value: object, *, field_name: str = "seeds") -> tuple[int, ...]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
        if not raw_values:
            raise ValueError(f"{field_name} must contain at least one seed")
        try:
            values = [int(part) for part in raw_values]
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain integers") from exc
    elif isinstance(value, list):
        if not value:
            raise ValueError(f"{field_name} must be a non-empty array")
        values = value
    else:
        raise ValueError(f"{field_name} must be a non-empty array")

    normalized: list[int] = []
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{field_name} must contain integers")
        if seed < 0:
            raise ValueError(f"{field_name} cannot contain negative values")
        if seed not in normalized:
            normalized.append(seed)
    return tuple(normalized)


def _normalize_mo_selection_modes(value: object, *, field_name: str = "mo.selection_modes") -> tuple[str, ...]:
    if isinstance(value, str):
        raw_modes = (value.strip().lower(),)
    elif isinstance(value, list):
        if not value:
            raise ValueError(f"{field_name} must be a non-empty array")
        raw_modes = tuple(_require_string(mode, field_name=field_name).lower() for mode in value)
    else:
        raise ValueError(f"{field_name} must be all or a non-empty array")
    if raw_modes == ("all",):
        return ALL_MO_SELECTION_MODES
    if "all" in raw_modes:
        raise ValueError(f"{field_name} cannot combine all with explicit modes")
    normalized: list[str] = []
    valid_modes = set(ALL_MO_SELECTION_MODES)
    for mode in raw_modes:
        if mode not in valid_modes:
            raise ValueError(
                f"{field_name} has invalid value {mode!r}; choose all, compromise, best_depth, or best_cnot_count"
            )
        if mode not in normalized:
            normalized.append(mode)
    return tuple(normalized)


def _selection_mode_from_objective(objective_name: str) -> str:
    if objective_name == "depth":
        return "best_depth"
    if objective_name == "cnot_count":
        return "best_cnot_count"
    raise ValueError("Only depth and cnot_count are supported as campaign MO selection objectives")


def _policy_for_selection_mode(mode: str) -> tuple[LayoutSelectionPolicy, str | None]:
    if mode == "compromise":
        return LayoutSelectionPolicy.COMPROMISE, None
    if mode == "best_depth":
        return LayoutSelectionPolicy.BEST_ON_OBJECTIVE, "depth"
    if mode == "best_cnot_count":
        return LayoutSelectionPolicy.BEST_ON_OBJECTIVE, "cnot_count"
    raise ValueError(f"Unsupported MO selection mode: {mode}")


def _build_batch_campaign_config(entry: dict[str, Any]) -> CampaignConfig:
    circuit = _require_mapping(entry.get("circuit"), field_name="circuit")
    family = _normalize_choice(circuit.get("family"), field_name="circuit.family", valid_values=_CIRCUIT_FAMILIES)
    num_qubits = _optional_int(circuit, "num_qubits", 0, minimum=1)
    circuit_specs = (CampaignCircuitSpec(family=family, num_qubits=num_qubits),)
    mode = _normalize_choice(entry.get("mode", "default"), field_name="mode", valid_values=("default", "advanced"))
    topology_source = _normalize_choice(
        entry.get("topology_source", "backend"),
        field_name="topology_source",
        valid_values=("backend",),
    )
    backend_names = _normalize_backend_names(entry.get("backend_names", [_DEFAULT_BACKEND]))
    base_config = build_default_campaign_config(circuit_specs=circuit_specs, backend_name=backend_names[0])
    rl = _require_mapping(entry.get("rl", {}), field_name="rl")
    mo = _require_mapping(entry.get("mo", {}), field_name="mo")
    rl_algorithm = _normalize_choice(
        rl.get("algorithm", base_config.rl_algorithm),
        field_name="rl.algorithm",
        valid_values=_RL_ALGORITHMS,
    )
    rl_frontier_mode = _normalize_choice(
        rl.get("frontier_mode", base_config.rl_frontier_mode),
        field_name="rl.frontier_mode",
        valid_values=_FRONTIER_MODES,
    )
    mo_effort_mode = _normalize_choice(
        mo.get("effort_mode", base_config.mo_effort_mode),
        field_name="mo.effort_mode",
        valid_values=_MO_EFFORT_MODES,
    )
    if "seeds" in entry:
        seeds = _normalize_seed_values(entry["seeds"])
    else:
        seeds = (_optional_int(entry, "seed", base_config.seed, minimum=0),)
    if "selection_modes" in mo:
        mo_selection_modes = _normalize_mo_selection_modes(mo["selection_modes"])
        layout_policy, mo_objective_name = _policy_for_selection_mode(mo_selection_modes[0])
    else:
        layout_policy_name = _normalize_choice(
            mo.get("layout_policy", base_config.layout_policy.value),
            field_name="mo.layout_policy",
            valid_values=_LAYOUT_POLICIES,
        )
        layout_policy = LayoutSelectionPolicy(layout_policy_name)
        mo_objective_name = mo.get("objective_name")
        if mo_objective_name is not None:
            mo_objective_name = _normalize_choice(
                mo_objective_name,
                field_name="mo.objective_name",
                valid_values=_available_objective_names(),
            )
        if layout_policy is LayoutSelectionPolicy.BEST_ON_OBJECTIVE:
            mo_selection_modes = (_selection_mode_from_objective(mo_objective_name),)
        else:
            mo_selection_modes = ("compromise",)

    return CampaignConfig(
        circuit_specs=base_config.circuit_specs,
        backend_names=backend_names,
        rl_algorithm=rl_algorithm,
        rl_total_timesteps=_optional_int(
            rl,
            "total_timesteps",
            base_config.rl_total_timesteps,
            minimum=1,
        ),
        rl_frontier_mode=rl_frontier_mode,
        rl_lookahead_window=_optional_int(
            rl,
            "lookahead_window",
            base_config.rl_lookahead_window,
            minimum=1,
        ),
        rl_max_steps=_optional_int(rl, "max_steps", base_config.rl_max_steps, minimum=1),
        rl_learning_rate=_optional_float(rl, "learning_rate", base_config.rl_learning_rate),
        rl_clip_range=_optional_float(rl, "clip_range", base_config.rl_clip_range),
        rl_target_kl=_optional_float(rl, "target_kl", base_config.rl_target_kl),
        rl_n_eval_episodes=_optional_int(
            rl,
            "n_eval_episodes",
            base_config.rl_n_eval_episodes,
            minimum=1,
        ),
        seed=seeds[0],
        seeds=seeds,
        mo_use_quick=_optional_bool(mo, "use_quick", base_config.mo_use_quick),
        mo_population_size=_optional_int(
            mo,
            "population_size",
            base_config.mo_population_size,
            minimum=1,
        ),
        mo_n_generations=_optional_int(
            mo,
            "n_generations",
            base_config.mo_n_generations,
            minimum=1,
        ),
        mo_effort_mode=mo_effort_mode,
        layout_policy=layout_policy,
        mo_objective_name=mo_objective_name,
        mo_selection_modes=mo_selection_modes,
        parallel_workers=_optional_int(entry, "parallel_workers", 1, minimum=1),
        mode=mode,
        topology_source=topology_source,
    )


def _build_batch_campaign(entry: dict[str, Any]) -> Campaign:
    campaign_id = _require_string(entry.get("campaign_id"), field_name="campaign_id")
    return Campaign.from_config(campaign_id=campaign_id, config=_build_batch_campaign_config(entry))


def load_campaign_batch(path: Path | str) -> tuple[Campaign, ...]:
    batch_path = Path(path)
    try:
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {batch_path}: {exc}") from exc

    root = _require_mapping(payload, field_name="batch")
    raw_campaigns = root.get("campaigns")
    if not isinstance(raw_campaigns, list) or not raw_campaigns:
        raise ValueError("campaigns must be a non-empty array")

    campaigns: list[Campaign] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, raw_campaign in enumerate(raw_campaigns):
        try:
            campaign = _build_batch_campaign(_require_mapping(raw_campaign, field_name=f"campaigns[{index}]"))
        except (TypeError, ValueError) as exc:
            errors.append(f"campaigns[{index}]: {exc}")
            continue
        if campaign.campaign_id in seen_ids:
            errors.append(f"campaigns[{index}]: duplicate campaign_id: {campaign.campaign_id}")
            continue
        seen_ids.add(campaign.campaign_id)
        campaigns.append(campaign)

    if errors:
        raise ValueError("\n".join(errors))
    return tuple(campaigns)


def _write_batch_summary(*, output_root: Path, report: BatchReport) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def run_campaign_batch(
    campaigns: tuple[Campaign, ...] | list[Campaign],
    *,
    output_root: Path | str = "campaigns",
    dry_run: bool = False,
    run_campaign_fn: Callable[..., object] = run_campaign,
    run_campaign_matrix_fn: Callable[..., object] = run_campaign_matrix,
) -> BatchReport:
    output_path = Path(output_root)
    results: list[BatchCampaignResult] = []
    for campaign in campaigns:
        campaign_output_dir = output_path / campaign.campaign_id
        if is_matrix_campaign_config(campaign.config):
            summary_document = str(campaign_output_dir / "matrix_summary.md")
            structured_output = str(campaign_output_dir / "matrix_summary.json")
        else:
            summary_document = str(campaign_output_dir / "summary.md")
            structured_output = str(campaign_output_dir / "campaign.json")
        if dry_run:
            results.append(
                BatchCampaignResult(
                    campaign_id=campaign.campaign_id,
                    status="validated",
                    summary_document=summary_document,
                    structured_output=structured_output,
                )
            )
            continue

        try:
            if is_matrix_campaign_config(campaign.config):
                report = run_campaign_matrix_fn(campaign, output_root=output_path)
                report_status = report.status
            else:
                report = run_campaign_fn(campaign, output_root=output_path)
                report_status = report.campaign_status
        except Exception as exc:
            results.append(
                BatchCampaignResult(
                    campaign_id=campaign.campaign_id,
                    status="failed",
                    summary_document=summary_document,
                    structured_output=structured_output,
                    error=str(exc),
                )
            )
            continue

        results.append(
            BatchCampaignResult(
                campaign_id=campaign.campaign_id,
                status=report_status,
                summary_document=summary_document,
                structured_output=structured_output,
            )
        )

    failed_campaigns = sum(1 for result in results if result.status == "failed")
    completed_campaigns = sum(1 for result in results if result.status in {"completed", "validated"})
    status = "dry_run" if dry_run else ("failed" if failed_campaigns else "completed")
    batch_summary_path = output_path / "batch_summary.json"
    batch_report = BatchReport(
        status=status,
        dry_run=dry_run,
        total_campaigns=len(results),
        completed_campaigns=completed_campaigns,
        failed_campaigns=failed_campaigns,
        results=tuple(results),
        summary_path=str(batch_summary_path),
    )
    _write_batch_summary(output_root=output_path, report=batch_report)
    return batch_report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input_path", help="JSON file with one or more Campaign definitions.")
    parser.add_argument("--output-root", default="campaigns")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_campaign_cli_from_args(
    argv: list[str] | None = None,
    *,
    input_fn=input,
    output_fn=print,
    run_campaign_fn: Callable[..., object] = run_campaign,
    campaign_id_factory=_default_campaign_id,
) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.input_path is None:
        if args.dry_run:
            output_fn("--dry-run requires --input.")
            return 1
        return run_interactive_campaign_cli(
            input_fn=input_fn,
            output_fn=output_fn,
            run_campaign_fn=run_campaign_fn,
            campaign_id_factory=campaign_id_factory,
            output_root=args.output_root,
        )

    try:
        campaigns = load_campaign_batch(args.input_path)
    except ValueError as exc:
        output_fn(f"Invalid campaign batch: {exc}")
        return 1

    if args.dry_run:
        for campaign in campaigns:
            output_fn(f"Validated campaign: {campaign.campaign_id}")
    else:
        output_fn(f"Executing {len(campaigns)} campaign(s) from {args.input_path}...")

    report = run_campaign_batch(
        campaigns,
        output_root=args.output_root,
        dry_run=args.dry_run,
        run_campaign_fn=run_campaign_fn,
        run_campaign_matrix_fn=lambda campaign, *, output_root: run_campaign_matrix(
            campaign,
            output_root=output_root,
            run_campaign_fn=run_campaign_fn if run_campaign_fn is not run_campaign else None,
        ),
    )
    for result in report.results:
        output_fn(f"Campaign ID: {result.campaign_id}")
        output_fn(f"Final Status: {result.status}")
        if result.error is not None:
            output_fn(f"Error: {result.error}")
        output_fn(f"Summary Document: {result.summary_document}")
        output_fn(f"Structured Campaign Output: {result.structured_output}")
    output_fn(f"Batch Summary: {report.summary_path}")
    return 1 if report.status == "failed" else 0


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
    campaign_output_dir = Path(output_root) / campaign.campaign_id
    if is_matrix_campaign_config(campaign.config):
        report = run_campaign_matrix(
            campaign,
            output_root=output_root,
            run_campaign_fn=run_campaign_fn if run_campaign_fn is not run_campaign else None,
        )
        final_status = report.status
        summary_document = campaign_output_dir / "matrix_summary.md"
        structured_output = campaign_output_dir / "matrix_summary.json"
    else:
        report = run_campaign_fn(campaign, output_root=output_root)
        final_status = report.campaign_status
        summary_document = campaign_output_dir / "summary.md"
        structured_output = campaign_output_dir / "campaign.json"
    output_fn(f"Campaign ID: {campaign.campaign_id}")
    output_fn(f"Final Status: {final_status}")
    output_fn(f"Summary Document: {summary_document}")
    output_fn(f"Structured Campaign Output: {structured_output}")
    return 0


def main() -> int:
    return run_campaign_cli_from_args()


if __name__ == "__main__":
    raise SystemExit(main())
