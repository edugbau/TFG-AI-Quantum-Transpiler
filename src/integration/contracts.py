from dataclasses import dataclass, field
from enum import Enum

from src.integration.mo_effort import (
    DEFAULT_MO_N_GENERATIONS,
    DEFAULT_MO_POPULATION_SIZE,
    MIN_CUSTOM_MO_POPULATION_SIZE,
)
from src.integration.synthetic_topology import SyntheticTopologySpec


_ALLOWED_SCENARIO_NAMES = frozenset({"Baseline", "MO_Only", "RL_Only", "MO+RL"})


class CircuitSource(str, Enum):
    LIBRARY = "library"
    QASM_FILE = "qasm_file"


class CircuitFormat(str, Enum):
    AUTO = "auto"
    QASM2 = "qasm2"
    QASM3 = "qasm3"


class LayoutSelectionPolicy(str, Enum):
    COMPROMISE = "compromise"
    BEST_ON_OBJECTIVE = "best_on_objective"


@dataclass(slots=True)
class ScenarioRequest:
    scenario_name: str
    backend_name: str
    circuit_name: str | None = None
    num_qubits: int | None = None
    seed: int = 42
    layout_policy: LayoutSelectionPolicy = LayoutSelectionPolicy.COMPROMISE
    mo_use_quick: bool = True
    mo_population_size: int = DEFAULT_MO_POPULATION_SIZE
    mo_n_generations: int = DEFAULT_MO_N_GENERATIONS
    initial_layout: list[int] | None = None
    rl_model_path: str | None = None
    mo_objective_index: int = 0
    circuit_source: CircuitSource = CircuitSource.LIBRARY
    circuit_path: str | None = None
    circuit_format: CircuitFormat = CircuitFormat.AUTO
    synthetic_topology: SyntheticTopologySpec | None = None

    def __post_init__(self) -> None:
        self.circuit_source = CircuitSource(self.circuit_source)
        self.circuit_format = CircuitFormat(self.circuit_format)
        self.layout_policy = LayoutSelectionPolicy(self.layout_policy)
        if self.scenario_name not in _ALLOWED_SCENARIO_NAMES:
            raise ValueError("scenario_name must be one of Baseline, MO_Only, RL_Only, MO+RL")
        if self.mo_objective_index < 0:
            raise ValueError("mo_objective_index must be non-negative")
        if self.mo_population_size < MIN_CUSTOM_MO_POPULATION_SIZE:
            raise ValueError(
                "mo_population_size must be at least "
                f"{MIN_CUSTOM_MO_POPULATION_SIZE}"
            )
        if self.mo_n_generations <= 0:
            raise ValueError("mo_n_generations must be greater than zero")
        if self.circuit_source is CircuitSource.QASM_FILE:
            if self.circuit_path is None:
                raise ValueError("circuit_path is required when circuit_source is qasm_file")
            if self.circuit_name is not None:
                raise ValueError("circuit_name is not accepted when circuit_source is qasm_file")
            if self.num_qubits is not None:
                raise ValueError("num_qubits is not accepted when circuit_source is qasm_file")
        else:
            if self.circuit_name is None:
                raise ValueError("circuit_name is required when circuit_source is library")
            if self.num_qubits is None:
                raise ValueError("num_qubits is required when circuit_source is library")
            if self.circuit_path is not None:
                raise ValueError("library circuit_source does not accept circuit_path")
            if self.circuit_format is not CircuitFormat.AUTO:
                raise ValueError("library circuit_source does not accept non-default circuit_format")
        if self.num_qubits is not None and self.num_qubits <= 0:
            raise ValueError("num_qubits must be greater than zero")
        if self.synthetic_topology is not None and self.backend_name != self.synthetic_topology.backend_name:
            raise ValueError("backend_name must match synthetic_topology.backend_name")
        if self.initial_layout is not None:
            if self.num_qubits is not None and len(self.initial_layout) != self.num_qubits:
                raise ValueError("initial_layout length must match num_qubits")
            if any(entry < 0 for entry in self.initial_layout):
                raise ValueError("initial_layout cannot contain negative entries")
            if len(set(self.initial_layout)) != len(self.initial_layout):
                raise ValueError("initial_layout contains duplicated physical qubits")
        self._validate_scenario_specific_constraints()

    def _validate_scenario_specific_constraints(self) -> None:
        if self.scenario_name == "Baseline":
            if self.initial_layout is not None:
                raise ValueError("Baseline does not accept initial_layout")
            if self.rl_model_path is not None:
                raise ValueError("Baseline does not accept rl_model_path")
            if self.layout_policy is not LayoutSelectionPolicy.COMPROMISE:
                raise ValueError("Baseline does not accept non-default layout_policy")
            if self.mo_use_quick is not True:
                raise ValueError("Baseline does not accept non-default mo_use_quick")
            if self.mo_population_size != DEFAULT_MO_POPULATION_SIZE:
                raise ValueError("Baseline does not accept non-default mo_population_size")
            if self.mo_n_generations != DEFAULT_MO_N_GENERATIONS:
                raise ValueError("Baseline does not accept non-default mo_n_generations")
            if self.mo_objective_index != 0:
                raise ValueError("Baseline does not accept non-default mo_objective_index")
            return

        if self.scenario_name == "MO_Only" and self.rl_model_path is not None:
            raise ValueError("MO_Only does not accept rl_model_path")
        if self.scenario_name == "MO_Only" and self.initial_layout is not None:
            raise ValueError("MO_Only does not accept initial_layout")

        if self.scenario_name == "RL_Only":
            if self.rl_model_path is None:
                raise ValueError("RL_Only requires rl_model_path")
            if self.circuit_source is CircuitSource.QASM_FILE:
                raise ValueError("RL_Only does not accept qasm_file as a public circuit_source")
            if self.layout_policy is not LayoutSelectionPolicy.COMPROMISE:
                raise ValueError("RL_Only does not accept non-default layout_policy")
            if self.mo_use_quick is not True:
                raise ValueError("RL_Only does not accept non-default mo_use_quick")
            if self.mo_population_size != DEFAULT_MO_POPULATION_SIZE:
                raise ValueError("RL_Only does not accept non-default mo_population_size")
            if self.mo_n_generations != DEFAULT_MO_N_GENERATIONS:
                raise ValueError("RL_Only does not accept non-default mo_n_generations")
            if self.mo_objective_index != 0:
                raise ValueError("RL_Only does not accept non-default mo_objective_index")
            return

        if self.scenario_name == "MO+RL":
            if self.rl_model_path is None:
                raise ValueError("MO+RL requires rl_model_path")
            if self.initial_layout is not None:
                raise ValueError("MO+RL does not accept initial_layout")
            if self.circuit_source is CircuitSource.QASM_FILE:
                raise ValueError("MO+RL does not accept qasm_file as a public circuit_source")


@dataclass(slots=True)
class RoutingEpisodeSummary:
    initial_layout: list[int] | None
    final_layout: list[int] | None
    steps_executed: int
    total_reward: float
    completed: bool
    truncated: bool
    total_swaps: int
    gates_executed_count: int
    truncation_reason: str | None = None
    termination_reason: str | None = None
    swap_trace: list[tuple[int, int]] = field(default_factory=list)
    executed_gate_trace: list[tuple[str, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.steps_executed < 0:
            raise ValueError("steps_executed cannot be negative")
        if self.total_swaps < 0:
            raise ValueError("total_swaps cannot be negative")
        if self.gates_executed_count < 0:
            raise ValueError("gates_executed_count cannot be negative")
        if self.completed and self.truncated:
            raise ValueError("completed and truncated cannot both be True")
        if self.truncation_reason not in {None, "max_steps", "stagnation"}:
            raise ValueError("truncation_reason must be one of None, max_steps, stagnation")
        if self.truncation_reason is not None and not self.truncated:
            raise ValueError("truncation_reason requires truncated=True")
        if self.termination_reason not in {None, "stagnation"}:
            raise ValueError("termination_reason must be one of None, stagnation")
        if self.termination_reason is not None and (self.completed or self.truncated):
            raise ValueError("termination_reason requires a non-completed, non-truncated episode")
        self._validate_layout(self.initial_layout, "initial_layout")
        self._validate_layout(self.final_layout, "final_layout")
        if self.initial_layout is not None and self.final_layout is not None:
            if len(self.initial_layout) != len(self.final_layout):
                raise ValueError("initial_layout and final_layout must have the same length")
        self.swap_trace = [self._normalize_swap_edge(edge) for edge in self.swap_trace]
        if self.total_swaps != len(self.swap_trace):
            raise ValueError("total_swaps must match the number of entries in swap_trace")
        self.executed_gate_trace = [
            self._normalize_gate_tuple(gate) for gate in self.executed_gate_trace
        ]

    @staticmethod
    def _validate_layout(layout: list[int] | None, field_name: str) -> None:
        if layout is None:
            return
        if any(entry < 0 for entry in layout):
            raise ValueError(f"{field_name} cannot contain negative entries")
        if len(set(layout)) != len(layout):
            raise ValueError(f"{field_name} cannot contain duplicated entries")

    @staticmethod
    def _normalize_swap_edge(edge: tuple[int, int] | list[int]) -> tuple[int, int]:
        if len(edge) != 2:
            raise ValueError("swap_trace entries must contain exactly two physical qubits")

        left = int(edge[0])
        right = int(edge[1])
        if left < 0 or right < 0:
            raise ValueError("swap_trace cannot contain negative physical qubits")
        if left == right:
            raise ValueError("swap_trace cannot contain degenerate swaps")
        return (left, right)

    @staticmethod
    def _normalize_gate_tuple(gate: tuple[str, int, int] | list[object]) -> tuple[str, int, int]:
        if len(gate) != 3:
            raise ValueError("executed_gate_trace entries must contain gate_name, q1, q2")

        gate_name = str(gate[0])
        logical_q1 = int(gate[1])
        logical_q2 = int(gate[2])
        if logical_q1 < 0 or logical_q2 < 0:
            raise ValueError("executed_gate_trace cannot contain negative logical qubits")
        return (gate_name, logical_q1, logical_q2)


@dataclass(slots=True)
class ScenarioResult:
    scenario_name: str
    circuit_name: str
    backend_name: str
    seed: int
    success: bool
    selected_layout: list[int] | None
    transpilation_metrics: dict[str, str | int | float | list[int]] | None
    transpilation_artifact: dict[str, object] | None = None
    routing_summary: RoutingEpisodeSummary | None = None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scenario_name not in _ALLOWED_SCENARIO_NAMES:
            raise ValueError("scenario_name must be one of Baseline, MO_Only, RL_Only, MO+RL")
        if self.success and self.errors:
            raise ValueError("success=True cannot include errors")
        if self.selected_layout is None:
            return
        if any(entry < 0 for entry in self.selected_layout):
            raise ValueError("selected_layout cannot contain negative entries")
        if len(set(self.selected_layout)) != len(self.selected_layout):
            raise ValueError("selected_layout cannot contain duplicated entries")
