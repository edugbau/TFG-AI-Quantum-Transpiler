from dataclasses import dataclass, field
from enum import Enum


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
    initial_layout: list[int] | None = None
    rl_model_path: str | None = None
    mo_objective_index: int = 0
    circuit_source: CircuitSource = CircuitSource.LIBRARY
    circuit_path: str | None = None
    circuit_format: CircuitFormat = CircuitFormat.AUTO

    def __post_init__(self) -> None:
        self.circuit_source = CircuitSource(self.circuit_source)
        self.circuit_format = CircuitFormat(self.circuit_format)
        if self.scenario_name not in _ALLOWED_SCENARIO_NAMES:
            raise ValueError("scenario_name must be one of Baseline, MO_Only, RL_Only, MO+RL")
        if self.mo_objective_index < 0:
            raise ValueError("mo_objective_index must be non-negative")
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

    def __post_init__(self) -> None:
        if self.steps_executed < 0:
            raise ValueError("steps_executed cannot be negative")
        if self.total_swaps < 0:
            raise ValueError("total_swaps cannot be negative")
        if self.gates_executed_count < 0:
            raise ValueError("gates_executed_count cannot be negative")
        if self.completed and self.truncated:
            raise ValueError("completed and truncated cannot both be True")
        self._validate_layout(self.initial_layout, "initial_layout")
        self._validate_layout(self.final_layout, "final_layout")
        if self.initial_layout is not None and self.final_layout is not None:
            if len(self.initial_layout) != len(self.final_layout):
                raise ValueError("initial_layout and final_layout must have the same length")

    @staticmethod
    def _validate_layout(layout: list[int] | None, field_name: str) -> None:
        if layout is None:
            return
        if any(entry < 0 for entry in layout):
            raise ValueError(f"{field_name} cannot contain negative entries")
        if len(set(layout)) != len(layout):
            raise ValueError(f"{field_name} cannot contain duplicated entries")


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
