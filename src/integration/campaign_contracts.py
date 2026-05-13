from __future__ import annotations

from dataclasses import dataclass

from src.integration.contracts import LayoutSelectionPolicy
from src.integration.mo_effort import MIN_CUSTOM_MO_POPULATION_SIZE
from src.integration.synthetic_topology import SyntheticTopologySpec, validate_synthetic_topology_capacity


_ALLOWED_CASE_RESULT_STATUSES = frozenset({"completed", "failed", "incomplete", "cancelled"})
_ALLOWED_CAMPAIGN_MODES = frozenset({"default", "advanced"})
_ALLOWED_MO_EFFORT_MODES = frozenset({"auto", "custom"})
_ALLOWED_TOPOLOGY_SOURCES = frozenset({"backend", "synthetic"})
_ALLOWED_CAMPAIGN_STATUSES = frozenset({"pending", "running", "completed", "failed", "cancelled", "interrupted"})
_TERMINAL_CAMPAIGN_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
_SUMMARY_CAMPAIGN_STATUSES = _TERMINAL_CAMPAIGN_STATUSES | {"running"}


@dataclass(frozen=True, slots=True)
class CampaignCircuitSpec:
    family: str
    num_qubits: int

    def __post_init__(self) -> None:
        normalized_family = self.family.strip()
        if not normalized_family:
            raise ValueError("CampaignCircuitSpec family cannot be blank")
        if self.num_qubits <= 0:
            raise ValueError("CampaignCircuitSpec num_qubits must be greater than zero")
        object.__setattr__(self, "family", normalized_family)


@dataclass(frozen=True, slots=True)
class CampaignCase:
    case_id: str
    circuit_family: str
    num_qubits: int
    backend_name: str


@dataclass(frozen=True, slots=True)
class CampaignCaseResult:
    case_id: str
    status: str

    def __post_init__(self) -> None:
        normalized_case_id = self.case_id.strip()
        if not normalized_case_id:
            raise ValueError("CampaignCaseResult case_id cannot be blank")
        if self.status not in _ALLOWED_CASE_RESULT_STATUSES:
            raise ValueError(
                "CampaignCaseResult status must be one of completed, failed, incomplete, cancelled"
            )
        object.__setattr__(self, "case_id", normalized_case_id)


@dataclass(slots=True)
class CampaignSummary:
    status: str
    total_cases: int
    comparable_completed_cases: int
    failed_cases: int
    incomplete_cases: int
    cancelled_cases: int
    case_results: tuple[CampaignCaseResult, ...] | None = None

    def __post_init__(self) -> None:
        if self.status not in _SUMMARY_CAMPAIGN_STATUSES:
            raise ValueError("CampaignSummary status must be one of running, completed, failed, cancelled, interrupted")
        for field_name in (
            "total_cases",
            "comparable_completed_cases",
            "failed_cases",
            "incomplete_cases",
            "cancelled_cases",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.case_results is None:
            self.case_results = ()
        else:
            self.case_results = tuple(self.case_results)
        aggregate_cases = (
            self.comparable_completed_cases
            + self.failed_cases
            + self.incomplete_cases
            + self.cancelled_cases
        )
        if aggregate_cases > self.total_cases:
            raise ValueError("total_cases cannot be smaller than the sum of aggregate campaign counts")
        if self.status == "completed" and aggregate_cases != self.total_cases:
            raise ValueError("total_cases must equal the sum of aggregate campaign counts for completed summaries")
        if len(self.case_results) > self.total_cases:
            raise ValueError("case_results cannot contain more entries than total_cases")


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
    mo_use_quick: bool
    mo_population_size: int
    mo_n_generations: int
    layout_policy: LayoutSelectionPolicy
    mo_effort_mode: str = "auto"
    mo_objective_name: str | None = None
    mode: str = "default"
    topology_source: str = "backend"
    synthetic_topology: SyntheticTopologySpec | None = None

    def __post_init__(self) -> None:
        if not self.circuit_specs:
            raise ValueError("Campaign requires at least one circuit_specs entry")
        if not self.backend_names:
            raise ValueError("Campaign requires at least one backend_names entry")
        if self.mode not in _ALLOWED_CAMPAIGN_MODES:
            raise ValueError("Campaign mode must be one of default, advanced")
        normalized_topology_source = self.topology_source.strip().lower()
        if normalized_topology_source not in _ALLOWED_TOPOLOGY_SOURCES:
            raise ValueError("CampaignConfig topology_source must be one of backend, synthetic")
        if self.mo_effort_mode not in _ALLOWED_MO_EFFORT_MODES:
            raise ValueError("CampaignConfig mo_effort_mode must be one of auto, custom")
        normalized_rl_algorithm = self.rl_algorithm.strip()
        if not normalized_rl_algorithm:
            raise ValueError("CampaignConfig rl_algorithm cannot be blank")
        normalized_frontier_mode = self.rl_frontier_mode.strip()
        if not normalized_frontier_mode:
            raise ValueError("CampaignConfig rl_frontier_mode cannot be blank")
        if self.rl_total_timesteps <= 0:
            raise ValueError("CampaignConfig rl_total_timesteps must be greater than zero")
        if self.rl_lookahead_window <= 0:
            raise ValueError("CampaignConfig rl_lookahead_window must be greater than zero")
        if self.rl_max_steps <= 0:
            raise ValueError("CampaignConfig rl_max_steps must be greater than zero")
        if self.seed < 0:
            raise ValueError("CampaignConfig seed cannot be negative")
        if not isinstance(self.mo_use_quick, bool):
            raise ValueError("CampaignConfig mo_use_quick must be a boolean")
        if self.mo_population_size <= 0:
            raise ValueError("CampaignConfig mo_population_size must be greater than zero")
        if self.mo_effort_mode == "custom" and self.mo_population_size < MIN_CUSTOM_MO_POPULATION_SIZE:
            raise ValueError(
                "CampaignConfig mo_population_size must be at least "
                f"{MIN_CUSTOM_MO_POPULATION_SIZE} when mo_effort_mode is custom"
            )
        if self.mo_n_generations <= 0:
            raise ValueError("CampaignConfig mo_n_generations must be greater than zero")
        normalized_layout_policy = LayoutSelectionPolicy(self.layout_policy)
        normalized_objective_name = self.mo_objective_name
        if normalized_objective_name is not None:
            normalized_objective_name = normalized_objective_name.strip()
        normalized_backends = [backend_name.strip() for backend_name in self.backend_names]
        if any(not backend_name for backend_name in normalized_backends):
            raise ValueError("Campaign backend_names cannot contain blank entries")
        if len(set(normalized_backends)) != len(normalized_backends):
            raise ValueError("Campaign backend_names cannot contain duplicates")
        normalized_circuit_specs = tuple(self.circuit_specs)
        circuit_keys = [(spec.family, spec.num_qubits) for spec in normalized_circuit_specs]
        if len(set(circuit_keys)) != len(circuit_keys):
            raise ValueError("CampaignConfig circuit_specs cannot contain duplicates")
        if normalized_topology_source == "backend":
            if self.synthetic_topology is not None:
                raise ValueError("CampaignConfig synthetic_topology is only accepted for synthetic topology_source")
        else:
            if self.mode != "advanced":
                raise ValueError("CampaignConfig synthetic topology_source is only accepted in advanced mode")
            if self.synthetic_topology is None:
                raise ValueError("CampaignConfig synthetic_topology is required for synthetic topology_source")
            if len(normalized_backends) != 1:
                raise ValueError("CampaignConfig synthetic topology_source requires exactly one backend_names entry")
            if normalized_backends[0] != self.synthetic_topology.backend_name:
                raise ValueError("CampaignConfig backend_names must contain the synthetic topology backend_name")
            validate_synthetic_topology_capacity(
                self.synthetic_topology,
                required_qubits=max(spec.num_qubits for spec in normalized_circuit_specs),
            )
        if normalized_layout_policy is LayoutSelectionPolicy.BEST_ON_OBJECTIVE:
            if not normalized_objective_name:
                raise ValueError("CampaignConfig mo_objective_name is required for best_on_objective")
        elif normalized_objective_name is not None:
            raise ValueError("CampaignConfig mo_objective_name is only accepted for best_on_objective")
        object.__setattr__(self, "backend_names", tuple(normalized_backends))
        object.__setattr__(self, "circuit_specs", normalized_circuit_specs)
        object.__setattr__(self, "rl_algorithm", normalized_rl_algorithm)
        object.__setattr__(self, "rl_frontier_mode", normalized_frontier_mode)
        object.__setattr__(self, "layout_policy", normalized_layout_policy)
        object.__setattr__(self, "mo_objective_name", normalized_objective_name)
        object.__setattr__(self, "topology_source", normalized_topology_source)


@dataclass(slots=True)
class Campaign:
    campaign_id: str
    config: CampaignConfig
    status: str = "pending"
    summary: CampaignSummary | None = None

    def __post_init__(self) -> None:
        normalized_campaign_id = self.campaign_id.strip()
        if not normalized_campaign_id:
            raise ValueError("Campaign campaign_id cannot be blank")
        if self.status not in _ALLOWED_CAMPAIGN_STATUSES:
            raise ValueError("Campaign status must be one of pending, running, completed, failed, cancelled, interrupted")
        if self.status in _TERMINAL_CAMPAIGN_STATUSES and self.summary is None:
            raise ValueError("Campaign status requires summary for terminal states")
        if self.status == "pending" and self.summary is not None:
            raise ValueError("Campaign summary is not allowed for pending state")
        if self.summary is not None and self.summary.status != self.status:
            raise ValueError("Campaign summary status must match Campaign status")
        self.campaign_id = normalized_campaign_id

    @classmethod
    def from_config(cls, *, campaign_id: str, config: CampaignConfig) -> "Campaign":
        return cls(campaign_id=campaign_id, config=config)

    def build_cases(self) -> list[CampaignCase]:
        cases: list[CampaignCase] = []
        for circuit_spec in self.config.circuit_specs:
            circuit_name = f"{circuit_spec.family}_{circuit_spec.num_qubits}"
            for backend_name in self.config.backend_names:
                cases.append(
                    CampaignCase(
                        case_id=f"{circuit_name}__{backend_name}",
                        circuit_family=circuit_spec.family,
                        num_qubits=circuit_spec.num_qubits,
                        backend_name=backend_name,
                    )
                )
        return cases
