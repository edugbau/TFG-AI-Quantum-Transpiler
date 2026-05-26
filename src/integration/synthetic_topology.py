from __future__ import annotations

from dataclasses import dataclass, field

from qiskit.transpiler import CouplingMap


SYNTHETIC_BASIS_GATES: tuple[str, ...] = ("id", "rz", "sx", "x", "cx")
SYNTHETIC_TOPOLOGY_SHAPES: tuple[str, ...] = (
    "full",
    "line",
    "ring",
    "t",
    "grid",
    "heavy_hex",
    "heavy_square",
    "hexagonal_lattice",
)


def _require_positive(value: int | None, field_name: str) -> int:
    if value is None or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _require_odd_positive(value: int | None, field_name: str) -> int:
    value = _require_positive(value, field_name)
    if value % 2 == 0:
        raise ValueError(f"{field_name} must be an odd positive integer")
    return value


def _require_minimum(value: int | None, field_name: str, minimum: int) -> int:
    value = _require_positive(value, field_name)
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _bidirectional_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    directed_edges: list[tuple[int, int]] = []
    for left, right in edges:
        directed_edges.append((left, right))
        directed_edges.append((right, left))
    return directed_edges


def _balanced_t_bar_width(num_qubits: int) -> int:
    candidates: list[tuple[int, int, int]] = []
    for bar_width in range(3, num_qubits + 1, 2):
        stem_height = num_qubits - bar_width + 1
        if stem_height < 3:
            continue
        candidates.append((abs(bar_width - stem_height), -bar_width, bar_width))
    if not candidates:
        raise ValueError("num_qubits must support a T topology with bar and stem of at least 3 qubits")
    return min(candidates)[2]


def _build_t_coupling_map(num_qubits: int) -> CouplingMap:
    bar_width = _balanced_t_bar_width(num_qubits)
    center = bar_width // 2
    undirected_edges: list[tuple[int, int]] = [(qubit, qubit + 1) for qubit in range(bar_width - 1)]

    previous = center
    for qubit in range(bar_width, num_qubits):
        undirected_edges.append((previous, qubit))
        previous = qubit

    return CouplingMap(_bidirectional_edges(undirected_edges))


@dataclass(frozen=True, slots=True)
class SyntheticTopologySpec:
    shape: str
    num_qubits: int | None = None
    rows: int | None = None
    cols: int | None = None
    distance: int | None = None
    basis_gates: tuple[str, ...] = field(default_factory=lambda: SYNTHETIC_BASIS_GATES)

    def __post_init__(self) -> None:
        normalized_shape = self.shape.strip().lower().replace("-", "_")
        if normalized_shape not in SYNTHETIC_TOPOLOGY_SHAPES:
            raise ValueError(
                "SyntheticTopologySpec shape must be one of "
                + ", ".join(SYNTHETIC_TOPOLOGY_SHAPES)
            )

        basis_gates = tuple(gate.strip().lower() for gate in self.basis_gates if gate.strip())
        if not basis_gates:
            raise ValueError("SyntheticTopologySpec basis_gates cannot be empty")
        if "cx" not in basis_gates:
            raise ValueError("SyntheticTopologySpec basis_gates must include cx")

        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "basis_gates", basis_gates)

        if normalized_shape in {"full", "line", "ring", "t"}:
            if normalized_shape == "t":
                num_qubits = _require_minimum(self.num_qubits, "num_qubits", 5)
            else:
                num_qubits = _require_positive(self.num_qubits, "num_qubits")
            object.__setattr__(self, "num_qubits", num_qubits)
            object.__setattr__(self, "rows", None)
            object.__setattr__(self, "cols", None)
            object.__setattr__(self, "distance", None)
            return

        if normalized_shape in {"grid", "hexagonal_lattice"}:
            object.__setattr__(self, "rows", _require_positive(self.rows, "rows"))
            object.__setattr__(self, "cols", _require_positive(self.cols, "cols"))
            object.__setattr__(self, "num_qubits", None)
            object.__setattr__(self, "distance", None)
            return

        object.__setattr__(self, "distance", _require_odd_positive(self.distance, "distance"))
        object.__setattr__(self, "num_qubits", None)
        object.__setattr__(self, "rows", None)
        object.__setattr__(self, "cols", None)

    @property
    def backend_name(self) -> str:
        if self.shape in {"full", "line", "ring", "t"}:
            return f"synthetic_{self.shape}_{self.num_qubits}q"
        if self.shape == "grid":
            return f"synthetic_grid_{self.rows}x{self.cols}"
        if self.shape == "hexagonal_lattice":
            return f"synthetic_hexagonal_lattice_{self.rows}x{self.cols}"
        return f"synthetic_{self.shape}_d{self.distance}"

    @property
    def physical_qubits(self) -> int:
        return int(self.build_coupling_map().size())

    def build_coupling_map(self) -> CouplingMap:
        if self.shape == "full":
            return CouplingMap.from_full(self.num_qubits)
        if self.shape == "line":
            return CouplingMap.from_line(self.num_qubits)
        if self.shape == "ring":
            return CouplingMap.from_ring(self.num_qubits)
        if self.shape == "t":
            return _build_t_coupling_map(self.num_qubits)
        if self.shape == "grid":
            return CouplingMap.from_grid(self.rows, self.cols)
        if self.shape == "heavy_hex":
            return CouplingMap.from_heavy_hex(self.distance)
        if self.shape == "heavy_square":
            return CouplingMap.from_heavy_square(self.distance)
        return CouplingMap.from_hexagonal_lattice(self.rows, self.cols)

    def build_backend(self) -> "SyntheticBackend":
        return SyntheticBackend(spec=self, coupling_map=self.build_coupling_map())

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "topology_source": "synthetic",
            "shape": self.shape,
            "backend_name": self.backend_name,
            "physical_qubits": self.physical_qubits,
            "basis_gates": list(self.basis_gates),
        }
        if self.num_qubits is not None:
            metadata["num_qubits"] = self.num_qubits
        if self.rows is not None:
            metadata["rows"] = self.rows
        if self.cols is not None:
            metadata["cols"] = self.cols
        if self.distance is not None:
            metadata["distance"] = self.distance
        return metadata


@dataclass(frozen=True, slots=True)
class SyntheticBackend:
    spec: SyntheticTopologySpec
    coupling_map: CouplingMap

    @property
    def name(self) -> str:
        return self.spec.backend_name

    @property
    def num_qubits(self) -> int:
        return int(self.coupling_map.size())

    @property
    def basis_gates(self) -> tuple[str, ...]:
        return self.spec.basis_gates

    @property
    def backend_kind(self) -> str:
        return "synthetic"

    @property
    def topology_metadata(self) -> dict[str, object]:
        return self.spec.to_metadata()


def validate_synthetic_topology_capacity(
    synthetic_topology: SyntheticTopologySpec,
    *,
    required_qubits: int,
) -> None:
    if synthetic_topology.physical_qubits < required_qubits:
        raise ValueError(
            "Synthetic topology does not have enough physical qubits: "
            f"requires at least {required_qubits}, got {synthetic_topology.physical_qubits}"
        )
