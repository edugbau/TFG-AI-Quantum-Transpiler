"""
Tests unitarios para el módulo qiskit_interface
=================================================

Cobertura de los tres sub-módulos:
  - circuit_utils (test_circuit_*)
  - backend_info  (test_backend_*)
  - transpiler    (test_transpiler_*)

Se utilizan exclusivamente Fake Backends (sin API keys ni conexión).
Cada test es independiente y reproducible (seeds fijadas).

Ejecución:
  pytest tests/test_qiskit_interface/ -v

Autor: Eduardo González Bautista
Fecha: 2026-02-08
"""

import pytest
from importlib.util import find_spec
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports del módulo bajo test
# ---------------------------------------------------------------------------
from src.qiskit_interface.circuit_utils import (
    CircuitMetrics,
    create_ghz_circuit,
    create_qft_circuit,
    create_random_circuit,
    create_clifford_circuit,
    extract_metrics,
    count_two_qubit_gates,
    export_circuit_to_qasm2,
    export_circuit_to_qasm3,
    load_circuit,
    load_circuit_from_qasm2,
    load_circuit_from_qasm3,
    save_circuit_to_qasm2,
    circuit_to_dag_convert,
    dag_to_circuit_convert,
    circuit_to_text,
    circuits_from_library,
)
from src.qiskit_interface.backend_info import (
    BackendInfo,
    get_backend,
    list_available_backends,
    get_coupling_map,
    get_coupling_edges,
    get_basis_gates,
    get_qubit_properties,
    get_gate_errors,
    get_gate_durations,
    extract_backend_info,
    get_heaviest_hex_layout,
    get_error_for_layout,
)
from src.qiskit_interface.transpiler import (
    TranspilationResult,
    transpile_circuit,
    transpile_post_routing,
    transpile_all_levels,
    transpile_batch,
    compare_transpilation_results,
    transpile_with_custom_layout,
    print_transpilation_comparison,
    run_baseline,
    list_available_baselines,
    run_named_baseline,
)
import src.qiskit_interface as qiskit_interface

from qiskit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError


# ===========================================================================
#  Fixtures
# ===========================================================================

@pytest.fixture
def simple_circuit() -> QuantumCircuit:
    """Circuito simple de 3 qubits para tests rápidos."""
    qc = QuantumCircuit(3, name="test_simple")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


@pytest.fixture
def ghz5() -> QuantumCircuit:
    """Circuito GHZ de 5 qubits."""
    return create_ghz_circuit(5)


@pytest.fixture
def backend_torino():
    """Instancia de FakeTorino."""
    return get_backend("fake_torino")


@pytest.fixture
def backend_info_torino(backend_torino) -> BackendInfo:
    """BackendInfo completo de FakeTorino."""
    return extract_backend_info(backend_torino)


# ===========================================================================
#  Tests — circuit_utils
# ===========================================================================

class TestCircuitCreation:
    """Tests de creación de circuitos de benchmark."""

    def test_ghz_circuit_creation(self):
        """GHZ de n qubits tiene n-1 CNOTs y profundidad n."""
        qc = create_ghz_circuit(5)
        assert qc.num_qubits == 5
        # H + 4 CX = 5 puertas, profundidad = 5 (H + 4 CX secuenciales)
        assert qc.depth() == 5
        ops = qc.count_ops()
        assert ops.get("cx", 0) == 4
        assert ops.get("h", 0) == 1

    def test_ghz_minimum_qubits(self):
        """GHZ con menos de 2 qubits debe lanzar ValueError."""
        with pytest.raises(ValueError):
            create_ghz_circuit(1)

    def test_qft_circuit_creation(self):
        """QFT genera un circuito con el número correcto de qubits."""
        qc = create_qft_circuit(4)
        assert qc.num_qubits == 4
        assert qc.depth() > 0

    def test_qft_inverse(self):
        """QFT inversa tiene la misma estructura básica."""
        qc = create_qft_circuit(3, inverse=True)
        assert qc.num_qubits == 3
        assert qc.depth() > 0

    def test_random_circuit(self):
        """Circuito aleatorio con seed es reproducible."""
        qc1 = create_random_circuit(4, 3, seed=123)
        qc2 = create_random_circuit(4, 3, seed=123)
        # Mismo seed → mismo circuito
        assert qc1.count_ops() == qc2.count_ops()

    def test_random_circuit_different_seeds(self):
        """Seeds distintas producen circuitos distintos (en general)."""
        qc1 = create_random_circuit(5, 5, seed=1)
        qc2 = create_random_circuit(5, 5, seed=99)
        # Es extremadamente improbable que sean idénticos
        # pero no impossible, así que solo verificamos que se crean
        assert qc1.num_qubits == qc2.num_qubits == 5

    def test_clifford_circuit(self):
        """Clifford aleatorio genera un circuito válido."""
        qc = create_clifford_circuit(3, seed=42)
        assert qc.num_qubits == 3
        assert qc.depth() > 0
        # Los Clifford solo usan puertas del grupo de Clifford
        allowed = {"h", "s", "sdg", "cx", "cz", "x", "y", "z", "swap", "id"}
        for gate_name in qc.count_ops():
            assert gate_name in allowed, f"Puerta inesperada: {gate_name}"

    def test_circuits_from_library(self):
        """La biblioteca de circuitos genera todos los tipos esperados."""
        lib = circuits_from_library(4, seed=42)
        expected_keys = {"ghz", "qft", "qft_inv", "random_shallow", "random_deep", "clifford"}
        assert set(lib.keys()) == expected_keys
        for name, qc in lib.items():
            assert qc.num_qubits == 4, f"Circuito '{name}' tiene {qc.num_qubits} qubits"


class TestCircuitMetrics:
    """Tests de extracción de métricas."""

    def test_extract_metrics_basic(self, simple_circuit):
        """Métricas básicas de un circuito simple."""
        metrics = extract_metrics(simple_circuit)
        assert isinstance(metrics, CircuitMetrics)
        assert metrics.num_qubits == 3
        assert metrics.depth == 3  # H + CX + CX
        assert metrics.two_qubit_gates == 2
        assert metrics.nonlocal_gates == 2
        assert metrics.total_gates == 3  # H + 2 CX

    def test_count_two_qubit_gates(self, simple_circuit):
        """Conteo preciso de puertas de 2 qubits."""
        count = count_two_qubit_gates(simple_circuit)
        assert count == 2

    def test_metrics_empty_circuit(self):
        """Circuito vacío tiene todas las métricas en 0."""
        qc = QuantumCircuit(2)
        metrics = extract_metrics(qc)
        assert metrics.depth == 0
        assert metrics.total_gates == 0
        assert metrics.two_qubit_gates == 0

    def test_metrics_to_dict(self, simple_circuit):
        """to_dict() devuelve un diccionario serializable."""
        metrics = extract_metrics(simple_circuit)
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "depth" in d
        assert "two_qubit_gates" in d

    def test_metrics_summary(self, simple_circuit):
        """summary() devuelve una cadena legible."""
        metrics = extract_metrics(simple_circuit)
        s = metrics.summary()
        assert "Profundidad" in s
        assert "Puertas 2-qubit" in s


class TestCircuitIO:
    """Tests de importación/exportación QASM."""

    def test_load_circuit_library_attaches_metadata(self):
        """load_circuit desde biblioteca adjunta metadatos de procedencia."""
        loaded = load_circuit(
            "library",
            circuit_name="ghz",
            num_qubits=3,
            seed=42,
        )

        assert loaded.num_qubits == 3
        assert loaded.metadata is not None
        assert loaded.metadata["source_kind"] == "library"
        assert loaded.metadata["source_format"] == "library"
        assert loaded.metadata["resolved_circuit_name"] == "ghz"

    def test_load_circuit_qasm_file_auto_detects_qasm2_and_attaches_metadata(self, tmp_path):
        """load_circuit detecta QASM2 automáticamente y adjunta metadatos."""
        qasm_path = tmp_path / "simple_from_text.txt"
        qasm_path.write_text(
            '\n'.join([
                'OPENQASM 2.0;',
                'include "qelib1.inc";',
                'qreg q[2];',
                'h q[0];',
                'cx q[0],q[1];',
            ]),
            encoding="utf-8",
        )

        loaded = load_circuit(
            "qasm_file",
            circuit_path=qasm_path,
            circuit_format="auto",
        )

        assert loaded.num_qubits == 2
        assert loaded.metadata is not None
        assert loaded.metadata["source_kind"] == "qasm_file"
        assert loaded.metadata["source_format"] == "qasm2"
        assert loaded.metadata["source_path"] == str(qasm_path)
        assert loaded.metadata["resolved_circuit_name"] == "simple_from_text"

    def test_load_circuit_qasm_file_qasm2_preserves_relative_include_resolution(self, tmp_path):
        """load_circuit debe preservar includes relativos en cargas desde fichero."""
        include_path = tmp_path / "custom_defs.inc"
        include_path.write_text(
            "// sibling include resolved relative to the main file\n",
            encoding="utf-8",
        )
        qasm_path = tmp_path / "with_relative_include.qasm"
        qasm_path.write_text(
            '\n'.join([
                'OPENQASM 2.0;',
                'include "qelib1.inc";',
                'include "custom_defs.inc";',
                'qreg q[1];',
                'h q[0];',
            ]),
            encoding="utf-8",
        )

        loaded = load_circuit(
            "qasm_file",
            circuit_path=qasm_path,
            circuit_format="qasm2",
        )

        assert loaded.num_qubits == 1
        assert loaded.count_ops().get("h", 0) == 1
        assert loaded.metadata is not None
        assert loaded.metadata["source_path"] == str(qasm_path)

    def test_load_circuit_qasm_file_auto_detects_qasm3_from_contents(self, tmp_path):
        """load_circuit detecta QASM3 desde el contenido del fichero."""
        qasm_path = tmp_path / "simple_qasm3.txt"
        qasm_path.write_text(
            '\n'.join([
                'OPENQASM 3.0;',
                'include "stdgates.inc";',
                'qubit[2] q;',
                'h q[0];',
                'cx q[0], q[1];',
            ]),
            encoding="utf-8",
        )

        if find_spec("qiskit_qasm3_import") is None:
            with pytest.raises(MissingOptionalLibraryError):
                load_circuit(
                    "qasm_file",
                    circuit_path=qasm_path,
                    circuit_format="auto",
                )
            return

        loaded = load_circuit(
            "qasm_file",
            circuit_path=qasm_path,
            circuit_format="auto",
        )

        assert loaded.num_qubits == 2
        assert loaded.metadata is not None
        assert loaded.metadata["source_kind"] == "qasm_file"
        assert loaded.metadata["source_format"] == "qasm3"
        assert loaded.metadata["source_path"] == str(qasm_path)
        assert loaded.metadata["resolved_circuit_name"] == "simple_qasm3"

    def test_load_circuit_library_requires_circuit_name(self):
        """load_circuit exige circuit_name para circuitos de biblioteca."""
        with pytest.raises(ValueError, match="circuit_name is required"):
            load_circuit("library", num_qubits=3)

    def test_load_circuit_library_requires_num_qubits(self):
        """load_circuit exige num_qubits para circuitos de biblioteca."""
        with pytest.raises(ValueError, match="num_qubits is required"):
            load_circuit("library", circuit_name="ghz")

    def test_load_circuit_library_selects_requested_generator_lazily(self):
        """load_circuit no debe fallar por generadores no solicitados de la biblioteca."""
        loaded = load_circuit(
            "library",
            circuit_name="qft",
            num_qubits=1,
            seed=42,
        )

        assert loaded.num_qubits == 1
        assert loaded.metadata is not None
        assert loaded.metadata["resolved_circuit_name"] == "qft"

    def test_load_circuit_qasm_file_requires_circuit_path(self):
        """load_circuit exige circuit_path para source_kind qasm_file."""
        with pytest.raises(ValueError, match="circuit_path is required"):
            load_circuit("qasm_file")

    def test_load_circuit_qasm_file_rejects_invalid_format(self, tmp_path):
        """load_circuit valida explícitamente circuit_format."""
        qasm_path = tmp_path / "simple.qasm"
        qasm_path.write_text(
            '\n'.join([
                'OPENQASM 2.0;',
                'include "qelib1.inc";',
                'qreg q[1];',
            ]),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="circuit_format must be 'auto', 'qasm2', or 'qasm3'"):
            load_circuit(
                "qasm_file",
                circuit_path=qasm_path,
                circuit_format="invalid",
            )

    def test_load_circuit_qasm_file_auto_rejects_unknown_contents(self, tmp_path):
        """load_circuit falla si no puede detectar el formato desde contenido."""
        qasm_path = tmp_path / "unknown_format.txt"
        qasm_path.write_text("not a qasm program", encoding="utf-8")

        with pytest.raises(ValueError, match="No se pudo detectar el formato QASM"):
            load_circuit(
                "qasm_file",
                circuit_path=qasm_path,
                circuit_format="auto",
            )

    def test_load_circuit_qasm_file_auto_detects_qasm2_with_utf8_bom(self, tmp_path):
        """La autodetección de QASM tolera BOM UTF-8 al inicio del fichero."""
        qasm_path = tmp_path / "bom_prefixed.txt"
        qasm_path.write_text(
            '\ufeff' + '\n'.join([
                'OPENQASM 2.0;',
                'include "qelib1.inc";',
                'qreg q[1];',
            ]),
            encoding="utf-8",
        )

        loaded = load_circuit(
            "qasm_file",
            circuit_path=qasm_path,
            circuit_format="auto",
        )

        assert loaded.num_qubits == 1
        assert loaded.metadata is not None
        assert loaded.metadata["source_format"] == "qasm2"

    def test_qasm2_roundtrip(self, simple_circuit):
        """Exportar a QASM2 e importar de vuelta preserva la estructura."""
        qasm_str = export_circuit_to_qasm2(simple_circuit)
        assert "OPENQASM 2.0" in qasm_str

        loaded = load_circuit_from_qasm2(qasm_str)
        assert loaded.num_qubits == simple_circuit.num_qubits
        assert loaded.depth() == simple_circuit.depth()

    def test_qasm2_file_not_found(self):
        """Cargar un fichero QASM inexistente lanza FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_circuit_from_qasm2(Path("no_existe.qasm"))

    def test_load_circuit_from_qasm2_reads_existing_non_qasm_file(self, tmp_path):
        """El loader QASM2 trata cualquier ruta existente como fichero."""
        qasm_path = tmp_path / "simple_qasm2.txt"
        qasm_path.write_text(
            '\n'.join([
                'OPENQASM 2.0;',
                'include "qelib1.inc";',
                'qreg q[2];',
                'h q[0];',
            ]),
            encoding="utf-8",
        )

        loaded = load_circuit_from_qasm2(qasm_path)

        assert loaded.num_qubits == 2

    def test_export_qasm3_roundtrip_from_string(self, simple_circuit):
        """Exportar a QASM3 produce una cadena que puede recargarse."""
        # Arrange
        qasm3_str = export_circuit_to_qasm3(simple_circuit)

        # Act / Assert
        if find_spec("qiskit_qasm3_import") is None:
            with pytest.raises(MissingOptionalLibraryError):
                load_circuit_from_qasm3(qasm3_str)
            return

        loaded = load_circuit_from_qasm3(qasm3_str)

        assert "OPENQASM 3" in qasm3_str
        assert loaded.num_qubits == simple_circuit.num_qubits
        assert loaded.depth() == simple_circuit.depth()

    def test_load_circuit_from_qasm3_reads_existing_non_qasm_file(self, tmp_path):
        """El loader QASM3 trata cualquier ruta existente como fichero."""
        qasm_path = tmp_path / "simple_qasm3.txt"
        qasm_path.write_text(
            '\n'.join([
                'OPENQASM 3.0;',
                'include "stdgates.inc";',
                'qubit[2] q;',
                'h q[0];',
            ]),
            encoding="utf-8",
        )

        if find_spec("qiskit_qasm3_import") is None:
            with pytest.raises(MissingOptionalLibraryError):
                load_circuit_from_qasm3(qasm_path)
            return

        loaded = load_circuit_from_qasm3(qasm_path)

        assert loaded.num_qubits == 2

    def test_save_qasm2_creates_file_that_can_be_loaded(self, simple_circuit, tmp_path):
        """Guardar a QASM2 crea un fichero reutilizable."""
        # Arrange
        output_path = tmp_path / "nested" / "simple.qasm"

        # Act
        saved_path = save_circuit_to_qasm2(simple_circuit, output_path)
        loaded = load_circuit_from_qasm2(saved_path)

        # Assert
        assert saved_path == output_path
        assert saved_path.is_file()
        assert loaded.num_qubits == simple_circuit.num_qubits
        assert loaded.depth() == simple_circuit.depth()


class TestCircuitConversion:
    """Tests de conversión Circuit ↔ DAG."""

    def test_circuit_to_dag_roundtrip(self, simple_circuit):
        """Convertir a DAG y de vuelta preserva la profundidad."""
        dag = circuit_to_dag_convert(simple_circuit)
        recovered = dag_to_circuit_convert(dag)
        assert recovered.depth() == simple_circuit.depth()
        assert recovered.num_qubits == simple_circuit.num_qubits

    def test_circuit_to_text(self, simple_circuit):
        """Representación textual no está vacía."""
        text = circuit_to_text(simple_circuit)
        assert len(text) > 0
        assert "q" in text.lower()  # Debe mencionar qubits


# ===========================================================================
#  Tests — backend_info
# ===========================================================================

class TestBackendInfo:
    """Tests de información de backends."""

    def test_list_backends(self):
        """Hay al menos 3 backends disponibles."""
        backends = list_available_backends()
        assert len(backends) >= 3
        assert "fake_torino" in backends
        assert "fake_sherbrooke" in backends
        assert "fake_brisbane" in backends

    def test_get_backend_valid(self):
        """Instanciar un backend válido no lanza excepciones."""
        backend = get_backend("fake_torino")
        assert backend.num_qubits == 133

    def test_get_backend_invalid(self):
        """Nombre de backend inválido lanza ValueError."""
        with pytest.raises(ValueError):
            get_backend("backend_inexistente")

    def test_coupling_map(self, backend_torino):
        """CouplingMap tiene el tamaño correcto."""
        cm = get_coupling_map(backend_torino)
        assert cm.size() == 133

    def test_coupling_edges(self, backend_torino):
        """Las aristas del coupling map son tuplas de 2 enteros."""
        edges = get_coupling_edges(backend_torino)
        assert len(edges) > 0
        for edge in edges:
            assert len(edge) == 2
            assert isinstance(edge[0], int)
            assert isinstance(edge[1], int)

    def test_basis_gates(self, backend_torino):
        """Las basis gates incluyen puertas de 1Q (sx, rz) y 2Q (cz)."""
        gates = get_basis_gates(backend_torino)
        assert "sx" in gates
        assert "rz" in gates
        assert "cz" in gates

    def test_qubit_properties(self, backend_torino):
        """Se extraen T1 y T2 para todos los qubits."""
        props = get_qubit_properties(backend_torino)
        assert "t1" in props
        assert "t2" in props
        # Debe haber propiedades para los 133 qubits
        assert len(props["t1"]) == 133

    def test_gate_errors(self, backend_torino):
        """Se extraen errores de puertas."""
        errors = get_gate_errors(backend_torino)
        assert "cz" in errors  # Torino usa CZ
        assert "sx" in errors  # SX es universal
        # Los errores deben ser valores positivos
        for qargs, error in errors["cz"].items():
            assert error >= 0

    def test_gate_durations(self, backend_torino):
        """Se extraen duraciones de puertas."""
        durations = get_gate_durations(backend_torino)
        assert "cz" in durations
        for qargs, dur in durations["cz"].items():
            assert dur > 0

    def test_extract_backend_info(self, backend_info_torino):
        """BackendInfo contiene toda la información esperada."""
        info = backend_info_torino
        assert info.name is not None
        assert info.num_qubits == 133
        assert info.coupling_map is not None
        assert len(info.coupling_edges) > 0
        assert info.two_qubit_gate == "cz"
        assert len(info.gate_errors_2q) > 0
        assert len(info.qubit_t1) > 0

    def test_backend_info_summary(self, backend_info_torino):
        """El resumen del backend contiene información legible."""
        s = backend_info_torino.summary()
        assert "Qubits" in s
        assert "cz" in s

    def test_backend_info_to_summary_dict_exposes_stable_aggregates(self, backend_info_torino):
        """El resumen serializable del backend expone agregados estables para artefactos."""
        summary = backend_info_torino.to_summary_dict()

        assert summary["backend_name"] == "fake_torino"
        assert summary["num_qubits"] == 133
        assert summary["two_qubit_gate"] == "cz"
        assert "basis_gates" in summary
        assert summary["coupling_edges_count"] == len(backend_info_torino.coupling_edges)
        assert summary["min_error_2q"] <= summary["avg_error_2q"] <= summary["max_error_2q"]
        assert summary["avg_t1"] > 0
        assert summary["avg_t2"] > 0


class TestBackendUtilities:
    """Tests de utilidades de backend."""

    def test_heaviest_hex_layout(self, backend_torino):
        """Selecciona el número correcto de qubits."""
        layout = get_heaviest_hex_layout(backend_torino, 5)
        assert len(layout) == 5
        # Todos los qubits seleccionados deben ser válidos
        for q in layout:
            assert 0 <= q < 133

    def test_heaviest_hex_layout_too_many(self, backend_torino):
        """Pedir más qubits de los disponibles lanza ValueError."""
        with pytest.raises(ValueError):
            get_heaviest_hex_layout(backend_torino, 200)

    def test_error_for_layout(self, backend_info_torino):
        """Calcular errores para un layout devuelve métricas sensatas."""
        layout = [0, 1, 2, 3, 4]
        stats = get_error_for_layout(backend_info_torino, layout)
        assert "avg_error_1q" in stats
        assert "avg_error_2q" in stats
        assert "avg_t1" in stats
        assert "avg_t2" in stats
        assert stats["avg_error_1q"] >= 0
        assert stats["avg_t1"] > 0


# ===========================================================================
#  Tests — transpiler
# ===========================================================================

class TestTranspilation:
    """Tests de transpilación estándar."""

    def test_transpile_basic(self, simple_circuit, backend_torino):
        """Transpilación básica produce un resultado válido."""
        result = transpile_circuit(
            simple_circuit,
            backend=backend_torino,
            optimization_level=1,
            seed=42,
        )
        assert isinstance(result, TranspilationResult)
        assert result.transpiled_circuit is not None
        assert result.transpiled_metrics is not None
        assert result.original_metrics is not None
        assert result.elapsed_time_s > 0

    def test_transpile_all_levels(self, simple_circuit, backend_torino):
        """Transpilar a los 4 niveles produce 4 resultados."""
        results = transpile_all_levels(
            simple_circuit,
            backend=backend_torino,
            seed=42,
        )
        assert len(results) == 4
        assert all(level in results for level in range(4))

    def test_transpile_uses_basis_gates(self, simple_circuit, backend_torino):
        """El circuito transpilado solo usa puertas nativas del backend."""
        result = transpile_circuit(
            simple_circuit,
            backend=backend_torino,
            optimization_level=1,
        )
        # Las puertas del circuito transpilado deben ser subset de las
        # operaciones del target
        target_ops = set(backend_torino.target.operation_names)
        transpiled_ops = set(result.transpiled_metrics.gate_counts.keys())
        # Eliminar barrier que a veces se añade
        transpiled_ops.discard("barrier")
        assert transpiled_ops.issubset(target_ops), (
            f"Puertas no nativas encontradas: {transpiled_ops - target_ops}"
        )

    def test_transpile_with_backend_name(self, simple_circuit):
        """Se puede transpilar indicando solo el nombre del backend."""
        result = transpile_circuit(
            simple_circuit,
            backend_name="fake_sherbrooke",
            optimization_level=1,
        )
        assert result.transpiled_circuit is not None
        assert "sherbrooke" in result.backend_name.lower()

    def test_transpile_reproducibility(self, simple_circuit, backend_torino):
        """Misma seed produce resultados idénticos."""
        r1 = transpile_circuit(simple_circuit, backend=backend_torino, seed=42)
        r2 = transpile_circuit(simple_circuit, backend=backend_torino, seed=42)
        assert r1.transpiled_metrics.depth == r2.transpiled_metrics.depth
        assert r1.transpiled_metrics.two_qubit_gates == r2.transpiled_metrics.two_qubit_gates


class TestTranspilationWithLayout:
    """Tests de transpilación con layout personalizado."""

    def test_transpile_with_custom_layout(self, simple_circuit, backend_torino):
        """Transpilar con layout inicial personalizado funciona."""
        # Layout: qubits lógicos 0,1,2 → físicos 10,11,12
        layout = [10, 11, 12]
        result = transpile_with_custom_layout(
            simple_circuit,
            layout=layout,
            backend=backend_torino,
            optimization_level=1,
        )
        assert result.transpiled_circuit is not None
        assert result.initial_layout == layout

    def test_transpile_with_custom_layout_rejects_wrong_length(self, simple_circuit, backend_torino):
        """Un layout con longitud incorrecta se rechaza explícitamente."""
        # Arrange
        layout = [10, 11]

        # Act / Assert
        with pytest.raises(ValueError, match="same length"):
            transpile_with_custom_layout(
                simple_circuit,
                layout=layout,
                backend=backend_torino,
                optimization_level=1,
            )

    def test_transpile_with_custom_layout_rejects_duplicate_qubits(self, simple_circuit, backend_torino):
        """Un layout no puede reutilizar el mismo qubit físico."""
        # Arrange
        layout = [10, 10, 12]

        # Act / Assert
        with pytest.raises(ValueError, match="duplicate"):
            transpile_with_custom_layout(
                simple_circuit,
                layout=layout,
                backend=backend_torino,
                optimization_level=1,
            )

    def test_transpile_with_custom_layout_rejects_out_of_range_qubits(self, simple_circuit, backend_torino):
        """Los qubits físicos deben existir en el backend."""
        # Arrange
        layout = [10, 11, backend_torino.num_qubits]

        # Act / Assert
        with pytest.raises(ValueError, match="range"):
            transpile_with_custom_layout(
                simple_circuit,
                layout=layout,
                backend=backend_torino,
                optimization_level=1,
            )

    def test_transpile_post_routing_skips_layout_and_routing_but_keeps_metrics_reference(
        self,
        simple_circuit,
        backend_torino,
    ):
        routed = QuantumCircuit(backend_torino.num_qubits, name="routed")
        routed.h(10)
        routed.cx(10, 11)
        routed.swap(11, 12)

        result = transpile_post_routing(
            routed,
            backend=backend_torino,
            optimization_level=1,
            seed=42,
            reference_circuit=simple_circuit,
            initial_layout=[10, 11, 12],
            final_layout=[10, 12, 11],
        )

        assert result.original_circuit is simple_circuit
        assert result.initial_layout == [10, 11, 12]
        assert result.final_layout == [10, 12, 11]
        assert result.transpiled_circuit is not None
        assert result.transpiled_metrics is not None

    def test_transpile_circuit_reports_final_layout_from_qiskit_final_index_layout(
        self,
        simple_circuit,
        backend_torino,
    ):
        result = transpile_circuit(
            simple_circuit,
            backend=backend_torino,
            optimization_level=1,
            initial_layout=[10, 20, 30],
            seed=42,
        )

        layout_obj = result.transpiled_circuit.layout
        assert layout_obj is not None
        assert callable(getattr(layout_obj, "final_index_layout", None))
        assert result.final_layout == layout_obj.final_index_layout()[: simple_circuit.num_qubits]

    def test_transpile_circuit_reports_qiskit_initial_layout(self, simple_circuit, backend_torino):
        result = transpile_circuit(
            simple_circuit,
            backend=backend_torino,
            optimization_level=1,
            seed=42,
        )

        layout_obj = result.transpiled_circuit.layout
        assert layout_obj is not None
        assert callable(getattr(layout_obj, "initial_index_layout", None))
        assert result.qiskit_initial_layout == layout_obj.initial_index_layout(filter_ancillas=True)[
            : simple_circuit.num_qubits
        ]
        assert result.to_artifact_dict()["transpilation"]["qiskit_initial_layout"] == result.qiskit_initial_layout

    def test_transpilation_metrics_distinguish_active_qubits_from_materialized_backend_width(
        self,
        simple_circuit,
        backend_torino,
    ):
        result = transpile_with_custom_layout(
            simple_circuit,
            layout=[10, 11, 12],
            backend=backend_torino,
            optimization_level=1,
            seed=42,
        )

        assert result.transpiled_metrics.num_qubits == backend_torino.num_qubits
        assert result.transpiled_metrics.active_qubits == 3
        assert result.to_dict()["trans_active_qubits"] == 3


class TestTranspilationBatch:
    """Tests de transpilación batch y baseline."""

    def test_transpile_batch(self, backend_torino):
        """Transpilación batch de múltiples circuitos."""
        circuits = {
            "ghz3": create_ghz_circuit(3),
            "ghz4": create_ghz_circuit(4),
        }
        results = transpile_batch(
            circuits,
            backend=backend_torino,
            optimization_level=1,
        )
        assert len(results) == 2
        assert "ghz3" in results
        assert "ghz4" in results

    def test_run_baseline_returns_one_row_per_backend_level_combo(self, simple_circuit):
        """run_baseline devuelve una fila por combinación evaluada."""
        # Arrange
        backend_names = ["fake_torino"]
        optimization_levels = [0, 1]

        # Act
        rows = run_baseline(
            simple_circuit,
            backend_names=backend_names,
            optimization_levels=optimization_levels,
            seed=42,
        )

        # Assert
        assert len(rows) == 2
        assert {row["optimization_level"] for row in rows} == {0, 1}
        assert {row["backend_name"] for row in rows} == {"fake_torino"}
        assert {row["circuit_name"] for row in rows} == {simple_circuit.name}
        for row in rows:
            assert "trans_depth" in row
            assert "trans_two_qubit_gates" in row

    def test_run_named_baseline_tags_result_with_baseline_name(self, simple_circuit):
        """run_named_baseline propaga el nombre del baseline al resultado plano."""
        rows = run_named_baseline(
            "qiskit_level_2",
            simple_circuit,
            backend_names=["fake_torino"],
            seed=42,
        )

        assert len(rows) == 1
        assert rows[0]["baseline_name"] == "qiskit_level_2"
        assert rows[0]["optimization_level"] == 2

    def test_run_named_baseline_supports_custom_layout_baseline(self, simple_circuit):
        """El baseline nombrado de layout reutiliza transpile_with_custom_layout."""
        layout = [10, 11, 12]

        rows = run_named_baseline(
            "custom_layout_level_1",
            simple_circuit,
            backend_names=["fake_torino"],
            layout=layout,
            seed=42,
        )

        assert len(rows) == 1
        assert rows[0]["baseline_name"] == "custom_layout_level_1"
        assert rows[0]["optimization_level"] == 1
        assert rows[0]["initial_layout"] == layout

    def test_list_available_baselines_exposes_named_catalog(self):
        """El catálogo público de baselines nombrados es pequeño y estable."""
        baselines = list_available_baselines()

        assert isinstance(baselines, list)
        assert baselines == [
            "qiskit_level_0",
            "qiskit_level_1",
            "qiskit_level_2",
            "qiskit_level_3",
            "custom_layout_level_1",
        ]

    def test_run_baseline_flat_rows_do_not_require_backend_info_extraction(self, simple_circuit, monkeypatch):
        """Los resultados planos no deben forzar la extracción de hardware summary."""
        def fail_extract(_backend):
            raise AssertionError("extract_backend_info should not be called for flat rows")

        monkeypatch.setattr("src.qiskit_interface.transpiler.extract_backend_info", fail_extract)

        rows = run_baseline(
            simple_circuit,
            backend_names=["fake_torino"],
            optimization_levels=[1],
            seed=42,
        )

        assert len(rows) == 1
        assert rows[0]["backend_name"] == "fake_torino"

    def test_compare_results(self, simple_circuit, backend_torino):
        """Comparación de resultados genera filas para DataFrame."""
        results = transpile_all_levels(simple_circuit, backend=backend_torino)
        rows = compare_transpilation_results(results)
        assert len(rows) == 4
        for row in rows:
            assert "trans_depth" in row
            assert "trans_two_qubit_gates" in row

    def test_print_transpilation_comparison_prints_table(self, simple_circuit, backend_torino, capsys):
        """La utilidad imprime una tabla legible con las métricas clave."""
        # Arrange
        results = transpile_all_levels(simple_circuit, backend=backend_torino, seed=42)

        # Act
        print_transpilation_comparison(results)
        captured = capsys.readouterr()

        # Assert
        assert "COMPARACIÓN DE TRANSPILACIONES" in captured.out
        assert "Depth" in captured.out
        assert "2Q Gates" in captured.out
        assert "0" in captured.out


class TestPackageApi:
    """Cobertura mínima de la API pública reexportada en el paquete raíz."""

    def test_package_root_load_circuit_reexports_qasm_file_auto_loader(self, tmp_path):
        """El paquete raíz reexporta load_circuit con autodetección QASM."""
        qasm_path = tmp_path / "package_root_loader.txt"
        qasm_path.write_text(
            '\n'.join([
                'OPENQASM 2.0;',
                'include "qelib1.inc";',
                'qreg q[2];',
                'h q[0];',
                'cx q[0],q[1];',
            ]),
            encoding="utf-8",
        )

        loaded = qiskit_interface.load_circuit(
            "qasm_file",
            circuit_path=qasm_path,
            circuit_format="auto",
        )

        assert loaded.num_qubits == 2
        assert loaded.metadata is not None
        assert loaded.metadata["source_kind"] == "qasm_file"
        assert loaded.metadata["source_format"] == "qasm2"
        assert loaded.metadata["source_path"] == str(qasm_path)
        assert loaded.metadata["resolved_circuit_name"] == "package_root_loader"

    def test_package_root_reexports_new_public_apis(self, simple_circuit, tmp_path):
        """El paquete raíz expone las nuevas APIs públicas esperadas."""
        # Arrange
        output_path = tmp_path / "package_root_export.qasm"

        # Act
        qasm2_str = qiskit_interface.export_circuit_to_qasm2(simple_circuit)
        qasm3_str = qiskit_interface.export_circuit_to_qasm3(simple_circuit)
        saved_path = qiskit_interface.save_circuit_to_qasm2(simple_circuit, output_path)
        reloaded_qasm2 = qiskit_interface.load_circuit_from_qasm2(qasm2_str)

        # Assert
        assert callable(qiskit_interface.run_baseline)
        assert callable(qiskit_interface.run_named_baseline)
        assert callable(qiskit_interface.list_available_baselines)
        assert callable(qiskit_interface.print_transpilation_comparison)
        assert "OPENQASM 2.0" in qasm2_str
        assert "OPENQASM 3" in qasm3_str
        assert saved_path == output_path
        assert reloaded_qasm2.depth() == simple_circuit.depth()

    def test_package_root_qasm3_loader_matches_environment_contract(self, simple_circuit):
        """La carga QASM3 reexportada es estable con o sin importador opcional."""
        # Arrange
        qasm3_str = qiskit_interface.export_circuit_to_qasm3(simple_circuit)

        # Act / Assert
        if find_spec("qiskit_qasm3_import") is None:
            with pytest.raises(MissingOptionalLibraryError):
                qiskit_interface.load_circuit_from_qasm3(qasm3_str)
            return

        loaded = qiskit_interface.load_circuit_from_qasm3(qasm3_str)

        assert loaded.num_qubits == simple_circuit.num_qubits

    def test_package_root_print_transpilation_comparison_smoke(self, simple_circuit, backend_torino, capsys):
        """La utilidad reexportada imprime una comparación observable."""
        # Arrange
        results = {"baseline": transpile_circuit(simple_circuit, backend=backend_torino, seed=42)}

        # Act
        qiskit_interface.print_transpilation_comparison(results)
        captured = capsys.readouterr()

        # Assert
        assert "baseline" in captured.out
        assert "COMPARACIÓN DE TRANSPILACIONES" in captured.out


class TestTranspilationResult:
    """Tests de la dataclass TranspilationResult."""

    def test_depth_reduction(self, simple_circuit, backend_torino):
        """depth_reduction calcula la razón correcta."""
        result = transpile_circuit(
            simple_circuit, backend=backend_torino, optimization_level=1
        )
        # La reducción puede ser negativa (el transpilador añade puertas)
        assert isinstance(result.depth_reduction, float)

    def test_two_qubit_overhead(self, simple_circuit, backend_torino):
        """two_qubit_gate_overhead devuelve un multiplicador numérico."""
        result = transpile_circuit(
            simple_circuit, backend=backend_torino, optimization_level=1
        )
        assert isinstance(result.two_qubit_gate_overhead, float)
        assert result.two_qubit_gate_overhead >= 0

    def test_to_dict(self, simple_circuit, backend_torino):
        """to_dict() genera un diccionario plano."""
        result = transpile_circuit(
            simple_circuit, backend=backend_torino, optimization_level=1
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "backend_name" in d
        assert "optimization_level" in d
        assert "trans_depth" in d

    def test_to_artifact_dict_exposes_structured_transpilation_artifact(self, tmp_path):
        """El artefacto estructurado incluye procedencia y métricas comparables."""
        qasm_path = tmp_path / "artifact_input.qasm"
        qasm_path.write_text(
            '\n'.join([
                'OPENQASM 2.0;',
                'include "qelib1.inc";',
                'qreg q[3];',
                'h q[0];',
                'cx q[0],q[1];',
                'cx q[1],q[2];',
            ]),
            encoding="utf-8",
        )
        circuit = load_circuit("qasm_file", circuit_path=qasm_path, circuit_format="qasm2")
        result = transpile_circuit(
            circuit,
            backend_name="fake_torino",
            optimization_level=2,
            seed=42,
        )
        result.baseline_name = "qiskit_level_2"

        assert result.hardware_summary is None

        artifact = result.to_artifact_dict()

        assert artifact["artifact_version"] == "transpilation_result.v1"
        assert artifact["baseline_name"] == "qiskit_level_2"
        assert artifact["circuit"]["source_kind"] == "qasm_file"
        assert artifact["circuit"]["source_path"] == str(qasm_path)
        assert artifact["backend"]["backend_name"] == "fake_torino"
        assert "coupling_edges_count" in artifact["backend"]
        assert "avg_error_2q" in artifact["backend"]
        assert "avg_t1" in artifact["backend"]
        assert artifact["metrics"]["original"] == result.original_metrics.to_dict()
        assert artifact["metrics"]["transpiled"] == result.transpiled_metrics.to_dict()
        assert artifact["metrics"]["original"]["active_qubits"] == 3
        assert artifact["metrics"]["transpiled"]["active_qubits"] >= 3

    def test_to_artifact_dict_falls_back_when_backend_summary_cannot_be_extracted(self, simple_circuit, backend_torino, monkeypatch):
        """El artefacto tolera backends sin extracción detallada previa."""
        result = transpile_circuit(simple_circuit, backend=backend_torino, optimization_level=1, seed=42)

        def fail_extract(_backend):
            raise RuntimeError("backend info unavailable")

        monkeypatch.setattr("src.qiskit_interface.transpiler.extract_backend_info", fail_extract)

        artifact = result.to_artifact_dict()

        assert artifact["backend"] == {"backend_name": "fake_torino"}

    def test_summary(self, simple_circuit, backend_torino):
        """summary() devuelve un string legible."""
        result = transpile_circuit(
            simple_circuit, backend=backend_torino, optimization_level=1
        )
        s = result.summary()
        assert "Backend" in s
        assert "Profundidad" in s
