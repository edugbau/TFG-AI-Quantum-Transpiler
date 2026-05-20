"""
qiskit_interface - contrato de circuitos, backends y transpilacion
===================================================================

Este paquete expone la capa comun entre Qiskit y el resto del proyecto.
Su superficie publica agrupa:

  - **circuit_utils**: carga, creacion, conversion y metricas de circuitos.
  - **backend_info**: metadata de hardware simulado y layouts heuristicos.
  - **transpiler**: baseline de Qiskit, evaluacion de layouts externos y
    helpers de comparacion.

El resto del proyecto consume esta fachada para generar circuitos,
evaluar layouts y construir artefactos serializables sin acoplarse a los
detalles internos de Qiskit.
"""

# ===================================================================
#  Re-exportaciones publicas
# ===================================================================
#  Se importan aqui los simbolos mas usados para que los usuarios
#  del modulo puedan hacer:
#      from src.qiskit_interface import create_ghz_circuit
#  en lugar de:
#      from src.qiskit_interface.circuit_utils import create_ghz_circuit
# ===================================================================

# --- circuit_utils ---
from .circuit_utils import (
    CircuitMetrics,
    load_circuit,
    load_circuit_from_qasm2,
    load_circuit_from_qasm3,
    export_circuit_to_qasm2,
    export_circuit_to_qasm3,
    save_circuit_to_qasm2,
    create_ghz_circuit,
    create_qft_circuit,
    create_random_circuit,
    create_clifford_circuit,
    count_two_qubit_gates,
    extract_metrics,
    circuit_to_dag_convert,
    dag_to_circuit_convert,
    circuit_to_text,
    circuit_to_matplotlib,
    print_circuit_info,
    circuits_from_library,
)

# --- backend_info ---
from .backend_info import (
    BackendInfo,
    AVAILABLE_BACKENDS,
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
    print_backend_info,
)

# --- transpiler ---
from .transpiler import (
    TranspilationResult,
    OPTIMIZATION_LEVELS,
    transpile_circuit,
    transpile_post_routing,
    transpile_all_levels,
    transpile_batch,
    compare_transpilation_results,
    print_transpilation_comparison,
    transpile_with_custom_layout,
    run_baseline,
    list_available_baselines,
    run_named_baseline,
)

# ===================================================================
#  __all__ - API publica explicita
# ===================================================================
__all__ = [
    # circuit_utils
    "CircuitMetrics",
    "load_circuit",
    "load_circuit_from_qasm2",
    "load_circuit_from_qasm3",
    "export_circuit_to_qasm2",
    "export_circuit_to_qasm3",
    "save_circuit_to_qasm2",
    "create_ghz_circuit",
    "create_qft_circuit",
    "create_random_circuit",
    "create_clifford_circuit",
    "count_two_qubit_gates",
    "extract_metrics",
    "circuit_to_dag_convert",
    "dag_to_circuit_convert",
    "circuit_to_text",
    "circuit_to_matplotlib",
    "print_circuit_info",
    "circuits_from_library",
    # backend_info
    "BackendInfo",
    "AVAILABLE_BACKENDS",
    "get_backend",
    "list_available_backends",
    "get_coupling_map",
    "get_coupling_edges",
    "get_basis_gates",
    "get_qubit_properties",
    "get_gate_errors",
    "get_gate_durations",
    "extract_backend_info",
    "get_heaviest_hex_layout",
    "get_error_for_layout",
    "print_backend_info",
    # transpiler
    "TranspilationResult",
    "OPTIMIZATION_LEVELS",
    "transpile_circuit",
    "transpile_post_routing",
    "transpile_all_levels",
    "transpile_batch",
    "compare_transpilation_results",
    "print_transpilation_comparison",
    "transpile_with_custom_layout",
    "run_baseline",
    "list_available_baselines",
    "run_named_baseline",
]
