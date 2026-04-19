"""
qiskit_interface — Módulo 1: Interfaz con Qiskit
=================================================

Módulo de interfaz con el ecosistema Qiskit 2.x para el TFG
"Transpilación Cuántica Híbrida".

Este paquete proporciona tres sub-módulos:

  - **circuit_utils**: Carga, creación, conversión y extracción de
    métricas de circuitos cuánticos.
  - **backend_info**: Consulta de información de backends cuánticos
    simulados (topología, puertas nativas, errores, T1/T2).
  - **transpiler**: Transpilación estándar de Qiskit como baseline
    y utilidades de comparación.

Uso típico::

    from src.qiskit_interface import (
        create_ghz_circuit,
        extract_metrics,
        get_backend,
        extract_backend_info,
        transpile_circuit,
        transpile_all_levels,
    )

    # Crear circuito y extraer métricas
    ghz = create_ghz_circuit(5)
    metrics = extract_metrics(ghz)
    print(metrics.summary())

    # Obtener info del backend
    backend = get_backend("fake_torino")
    info = extract_backend_info(backend)

    # Transpilar y comparar
    result = transpile_circuit(ghz, backend=backend, optimization_level=2)
    print(result.summary())
"""

# ===================================================================
#  Re-exportaciones públicas
# ===================================================================
#  Se importan aquí los símbolos más usados para que los usuarios
#  del módulo puedan hacer:
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
#  __all__ — API pública explícita
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
    "transpile_all_levels",
    "transpile_batch",
    "compare_transpilation_results",
    "print_transpilation_comparison",
    "transpile_with_custom_layout",
    "run_baseline",
    "list_available_baselines",
    "run_named_baseline",
]
