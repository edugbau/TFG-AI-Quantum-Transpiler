# qiskit_interface

`qiskit_interface` es la capa de contrato entre Qiskit y el resto del proyecto. Convierte circuitos, metadata de backend y resultados de transpilacion en artefactos que pueden consumir `mo_module` e `integration` sin depender de detalles internos de Qiskit.

## Piezas publicas

| Archivo | Responsabilidad | Simbolos clave |
| --- | --- | --- |
| `circuit_utils.py` | Carga, generacion y conversion de circuitos, mas metricas comunes | `CircuitMetrics`, `load_circuit`, `load_circuit_from_qasm2`, `load_circuit_from_qasm3`, `export_circuit_to_qasm2`, `export_circuit_to_qasm3`, `create_ghz_circuit`, `create_qft_circuit`, `create_random_circuit`, `create_clifford_circuit`, `extract_metrics` |
| `backend_info.py` | Consulta unificada de hardware simulado | `BackendInfo`, `AVAILABLE_BACKENDS`, `get_backend`, `list_available_backends`, `extract_backend_info`, `get_heaviest_hex_layout`, `get_error_for_layout` |
| `transpiler.py` | Baseline de Qiskit, evaluacion de layouts externos y comparaciones | `TranspilationResult`, `transpile_circuit`, `transpile_with_custom_layout`, `transpile_post_routing`, `run_baseline`, `list_available_baselines`, `run_named_baseline` |
| `__init__.py` | Fachada publica del paquete | Reexports de los contratos mas usados |

## Contratos centrales

### `CircuitMetrics`

Resume un circuito como datos serializables. Sus campos principales son:

- `depth`
- `num_qubits`
- `total_gates`
- `two_qubit_gates`
- `nonlocal_gates`
- `cnot_equivalent`
- `gate_counts`
- `width`
- `num_clbits`
- `active_qubits`

Importante: `two_qubit_gates` cuenta operaciones de 2 o mas qubits. `cnot_equivalent` se mantiene como metrica separada para informes y compatibilidad con el resto del proyecto.

### `BackendInfo`

Agrupa la informacion necesaria para evaluar layouts y transpilaciones:

- `coupling_map` y `coupling_edges`
- `basis_gates`
- `two_qubit_gate`
- `single_qubit_gates`
- `qubit_t1`, `qubit_t2`, `qubit_frequency`
- `gate_errors_1q`, `gate_errors_2q`
- `dt`
- `backend_kind`
- `topology_metadata`

### `TranspilationResult`

Concentra el resultado completo de una transpilacion:

- circuito original y circuito transpileado
- metricas originales y metricas finales
- `optimization_level`
- `backend_name`
- `initial_layout`, `qiskit_initial_layout` y `final_layout`
- `baseline_name`
- `hardware_summary`
- `elapsed_time_s`

El resultado puede serializarse con `to_dict()` y tambien con un artefacto estructurado para consumo de `integration`.

## Flujo general

1. `circuit_utils.py` crea o carga el circuito.
2. `backend_info.py` resuelve el hardware simulado y sus propiedades.
3. `transpiler.py` ejecuta la baseline de Qiskit o evalua un layout externo.
4. `integration` consume el resultado para escenarios `Baseline`, `MO_Only` y la etapa post-routing de `MO+RL`.
5. `mo_module` consume metricas y helpers de transpilacion para puntuar layouts.

## Decisiones de diseno

- Se usa la API moderna de Qiskit 2.x, incluyendo `backend.target`, `qasm2`/`qasm3` y `synth_qft_full`.
- El catalogo de backends publicos se limita a fake backends para mantener reproducibilidad y evitar credenciales.
- El modulo acepta backends tipo fake o sinteticos por duck typing cuando `integration` o sus capas de apoyo necesitan trabajar con topologias derivadas.
- `transpile_with_custom_layout` evalua un layout suministrado por el llamador sin asumir que proviene de MO.
- `transpile_post_routing` existe para el flujo RL: Qiskit completa la parte posterior al routing cuando `integration` reconstruye un circuito ruteado.

## Limites y alcance

- `qiskit_interface` no orquesta MO, RL ni Campaigns.
- `load_circuit` soporta entrada de biblioteca y `qasm_file`; los escenarios RL publicos siguen sin exponer una entrada QASM equivalente.
- El catalogo publico de backends se limita a `fake_torino`, `fake_sherbrooke` y `fake_brisbane`.
- `run_baseline` y `run_named_baseline` son la forma canonica de comparar contra Qiskit.
- `transpile_post_routing` y `run_named_baseline` son relevantes para `integration`; `mo_module` los usa solo como parte de la evaluacion de layouts.

## Lectura recomendada

1. Empieza por `CircuitMetrics`, `BackendInfo` y `TranspilationResult`.
2. Sigue con `load_circuit` y `get_backend`.
3. Termina con `transpile_with_custom_layout`, `transpile_post_routing` y `run_named_baseline` para entender el flujo de comparacion.
