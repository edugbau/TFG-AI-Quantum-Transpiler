# Chapter 1 Base Transpilation Design

**Goal:** ampliar la capa base de transpilación para aceptar circuitos desde librería interna o ficheros QASM 2/3, producir artefactos estructurados y formalizar un catálogo pequeño de baselines sobre los fake backends actuales.

## Contexto

El repositorio ya dispone de una fachada útil en `src/qiskit_interface/` para:

- cargar y exportar circuitos QASM;
- generar benchmarks internos;
- consultar propiedades hardware de `fake_torino`, `fake_sherbrooke` y `fake_brisbane`;
- transpilar circuitos y comparar métricas.

También existe una `integration v1` capaz de orquestar `Baseline`, `MO_Only`, `RL_Only` y `MO+RL` para routing. Sin embargo, hoy la entrada visible de `integration` sigue limitada a la librería interna de circuitos y la salida Qiskit sigue siendo principalmente plana (`to_dict()`), lo que limita la trazabilidad experimental y deja el baseline demasiado implícito.

## Alcance aprobado

Este diseño cubre únicamente:

- soporte visible para `library` y `qasm_file` como entradas de circuito;
- soporte de formatos `qasm2`, `qasm3` y autodetección `auto`;
- mantenimiento del catálogo actual de fake backends, sin ampliarlo;
- artefacto estructurado adicional para resultados de transpilación;
- catálogo pequeño de baselines explícitos sobre el transpiler actual;
- exposición mínima de esa evolución en `src/integration/`.

## No objetivos

Quedan fuera de esta iteración:

- añadir nuevos backends más allá de `fake_torino`, `fake_sherbrooke` y `fake_brisbane`;
- incorporar benchmarks externos más allá de QASM 2/3;
- reconstruir circuitos finales para `RL_Only` o `MO+RL`;
- cerrar la comparabilidad homogénea entre escenarios Qiskit y RL;
- mover responsabilidades de orquestación fuera de `src/integration/`.

## Restricciones de diseño

1. `src/qiskit_interface/` sigue siendo dueño de parsing QASM, backends, métricas y baselines.
2. `src/integration/` sigue siendo dueño de la orquestación de escenarios y del handoff MO -> RL.
3. La convención `layout[i] = physical_qubit_for_logical_qubit_i` no cambia.
4. `TranspilationResult.to_dict()` debe seguir siendo compatible con el uso actual en tests, pandas e integración.
5. Los escenarios RL mantienen su contrato público actual: `episode summaries`, no circuitos finales.

## Diseño propuesto

### 1. Entrada normalizada de circuitos

Se añadirá un cargador normalizado en `src/qiskit_interface/circuit_utils.py`:

- `load_circuit(source_kind, *, circuit_name=None, num_qubits=None, seed=42, circuit_path=None, circuit_format="auto") -> QuantumCircuit`

Este cargador soportará dos modos:

- `library`: usa `circuits_from_library(...)` y exige `circuit_name` y `num_qubits`.
- `qasm_file`: exige `circuit_path`, permite `circuit_format` en `auto`, `qasm2` o `qasm3`, y deriva el nombre resuelto desde el fichero cuando no se indique otro explícitamente.

Para minimizar el acoplamiento, la metadata del input se adjuntará al propio `QuantumCircuit` mediante `circuit.metadata`, con al menos:

- `resolved_circuit_name`
- `source_kind`
- `source_format`
- `source_path`

`src/integration/scenarios.py` consumirá este cargador y dejará de parsear o resolver entradas por su cuenta.

### 2. Contrato de artefactos de transpilación

`TranspilationResult` mantendrá `to_dict()` como salida plana y añadirá:

- `baseline_name: str | None`
- `hardware_summary: dict | None`
- `to_artifact_dict() -> dict`

El artefacto estructurado tendrá esta forma conceptual:

```python
{
    "artifact_version": "transpilation_result.v1",
    "baseline_name": "qiskit_level_1",
    "circuit": {
        "resolved_circuit_name": "ghz",
        "source_kind": "library",
        "source_format": "library",
        "source_path": None,
        "num_qubits": 5,
    },
    "backend": {
        "backend_name": "fake_torino",
        "num_qubits": 133,
        "two_qubit_gate": "cz",
        "basis_gates": ["cz", "id", "rz", "sx", "x"],
        "coupling_edges_count": 144,
        "avg_error_2q": 0.01,
        "max_error_2q": 0.02,
        "avg_t1": 0.0001,
        "avg_t2": 0.00009,
    },
    "transpilation": {
        "optimization_level": 1,
        "seed": 42,
        "elapsed_time_s": 0.12,
        "initial_layout": None,
        "final_layout": [0, 1, 2, 3, 4],
    },
    "metrics": {
        "original": {...},
        "transpiled": {...},
    },
}
```

La nueva salida estructurada no sustituye a `to_dict()`; la complementa.

### 3. Catálogo pequeño de baselines

Se formalizará un catálogo mínimo en `src/qiskit_interface/transpiler.py`:

- `qiskit_level_0`
- `qiskit_level_1`
- `qiskit_level_2`
- `qiskit_level_3`
- `custom_layout_level_1`

Se añadirán:

- `BASELINE_PRESETS`
- `list_available_baselines()`
- `run_named_baseline(...)`

`run_named_baseline(...)` delegará internamente en `transpile_circuit(...)` o `transpile_with_custom_layout(...)` y etiquetará el `TranspilationResult` con `baseline_name`.

### 4. Resumen hardware-aware estable

Se añadirá un resumen hardware-aware estable derivado de `BackendInfo`, sin volcar objetos internos de Qiskit ni ampliar el modelo de backends.

El resumen incluirá:

- `backend_name`
- `num_qubits`
- `two_qubit_gate`
- `basis_gates`
- `coupling_edges_count`
- `min_error_2q`, `avg_error_2q`, `max_error_2q`
- `avg_t1`, `avg_t2`

Este resumen servirá tanto para `to_artifact_dict()` como para futuras comparativas experimentales.

### 5. Exposición mínima en integration

`src/integration/contracts.py` ampliará `ScenarioRequest` con:

- `circuit_source`: `library` o `qasm_file`
- `circuit_path`: ruta opcional
- `circuit_format`: `auto`, `qasm2` o `qasm3`

Validaciones clave:

- `library` exige `circuit_name` y `num_qubits`.
- `qasm_file` exige `circuit_path`.
- `library` no acepta `circuit_path`.

`ScenarioResult` añadirá `transpilation_artifact` como salida opcional, manteniendo `transpilation_metrics` como salida plana separada.

`Baseline` y `MO_Only` cargarán circuitos mediante el cargador normalizado y devolverán tanto `transpilation_metrics` como `transpilation_artifact`.

`RL_Only` y `MO+RL` podrán consumir también circuitos resueltos desde QASM, pero no cambiarán su salida pública actual.

## Flujo de datos

1. `runner.py` parsea la combinación de flags de entrada.
2. `ScenarioRequest` valida coherencia por escenario y por tipo de input.
3. `scenarios._load_circuit()` delega en `qiskit_interface.load_circuit(...)`.
4. El `QuantumCircuit` vuelve con metadata normalizada del input.
5. `Baseline` usa `run_named_baseline(..., baseline_name="qiskit_level_1")`.
6. `MO_Only` usa `run_named_baseline(..., baseline_name="custom_layout_level_1", layout=selected_layout)`.
7. `ScenarioResult` conserva métricas planas y añade el artefacto estructurado.

## Errores esperados

- combinación inválida de flags para `library` o `qasm_file`;
- fichero QASM inexistente;
- formato QASM no detectable en modo `auto`;
- `qasm3` sin importador opcional cuando el entorno no lo soporte;
- baseline desconocido o baseline con layout obligatorio no proporcionado.

Los errores deben ser tempranos y legibles, distinguiendo claramente entre error de configuración y error de carga/parsing.

## Estrategia de pruebas

### qiskit_interface

- carga normalizada desde librería y desde fichero QASM2;
- autodetección de formato QASM;
- carga QASM3 condicionada por dependencia opcional;
- `to_artifact_dict()` con estructura estable;
- catálogo de baselines explícitos;
- compatibilidad de `to_dict()`.

### integration

- validación de `ScenarioRequest` para `library` y `qasm_file`;
- soporte de `qasm_file` en `Baseline` y `MO_Only`;
- serialización de `transpilation_artifact`;
- compatibilidad del runner CLI con el modo actual y el nuevo modo QASM;
- ausencia de cambios semánticos en `RL_Only` y `MO+RL`.

### documentación

- actualización de `README.md`, `src/qiskit_interface/README.md` y `src/integration/README.md` para reflejar el nuevo alcance sin prometer comparabilidad completa con RL.

## Riesgos y mitigaciones

- Riesgo: romper la CLI actual.
  - Mitigación: mantener `library` como comportamiento por defecto y conservar `--circuit` / `--num-qubits` para ese caso.

- Riesgo: mezclar parsing QASM con lógica de integración.
  - Mitigación: toda la carga de circuitos seguirá concentrada en `qiskit_interface`.

- Riesgo: inflar demasiado el catálogo de baselines.
  - Mitigación: limitar la iteración a cinco presets explícitos.

- Riesgo: aparentar comparabilidad completa con RL.
  - Mitigación: mantener la separación entre `transpilation_metrics` y `routing_summary`, y conservar la nota de limitación RL.

## Criterios de éxito

- `Baseline` y `MO_Only` aceptan tanto `library` como `qasm_file`.
- El proyecto sigue usando exclusivamente los tres fake backends actuales.
- `TranspilationResult.to_dict()` sigue funcionando para el consumo actual.
- Existe un `to_artifact_dict()` estable y serializable.
- Los baselines Qiskit quedan formalizados y etiquetados explícitamente.
- Los tests de `tests/test_qiskit_interface` y `tests/test_integration` siguen verdes.
