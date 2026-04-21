# Documentación Interna del Módulo `integration`

Este documento detalla las responsabilidades, contratos, pipelines y seams de desacoplamiento del módulo `integration`. Este módulo actúa como la capa de orquestación entre `qiskit_interface`, `mo_module` y `rl_module`, y es el dueño del handoff MO -> RL para los escenarios de evaluación definidos en la v1 de `routing`.

## Visión General

La versión actual de `integration` implementa únicamente escenarios de evaluación para `routing`:

- `Baseline`
- `MO_Only`
- `RL_Only`
- `MO+RL`

El módulo no implementa entrenamiento RL, no cubre `synthesis` y no reconstruye todavía circuitos finales a partir de la salida del entorno RL. En esta versión, los escenarios basados en RL devuelven resúmenes de episodio (*episode summaries*), no artefactos `QuantumCircuit` finales.

QASM input is available for the Qiskit-facing scenarios of this v1, es decir, para `Baseline` y `MO_Only` cuando la petición usa `circuit_source="qasm_file"`. Los escenarios `RL_Only` y `MO+RL` siguen consumiendo la representación de circuito usada por el evaluador de routing y, como todavía no materializan circuitos finales, no exponen una entrada QASM equivalente en la superficie pública actual.

El backend catalog is intentionally limited a los fake backends actuales (`fake_torino`, `fake_sherbrooke`, `fake_brisbane`). Esta restricción mantiene la evaluación reproducible, evita dependencias de credenciales y refleja el catálogo controlado que publica `qiskit_interface` en esta etapa del proyecto.

## Estructura del Módulo

```
src/integration/
├── __init__.py              # API pública mínima del módulo.
├── contracts.py            # DTOs y validaciones de entrada/salida.
├── backend_adapter.py      # Traducción backend -> datos consumibles por integration.
├── layout_policy.py        # Selección de layout único a partir de la salida de MO.
├── routing_evaluator.py    # Ejecución de episodios RL de routing.
├── scenarios.py            # Orquestación de escenarios Baseline / MO / RL / MO+RL.
├── runner.py               # CLI reproducible para lanzar escenarios.
├── README.md               # Overview breve del alcance de la v1.
└── docs/
    └── internal_documentation.md
```

## Funcionalidades Principales

### 1. Contratos y Validación (`contracts.py`)
Define la superficie propia del módulo para evitar propagar estructuras internas de otros módulos.

- **`LayoutSelectionPolicy`**: enum con políticas de selección del layout MO (`compromise`, `best_on_objective`).
- **`ScenarioRequest`**: describe una ejecución de escenario.
  - valida nombres de escenario soportados;
  - valida `num_qubits`, `initial_layout` y coherencia de parámetros por escenario;
  - obliga a `rl_model_path` en `RL_Only` y `MO+RL`.
- **`RoutingEpisodeSummary`**: resume el resultado de un episodio RL.
  - incluye `initial_layout`, `final_layout`, `steps_executed`, `total_reward`, `completed`, `truncated`, `total_swaps` y `gates_executed_count`.
- **`ScenarioResult`**: salida unificada de cualquier escenario.
  - separa explícitamente `transpilation_metrics` de `routing_summary`;
  - evita estados imposibles como `success=True` con errores presentes.

### 2. Adaptación de Backend (`backend_adapter.py`)
Ubicación: `backend_adapter.py`

Centraliza la traducción desde el backend de Qiskit a una estructura pequeña y estable para `integration`.

- **`BackendBundle`**: agrupa `backend_name`, `backend`, `coupling_edges` y `basis_gates`.
- **`resolve_backend_bundle()`**:
  - obtiene el backend con `qiskit_interface.get_backend()`;
  - extrae aristas con `get_coupling_edges()`;
  - extrae `basis_gates` para compatibilidad futura con `synthesis`.

Esta capa evita que los escenarios tengan que conocer detalles de acceso a backends o de extracción de topología.

### 3. Selección del Layout MO (`layout_policy.py`)
Ubicación: `layout_policy.py`

Convierte la salida multiobjetivo de `mo_module` en un layout único consumible por el resto de `integration`.

- **`select_layout_from_mo_result()`**:
  - usa `get_compromise_layout()` para la política por defecto;
  - usa `get_best_layout(objective_index=...)` para selección dirigida por objetivo;
  - rechaza políticas no soportadas con `ValueError`.

Esta separación es deliberada: el resto de `integration` no necesita saber cómo se representa internamente el frente de Pareto.

### 4. Evaluación RL de Routing (`routing_evaluator.py`)
Ubicación: `routing_evaluator.py`

Encapsula la ejecución de un episodio RL en modo `routing`.

- **`_create_routing_env()`**:
  - crea un `QuantumTranspilationEnv` en `mode="routing"`;
  - importa el entorno RL de forma perezosa para reducir el acoplamiento de importación.
- **`evaluate_routing_episode()`**:
  - acepta circuito, `coupling_edges`, agente, seed e `initial_layout` opcional;
  - inyecta `initial_layout` vía `reset(options={"initial_layout": ...})`;
  - ejecuta `agent.predict(..., deterministic=True)` hasta terminación o truncado;
  - normaliza layouts a listas de `int` Python para que sean serializables a JSON;
  - contabiliza también las puertas ejecutadas durante `reset()` si el entorno ya avanza parcialmente antes del primer `step()`.

La salida de esta capa es siempre un `RoutingEpisodeSummary`. No reconstruye circuitos ni expone detalles de GUI.

### 5. Orquestación de Escenarios (`scenarios.py`)
Ubicación: `scenarios.py`

Contiene la lógica de composición de los cuatro escenarios soportados.

- **`_load_circuit()`**: carga circuitos mediante `qiskit_interface.load_circuit(...)`, resolviendo tanto `library` como `qasm_file` y propagando la metadata normalizada de entrada.
- **`_load_agent()`**: carga el modelo RL usando `QuantumRLAgent.load(...)`, con importación perezosa.
- **`_load_routing_contract()`**: intenta leer el sidecar `run_metadata.json` del checkpoint RL y usa ese saved routing contract cuando está disponible; si no existe, aplica legacy defaults para mantener compatibilidad con checkpoints previos.
- **`_require_scenario()`**: evita ejecutar el runner equivocado para un `ScenarioRequest` dado.
- **`_validate_selected_layout()`**: valida ancho, duplicados, negatividad y rango físico del layout seleccionado por MO antes de entregarlo a Qiskit o RL.
- **`_run_mo()`**: encapsula la elección entre `optimize_layout_quick()` y `optimize_layout()`.

Funciones principales:

- **`run_baseline_scenario()`**
  - ejecuta `run_named_baseline(..., "qiskit_level_1")` sin layout externo;
  - devuelve `transpilation_metrics` y `transpilation_artifact`.
- **`run_mo_only_scenario()`**
  - ejecuta MO;
  - selecciona un layout;
  - evalúa ese layout con `run_named_baseline(..., "custom_layout_level_1", layout=...)`;
  - devuelve `selected_layout`, `transpilation_metrics` y `transpilation_artifact`.
- **`run_rl_only_scenario()`**
  - carga modelo RL;
  - resuelve el contrato de routing desde `run_metadata.json` cuando está presente o desde legacy defaults cuando falta el sidecar;
  - evalúa el episodio con el layout inicial proporcionado por el llamador o el layout trivial del entorno;
  - devuelve `routing_summary` y publica el fallback mediante `ScenarioResult.notes` cuando falta el sidecar.
- **`run_mo_rl_scenario()`**
  - ejecuta MO;
  - selecciona un layout;
  - lo inyecta en RL como `initial_layout`;
  - reutiliza el mismo contrato de routing persistido y, si falta el sidecar, reporta el fallback mediante `ScenarioResult.notes`;
  - devuelve `selected_layout`, `routing_summary` y la nota de limitación RL.

## Pipelines (Flujos de Trabajo)

### A. Pipeline `Baseline`

1. Crear `ScenarioRequest` desde la CLI o desde código.
2. Resolver backend con `resolve_backend_bundle()`.
3. Cargar circuito mediante `_load_circuit()`, resolviendo `library` o `qasm_file` vía `qiskit_interface.load_circuit(...)`.
4. Ejecutar `qiskit_interface.run_named_baseline(..., baseline_name="qiskit_level_1")`.
5. Devolver `ScenarioResult` con `transpilation_metrics` y `transpilation_artifact`.

### B. Pipeline `MO_Only`

1. Crear `ScenarioRequest`.
2. Resolver backend.
3. Cargar circuito mediante `_load_circuit()`.
4. Ejecutar MO (`optimize_layout_quick()` o `optimize_layout()`).
5. Seleccionar un layout mediante `layout_policy`.
6. Validar el layout antes de entregarlo a dependencias externas.
7. Ejecutar `qiskit_interface.run_named_baseline(..., baseline_name="custom_layout_level_1", layout=selected_layout)`.
8. Devolver `ScenarioResult` con `selected_layout`, métricas Qiskit y `transpilation_artifact`.

### C. Pipeline `RL_Only`

1. Crear `ScenarioRequest` con `rl_model_path`.
2. Resolver backend.
3. Cargar circuito y agente RL.
4. Resolver el contrato de routing desde `run_metadata.json` si existe; en caso contrario, aplicar legacy defaults.
5. Ejecutar `evaluate_routing_episode()`.
6. Devolver `ScenarioResult` con `routing_summary` y una nota adicional si hubo fallback.

### D. Pipeline `MO+RL`

1. Crear `ScenarioRequest` con `rl_model_path`.
2. Resolver backend.
3. Cargar circuito.
4. Ejecutar MO y seleccionar layout.
5. Validar el layout seleccionado.
6. Cargar agente RL.
7. Resolver el contrato de routing desde el sidecar cuando esté disponible y conservar el fallback a legacy defaults cuando falte.
8. Ejecutar `evaluate_routing_episode()` con ese layout como `initial_layout`.
9. Devolver `ScenarioResult` con `selected_layout`, `routing_summary` y una nota adicional si hubo fallback.

## CLI y Reproducibilidad

La entrada visible del módulo es `runner.py`.

- **`build_parser()`** define la superficie CLI de la v1:
  - `--scenario`
  - `--circuit-source`
  - `--circuit`
  - `--circuit-path`
  - `--circuit-format`
  - `--num-qubits`
  - `--backend`
  - `--seed`
  - `--rl-model-path`
- `--circuit` y `--num-qubits` siguen siendo la entrada principal para `library`; `--circuit-path` y `--circuit-format` habilitan `qasm_file` para los escenarios Qiskit-facing.
- **`run_from_args()`** construye un `ScenarioRequest`, despacha al escenario correcto y serializa el `ScenarioResult`.
- **`main()`** imprime el payload en JSON y devuelve código de salida `0` cuando la ejecución tiene éxito.

La CLI está diseñada como un runner fino. No contiene lógica de negocio más allá de parseo, dispatch y serialización.

## Patrones de Diseño y Seams de Desacoplamiento

| Patrón / seam | Dónde | Propósito |
| :--- | :--- | :--- |
| **DTO / Dataclass** | `contracts.py` | Definir contratos estables de entrada y salida del módulo. |
| **Adapter** | `backend_adapter.py` | Encapsular backend, topología y `basis_gates` sin contaminar escenarios. |
| **Policy seam** | `layout_policy.py` | Separar selección de layout de la orquestación de escenarios. |
| **Evaluation seam** | `routing_evaluator.py` | Encapsular la dependencia con el entorno RL y su ciclo de episodio. |
| **Scenario orchestration** | `scenarios.py` | Mantener cada escenario como una composición pequeña y explícita. |
| **Thin CLI** | `runner.py` | Exponer el módulo sin acoplarlo a GUIs ni a detalles de entrenamiento. |

Dos decisiones de desacoplamiento importantes:

1. Las importaciones a RL más pesadas (`QuantumTranspilationEnv`, `QuantumRLAgent`) se hacen de forma perezosa dentro de seams privadas.
2. `integration` depende del contrato de layout y de resultados resumidos, no del estado interno completo de MO o RL.

## Funciones Llamadas Extensamente

| Función | Archivo | Descripción | Uso |
| :--- | :--- | :--- | :--- |
| **`resolve_backend_bundle`** | `backend_adapter.py` | Resuelve backend y topología útil para los escenarios. | Se usa al inicio de todos los escenarios. |
| **`select_layout_from_mo_result`** | `layout_policy.py` | Selecciona un layout único desde la salida de MO. | Se usa en `MO_Only` y `MO+RL`. |
| **`evaluate_routing_episode`** | `routing_evaluator.py` | Ejecuta un episodio RL y lo resume. | Núcleo de `RL_Only` y `MO+RL`. |
| **`run_baseline_scenario`** | `scenarios.py` | Orquesta el baseline Qiskit. | Entrada de referencia comparativa. |
| **`run_mo_only_scenario`** | `scenarios.py` | Orquesta MO + evaluación Qiskit. | Escenario de evaluación del layout MO. |
| **`run_rl_only_scenario`** | `scenarios.py` | Orquesta evaluación RL sin MO. | Escenario RL puro en `routing`. |
| **`run_mo_rl_scenario`** | `scenarios.py` | Orquesta el handoff MO -> RL. | Escenario híbrido principal de la v1. |
| **`run_from_args`** | `runner.py` | Punto de entrada programático de la CLI. | Ejecuta parseo, dispatch y serialización. |

## Integración con Otros Módulos

### Dependencias de `qiskit_interface`

- `get_backend()`, `get_coupling_edges()`, `get_basis_gates()` para resolver `BackendBundle`.
- `load_circuit()` para construir la entrada visible de la v1 desde `library` o `qasm_file`.
- `run_named_baseline()` para `Baseline` y para la evaluación Qiskit de `MO_Only`.

### Dependencias de `mo_module`

- `optimize_layout_quick()` / `optimize_layout()` para generar layouts candidatos.
- `OptimizationResult.get_compromise_layout()` y `get_best_layout()` como API pública consumida por `layout_policy.py`.

### Dependencias de `rl_module`

- `QuantumRLAgent.load()` para cargar modelos ya entrenados.
- `QuantumTranspilationEnv` en `mode="routing"` para ejecutar episodios.
- Soporte de `initial_layout` vía `reset(options=...)` como contrato de handoff.
- Sidecar `run_metadata.json` para persistir el contrato de evaluación de routing; si falta, `integration` conserva legacy defaults y reporta el fallback mediante una entrada adicional en `ScenarioResult.notes`.

## Limitaciones Conocidas de la v1

### 1. Salida RL no materializada como circuito final
Los escenarios `RL_Only` y `MO+RL` no devuelven todavía un circuito final reconstruido. La salida pública es un resumen de episodio.

Consecuencia:

- `Baseline` y `MO_Only` son comparables entre sí mediante métricas Qiskit.
- `RL_Only` y `MO+RL` son comparables entre sí mediante métricas de episodio.
- La comparación homogénea entre ambos grupos queda incompleta hasta una iteración futura.

### 2. Alcance limitado a `routing`
La v1 no orquesta todavía `synthesis`.

### 3. Entrada QASM limitada a escenarios Qiskit-facing
La v1 ya acepta `qasm_file` para `Baseline` y `MO_Only`. Ese soporte no se extiende todavía a `RL_Only` ni `MO+RL`, porque esos flujos siguen devolviendo `RoutingEpisodeSummary` y no un circuito final materializado.

### 4. Sin entrenamiento RL desde `integration`
`integration` consume modelos ya entrenados, pero no invoca el pipeline de entrenamiento.

### 5. Fallback para metadata ausente en checkpoints antiguos
Cuando un checkpoint RL no trae `run_metadata.json`, `integration` sigue evaluándolo con legacy defaults. Ese fallback se reporta mediante una nota pública en `ScenarioResult.notes`, por ejemplo `Legacy RL evaluation defaults were used because no run metadata sidecar was found.`.

## Evolución Futura

Líneas naturales de evolución del módulo:

1. Soporte de `synthesis` como nuevo conjunto de escenarios.
2. Entrada de circuitos desde QASM para escenarios RL cuando exista una representación final de circuito consistente.
3. Reconstrucción/exportación de circuitos finales desde RL.
4. Comparabilidad homogénea entre escenarios Qiskit y RL.
5. Posible capa visual encima del runner y de los DTOs ya existentes.
