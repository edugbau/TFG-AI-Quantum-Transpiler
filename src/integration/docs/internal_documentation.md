# Documentación Interna del Módulo `integration`

Este documento detalla las responsabilidades, contratos, pipelines y seams de desacoplamiento del módulo `integration`. Este módulo actúa como la capa de orquestación entre `qiskit_interface`, `mo_module` y `rl_module`, y es el dueño del handoff MO -> RL para los escenarios de evaluación definidos en la v1 de `routing`.

## Visión General

La versión actual de `integration` implementa únicamente escenarios de evaluación para `routing`:

- `Baseline`
- `MO_Only`
- `RL_Only`
- `MO+RL`

El módulo no implementa entrenamiento RL, no cubre `synthesis` y no reconstruye todavía circuitos finales a partir de la salida del entorno RL. En esta versión, los escenarios basados en RL devuelven resúmenes de episodio (*episode summaries*), no artefactos `QuantumCircuit` finales.

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

- **`_load_circuit()`**: carga circuitos desde `qiskit_interface.circuits_from_library(...)`.
- **`_load_agent()`**: carga el modelo RL usando `QuantumRLAgent.load(...)`, con importación perezosa.
- **`_require_scenario()`**: evita ejecutar el runner equivocado para un `ScenarioRequest` dado.
- **`_validate_selected_layout()`**: valida ancho, duplicados, negatividad y rango físico del layout seleccionado por MO antes de entregarlo a Qiskit o RL.
- **`_run_mo()`**: encapsula la elección entre `optimize_layout_quick()` y `optimize_layout()`.

Funciones principales:

- **`run_baseline_scenario()`**
  - transpila con `qiskit_interface` sin layout externo;
  - devuelve solo `transpilation_metrics`.
- **`run_mo_only_scenario()`**
  - ejecuta MO;
  - selecciona un layout;
  - evalúa ese layout con `transpile_with_custom_layout()`;
  - devuelve `selected_layout` y `transpilation_metrics`.
- **`run_rl_only_scenario()`**
  - carga modelo RL;
  - evalúa el episodio con el layout inicial proporcionado por el llamador o el layout trivial del entorno;
  - devuelve `routing_summary` y la nota de limitación RL.
- **`run_mo_rl_scenario()`**
  - ejecuta MO;
  - selecciona un layout;
  - lo inyecta en RL como `initial_layout`;
  - devuelve `selected_layout`, `routing_summary` y la nota de limitación RL.

## Pipelines (Flujos de Trabajo)

### A. Pipeline `Baseline`

1. Crear `ScenarioRequest` desde la CLI o desde código.
2. Resolver backend con `resolve_backend_bundle()`.
3. Cargar circuito desde la librería interna.
4. Ejecutar `qiskit_interface.transpile_circuit()`.
5. Devolver `ScenarioResult` con `transpilation_metrics`.

### B. Pipeline `MO_Only`

1. Crear `ScenarioRequest`.
2. Resolver backend.
3. Cargar circuito.
4. Ejecutar MO (`optimize_layout_quick()` o `optimize_layout()`).
5. Seleccionar un layout mediante `layout_policy`.
6. Validar el layout antes de entregarlo a dependencias externas.
7. Ejecutar `transpile_with_custom_layout()`.
8. Devolver `ScenarioResult` con métricas Qiskit y `selected_layout`.

### C. Pipeline `RL_Only`

1. Crear `ScenarioRequest` con `rl_model_path`.
2. Resolver backend.
3. Cargar circuito y agente RL.
4. Ejecutar `evaluate_routing_episode()`.
5. Devolver `ScenarioResult` con `routing_summary`.

### D. Pipeline `MO+RL`

1. Crear `ScenarioRequest` con `rl_model_path`.
2. Resolver backend.
3. Cargar circuito.
4. Ejecutar MO y seleccionar layout.
5. Validar el layout seleccionado.
6. Cargar agente RL.
7. Ejecutar `evaluate_routing_episode()` con ese layout como `initial_layout`.
8. Devolver `ScenarioResult` con `selected_layout`, `routing_summary` y nota de limitación.

## CLI y Reproducibilidad

La entrada visible del módulo es `runner.py`.

- **`build_parser()`** define la superficie CLI de la v1:
  - `--scenario`
  - `--circuit`
  - `--num-qubits`
  - `--backend`
  - `--seed`
  - `--rl-model-path`
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
- `circuits_from_library()` para construir la entrada visible de la v1.
- `transpile_circuit()` para `Baseline`.
- `transpile_with_custom_layout()` para `MO_Only`.

### Dependencias de `mo_module`

- `optimize_layout_quick()` / `optimize_layout()` para generar layouts candidatos.
- `OptimizationResult.get_compromise_layout()` y `get_best_layout()` como API pública consumida por `layout_policy.py`.

### Dependencias de `rl_module`

- `QuantumRLAgent.load()` para cargar modelos ya entrenados.
- `QuantumTranspilationEnv` en `mode="routing"` para ejecutar episodios.
- Soporte de `initial_layout` vía `reset(options=...)` como contrato de handoff.

## Limitaciones Conocidas de la v1

### 1. Salida RL no materializada como circuito final
Los escenarios `RL_Only` y `MO+RL` no devuelven todavía un circuito final reconstruido. La salida pública es un resumen de episodio.

Consecuencia:

- `Baseline` y `MO_Only` son comparables entre sí mediante métricas Qiskit.
- `RL_Only` y `MO+RL` son comparables entre sí mediante métricas de episodio.
- La comparación homogénea entre ambos grupos queda incompleta hasta una iteración futura.

### 2. Alcance limitado a `routing`
La v1 no orquesta todavía `synthesis`.

### 3. Entrada visible limitada a librería interna de circuitos
La CLI actual no acepta QASM. Ese soporte queda explícitamente diferido.

### 4. Sin entrenamiento RL desde `integration`
`integration` consume modelos ya entrenados, pero no invoca el pipeline de entrenamiento.

## Evolución Futura

Líneas naturales de evolución del módulo:

1. Soporte de `synthesis` como nuevo conjunto de escenarios.
2. Entrada de circuitos desde QASM.
3. Reconstrucción/exportación de circuitos finales desde RL.
4. Comparabilidad homogénea entre escenarios Qiskit y RL.
5. Posible capa visual encima del runner y de los DTOs ya existentes.
