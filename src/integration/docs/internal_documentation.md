# Documentación Interna del Módulo `integration`

Este documento detalla las responsabilidades, contratos, pipelines y seams de desacoplamiento del módulo `integration`. Este módulo actúa como la capa de orquestación entre `qiskit_interface`, `mo_module` y `rl_module`, y es el dueño tanto del handoff MO -> RL como de la orquestación de Campaigns y de la comparación de Scenarios en la v1 actual.

## Visión General

La versión actual de `integration` tiene dos capas públicas complementarias:

1. una capa de Scenarios para evaluación unitaria de `routing`;
2. una capa de Campaign para orquestar flujos reproducibles de `train+eval` por Campaign Case.

La capa de Scenarios sigue cubriendo:

- `Baseline`
- `MO_Only`
- `RL_Only`
- `MO+RL`

La capa de Campaign soporta una Train+Eval Campaign donde cada Campaign Case corresponde a una combinación `circuit x backend`. El conjunto canónico de comparación dentro de esa Campaign es `Baseline`, `MO_Only` y `MO+RL`. `RL_Only` sigue existiendo como Scenario, pero queda fuera del flujo guiado principal de Campaign.

Dentro de ese flujo guiado, `MO_Only` es el Scenario que selecciona el layout del Campaign Case. El training de Campaign para `MO+RL` arranca desde ese layout exacto, produce el Training Artifact del caso y la evaluación posterior de `MO+RL` reutiliza tanto ese mismo layout como ese artifacto resultante.

El módulo no implementa el entrenamiento RL en sí mismo ni cubre `synthesis` en esta capa pública. `integration` orquesta el training por Campaign Case a través de un seam explícito y consume el Training Artifact resultante, mientras que `rl_module` sigue siendo dueño de cómo se ejecuta el training y de cómo se producen los checkpoints. En esta versión, `RL_Only` sigue devolviendo resúmenes de episodio (*episode summaries*), mientras que `MO+RL` ya puede reconstruir el circuito ruteado desde la traza RL: usa `executed_gate_trace` cuando está disponible para reproducir exactamente las puertas ejecutadas, usa `swap_trace` para materializar los swaps físicos y ejecuta las fases post-routing de Qiskit cuando el episodio RL completa el routing.

En términos de ownership:

- `integration` owns Campaign orchestration, Scenario comparison, persistence, Summary Document generation y el handoff MO -> RL.
- `rl_module` owns RL training implementation and checkpoint production.
- `mo_module` owns layout generation and selection inputs.
- En el camino híbrido, MO entra en evaluación a través de `initial_layout`; ese contrato se mantiene explícito y sigue siendo propiedad de `integration`.

QASM input is available for the Qiskit-facing scenarios of this v1, es decir, para `Baseline` y `MO_Only` cuando la petición usa `circuit_source="qasm_file"`. Los escenarios `RL_Only` y `MO+RL` no exponen todavía una entrada QASM equivalente en la superficie pública actual.

The backend catalog is intentionally limited. La capa subyacente de `integration` trabaja con el catálogo actual de fake backends publicado por `qiskit_interface` (`fake_torino`, `fake_sherbrooke`, `fake_brisbane`). Sin embargo, la guided Campaign CLI expone hoy una superficie más estrecha y solo ofrece `fake_torino` y `fake_brisbane`. Esta restricción mantiene la ejecución guiada reproducible, evita dependencias de credenciales y refleja la superficie actualmente implementada en `campaign_cli.py`.

## Estructura del Módulo

```
src/integration/
├── __init__.py              # API pública mínima del módulo.
├── campaign_cli.py          # CLI guiada para construir y ejecutar Campaigns.
├── campaign_contracts.py    # DTOs de Campaign, Campaign Case y resumen agregado.
├── campaign_reporting.py    # Summary Document, agregados y persistencia pública.
├── campaign_runner.py       # Orquestación secuencial train+eval por Campaign Case.
├── contracts.py            # DTOs y validaciones de entrada/salida.
├── backend_adapter.py      # Traducción backend -> datos consumibles por integration.
├── layout_policy.py        # Selección de layout único a partir de la salida de MO.
├── routing_evaluator.py    # Ejecución de episodios RL de routing.
├── scenarios.py            # Orquestación de escenarios Baseline / MO / RL / MO+RL.
├── training_bridge.py      # Seam Campaign -> rl_module.training.
├── runner.py               # CLI reproducible para lanzar escenarios.
├── README.md               # Overview breve del alcance de la v1.
└── docs/
    └── internal_documentation.md
```

## Funcionalidades Principales

### 0. Campaigns y Resumen Público (`campaign_*`, `training_bridge.py`)
Define la capa Campaign sin mover la implementación de training dentro de `integration`.

- **`campaign_contracts.py`**:
  - modela `Campaign`, `CampaignConfig`, `CampaignCase` y `CampaignSummary`;
  - distingue `Default Campaign` y `Advanced Campaign` mediante `CampaignConfig.mode`;
  - enumera Campaign Cases como combinaciones `circuit x backend`.
- **`training_bridge.py`**:
  - actúa como seam entre Campaign y `src.rl_module.training`;
  - reenvía el layout seleccionado por `MO_Only` como `initial_layout` cuando el Campaign Case ejecuta el camino `MO+RL`;
  - devuelve el Training Artifact seleccionado, prefiriendo `best_model.zip` y con fallback a `final_model.zip`.
- **`campaign_runner.py`**:
  - ejecuta `Baseline`, `MO_Only`, training RL y `MO+RL` en secuencia por Campaign Case;
  - usa el layout exacto seleccionado por `MO_Only` para lanzar el training RL del camino `MO+RL` y reutiliza ese mismo layout en la evaluación híbrida del caso;
  - persiste el estado de la Campaign tras cada caso;
  - marca casos como `failed`, `incomplete` o `cancelled` en los caminos explícitos implementados por el runner, pero la no comparabilidad agregada también puede quedar reflejada solo en `CampaignSummary.incomplete_cases` e incidents aunque el `case_report.status` permanezca `completed`.
- **`campaign_reporting.py`**:
  - genera el Summary Document `summary.md` y el output estructurado `campaign.json`;
  - persiste también `cases/<case>/result.json`;
  - resume agregados comparables, detalle por caso, training notes e incidents.
- **`campaign_cli.py`**:
  - expone la guided CLI para construir una Campaign reproducible;
  - ofrece el camino `default` con valores canónicos compartidos y el camino `advanced` con ajuste explícito de MO y RL.

### 1. Contratos y Validación (`contracts.py`)
Define la superficie propia del módulo para evitar propagar estructuras internas de otros módulos.

- **`LayoutSelectionPolicy`**: enum con políticas de selección del layout MO (`compromise`, `best_on_objective`).
- **`ScenarioRequest`**: describe una ejecución de escenario.
  - valida nombres de escenario soportados;
  - valida `num_qubits`, `initial_layout` y coherencia de parámetros por escenario;
  - obliga a `rl_model_path` en `RL_Only` y `MO+RL`.
- **`RoutingEpisodeSummary`**: resume el resultado de un episodio RL.
  - incluye `initial_layout`, `final_layout`, `steps_executed`, `total_reward`, `completed`, `truncated`, `total_swaps`, `gates_executed_count`, `swap_trace` y `executed_gate_trace`;
  - garantiza `total_swaps == len(swap_trace)`, por lo que `total_swaps` representa swaps realmente materializados y reproducibles.
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
  - contabiliza también las puertas ejecutadas durante `reset()` si el entorno ya avanza parcialmente antes del primer `step()`;
  - captura `swap_trace` como la secuencia ordenada de swaps válidos materializables;
  - captura `executed_gate_trace` como la secuencia exacta de puertas lógicas realmente ejecutadas durante `reset()` y `step()`.

- **`build_routed_circuit()`**:
  - reconstruye un `QuantumCircuit` físico a partir de `initial_layout` y la traza RL;
  - prefiere `executed_gate_trace` para reproducir exactamente las puertas ejecutadas y usa `swap_trace` para insertar los swaps físicos necesarios;
  - mantiene un fallback de reconstrucción por frontier replay cuando no se proporciona `executed_gate_trace`;
  - devuelve también el `final_layout` reconstruido para que `scenarios.py` pueda validarlo antes del post-routing de Qiskit.

La salida de esta capa es siempre un `RoutingEpisodeSummary`. No reconstruye circuitos ni expone detalles de GUI.

### 5. Orquestación de Escenarios (`scenarios.py`)
Ubicación: `scenarios.py`

Contiene la lógica de composición de los cuatro escenarios soportados.

- **`_load_circuit()`**: carga circuitos mediante `qiskit_interface.load_circuit(...)`, resolviendo tanto `library` como `qasm_file` y propagando la metadata normalizada de entrada.
- **`_load_agent()`**: carga el modelo RL usando `QuantumRLAgent.load(...)`, con importación perezosa.
- **`resolve_routing_model_contract()`**: entrypoint definido en `rl_model_contract.py` y llamado desde `scenarios.py`; intenta leer el sidecar `run_metadata.json` del checkpoint RL y usa ese saved routing contract cuando está disponible; si el sidecar trae versioned masked routing metadata, `integration` consume esa variante explícita del contrato; si no existe, aplica legacy defaults para mantener compatibilidad con checkpoints previos.
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
  - resuelve el contrato de routing mediante `resolve_routing_model_contract()` desde `run_metadata.json` cuando está presente, incluyendo versioned masked routing metadata para checkpoints nuevos de `MaskablePPO`, o desde legacy defaults cuando falta el sidecar;
  - evalúa el episodio con el layout inicial proporcionado por el llamador o el layout trivial del entorno;
  - devuelve `routing_summary` y publica el fallback mediante `ScenarioResult.notes` cuando falta el sidecar.
- **`run_mo_rl_scenario()`**
  - ejecuta MO;
  - selecciona un layout;
  - lo inyecta en RL como `initial_layout`;
  - reutiliza el mismo contrato de routing persistido y, si falta el sidecar, reporta el fallback mediante `ScenarioResult.notes`;
  - si el episodio RL completa el routing, valida que `routing_summary.initial_layout` siga coincidiendo con el layout seleccionado, reconstruye el circuito ruteado desde `executed_gate_trace` + `swap_trace`, valida el `final_layout` reconstruido y ejecuta `qiskit_interface.transpile_post_routing(...)` para devolver métricas comparables y un artefacto de transpilación;
  - si el episodio RL no completa el routing, devuelve un `ScenarioResult` controlado sin artefacto de transpilación final.

Sobre comparabilidad de métricas Qiskit:

- `trans_num_qubits` y `trans_width` siguen describiendo la anchura física materializada del circuito transpìlado.
- `trans_active_qubits` describe cuántos qubits físicos quedan realmente activos tras la transpilación y es la métrica adecuada para comparar layouts dispersos.

## Pipelines (Flujos de Trabajo)

### Campaign layer

#### E. Train+Eval Campaign

1. Construir una `CampaignConfig` mediante el camino `default` o `advanced`.
2. Crear una `Campaign` y expandir sus Campaign Cases como combinaciones `circuit x backend`.
3. Para cada Campaign Case, ejecutar `Baseline` y `MO_Only`.
4. Tomar el layout exacto seleccionado por `MO_Only` para ese Campaign Case.
5. Lanzar training RL a través de `training_bridge.py` usando ese layout como `initial_layout`.
6. Seleccionar el Training Artifact, prefiriendo `best_model.zip` y haciendo fallback a `final_model.zip`.
7. Ejecutar `MO+RL` reutilizando ese mismo layout y ese Training Artifact en evaluación.
8. Persistir el estado público de la Campaign en `summary.md`, `campaign.json` y `cases/<case>/result.json`.
9. Reportar explícitamente casos `failed`, `incomplete` o `cancelled` cuando el runner así lo establezca, y además registrar en agregados e incidents los casos que terminan sin un bundle comparable completo.

#### F. Default Campaign vs Advanced Campaign

- **Default Campaign**: usa el conjunto canónico de defaults compartidos por la guided CLI para priorizar reproducibilidad y una configuración breve.
- **Advanced Campaign**: permite elegir explícitamente backend(s), algoritmo RL, timesteps, frontier mode, lookahead, max steps, seed, tamaño de MO y layout policy, dentro de la superficie actualmente expuesta por la guided CLI (`fake_torino`, `fake_brisbane`).
- En ambos casos, la comparación principal sigue siendo `Baseline`, `MO_Only` y `MO+RL`, y el Summary Document conserva los mismos artefactos públicos.

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
4. Resolver el contrato de routing con `resolve_routing_model_contract()` desde `run_metadata.json` si existe, consumiendo versioned masked routing metadata cuando aparezca; en caso contrario, aplicar legacy defaults.
5. Ejecutar `evaluate_routing_episode()`.
6. Devolver `ScenarioResult` con `routing_summary` y una nota adicional si hubo fallback.

### D. Pipeline `MO+RL`

1. Crear `ScenarioRequest` con `rl_model_path`.
2. Resolver backend.
3. Cargar circuito.
4. Ejecutar MO y seleccionar layout.
5. Validar el layout seleccionado.
6. Cargar agente RL.
7. Resolver el contrato de routing con `resolve_routing_model_contract()` desde el sidecar cuando esté disponible, incluyendo versioned masked routing metadata para checkpoints nuevos de `MaskablePPO`, y conservar el fallback a legacy defaults cuando falte.
8. Ejecutar `evaluate_routing_episode()` con ese layout como `initial_layout`.
9. Si el episodio completa el routing, reconstruir el circuito final desde `executed_gate_trace` + `swap_trace`, validar el `final_layout` reconstruido y ejecutar `qiskit_interface.transpile_post_routing(...)`.
10. Si el episodio no completa el routing, devolver un `ScenarioResult` controlado sin reconstrucción ni post-transpilación.

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
- `src.rl_module.training` a través de `training_bridge.py` para ejecutar training por Campaign Case sin mover la implementación del training a `integration`.
- Sidecar `run_metadata.json` para persistir el contrato de evaluación de routing; cuando el sidecar incluye versioned masked routing metadata, `integration` consume esa variante para checkpoints nuevos de `MaskablePPO`; si falta, `integration` conserva legacy defaults y reporta el fallback mediante una entrada adicional en `ScenarioResult.notes`.

## Limitaciones Conocidas de la v1

### 1. Materialización final limitada por escenario RL
`RL_Only` no devuelve todavía un circuito final reconstruido. `MO+RL` sí lo hace cuando el episodio RL completa el routing.

Consecuencia:

- `Baseline`, `MO_Only` y `MO+RL` pueden compararse mediante métricas Qiskit cuando `MO+RL` completa el routing.
- `RL_Only` sigue siendo un escenario de resumen de episodio.
- Si `MO+RL` no completa el routing, devuelve un resultado controlado sin métricas finales de transpilación.

### 2. Alcance limitado a `routing`
La v1 no orquesta todavía `synthesis`.

### 3. Entrada QASM limitada a escenarios Qiskit-facing
La v1 ya acepta `qasm_file` para `Baseline` y `MO_Only`. Ese soporte no se extiende todavía a `RL_Only` ni `MO+RL` en la superficie pública actual.

### 4. Entrenamiento RL no implementado dentro de `integration`
`integration` ya puede orquestar una Train+Eval Campaign y disparar training por Campaign Case, pero no implementa el training RL internamente: ese comportamiento sigue perteneciendo a `rl_module` y se consume a través de `training_bridge.py`.

### 5. Fallback para metadata ausente en checkpoints antiguos
Cuando un checkpoint RL no trae `run_metadata.json`, `integration` sigue evaluándolo con legacy defaults. Ese fallback se reporta mediante una nota pública en `ScenarioResult.notes`, por ejemplo `Legacy RL evaluation defaults were used because no run metadata sidecar was found.`.

Esta compatibilidad también cubre la coexistencia entre checkpoints nuevos de masked routing y checkpoints legacy de `PPO`/`DQN`: `MaskablePPO` queda reservado como estándar para el régimen enmascarado nuevo, mientras que los modelos legacy siguen entrando por los contratos legacy/default o por flujos unmasked sin cambiar la ownership pública del módulo.

## Evolución Futura

Líneas naturales de evolución del módulo:

1. Soporte de `synthesis` como nuevo conjunto de escenarios.
2. Entrada de circuitos desde QASM para escenarios RL cuando exista una representación final de circuito consistente.
3. Reconstrucción/exportación de circuitos finales desde RL.
4. Comparabilidad homogénea entre escenarios Qiskit y RL.
5. Posible capa visual encima del runner y de los DTOs ya existentes.
