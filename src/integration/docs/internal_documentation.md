# Documentacion interna de `integration`

`integration` es la capa de orquestacion del proyecto. Su trabajo no es reimplementar MO o RL, sino conectar `qiskit_interface`, `mo_module` y `rl_module` para ejecutar Scenarios y Campaigns reproducibles.

## Mapa de lectura

1. `contracts.py` y `campaign_contracts.py` definen la superficie publica.
2. `scenarios.py`, `routing_evaluator.py`, `hybrid_layout_probe.py`, `rl_guided_mo.py` y `rl_model_contract.py` cubren la capa de Scenario y el handoff MO -> RL.
3. `training_bridge.py`, `campaign_runner.py` y `campaign_reporting.py` cubren la capa de Campaign.
4. `campaign_cli.py` y `runner.py` exponen la entrada de usuario.

## 1. Contratos publicos

### `contracts.py`

Define los contratos minimos de evaluacion:

- `CircuitSource` y `CircuitFormat` separan la procedencia del circuito.
- `LayoutSelectionPolicy` expresa como se elige un layout de MO.
- `ScenarioRequest` valida la peticion antes de ejecutar `Baseline`, `MO_Only`, `RL_Only` o `MO+RL`.
- `RoutingEpisodeSummary` resume una traza RL de routing.
- `ScenarioResult` unifica la salida publica de cualquier Scenario.

Puntos clave:

- `ScenarioRequest` acepta `library` y `qasm_file` para los escenarios Qiskit-facing.
- `RL_Only` y `MO+RL` requieren `rl_model_path`.
- `Baseline` no acepta layout externo ni parametros RL.
- `RoutingEpisodeSummary` guarda `swap_trace` y `executed_gate_trace`; `total_swaps` debe coincidir con `len(swap_trace)`.

### `campaign_contracts.py`

Modela Campaigns reproducibles:

- `CampaignCircuitSpec` describe cada familia de circuito.
- `CampaignCase` representa una combinacion `circuit x backend`.
- `CampaignCaseResult` guarda el estado de cada case.
- `CampaignSummary` agrega comparabilidad, incidencias y resumen de status.
- `CampaignConfig` concentra configuracion de RL, MO, layout policy y topologia.
- `Campaign` agrupa la configuracion y expande los cases.

Puntos clave:

- `CampaignConfig` distingue `default` y `advanced`.
- `mo_effort_mode` puede ser `auto` o `custom`.
- `topology_source` puede ser `backend` o `synthetic`.
- `SyntheticTopologySpec` solo tiene sentido en modo `advanced`.
- La forma sintetica `t` usa `num_qubits >= 5`, genera `synthetic_t_Nq` y construye una T balanceada con desempate hacia barra superior mas ancha.

## 2. Capa de Scenario

### `backend_adapter.py`

Convierte el backend de Qiskit en una estructura pequena y estable para `integration`. Es el unico lugar donde se centraliza la adaptacion de coupling map y basis gates.

### `layout_policy.py`

Toma un `OptimizationResult` de `mo_module` y lo reduce a un layout unico. La decision por defecto es `compromise`; la alternativa es `best_on_objective`.

### `rl_model_contract.py`

Resuelve el contrato de un checkpoint RL.

- Lee `run_metadata.json` cuando existe.
- Si el metadata trae versioned masked routing, se usa esa variante: `v1` preserva checkpoints historicos, `v2` aplica anti-undo, `v3` anade anti-ciclo, terminacion por estancamiento y top-k SABRE opcional con fallbacks no vacios, `v4` incorpora decay SABRE y `v5` conserva aristas preparatorias a un salto.
- Si no hay metadata, se cae a defaults legacy para no romper checkpoints PPO/DQN antiguos.

### `routing_evaluator.py`

Ejecuta un episodio RL de routing y, si el episodio completa, reconstruye el circuito ruteado.

- `evaluate_routing_episode()` devuelve `RoutingEpisodeSummary`.
- Su selector de accion opcional permite reutilizar el mismo loop con una politica determinista.
- `build_routed_circuit()` replays de forma prioritaria `executed_gate_trace` y usa `swap_trace` para materializar swaps.
- El resultado final se vuelve comparable con Qiskit cuando el routing termina.

### `hybrid_layout_probe.py`

Implementa el modo Campaign opt-in `hybrid_probe`.

- deduplica layouts del frente de Pareto conservando el primer indice;
- deriva el routing subgraph actual para cada candidato;
- ejecuta una sonda SABRE-like determinista que prioriza desbloqueos, minimiza el scoring de la semantica masked-routing actual y desempata por indice de accion;
- reconstruye cada solucion completa y calcula metricas post-routing con el backend real;
- selecciona layouts MO por `(CNOT-equivalent, depth, swaps)`;
- evalua el layout inicial de Qiskit como control diagnostico, excluido de la seleccion;
- cae explicitamente a `compromise` si ninguna sonda MO completa el routing;
- persiste `hybrid_layout_probe.json`.

### `rl_guided_mo.py`

Implementa el modo Campaign experimental `rl_guided`.

- carga el Training Artifact congelado de `RL_Only`;
- evalua cada layout unico con un episodio RL determinista y cacheado;
- reconstruye los episodios completos y calcula `(trans_depth, trans_cnot_equivalent)` mediante post-routing Qiskit;
- penaliza episodios incompletos o errores con `(1e12, 1e12)` y excluye esos candidatos del compromiso valido;
- falla antes del fine-tuning si ningun candidato completa routing;
- persiste `rl_guided_mo.json` con frente valido, intentos, cache, checkpoint fuente y controles para los layouts Qiskit, `MO_Only` y RL-guided.

### `scenarios.py`

Contiene los cuatro Scenario publicos:

- `run_baseline_scenario()`
- `run_mo_only_scenario()`
- `run_rl_only_scenario()`
- `run_mo_rl_scenario()`

Flujo resumido:

1. cargar circuito y backend;
2. resolver layout MO o layout inicial Qiskit;
3. cargar el modelo RL si aplica;
4. ejecutar el Scenario;
5. devolver `ScenarioResult`.

Limitaciones importantes:

- `Baseline` y `MO_Only` aceptan `qasm_file`.
- Los escenarios RL siguen centrados en trazas y no exponen una entrada QASM publica equivalente.
- `RL_Only` reutiliza por defecto el layout inicial resuelto por `qiskit_level_1`; un layout inyectado explicitamente conserva prioridad.
- `MO+RL` reconstruye el circuito y ejecuta post-routing de Qiskit solo cuando el episodio completa.

## 3. Capa de Campaign

### `training_bridge.py`

Es el seam entre Campaign e `src.rl_module.training`.

- recibi el `CampaignCase` y la configuracion del Campaign;
- convierte la configuracion a hyperparams de entrenamiento;
- pasa el layout seleccionado por MO como `initial_layout`;
- selecciona `best_model.zip` con la tupla post-routing `(cnot_equivalent, depth, swaps)` y cae a `final_model.zip` si aun no existe una solucion valida.
- tolera episodios incompletos durante la busqueda inicial sin consumir paciencia de early stopping.
- puede cargar `initial_model_path` sobre un entorno nuevo y continuar el checkpoint completo con un presupuesto local adicional.
- tras la primera solucion, escala la paciencia post-routing al maximo entre 20 evaluaciones y el 20% del presupuesto solicitado para no cortar demasiado pronto campañas largas.

### `campaign_runner.py`

Orquesta la Campaign caso a caso.

- ejecuta `Baseline`, `MO_Only`, training RL y `MO+RL`;
- persiste el estado despues de cada case;
- mantiene la separacion entre casos comparables y no comparables;
- usa el layout exacto seleccionado por `MO_Only` como base del camino hibrido;
- deriva un path-expanded routing subgraph cuando es posible y cae al coupling map completo si la derivacion falla.

Tambien expone el runner interno agrupado por seed usado por Campaign matrix. Para cada case, este runner:

- ejecuta una sola vez `Baseline`, `RL_Only` y la optimizacion MO;
- selecciona los layouts de todos los modos sobre el mismo frente de Pareto;
- deduplica layouts fisicos equivalentes;
- pasa cada layout seleccionado como `initial_layout` al training hibrido y como `injected_layout` a `MO+RL`.

Cuando una hija usa `hybrid_probe`, el layout elegido por la sonda tambien se inyecta en `MO_Only`, training hibrido y `MO+RL`. El modo requiere `MaskablePPO`. El alias batch `all` conserva exclusivamente los tres modos historicos: `compromise`, `best_depth` y `best_cnot_count`.

Cuando una hija aislada usa `rl_guided`, el runner conserva el flujo `MO_Only` Qiskit como control, ejecuta la segunda busqueda NSGA-II con el checkpoint congelado de `RL_Only` y continua entrenamiento desde ese artefacto. El modo solo admite topologias sinteticas y mantiene el coupling map completo identico en pretraining, busqueda, fine-tuning y evaluacion final. `rl.total_timesteps` y `rl.finetune_timesteps` son presupuestos separados.

Los Scenarios aislados mantienen su fallback. Dentro del runner agrupado, `MO+RL` siempre recibe un layout inyectado y no vuelve a ejecutar MO.

### `campaign_reporting.py`

Construye la salida publica de la Campaign.

- `summary.md` como Summary Document;
- `campaign.json` como salida estructurada;
- `cases/<case>/result.json` para cada case.
- `cases/<case>/hybrid_layout_probe.json` para cases ejecutados con `hybrid_probe`.
- `cases/<case>/rl_guided_mo.json` para cases ejecutados con `rl_guided`.

En Campaign matrix, la sonda se persiste en el directorio compartido de la seed:
`runs/<campaign_id>__seed_<seed>__shared/cases/<case>/hybrid_layout_probe.json`.

Tambien agrega metricas comparables, incidencias y notas del training. Un Campaign o un case puede terminar como `completed` y aun asi no ser completamente comparable; eso se refleja en los agregados y en los incidents.

### `campaign_cli.py`

Expone la CLI guiada y el modo batch.

- `build_default_campaign_config()` construye la configuracion canonica.
- `run_interactive_campaign_cli()` maneja la experiencia guiada.
- `load_campaign_batch()` lee casos desde JSON.
- `run_campaign_batch()` ejecuta la lista de Campaigns.
- `run_campaign_cli_from_args()` y `main()` actuan como entrada publica.

La CLI actual expone una superficie controlada: `fake_torino` y `fake_brisbane`. La capa de Campaign tambien soporta topologias sinteticas en modo avanzado.
En batch JSON, `topology_source: "synthetic"` acepta `synthetic_topology` y puede derivar `backend_names` automaticamente; por ejemplo `{"shape": "t", "num_qubits": 11}` produce `synthetic_t_11q`.
Para routing `MaskablePPO`, el bloque `rl` tambien acepta `cycle_window`,
`stagnation_patience` y `sabre_top_k`. La paciencia puede omitirse para usar
el default adaptativo `max(8, 2 * num_qubits)`; el valor resuelto queda
persistido junto al checkpoint versionado.
La CLI guiada avanzada pregunta `SABRE top-k (blank to disable)` al seleccionar
`MaskablePPO`: un entero positivo activa la poda y Enter, `none`, `null` u
`off` la dejan desactivada.
El modo batch `mo.selection_modes: ["rl_guided"]` requiere ademas
`rl.finetune_timesteps`, `mode: "advanced"` y `topology_source: "synthetic"`.

### `runner.py`

Mantiene una CLI fina para scenarios individuales. Es util cuando se quiere evaluar un caso sin pasar por la experiencia completa de Campaign.

## 4. Decisiones de diseno

- `integration` es una capa de orquestacion, no de computo.
- El handoff MO -> RL pertenece aqui y no a `mo_module` ni a `rl_module`.
- La comparabilidad se expresa de forma explicita en el Summary Document.
- El contrato de `run_metadata.json` protege compatibilidad hacia atras.
- El grafo de routing path-expanded es un detalle de Campaign, no un cambio del contrato publico de Scenario.

## 5. Relacion con los otros modulos

- `qiskit_interface` aporta circuitos, backends y baselines.
- `mo_module` aporta layouts candidatos y seleccion de layout.
- `rl_module` aporta el entorno, el agente y el entrenamiento.
- `integration` conecta todo y publica los resultados.
