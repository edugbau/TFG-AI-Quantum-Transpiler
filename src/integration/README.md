# integration

`integration` is the orchestration layer that connects `qiskit_interface`, `mo_module` and `rl_module`. It owns Scenario execution, MO -> RL handoff, Campaign execution and the public persistence/reporting of those runs.

## Public surface

| Archivo | Proposito | Simbolos clave |
| --- | --- | --- |
| `contracts.py` | Contratos de Scenario y resultado de routing | `CircuitSource`, `CircuitFormat`, `LayoutSelectionPolicy`, `ScenarioRequest`, `RoutingEpisodeSummary`, `ScenarioResult` |
| `campaign_contracts.py` | Contratos de Campaign | `CampaignCircuitSpec`, `CampaignCase`, `CampaignCaseResult`, `CampaignSummary`, `CampaignConfig`, `Campaign` |
| `backend_adapter.py` | Adaptacion de backends de Qiskit a una estructura pequena y estable | `resolve_backend_bundle`, `BackendBundle` |
| `layout_policy.py` | Seleccion de un layout unico a partir de MO | `select_layout_from_mo_result` |
| `routing_evaluator.py` | Evaluacion y reconstruccion de episodios RL de routing | `evaluate_routing_episode`, `build_routed_circuit` |
| `routing_subgraph.py` | Derivacion del grafo de routing para Campaign | `build_path_expanded_subgraph` |
| `rl_model_contract.py` | Contrato de metadata de checkpoints y fallback legacy | `resolve_routing_model_contract`, `RoutingModelContract` |
| `training_bridge.py` | Seam entre Campaign e `rl_module.training` | `train_case`, `TrainingBridgeResult` |
| `scenarios.py` | Orquestacion de `Baseline`, `MO_Only`, `RL_Only` y `MO+RL` | `run_baseline_scenario`, `run_mo_only_scenario`, `run_rl_only_scenario`, `run_mo_rl_scenario` |
| `campaign_runner.py` | Train+eval Campaign por case | `run_campaign` |
| `campaign_reporting.py` | Summary Document y persistencia publica | `build_campaign_report`, `render_campaign_summary_markdown`, `write_campaign_outputs` |
| `campaign_cli.py` | CLI guiada y batch de Campaigns | `build_default_campaign_config`, `load_campaign_batch`, `run_campaign_batch`, `run_interactive_campaign_cli`, `run_campaign_cli_from_args`, `main` |
| `runner.py` | CLI fina de escenarios | entrada publica ligera |

## Scenario layer

La capa de Scenario cubre cuatro rutas:

- `Baseline`
- `MO_Only`
- `RL_Only`
- `MO+RL`

`Baseline` y `MO_Only` aceptan entrada `qasm_file`. Los escenarios basados en RL se centran en circuitos y trazas, no en una entrada QASM publica equivalente.

`RL_Only` y `MO+RL` reconstruyen el circuito ruteado cuando el episodio completa. Para ello, `routing_evaluator.py` prioriza `executed_gate_trace` y usa `swap_trace` para materializar swaps fisicos. Si el episodio no completa, devuelven un resultado controlado sin post-routing final.

## Campaign layer

`integration` tambien orquesta Train+Eval Campaigns reproducibles. Cada `Campaign Case` es una combinacion `circuit x backend`, y la comparacion canonica dentro de una Campaign es `Baseline`, `MO_Only`, `RL_Only` y `MO+RL`.

Dentro de esa secuencia:

1. `MO_Only` selecciona el layout.
2. `training_bridge.py` lo pasa a RL como `initial_layout`.
3. `campaign_runner.py` entrena el modelo del case.
4. `MO+RL` evalua el mismo layout y el Training Artifact resultante.
5. Si es posible, Campaign deriva un path-expanded routing subgraph; si no, cae al coupling map completo y deja nota del fallback.

## Contracts and metadata

- `ScenarioRequest`, `RoutingEpisodeSummary` y `ScenarioResult` validan el contrato publico de evaluacion.
- `CampaignConfig` distingue `default` y `advanced`, y tambien el modo `mo_effort_mode` (`auto` o `custom`).
- `SyntheticTopologySpec` permite usar topologias sinteticas en modo avanzado.
- `resolve_routing_model_contract()` lee `run_metadata.json` cuando existe y mantiene fallback legacy para checkpoints antiguos.
- La metadata versionada de masked routing se consume cuando esta disponible; si no, se mantiene la compatibilidad con modelos PPO/DQN legacy.

## CLI y persistencia

La guided CLI ofrece:

- un camino **Default Campaign** con valores canonicos compartidos;
- un camino **Advanced Campaign** con seleccion explicita de backends, RL, MO y topologia;
- ejecucion batch mediante `load_campaign_batch()` y `run_campaign_batch()`.

La superficie guiada actual expone `fake_torino` y `fake_brisbane` como backends visibles para la Campaign CLI.

Cada Campaign persiste:

- `summary.md` como Summary Document;
- `campaign.json` como salida estructurada;
- `cases/<case>/result.json` para cada caso.

El Summary Document deja clara la comparabilidad real de los casos. Un case puede terminar como `completed` y aun asi no ser comparable si falta un bundle completo de metricas.

## Decisiones de diseno

- `integration` es una capa de orquestacion, no una reimplementacion de MO o RL.
- La comparabilidad se expresa de forma explicita en el Summary Document e incidents.
- El backend catalogo publico es reducido para mantener reproducibilidad y evitar credenciales.
- El grafo de routing derivado es un detalle de Campaign; no cambia la semantica publica de `ScenarioRequest`.

## Lectura recomendada

1. Empieza por `contracts.py` y `campaign_contracts.py`.
2. Sigue con `scenarios.py`, `routing_evaluator.py` y `rl_model_contract.py`.
3. Termina con `campaign_runner.py`, `campaign_reporting.py` y `campaign_cli.py` para entender Campaign y persistencia.

## Documentacion de apoyo

- [docs/internal_documentation.md](docs/internal_documentation.md)
