# rl_module

`rl_module` implementa el entorno Gymnasium y el entrenamiento por refuerzo que resuelven routing y un primer modo de synthesis cuantica. Este modulo consume un `initial_layout` externo y no se encarga de generarlo; ese handoff pertenece a `integration`.

## Estructura publica

| Archivo | Proposito | Simbolos clave |
| --- | --- | --- |
| `environment.py` | Entorno principal de transpilacion cuantica | `QuantumTranspilationEnv` |
| `env_strategies.py` | Estrategias de observacion/accion por modo | `RoutingStrategy`, `SynthesisStrategy` |
| `frontier.py` | Frontera visible del circuito para routing | `SequentialFrontier`, `DagFrontier`, `LookaheadEntry` |
| `rewards.py` | Shaping de recompensas para routing y synthesis | `RoutingReward`, `SynthesisReward` |
| `agent.py` | Wrapper sobre Stable-Baselines3 | `QuantumRLAgent` |
| `training.py` | Pipeline de entrenamiento reproducible | `setup_training_pipeline`, `set_global_seeds` |
| `model_metadata.py` | Metadata de checkpoints y contratos de evaluacion | `build_run_metadata`, `save_run_metadata`, `load_run_metadata_for_model` |
| `synthesis_primitives.py` | Catalogo hardware-aware de primitivas | helpers de primitives |
| `synthesis_clifford.py` | Logica del modo synthesis v1 | estados y residual Clifford |
| `gui/` | GUI de inspeccion y evaluacion | `RLBenchmarkGUI` |

## Flujo general

1. `integration` construye el caso y pasa `initial_layout`, circuito y backend.
2. `environment.py` crea el episodio y delega semantica a una estrategia concreta.
3. `frontier.py` decide que puertas son visibles en routing.
4. `rewards.py` premia el progreso util y penaliza acciones malas o repetitivas.
5. `agent.py` entrena o evalua una politica SB3.
6. `training.py` guarda el modelo y `model_metadata.py` escribe `run_metadata.json`.
7. `integration` lee ese metadata para evaluar checkpoints nuevos o legacy.

## Routing y masked routing

El routing usa un action space discreto fijo sobre las aristas del coupling map. El nuevo regimen de `masked routing` no cambia ese catalogo de acciones: `action_masks()` solo aplica una hard mask determinista y frontier-aware para restringir que swaps pueden muestrearse en cada estado.

Puntos clave:

- `MaskablePPO` es el entrenador estandar para checkpoints nuevos de routing enmascarado.
- `frontier_restricted_edges.v4` conserva los filtros v3 y anade decay SABRE para penalizar reutilizacion serial de qubits fisicos.
- Cada filtro conserva la mascara anterior como fallback si la heuristica eliminaria todos los candidatos.
- Los checkpoints `frontier_restricted_edges.v1`, `frontier_restricted_edges.v2` y `frontier_restricted_edges.v3` mantienen su semantica historica al evaluarse.
- Los checkpoints legacy `PPO` y `DQN` siguen soportados mediante contratos legacy/default o evaluaciones unmasked.
- `DagFrontier` usa `front_layer()` para exponer paralelismo real; `SequentialFrontier` mantiene el comportamiento secuencial.
- `initial_layout` se respeta exactamente cuando llega desde fuera del modulo.

## Synthesis v1

`synthesis` es un modo acotado y entrenable:

- trabaja sobre circuitos Clifford;
- requiere `basis_gates` ademas del coupling map;
- usa un estado residual fisico como observacion principal;
- mantiene el layout fijo durante el episodio;
- no reutiliza la semantica de "cola de puertas pendientes" propia de routing.

## GUI e inspeccion

La GUI no es solo una demostracion: sirve para inspeccionar episodios, ver frontier, swaps, recompensas y progreso del residual. Las vistas de routing y synthesis comparten la misma app, pero muestran controles diferentes segun el modo.

## Decisiones de diseno

- Separacion por estrategias para no duplicar el entorno.
- Layout dual en O(1) con `current_layout` y `_inverse_layout`.
- Recompensa separada de la mecanica del entorno.
- Metadata versionada para mantener compatibilidad hacia atras.
- La orquestacion de Campaigns y el handoff MO -> RL no vive aqui.

## Documentacion relacionada

- [docs/internal_documentation.md](docs/internal_documentation.md): deep-dive tecnico del modulo.
- [docs/lookahead_frontier.md](docs/lookahead_frontier.md): observacion y frontera en routing.
- [docs/routing_stability_roadmap.md](docs/routing_stability_roadmap.md): roadmap de estabilidad del routing.
- [docs/synthesis_mode_status.md](docs/synthesis_mode_status.md): estado del modo synthesis.
- [docs/rl_module_explicacion_tfg.md](docs/rl_module_explicacion_tfg.md): nota de defensa y apunte de memoria.
