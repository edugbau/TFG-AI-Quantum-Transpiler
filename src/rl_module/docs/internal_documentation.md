# Documentacion interna de `rl_module`

`rl_module` es el nucleo RL del proyecto. Modela el problema de transpilacion cuantica como un entorno Gymnasium y separa routing, masked routing, synthesis, entrenamiento, metadata de checkpoints y GUI.

## Mapa de lectura

1. `environment.py`, `env_strategies.py`, `frontier.py` y `rewards.py` explican como se formula el problema.
2. `agent.py`, `training.py` y `model_metadata.py` cubren entrenamiento y contratos de evaluacion.
3. `synthesis_primitives.py` y `synthesis_clifford.py` explican el modo `synthesis`.
4. `gui/` cubre la inspeccion de episodios.

## 1. Arquitectura general

### Piezas principales

| Archivo | Rol |
| --- | --- |
| `environment.py` | Entorno principal `QuantumTranspilationEnv` |
| `env_strategies.py` | Estrategias por modo (`RoutingStrategy`, `SynthesisStrategy`) |
| `frontier.py` | Frontera visible para routing (`SequentialFrontier`, `DagFrontier`) |
| `rewards.py` | Shaping de recompensas |
| `agent.py` | Wrapper sobre Stable-Baselines3 |
| `training.py` | Pipeline reproducible de entrenamiento |
| `model_metadata.py` | Lectura y escritura de `run_metadata.json` |
| `synthesis_primitives.py` | Catalogo de primitivas hardware-aware |
| `synthesis_clifford.py` | Logica de residual Clifford para `synthesis` v1 |
| `gui/` | Visualizacion e inspeccion de episodios |

### Boundary con `integration`

`rl_module` consume un `initial_layout` externo, pero no decide su origen. El handoff MO -> RL pertenece a `integration`, que tambien decide como leer `run_metadata.json` y como reconstruir episodios para comparacion.

## 2. Routing y masked routing

### Idea de modelado

Routing se formula como una secuencia de decisiones sobre swaps:

- el action space es discreto y fijo sobre las aristas del coupling map;
- `action_masks()` solo restringe candidatos validos en cada estado;
- la semantica no cambia por episodio, lo que mantiene estables los indices de accion.

### Observacion

La observacion combina layout, frontier y contexto temporal:

- `layout` y `_inverse_layout` permiten consultas en O(1);
- `lookahead` expone las puertas visibles;
- `lookahead_physical` muestra el efecto fisico bajo el layout actual;
- `lookahead_executable` y `lookahead_routing_distance` explican si una puerta ya puede ejecutarse;
- `step_progress` aporta contexto temporal para diferenciar estados repetidos.

### Frontier

- `SequentialFrontier` mantiene una lectura lineal del circuito.
- `DagFrontier` usa `front_layer()` para exponer paralelismo real.

### Recompensas

La recompensa separa progreso util de ruido:

- bonifica puertas ejecutadas;
- penaliza SWAPs inutiles, layouts repetidos y deshacer el ultimo SWAP;
- castiga truncaciones y acciones invalidas;
- en `synthesis`, la recompensa se centra en reducir el residual.

### Masked routing

El regimen nuevo de `masked routing` no redefine el entorno:

- `MaskablePPO` es el trainer estandar para checkpoints nuevos;
- `PPO` y `DQN` siguen soportados como legacy;
- la mascara es determinista, frontier-aware y compatible con la codificacion fija de acciones.

## 3. Synthesis v1

`synthesis` es un modo acotado y distinto de routing.

- trabaja sobre circuitos Clifford;
- necesita `basis_gates` ademas del coupling map;
- usa primitivas hardware-aware en vez de swaps;
- mantiene el layout fijo durante el episodio;
- el criterio de progreso es el residual Clifford, no una cola de puertas pendientes.

`synthesis_primitives.py` define el catalogo discreto de primitivas. `synthesis_clifford.py` construye y evalua el estado residual fisico.

## 4. Entrenamiento y metadata

### `agent.py`

Encapsula Stable-Baselines3 y permite cargar o entrenar politicas para routing y synthesis.

### `training.py`

Responsable de:

- fijar semillas;
- crear entornos de entrenamiento y evaluacion;
- envolver con `Monitor`;
- registrar callbacks;
- guardar modelos y artefactos.

### `model_metadata.py`

Es el contrato que conecta entrenamiento y evaluacion.

- `build_run_metadata()` crea el metadata que luego se persiste como `run_metadata.json`.
- `save_run_metadata()` guarda el sidecar.
- `load_run_metadata_for_model()` lo recupera cuando `integration` evalua un checkpoint.

Este metadata puede incluir versiones de masked routing para checkpoints nuevos. Si falta, `integration` cae a defaults legacy.

## 5. GUI e inspeccion

La GUI no es un adorno.

- `rl_gui.py` coordina la aplicacion.
- `routing_view.py` expone controles de routing.
- `synthesis_view.py` expone controles de synthesis.
- `evaluation_panel.py` permite inspeccionar pasos, recompensas, layouts y trazas.

La GUI sirve para explicar por que una politica se comporta como lo hace y para detectar oscilaciones, patrones repetitivos y progreso real.

## 6. Decisiones de diseno

- Se separa la semantica por estrategias para evitar duplicar el entorno.
- Se mantiene una representacion de layout dual para acceder en O(1).
- El action space permanece fijo y la restriccion de candidatos se aplica con mascaras.
- La compatibilidad hacia atras es explicita mediante metadata versionada y fallback legacy.
- `rl_module` no orquesta Campaigns ni produce layouts; ese rol es de `integration`.

## 7. Documentacion relacionada

- [../README.md](../README.md): entrada canonica del modulo.
- [lookahead_frontier.md](lookahead_frontier.md): detalle de observacion y frontier.
- [routing_stability_roadmap.md](routing_stability_roadmap.md): roadmap de estabilidad y masking.
- [synthesis_mode_status.md](synthesis_mode_status.md): estado del modo synthesis.
- [rl_module_explicacion_tfg.md](rl_module_explicacion_tfg.md): nota de defensa y apunte para la memoria.

