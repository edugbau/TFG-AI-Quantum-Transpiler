# Lookahead y Frontier en `rl_module`

## Objetivo

El entorno de RL necesita que el efecto de un `SWAP` sea visible de forma
explícita para el agente. Antes, la observación solo exponía el layout global y
una ventana lógica de puertas pendientes. Eso obligaba al modelo a inferir por
sí mismo cómo cambiaban las puertas futuras tras cada `SWAP`, lo que hacía el
aprendizaje más difícil y favorecía oscilaciones.

La solución actual separa dos conceptos:

- **Layout**: mapeo `logical -> physical`
- **Frontier**: conjunto de puertas visibles que el agente debe considerar en el
  estado actual

Sobre esa base, el routing enmascarado introduce una restricción de candidatos al estilo SABRE sin alterar el action space público: el espacio discreto sigue siendo fijo sobre todas las aristas del coupling map, y `action_masks()` aplica una hard mask determinista y frontier-aware para desactivar swaps que no deben muestrearse en el estado actual.

## Tensores de observación

El entorno expone ahora los siguientes campos en la observación:

- `layout`
  - Array `int32[num_qubits]`
  - Cada posición es un qubit lógico y cada valor el qubit físico donde reside.

- `lookahead`
  - Array `int32[lookahead_window * 2]`
  - Codifica pares lógicos `(logical_q1, logical_q2)` de las puertas visibles.
  - Las puertas de 1 qubit se representan como `(q, q)`.

- `lookahead_physical`
  - Array `int32[lookahead_window * 2]`
  - Proyección física de `lookahead` bajo el layout actual.
  - Hace explícito qué par físico queda afectado tras cada `SWAP`.

- `lookahead_executable`
  - Array `float32[lookahead_window]`
  - `1.0` si la puerta visible puede ejecutarse con el layout actual.
  - `0.0` si aún requiere routing.

- `lookahead_routing_distance`
  - Array `float32[lookahead_window]`
  - Distancia de routing aproximada medida como `shortest_path_length - 1`.
  - Vale `0.0` si la puerta ya es ejecutable.
  - Para slots vacíos se mantiene en `0.0` y se usan junto con la máscara.

- `lookahead_valid_mask`
  - Array `float32[lookahead_window]`
  - `1.0` para posiciones ocupadas por puertas reales.
  - `0.0` para padding.

- `step_progress`
  - Escalar normalizado `float32[1]`
  - Indica progreso temporal del episodio.

## Por qué esto arregla la observabilidad del `SWAP`

Con esta representación, el agente ya no depende de inferir indirectamente el
efecto del `SWAP` a partir de `layout + lookahead lógico`.

Ahora, tras aplicar un `SWAP`:

- puede cambiar `lookahead_physical` aunque `lookahead` lógico siga igual,
- puede cambiar `lookahead_executable`,
- puede bajar `lookahead_routing_distance`.

Eso convierte el efecto del routing en una señal observacional directa.

## `frontier_mode`

El entorno soporta dos modos de frontera:

### `frontier_mode="sequential"`

- Usa la lista secuencial de instrucciones del circuito.
- La frontera visible son las primeras puertas pendientes.
- Mantiene compatibilidad con el comportamiento inicial del módulo.

### `frontier_mode="dag"`

- Construye un `DAGCircuit` con Qiskit.
- La frontera visible se basa en `dag.front_layer()`.
- Permite exponer paralelismo real entre puertas independientes.
- La ejecución puede consumir varias puertas de la front layer a la vez.

## Relación con `masked routing`

La máscara de acciones reutiliza exactamente esta información de frontera. No construye un catálogo variable de acciones ni reasigna índices por episodio; solo decide qué aristas del coupling map permanecen habilitadas en cada paso.

Esto hace que el régimen de **masked routing** sea compatible con la idea clásica de SABRE-style candidate restriction: restringir swaps candidatos con una heurística determinista basada en la frontera visible, manteniendo estable la codificación de acciones del entorno.

Para checkpoints nuevos de este régimen, `MaskablePPO` es el trainer estándar. Los checkpoints legacy de `PPO` y `DQN` siguen existiendo fuera de este contrato enmascarado y se evalúan con los contratos legacy/default o unmasked correspondientes.

`frontier_restricted_edges.v3` conserva este action space fijo y aplica filtros
acumulativos con fallback: anti-undo, anti-ciclo sobre
`(layout, frontier_revision)` y top-k SABRE opcional. La configuracion efectiva
se persiste en el sidecar del checkpoint para reproducir exactamente la
evaluacion. La version v3 tambien puede truncar episodios estancados cuando no
se ejecutan puertas ni se alcanza una nueva mejor distancia durante la
paciencia configurada.

## Comportamiento de `reset()`

- Si se inyecta `initial_layout`, el entorno lo respeta exactamente.
- El productor del `initial_layout` es externo al módulo; el handoff MO -> RL pertenecerá a `src/integration/`.
- Si no se inyecta, el layout por defecto es determinista: `[0, 1, 2, ...]`.
- Ya no se remezcla aleatoriamente el layout durante `reset()`.

## Limitación actual

`lookahead` y `frontier` siguen siendo una abstracción pensada para routing.
El modo `synthesis` ya está implementado en `environment.py` y
`env_strategies.py`, con semántica residual-céntrica propia y una condición de
terminación basada en que el residual Clifford llegue a la identidad. No
reutiliza la misma lectura de "cola de puertas pendientes" que routing.
