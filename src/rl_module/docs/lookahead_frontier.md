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

## Comportamiento de `reset()`

- Si se inyecta `initial_layout`, el entorno lo respeta exactamente.
- Si no se inyecta, el layout por defecto es determinista: `[0, 1, 2, ...]`.
- Ya no se remezcla aleatoriamente el layout durante `reset()`.

## Limitación actual

El modo `synthesis` sigue siendo un placeholder. La infraestructura de frontier
y observación se comparte con routing, pero la lógica de aplicar puertas,
comparar contra el objetivo y terminar episodios en synthesis todavía no está
implementada.
