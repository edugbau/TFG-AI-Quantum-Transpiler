# Documentación Interna: Módulo de Aprendizaje por Refuerzo (RL)

## Visión General
El `rl_module` se encarga de la fase de **Enrutamiento (Routing) y Síntesis** de circuitos cuánticos. Tras recibir un mapeo estático inicial (`initial_layout`) como input externo, el agente de RL se encarga de insertar puertas SWAP de manera dinámica para resolver los bloqueos de conectividad o transicionar a nuevas puertas. El handoff MO -> RL, cuando aplique, pertenece a `src/integration/`.

Para ofrecer una escalabilidad completa, la arquitectura del entorno ha sido diseñada aplicando el **Patrón Strategy**, separando la definición del problema en estrategias. Esto permite modificar si el agente actúa en modo "solo enrutamiento" (*Routing*) o si debe generar secuencias de puertas completas (*Synthesis*).

## Estructura de Proyecto
```
src/rl_module/
├── environment.py       # Entorno Gymnasium (QuantumTranspilationEnv) optimizado con búsquedas O(1) en Layout.
├── env_strategies.py    # Patrón Strategy que expone modos: RoutingStrategy, SynthesisStrategy.
├── frontier.py          # Proveedores de frontera (secuencial y DAG) para routing.
├── rewards.py           # Sistema desacoplado de heurísticas (RoutingReward, SynthesisReward).
├── agent.py             # Wrapper (QuantumRLAgent) orquestando Stable-Baselines3 (PPO / DQN) y soporte GPU/CUDA.
├── training.py          # Script de abstracción y pipeline de entrenamiento (logs en TensorBoard y Checkpoint callbacks).
├── gui/                 # GUI única con vistas especializadas por modo y panel de inspección de episodios.
└── docs/                # Documentación interna (como este archivo).
```

## GUI RL: una app, dos vistas especializadas

La interfaz `rl_gui.py` ya no debe entenderse como dos aplicaciones separadas. La GUI es una sola app (`RLBenchmarkGUI`) con configuración compartida y dos vistas especializadas:

- **Vista de routing**: expone controles específicos de frontera (`frontier_mode="sequential"` o `"dag"`) y métricas ligadas a lookahead, layout dinámico y SWAPs.
- **Vista de synthesis**: oculta los controles exclusivos de routing y centra la evaluación en primitivas hardware-aware y en el progreso del residual.

Ambas vistas comparten el mismo flujo general de entrenamiento/evaluación, pero cada una muestra únicamente los controles y los detalles de episodio que tienen significado semántico para su modo.

## Arquitectura de Escalabilidad (Patrón Strategy)

El entorno `QuantumTranspilationEnv` no impone directamente los espacios de observación ni de acción. Delega esta responsabilidad a las clases que heredan de `RLEnvStrategy`. Además, implementa en *O(1)* la búsqueda del mapeo cuántico utilizando arrays cruzados (`current_layout` y `_inverse_layout`).

La selección de puertas visibles y su ejecución ya no dependen de una única cola rígida. El entorno delega esa responsabilidad en proveedores de frontera:

- `SequentialFrontier`: usa la secuencia lineal de instrucciones.
- `DagFrontier`: usa `qiskit.converters.circuit_to_dag(...).front_layer()` para exponer paralelismo real.

### 1. Modo Enrutamiento (`mode="routing"`)
- **Action Space:** Discreto (`gym.spaces.Discrete`). El tamaño equivale al número de aristas bidireccionales del Coupling Map (sin duplicados). El agente inserta un SWAP y el layout dinámico se invierte.
- **Observation Space:** Diccionario con:
  - `layout`: Array del mapeo lógico→físico actual (tamaño `num_qubits`).
  - `lookahead`: Buffer vectorial lógico de tamaño fijo ($N \times 2$) sobre la frontera visible.
  - `lookahead_physical`: Proyección física de las mismas puertas bajo el layout actual.
  - `lookahead_executable`: Marca binaria de ejecutabilidad inmediata.
  - `lookahead_routing_distance`: Distancia de routing aproximada (`shortest_path_length - 1`).
  - `lookahead_valid_mask`: Máscara binaria para distinguir puertas reales de padding.
  - `step_progress`: Escalar normalizado $\in [0, 1]$ que indica `current_step / max_steps`. Proporciona **contexto temporal** al agente para distinguir estados idénticos visitados en momentos distintos del episodio, rompiendo oscilaciones cíclicas A→B→A.
- **Lógica de Ejecución:** El entorno busca qué puertas quedan desbloqueadas tras aplicar el SWAP y ejecuta repetitivamente (en cascada) sus dependencias.

### `frontier_mode`

El entorno puede operar con dos modos de frontera:

- `frontier_mode="sequential"`: usa la cola secuencial de instrucciones.
- `frontier_mode="dag"`: usa una `front_layer` real del DAG y hace visibles varias puertas independientes en paralelo.

Esto mejora la observabilidad del efecto del `SWAP`: el agente ya no ve solo el par lógico futuro, sino también su proyección física, su ejecutabilidad actual y su distancia de routing.

### Semántica del inspector de episodios en routing

Durante la evaluación interactiva, el `EpisodeInspectorPanel` registra un `EvaluationStepRecord` por paso y muestra metadatos propios de routing:

- **`layout_before` / `layout_after`**: estado del mapeo lógico->físico antes y después de la acción.
- **`visible_frontier_before`**: snapshot de la frontera visible antes del paso. Cada entrada incluye puerta, qubits lógicos, qubits físicos proyectados y si la puerta ya era ejecutable.
- **`swap_edge`**: arista física seleccionada por la acción discreta cuando el agente inserta un SWAP.
- **`executed_gates`**: puertas que el entorno pudo ejecutar en cascada tras actualizar el layout.
- **`routing_progress_delta`**: progreso neto de routing inducido por la acción.
- **`repeated_layout` / `undo_swap`**: señales de diagnóstico para detectar oscilaciones o deshacer el SWAP inmediatamente anterior.

La lectura correcta del inspector en routing es: primero se observa la frontera visible bajo el layout actual, luego la acción modifica el layout físico mediante un SWAP y, si eso desbloquea dependencias, el entorno ejecuta automáticamente las puertas habilitadas. Por eso el panel muestra frontera, layout, SWAP y puertas ejecutadas en ese orden lógico.

### 2. Modo Síntesis (`mode="synthesis"`)
- **Estado actual**: primer modo entrenable restringido a circuitos Clifford.
- **Conciencia de hardware**: requiere `coupling_map` y `basis_gates`; la topología sola no determina la puerta nativa de 2 qubits.
- **Espacio de acción**: `Discrete(N)` sobre un catálogo fijo de primitivas Clifford hardware-aware.
- **Criterio de éxito**: equivalencia Clifford por residual identidad en espacio físico.
- **Limitación actual**: el layout es fijo durante el episodio; no hay `swap` dinámico en synthesis v1.

### Semántica del inspector de episodios en synthesis

En `synthesis`, el inspector reutiliza el mismo panel pero cambia completamente la interpretación del paso:

- **`primitive_name`**: nombre de la primitiva aplicada desde el catálogo hardware-aware.
- **`primitive_physical_qargs`**: qubits físicos sobre los que se intenta aplicar la primitiva.
- **`primitive_cost`**: coste asociado a la primitiva elegida.
- **`residual_distance_before` / `residual_distance_after`**: distancia del residual antes y después del paso.
- **`residual_distance_delta`**: mejora neta del residual causada por la acción.

La visualización de `synthesis` es deliberadamente **residual-céntrica**. El panel no debe interpretarse como una cola de "puertas restantes" al estilo routing. El episodio progresa reduciendo el residual Clifford hacia la identidad física, y cada paso se describe por la primitiva aplicada y por cuánto reduce (o no) esa distancia residual.

## Sistema de Recompensas (`rewards.py`)

Extraído con el patrón *RewardStrategy*. Permite modificar heurísticas de recompensa sin alterar la mecánica de Gymnasium. Para `RoutingReward`:
- **SWAP aplicado:** Penalización ligera (ej. -1.0) incentivando caminos críticos cortos y menos puertas añadidas.
- **Puertas ejecutadas:** Fuerte bonificación (ej. +10.0) fomentando progreso.
- **Acción Inválida:** Penalización media (ej. -5.0) para prevenir movimientos fuera de rango o mal uso del coupling map.
- **Truncación (`max_steps` expirado):** Fuerte penalización en los límites temporales (ej. -20.0).
- **Circuito Completado:** Bonificación masiva a final de episodio (+50.0).

## Entrenamiento y Agente (SB3)
Se utilizan los wrappers del archivo `agent.py` y `training.py` basados en `stable_baselines3`.
El método `setup_training_pipeline()` permite acoplar automáticamente semillas de reproducibilidad (`set_global_seeds`), encapsular el entorno Gymnasium para reportes (`Monitor`), e inyectar `Dict` Policies a través del motor PyTorch que correrá predefinido en CUDA. Se levantan periódicamente *TensorBoard Logs* y *Checkpoint Calbacks* como sistema de salvado.
En `mode="synthesis"`, el pipeline también debe recibir `basis_gates` explícitas para construir el catálogo de primitivas tanto en entrenamiento como en evaluación.

Para DQN, se inyectan automáticamente hiperparámetros estabilizadores: `exploration_fraction=0.5` (mayor exploración durante el entrenamiento), `tau=0.05` (soft target updates para suavizar el aprendizaje) y `learning_starts=1000`.

## Gestión de Layouts Físicos vs. Lógicos

Cuando el circuito tiene menos qubits lógicos que el hardware físico (ej. Bell State de 2 qubits en un anillo de 5), los arrays se dimensionan correctamente:
- `current_layout` tiene tamaño `num_qubits` (lógicos).
- `_inverse_layout` tiene tamaño `num_physical_qubits` (hardware), con `-1` para posiciones vacías.
- SWAPs entre dos posiciones físicas **ambas vacías** (`lq1 == -1 and lq2 == -1`) se marcan como acción inválida y reciben penalización.

## Problema Conocido: Oscilación en Evaluación

### Síntoma
Durante el entrenamiento, la media móvil de recompensa sube y la longitud de episodio baja, indicando que el agente **aprende correctamente**. Sin embargo, al evaluar el modelo entrenado, el agente entra en un bucle infinito de oscilaciones (ej. Layout A → B → A → B...) sin resolver nunca el circuito.

### Causa Raíz
Durante el entrenamiento, PPO utiliza una **política estocástica**: muestrea acciones de su distribución de probabilidades, lo que rompe ciclos naturalmente por azar. Sin embargo, durante la evaluación (tanto con `deterministic=True` como `deterministic=False`), el agente tiende a repetir las mismas decisiones porque:

1. **Sin memoria explícita:** La red neuronal (MLP) no tiene estado recurrente. Si observa el mismo layout dos veces, emite la misma distribución de acciones. Aunque se añadió un escalar `step_progress` (progreso temporal normalizado `current_step / max_steps`) a la observación, este valor cambia muy lentamente en episodios largos (200 steps) y no es suficiente para diferenciar estados de forma efectiva.

2. **Política no convergida:** En entornos con espacio de estados pequeño (ej. Bell State en Ring-5 se resuelve en 1 SWAP), la red no tiene suficiente señal de gradiente para distinguir cuál de las 5 aristas es la correcta para **cada** layout inicial posible.

3. **Diferencia train vs eval:** El ruido estocástico del entrenamiento actúa como exploración implícita y enmascara el problema. En evaluación pura, el ruido desaparece y los ciclos emergen.

### Mitigaciones Implementadas
- **`step_progress` en observación** (`env_strategies.py`): Escalar $\in [0, 1]$ que da contexto temporal al agente. Ayuda parcialmente pero no resuelve el problema por completo.
- **Lookahead enriquecido**: `lookahead_physical`, `lookahead_executable`, `lookahead_routing_distance` y `lookahead_valid_mask` hacen explícito el efecto de cada `SWAP` sobre la frontera observable.
- **`frontier_mode="dag"`**: permite representar correctamente puertas paralelas en circuitos con dependencias no lineales.
- **Detección visual de ciclos en GUI** (`rl_gui.py`): durante la evaluación interactiva de **routing**, la GUI detecta si un layout se visita 3 veces y corta el episodio mostrando `"CICLO DETECTADO ⚠"`. Esto **no afecta** al entorno de entrenamiento y no se aplica a `synthesis`, donde el layout puede permanecer fijo por diseño y el progreso se mide sobre el residual, no sobre cambios de layout.
- **Penalización de SWAPs vacíos** (`environment.py`): Los SWAPs entre dos posiciones físicas sin qubit lógico se marcan como inválidos (-5.0 de penalización).
- **Shaping por repetición en routing** (`rewards.py` + `environment.py`): el entorno expone señales `repeated_layout`, `undo_swap` y `routing_progress_delta`, y `RoutingReward` ya penaliza repeticiones/undo-swap y bonifica progreso neto de routing.

### Posibles Soluciones Futuras (No Implementadas)
1. **Redes recurrentes (LSTM):** Usar `RecurrentPPO` de `sb3-contrib` para dar al agente memoria real de acciones pasadas.
2. **State hashing en la observación:** Incluir un hash de los últimos N layouts visitados como parte de la observación.
3. **Memoria explícita o contexto histórico richer:** Añadir contexto de historial sin depender solo de `step_progress`, por ejemplo mediante observaciones recurrentes o features de trayectoria más expresivas.
4. **Mayor entrenamiento y `max_steps` ajustado:** Con circuitos más complejos y más timesteps, la política podría converger lo suficiente para no oscilar.

---

## Elementos a Consultar (Preguntas para los tutores del TFG)

En esta recta final del desarrollo de `rl_module`, se enumeran las dudas y consideraciones que deberíamos evaluar y debatir frente a los tutores para encauzar correctamente la memoria y pruebas:

1. **Representación Estricta de DAG:**
   Actualmente, el mecanismo *Lookahead* del entorno extrae las siguientes puertas iterando la serie algorítmica de Qiskit por motivos de velocidad en el entrenamiento masivo. Sin embargo, para extraer el verdadero *paralelismo*, lo óptimo en la literatura ha sido utilizar el generador `DAGCircuit.front_layer()`.
   ¿Deberíamos reestructurar el Lookahead para ser puramente por DAG y asumir esa pérdida de rendimiento/velocidad del pipeline de entrenamiento, o se documenta nuestra heurística y por qué es funcional con esta topología?

2. **Sistema y Ajuste de Recompensas (`rewards.py`):**
   ¿Existen valores de pesos o penalizaciones iniciales recomendados basados en la experiencia con enrutadores previos? (Revisar los pesos iniciales: -1 por SWAP, +10 por ejecutar frente a la penalización de timeout). ¿Sugerencias sobre recompensas negativas incrementales?

3. **Algoritmos SB3 y Experimentos:**
   Nuestro Wrapper soporta tanto `PPO` como `DQN` en políticas de `MultiInput`. ¿Recomiendan enfocar toda la computación empírica y tunning en *Proximal Policy Optimization* (PPO) -que es el estándar general actual-, o prefieren ver un test de comparación gruesos o con DQN en el informe?

5. **Entrenamiento y Fuentes de `initial_layout`:**
   El sistema actualmente expone la inyección de un *Layout Estático Inicial* como input externo. A la hora de entrenar al agente de forma general (durante los miles de timesteps), ¿deberíamos exponerle distribuciones variadas de layouts iniciales sobre diferentes circuitos, o dejarlo entrenar primero con *layouts 1:1* ruidosos/adversariales para forzar su robustez al routing independientemente del productor aguas arriba? La orquestación de un futuro handoff MO -> RL pertenece a `src/integration/`.

6. **Oscilación en Evaluación (ver sección "Problema Conocido"):**
   El agente aprende correctamente durante el entrenamiento (reward crece, longitud de episodio baja) pero entra en bucle al evaluarlo. ¿Qué enfoque recomiendan: redes recurrentes (`RecurrentPPO`), penalización explícita por repetición de layouts, o considerarlo una limitación inherente del MLP sin memoria y documentarlo como tal?
