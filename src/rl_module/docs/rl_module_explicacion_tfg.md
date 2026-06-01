# Módulo RL: explicación técnica para la defensa del TFG

## 1. Idea principal

El `rl_module` convierte una parte de la transpilación cuántica en un problema de decisiones secuenciales. En lugar de aplicar una heurística fija, modela el proceso como un entorno de aprendizaje por refuerzo donde:

- el entorno representa el estado actual del problema;
- un agente observa ese estado;
- el agente elige una acción;
- el entorno responde con una nueva observación y una recompensa;
- el proceso se repite hasta completar el episodio o alcanzar un límite de pasos.

En este proyecto, el módulo RL trabaja en dos escenarios distintos:

- `routing`: el agente decide qué `SWAP` insertar para que las puertas del circuito objetivo puedan ejecutarse sobre una topología física limitada.
- `synthesis`: el agente decide qué primitivas nativas aplicar para construir un circuito equivalente al objetivo bajo restricciones del hardware.

La idea corta que conviene transmitir en la exposición es esta:

> El módulo RL aprende una política de decisión paso a paso para adaptar un circuito cuántico a un hardware concreto, ya sea reorganizando qubits mediante `SWAPs` o sintetizando directamente con primitivas nativas.

## 2. Encaje del módulo dentro del TFG

El repositorio está dividido en cuatro módulos con responsabilidades separadas:

- `qiskit_interface`: backends, métricas, transpilation y utilidades ligadas a Qiskit.
- `mo_module`: búsqueda multiobjetivo de layouts iniciales.
- `rl_module`: entorno RL, recompensas, agente, entrenamiento y evaluación.
- `integration`: futura orquestación entre módulos.

La separación importante en la defensa es esta:

- `mo_module` puede proponer un `initial_layout`.
- `rl_module` no genera layouts; los consume.
- el handoff entre módulos pertenece a `integration`, no a `rl_module`.

La convención compartida de layout es:

```python
layout[i] = physical_qubit_for_logical_qubit_i
```

Ese layout puede entrar en RL mediante:

```python
env.reset(options={"initial_layout": layout})
```

Esto es importante porque desacopla la búsqueda del layout inicial de la política de decisión posterior.

## 3. Qué entra y qué sale del módulo RL

### Entradas principales

- `target_circuit`: circuito objetivo que se quiere enrutar o sintetizar.
- `coupling_map`: conectividad física del hardware.
- `mode`: `"routing"` o `"synthesis"`.
- `initial_layout`: opcional; si no se da, se usa uno trivial.
- `basis_gates`: obligatorio en `synthesis`, porque la topología no basta para saber qué primitivas nativas existen.

### Salidas principales

- un agente entrenado (`PPO`, `DQN` o `MaskablePPO` para routing enmascarado) capaz de actuar en el entorno;
- episodios evaluables paso a paso;
- métricas de entrenamiento;
- trazas e inspección visual mediante la GUI.

## 4. Cómo se formula como problema de RL

La traducción a RL es bastante natural:

| Elemento de RL | En este proyecto |
| --- | --- |
| Entorno | `QuantumTranspilationEnv` |
| Estado | observación codificada del layout, frontera o residual |
| Acción | elegir un `SWAP` o una primitiva nativa |
| Recompensa | señal que incentiva progreso y penaliza acciones malas |
| Episodio | intento completo de resolver el circuito |
| Política | red neuronal entrenada con SB3 |

La clave conceptual es que el agente no ve "el circuito entero como texto", sino una representación numérica del estado actual del problema. A partir de esa representación aprende una política que intenta maximizar la recompensa acumulada.

## 5. Arquitectura interna del módulo

La arquitectura está dividida en piezas con responsabilidades bastante claras.

### `environment.py`

Contiene `QuantumTranspilationEnv`, que es el núcleo del módulo. Su trabajo es:

- mantener el estado del episodio;
- construir observaciones;
- aplicar acciones;
- decidir cuándo termina el episodio;
- invocar la función de recompensa.

Es la pieza que conecta todo lo demás.

### `env_strategies.py`

Implementa el patrón Strategy para que el entorno no quede acoplado a un único tipo de problema.

- `RoutingStrategy`: define el espacio de acción y observación para routing.
- `SynthesisStrategy`: define el espacio de acción y observación para synthesis.

Esto permite reutilizar el mismo entorno general con dos semánticas distintas.

### `frontier.py`

Abstrae la noción de "puertas visibles pendientes" en routing.

- `SequentialFrontier`: trata el circuito como una secuencia lineal.
- `DagFrontier`: usa el DAG de Qiskit y la `front_layer()` para exponer paralelismo real.

Esta separación es importante porque la calidad de la observación depende de cómo se representa la frontera del circuito.

### `rewards.py`

Separa la mecánica del entorno de la lógica de recompensa.

- `RoutingReward`
- `SynthesisReward`

Con esto se puede reajustar el shaping sin reescribir la dinámica del entorno.

### `synthesis_primitives.py`

Construye el catálogo discreto de primitivas hardware-aware para `synthesis`.

### `synthesis_clifford.py`

Implementa la lógica matemática específica del modo `synthesis` basado en Clifford:

- remapeo lógico -> físico del circuito objetivo;
- construcción del residual;
- conversión del residual a arrays observables por la red;
- criterio de completitud.

### `agent.py`

Encapsula Stable-Baselines3 en `QuantumRLAgent`. Soporta `PPO`, `DQN` y `MaskablePPO` para routing enmascarado, y detecta automáticamente si hay `CUDA` disponible.

### `training.py`

Contiene el pipeline de entrenamiento:

- fijar semillas;
- crear entorno de entrenamiento y evaluación;
- envolver con `Monitor`;
- configurar callbacks;
- guardar artefactos del modelo.

### `gui/`

La GUI no es solo una demo. También sirve como herramienta de inspección e interpretabilidad.

- `rl_gui.py`: shell principal de la aplicación.
- `routing_view.py`: controles específicos de routing.
- `synthesis_view.py`: controles específicos de synthesis.
- `evaluation_panel.py`: inspector estructurado de episodios.

## 6. Representación interna del layout

Una decisión técnica importante del módulo es mantener dos arrays sincronizados:

```python
current_layout[logical_qubit] = physical_qubit
_inverse_layout[physical_qubit] = logical_qubit
```

Esto permite búsquedas y actualizaciones en tiempo constante `O(1)`.

Su utilidad práctica es clara:

- para saber dónde está un qubit lógico, se consulta `current_layout`;
- para saber qué qubit lógico ocupa una posición física, se consulta `_inverse_layout`;
- al aplicar un `SWAP`, ambos arrays se actualizan inmediatamente.

Cuando el hardware tiene más qubits físicos que el circuito lógico, las posiciones vacías se representan con `-1` en `_inverse_layout`.

Esto también permite detectar acciones inválidas, por ejemplo:

- un `SWAP` entre dos posiciones físicas vacías;
- una primitiva de `synthesis` aplicada sobre un qubit físico que no contiene ningún qubit lógico activo.

## 7. Flujo completo de funcionamiento

El flujo interno del módulo puede resumirse así:

```text
target_circuit + coupling_map + modo + layout inicial opcional
                |
                v
        QuantumTranspilationEnv.reset()
                |
                v
          construccion de observacion
                |
                v
          agente.predict(observacion)
                |
                v
         QuantumTranspilationEnv.step()
                |
                v
    nueva observacion + reward + info + fin/no fin
                |
                v
               repetir
```

### 7.1 Fase `reset()`

En el `reset()` el entorno:

- reinicia contadores del episodio;
- extrae las puertas del circuito objetivo;
- construye la frontera adecuada;
- carga el `initial_layout` si viene dado desde fuera;
- si no hay layout externo, usa el layout trivial `0 -> 0`, `1 -> 1`, etc.;
- reconstruye `_inverse_layout`;
- inicializa el estado específico del modo.

La divergencia por modo es importante:

- en `routing`, tras el reset el entorno intenta ejecutar automáticamente las puertas ya desbloqueadas en la front layer;
- en `synthesis`, construye un `CliffordSynthesisState` con el objetivo remapeado a qubits físicos.

### 7.2 Fase `step()`

En cada paso el entorno:

- incrementa `current_step`;
- decodifica la acción usando la estrategia activa;
- aplica la lógica específica del modo;
- calcula si el episodio terminó o se truncó;
- construye la nueva observación;
- calcula la recompensa;
- devuelve `obs, reward, terminated, truncated, info`.

El diccionario `info` es especialmente útil para depuración e inspección visual, porque guarda metadatos detallados del paso.

## 8. Funcionamiento del modo `routing`

## 8.1 Objetivo

El objetivo de `routing` no es reescribir el circuito completo desde cero, sino decidir qué `SWAPs` insertar para que las puertas pendientes se vuelvan ejecutables sobre la conectividad física disponible.

## 8.2 Espacio de acción

El espacio de acción es discreto:

- cada acción representa una arista física única del `coupling_map`;
- elegir esa acción significa aplicar un `SWAP` sobre esa arista.

La estrategia deduplica aristas bidireccionales y las ordena de forma determinista.

## 8.3 Observación

La observación en routing es un `Dict` con estas claves:

- `layout`: mapeo lógico -> físico actual.
- `lookahead`: ventana de próximas puertas en términos lógicos.
- `lookahead_physical`: proyección física de esa misma ventana.
- `lookahead_executable`: indica qué entradas son ejecutables inmediatamente.
- `lookahead_routing_distance`: aproximación de cuántos movimientos faltan para conectar una puerta bloqueada.
- `lookahead_valid_mask`: distingue puertas reales de padding.
- `step_progress`: progreso temporal normalizado del episodio.

La idea de esta observación es que el agente no solo vea "qué puertas vienen", sino también cómo se proyectan físicamente bajo el layout actual y qué tan lejos están de poder ejecutarse.

## 8.4 Frontera visible

El módulo soporta dos interpretaciones de frontera:

- `sequential`: mira una cola lineal de instrucciones.
- `dag`: usa la `front_layer()` real del DAG y puede exponer varias puertas independientes al mismo tiempo.

Esto es importante en la defensa porque muestra que el trabajo no se limita a una cola ingenua: el módulo puede razonar con dependencias más cercanas a la estructura real del circuito.

## 8.5 Semántica del paso

Cuando el agente elige un `SWAP`:

- el entorno localiza qué qubits lógicos están en los dos nodos físicos;
- actualiza `current_layout` y `_inverse_layout`;
- detecta si el nuevo layout repite uno reciente;
- detecta si el `SWAP` actual deshace el anterior;
- intenta ejecutar en cascada todas las puertas que hayan quedado desbloqueadas.

Esa ejecución en cascada es una idea importante: el agente no recibe crédito solo por mover qubits, sino por desbloquear ejecución real del circuito.

## 8.6 Métrica de progreso

Además del número de puertas ejecutadas, el entorno calcula `routing_progress_delta`.

La lógica es:

- se agrega una señal de distancia de routing sobre la frontera visible;
- si esa distancia agregada baja, hay progreso;
- si sube o se repite layout, el agente recibe una señal peor.

Esto introduce reward shaping más fino que una simple recompensa binaria por completar o no completar el circuito.

## 8.7 Recompensa en routing

La recompensa por defecto combina varios términos:

- penalización por `SWAP` aplicado: `-1.0`;
- recompensa por puerta ejecutada: `+10.0` por puerta;
- penalización por acción inválida: `-5.0`;
- penalización por repetir layout reciente: `-1.0`;
- penalización por deshacer el `SWAP` anterior: `-1.0`;
- penalización adicional por `SWAP` improductivo: `-0.25`;
- shaping proporcional a `routing_progress_delta`: peso `0.5`;
- penalización proporcional a la distancia pendiente de la nueva frontera tras ejecutar puertas: peso `0.25`;
- penalización proporcional al incremento de profundidad crítica estimada: peso `0.1`;
- bonus de completitud: `+50.0`;
- penalización por truncación: `-30.0`;
- penalización por estancamiento: `-20.0`;
- penalización adicional por puerta pendiente al fallar: `-1.0`.

La filosofía es clara: castigar movimientos inútiles o cíclicos y premiar acciones que realmente acerquen la ejecución del circuito.

## 8.8 Criterio de finalización

En `routing`, el episodio termina cuando no quedan puertas pendientes en la frontera.

## 9. Funcionamiento del modo `synthesis`

## 9.1 Objetivo

El modo `synthesis` no intenta desbloquear puertas de un circuito dado mediante `SWAPs`, sino construir una secuencia de primitivas nativas que sea equivalente al circuito objetivo.

Actualmente es una primera versión entrenable con un alcance deliberadamente acotado:

- solo soporta circuitos Clifford;
- requiere `basis_gates` explícitas;
- mantiene el layout fijo durante el episodio;
- mide el progreso mediante un residual, no mediante una frontera de puertas pendientes.

## 9.2 Por qué `basis_gates` es obligatorio

En `routing`, conocer la conectividad física suele bastar para decidir qué `SWAPs` son posibles.

En `synthesis`, no basta con saber qué qubits están conectados. También hace falta saber qué puertas de 2 qubits son nativas del backend. Por eso el constructor exige `basis_gates` cuando `mode="synthesis"`.

## 9.3 Catálogo de primitivas

`SynthesisStrategy` construye un catálogo discreto de primitivas con `build_clifford_primitive_catalog()`.

En la versión actual, el catálogo puede incluir:

- `x` sobre cada qubit físico;
- `sx` sobre cada qubit físico;
- `rz(pi/2)`, `rz(pi)` y `rz(3*pi/2)` sobre cada qubit físico;
- una puerta nativa de 2 qubits sobre cada arista física válida: `cz`, `ecr` o `cx`.

Los costes por defecto del catálogo son:

- `x`: `1.0`
- `sx`: `1.0`
- `rz(...)`: `0.0`
- puerta de 2 qubits: `3.0`

Esto permite que la recompensa penalice naturalmente primitivas más caras.

## 9.4 Estado de síntesis basado en Clifford

La pieza clave es `CliffordSynthesisState`, que mantiene:

- `target`: Clifford objetivo remapeado al espacio físico;
- `current`: Clifford actual construido por las primitivas aplicadas.

El residual se define como:

```python
residual = current.adjoint().compose(target)
```

La interpretación es:

- si el residual es la identidad, el circuito sintetizado ya equivale al objetivo;
- si no es la identidad, todavía falta trabajo por hacer.

La distancia al objetivo se mide comparando el residual con la identidad a nivel de matriz simpléctica y fase.

## 9.5 Por qué se trabaja en espacio físico

El residual no se define solo sobre qubits lógicos, sino sobre qubits físicos. Esto tiene sentido porque:

- las acciones se aplican sobre qubits físicos;
- puede haber más qubits físicos que lógicos;
- los qubits físicos no usados deben permanecer coherentes con la identidad.

Esta es una decisión técnica importante y bastante defendible ante tutores.

## 9.6 Observación en synthesis

La observación de `synthesis` contiene:

- `layout`: lógico -> físico.
- `physical_to_logical`: físico -> lógico.
- `residual_symplectic`: matriz simpléctica aplanada del residual.
- `residual_phase`: bits de fase del residual.
- `step_progress`: progreso temporal normalizado.

Aquí la red no ve una frontera de puertas, sino una representación algebraica del error residual respecto al objetivo.

## 9.7 Validación de acciones

Una primitiva puede ser inválida si:

- intenta actuar sobre un qubit físico vacío;
- intenta usar una arista no conectada;
- el circuito objetivo no es Clifford y no puede convertirse correctamente.

Las acciones inválidas no completan el episodio ni cambian el estado actual, pero sí reciben penalización.

## 9.8 Recompensa en synthesis

La recompensa por defecto en `synthesis` es residual-céntrica:

- penalización por acción inválida: `-5.0`;
- penalización base por paso válido: `-0.25`;
- penalización proporcional al coste de la primitiva: peso `0.5`;
- recompensa proporcional a `residual_distance_delta`: peso `1.0`;
- bonus de completitud: `+100.0`;
- penalización por truncación: `-30.0`.

La intuición es:

- cada paso tiene un coste;
- las primitivas caras cuestan más;
- reducir el residual se recompensa;
- completar exactamente la síntesis se recompensa mucho.

## 9.9 Criterio de finalización

En `synthesis`, el episodio termina cuando el residual es la identidad. Si se alcanza `max_steps` sin lograrlo, el episodio se trunca.

## 10. Entrenamiento del agente

El entrenamiento se construye alrededor de `QuantumRLAgent` y `setup_training_pipeline()`.

El pipeline hace lo siguiente:

- fija semillas globales con `set_global_seeds()` para reproducibilidad;
- crea un entorno de entrenamiento y otro de evaluación;
- envuelve ambos con `Monitor`;
- configura `CheckpointCallback` y `EvalCallback`;
- entrena el modelo con Stable-Baselines3;
- guarda el modelo final y, si existe, el mejor modelo evaluado.

Los algoritmos soportados son:

- `PPO`
- `DQN`
- `MaskablePPO` (solo routing enmascarado)

El agente usa por defecto `MultiInputPolicy`, porque las observaciones son diccionarios con varias entradas heterogéneas.

Además, el wrapper detecta automáticamente si puede usar `cuda` o debe usar `cpu`.

En `DQN`, el wrapper inyecta algunos hiperparámetros de estabilización por defecto:

- `exploration_fraction=0.5`
- `tau=0.05`
- `learning_starts=1000`

## 11. Evaluación e interpretabilidad

Una parte valiosa del módulo es que no se limita a entrenar: también permite inspeccionar qué está haciendo realmente el agente.

## 11.1 GUI unificada con dos vistas

La GUI es una sola aplicación, `RLBenchmarkGUI`, pero con dos sub-vistas especializadas:

- `RoutingView`: muestra controles ligados a frontera y lookahead.
- `SynthesisView`: muestra controles ligados a perfiles de `basis_gates`.

Esto evita mezclar semánticas que en realidad son distintas.

## 11.2 Inspector de episodios

El `EpisodeInspectorPanel` guarda un `EvaluationStepRecord` por paso y permite revisar el episodio de manera estructurada.

En `routing`, cada paso puede mostrar:

- `layout_before` y `layout_after`;
- `visible_frontier_before`;
- `swap_edge`;
- `executed_gates`;
- `routing_progress_delta`;
- señales de ciclo como `repeated_layout` y `undo_swap`.

En `synthesis`, cada paso puede mostrar:

- `primitive_name`;
- `primitive_physical_qargs`;
- `primitive_cost`;
- `residual_distance_before`;
- `residual_distance_after`;
- `residual_distance_delta`.

Esto es especialmente útil en la defensa porque permite enseñar no solo el resultado final, sino la lógica interna del agente paso a paso.

## 11.3 Evaluación determinista

Durante la evaluación, la GUI ejecuta la política en modo determinista y puede cargar preferentemente el `best_model.zip` si existe.

Esto permite comparar:

- lo que el agente aprendió durante entrenamiento;
- cómo se comporta cuando ya no hay exploración estocástica.

## 12. Decisiones técnicas importantes del diseño

Hay varias decisiones arquitectónicas que conviene remarcar en la exposición:

### 12.1 Patrón Strategy

Permite reutilizar un mismo entorno general con dos semánticas muy diferentes sin duplicar toda la clase principal.

### 12.2 Separación entre dinámica y recompensa

La dinámica del entorno está en `environment.py`, mientras que la función de recompensa vive en `rewards.py`. Esto facilita experimentar con shaping sin reescribir la mecánica del episodio.

### 12.3 Representación O(1) del layout

Mantener `current_layout` y `_inverse_layout` reduce el coste de consultas y actualizaciones durante el episodio. Esto es especialmente importante porque el entorno puede ejecutarse miles de veces durante entrenamiento.

### 12.4 Frontera secuencial o DAG

La frontera no está hardcodeada como una cola lineal. El modo `dag` permite aproximarse mejor al paralelismo real del circuito.

### 12.5 Síntesis hardware-aware real

En `synthesis` no se asume que la conectividad basta. El módulo también exige `basis_gates`, lo que hace la formulación más realista desde el punto de vista del hardware.

### 12.6 Interpretabilidad incorporada

La GUI y el inspector no son un añadido cosmético: son una capa de análisis que ayuda a validar y explicar el comportamiento del agente.

## 13. Limitaciones actuales y problemas conocidos

También conviene explicar con honestidad qué límites tiene el módulo en su estado actual.

### 13.1 Síntesis acotada a Clifford

El modo `synthesis` actual solo cubre circuitos Clifford. No aborda todavía síntesis general no-Clifford.

### 13.2 Layout fijo en synthesis

En `synthesis` v1 no hay `SWAP` dinámico dentro del episodio. El layout se fija al inicio y la política solo decide primitivas.

### 13.3 Problema de oscilación en evaluación

Existe un comportamiento conocido en `routing`: durante entrenamiento, la política estocástica puede parecer aprender bien, pero en evaluación determinista puede entrar en ciclos del tipo `A -> B -> A -> B`.

La explicación principal es:

- la red no tiene memoria recurrente;
- un MLP puede tratar estados muy parecidos como si fueran el mismo;
- el ruido de exploración del entrenamiento puede ocultar ciclos que luego aparecen al evaluar.

Como mitigaciones actuales, el módulo ya incluye:

- `step_progress` en la observación;
- señales `repeated_layout` y `undo_swap`;
- reward shaping por progreso de routing;
- detección visual de ciclos en la GUI para `routing`.

### 13.4 Integración MO -> RL aún no orquestada

El módulo RL ya está preparado para consumir un `initial_layout`, pero la coordinación completa entre productores y consumidores sigue perteneciendo al módulo `integration`.

## 14. Qué aporta este módulo al TFG

Visto como contribución técnica, el módulo RL aporta varias cosas a la memoria y a la defensa:

- una formulación explícita de routing como problema RL con observación enriquecida;
- una arquitectura modular y escalable basada en estrategias;
- soporte para dos modos semánticamente distintos dentro del mismo marco;
- una primera implementación entrenable de síntesis Clifford hardware-aware;
- herramientas de evaluación e inspección paso a paso.

No es solo un script de entrenamiento. Es una infraestructura experimental para estudiar decisiones de transpilación con aprendizaje por refuerzo.

## 15. Guion oral breve para explicárselo a los tutores

Una forma clara de contarlo en voz alta sería esta:

> El módulo RL toma un circuito objetivo y una topología física, y convierte el problema de transpilarlo en una secuencia de decisiones. En modo routing, el agente aprende qué `SWAPs` insertar para desbloquear puertas del circuito. En modo synthesis, aprende qué primitivas nativas aplicar para reconstruir un circuito equivalente. Todo esto se formaliza como un entorno Gymnasium con observaciones, acciones, recompensas y episodios. La arquitectura está separada por responsabilidades: el entorno gestiona la dinámica, las estrategias definen cómo se observa y se actúa, las recompensas hacen el shaping, y el agente se entrena con Stable-Baselines3. Además, la GUI permite inspeccionar cada paso para entender por qué el agente toma ciertas decisiones y detectar problemas como bucles u oscilaciones.

## 16. Mensajes clave que conviene remarcar en la defensa

- El módulo RL no genera layouts: los consume.
- `routing` y `synthesis` comparten infraestructura, pero no comparten semántica.
- El estado no se define de forma ingenua: incluye contexto físico y señales de progreso.
- La síntesis actual ya es hardware-aware, porque usa `coupling_map` y `basis_gates`.
- La GUI aporta interpretabilidad, no solo visualización.
- El diseño ya deja preparada una base razonable para extensiones futuras.

## 17. Posibles preguntas de tutores y respuesta corta

### "¿Por qué RL y no solo una heurística fija?"

Porque el problema se ha formulado como una secuencia de decisiones dependientes del estado actual, y RL permite aprender políticas condicionadas por observaciones del circuito, del layout y del hardware, en lugar de codificar una regla estática para todos los casos.

### "¿Qué diferencia real hay entre routing y synthesis?"

En routing se modifica el layout dinámico mediante `SWAPs` para poder ejecutar un circuito ya dado. En synthesis no se intenta desbloquear una cola de puertas, sino construir activamente un circuito equivalente usando primitivas nativas.

### "¿Cuál es el núcleo técnico del diseño?"

La combinación de un entorno Gymnasium modular, una representación eficiente del layout, observaciones ricas, reward shaping y una separación limpia entre estrategias, recompensas y entrenamiento.

### "¿Qué limitación reconoces ahora mismo?"

Que `synthesis` sigue acotado a Clifford con layout fijo, y que en `routing` todavía existe riesgo de oscilación en evaluación determinista si la política no ha aprendido suficiente contexto temporal o histórico.
