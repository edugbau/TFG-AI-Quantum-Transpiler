# Documentación Interna: Módulo de Aprendizaje por Refuerzo (RL)

## Visión General
El `rl_module` se encarga de la segunda fase del pipeline híbrido: el **Enrutamiento (Routing) y Síntesis** de circuitos cuánticos. Tras recibir un mapeo estático inicial (Layout) generado por el módulo de Optimización Multiobjetivo (MO), el agente de RL se encarga de insertar puertas SWAP de manera dinámica para resolver los bloqueos de conectividad, minimizando la profundidad final del circuito.

Para garantizar la viabilidad futura del proyecto, la arquitectura del entorno ha sido diseñada aplicando el **Patrón Strategy**, permitiendo con un solo parámetro escalar el módulo para que el agente realice **Síntesis Completa** (generar las puertas desde cero) en lugar de solo enrutamiento.

## Estructura de Archivos

```
src/rl_module/
├── environment.py       # Entorno principal (QuantumTranspilationEnv) heredando de gymnasium.Env
├── env_strategies.py    # Patrón Strategy (RoutingStrategy y SynthesisStrategy)
├── rewards.py           # Funciones de recompensa desacopladas (RoutingReward, SynthesisReward)
└── docs/                # Documentación interna (este archivo)
```

## Arquitectura de Escalabilidad (Patrón Strategy)

El entorno `QuantumTranspilationEnv` no hardcodea los espacios de observación ni de acción. Delega esta responsabilidad a las clases hijas de `RLEnvStrategy`.

### 1. Modo Enrutamiento (`mode="routing"`) [Por defecto]
Es la implementación principal y actual para acoplarse con el Módulo MO.
- **Action Space:** Discreto (`gym.spaces.Discrete`). El tamaño equivale al número de conexiones (aristas) en el Coupling Map del hardware. En cada paso, el agente selecciona una arista para aplicar un **SWAP** entre los dos qubits físicos conectados.
- **Observation Space:** Diccionario compuesto por:
  1. `layout`: Un array con el mapeo actual lógico $\rightarrow$ físico.
  2. `lookahead`: Un array de tamaño $N \times 2$, representando la ventana de las próximas $N$ puertas lógicas a ejecutar (los pares de qubits lógicos que necesitan interaccionar).
- **Lógica de Ejecución (Step):** Tras aplicar el SWAP, el entorno comprueba automáticamente si las puertas pendientes en la *front-layer* del circuito ahora cumplen las restricciones topológicas y, de ser así, las "ejecuta" (las elimina de la cola de espera).

### 2. Modo Síntesis (`mode="synthesis"`) [Plantilla de escalabilidad]
Preparado para el futuro, donde el agente no solo mueve qubits, sino que deduce el circuito entero (e.g. desde un Tableau de Clifford o Matriz Unitaria).
- **Action Space:** Multi-Discreto (`gym.spaces.MultiDiscrete`). El agente elige simultáneamente [Tipo de Puerta (ej. CX, RX, RZ), Qubit Físico 1, Qubit Físico 2].
- **Observation Space:** Extensible. Actualmente hace scaffolding de Routing, pero está diseñado para aceptar representaciones de matrices unitarias o grafos de dependencia (DAGs).

## Sistema de Recompensas (`rewards.py`)

Extraído fuera del `step()` en clases hijas de `RewardStrategy`. Esto permite modificar heurísticas de recompensa sin tocar el núcleo del entorno:

**Para Routing (`RoutingReward`):**
- **SWAP aplicado:** Penalización leve (ej. -1) para incentivar caminos cortos.
- **Puertas ejecutadas:** Fuerte bonificación (ej. +10 por cada puerta que logre encajar en la topología).
- **Acción Inválida:** Penalización (-5) para desincentivar SWAPs redundantes.
- **Circuito Completado:** Bonificación masiva de victoria (+50).

## Integración con el Módulo MO (`skill_mo_rl_pipeline`)

El entorno está preparado para recibir el individuo ganador (Frente de Pareto) de `pymoo` como estado inicial. 
Esto se realiza inyectando el array del layout a través del parámetro `options` en la función `reset()`:

```python
env = QuantumTranspilationEnv(target_circuit=qc, coupling_map=cmap, mode="routing")

# El array devuelto por pymoo [1, 0, 3, 2] se inyecta directamente
obs, info = env.reset(options={"initial_layout": [1, 0, 3, 2]})
```

Esto cumple a rajatabla con la Skill definida, permitiendo al agente RL comenzar con un layout de alta calidad en lugar de uno trivial (`[0, 1, 2...]`), lo que reduce drásticamente los pasos necesarios para enrutar el circuito.

## Próximos Pasos (Pendientes)
1. **Representación Estricta con DAG:** En la maqueta actual, el *Lookahead* funciona extrayendo secuencialmente de la lista de instrucciones de Qiskit. Para aprovechar el paralelismo natural cuántico, será óptimo reemplazar la lista por una conversión a `DAGCircuit` de Qiskit y utilizar `dag.front_layer()`.
2. **Wrapper de Stable-Baselines3:** Crear el script que inicialice el agente PPO/DQN alimentando este entorno con soporte CUDA y callbacks de TensorBoard (`agent.py` y `training.py`).
