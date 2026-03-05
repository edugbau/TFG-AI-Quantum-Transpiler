# Skill: RL Quantum Synthesis
**Contexto para el Módulo 2 (`rl_module`).**

## Objetivo
Implementación correcta y estandarizada del Entorno de Aprendizaje por Refuerzo para síntesis de circuitos cuánticos Clifford.

## Librerías Principales
- **Gymnasium** (NO `gym` clásico de OpenAI).
- **Stable-Baselines3** (SB3).
- **PyTorch** con CUDA (`torch`).

## Reglas de Implementación

1. **Definición del Entorno (`gymnasium.Env`)**
   - **`__init__`**: Definir `observation_space` y `action_space` usando `gymnasium.spaces` (ej. `Discrete`, `Box`, `MultiDiscrete`).
   - **Estado (Observation):** La observación debe codificar el estado del circuito actual, el mapeo de los qubits y (si aplica) la "distancia" al objetivo o el remanente de conectividad (Coupling Map).
   - **Acción (Action):** La acción debe ser discreta o estructurada para representar qué puerta aplicar y en qué qubits lógicos/físicos. Por ejemplo, insertar un SWAP o cambiar la topología.
   - **`step(action)`**: Retorna `observation, reward, terminated, truncated, info`. Aplicar el paso al circuito/layout y calcular métricas.
   - **`reset(seed=seed, options=options)`**: Retorna `observation, info`. Debe restablecer el entorno a un estado inicial limpio o predefinido (ver Módulo de Integración).

2. **Recompensas (Reward Function)**
   - Recompensa **Densa:** Cambio positivo/negativo en la profundidad del circuito o recuento de puertas equivalentes de dos qubits.
   - Recompensa **Rala (Sparse):** Gran bonificación al conseguir la fidelidad o síntesis exacta del circuito Clifford.
   - Penalización por transiciones inválidas o bucles de SWAPs redundantes.

3. **Agente y Entrenamiento (SB3)**
   - **Algoritmos recomendados:** PPO o DQN (usualmente PPO para espacios de acciones más complejos, aunque las acciones discretas aplican a ambos).
   - **Aceleración HW:** Usar `device="cuda"` cuando se instancie el modelo SB3 o el MLP interno, si detecta GPU.
   - **Evaluación y Logging:** Usar los `EvalCallback` y `CheckpointCallback` de SB3. Exportar a TensorBoard (`tensorboard_log` en el modelo SB3).
