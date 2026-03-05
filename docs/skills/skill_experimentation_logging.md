# Skill: Experimentation Logging
**Contexto transversal para Análisis y Benchmark (`integration` / transversal).**

## Objetivo
Unificar la generación de gráficas (Matplotlib), el guardado de datos tabulares (Pandas) y la reproducibilidad de todo experimento mediante semillas estrictas y logs de TensorBoard.

## Reglas de Implementación

1. **Semillas Aleatorias Globales (Seeds)**
   - **Absolutamente OBLIGATORIO** fijar las semillas antes de cualquier ejecución que deba ser reproducible (incluyendo scripts de Optuna, pymoo, Gymnasium, y PyTorch).
   ```python
   import random
   import numpy as np
   import torch
   from pymoo.core.problem import Problem

   def set_seed(seed=42):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       # Evitar la no-determinismo de cudnn:
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
       # Gymnasium usa el propio parámetro `seed` en env.reset(seed=...)
       # pymoo requiere usar la clase `pymoo.util.misc.set_random_seed(seed)`
   ```

2. **Registro de RL (Stable-Baselines3)**
   - Siempre activar `tensorboard_log` en la inicialización de PPO/DQN (e.g. `PPO(..., tensorboard_log="./experiments/logs/rl_logs/")`).
   - Usar un callback periódico `CheckpointCallback` para guardar los pesos del modelo cada N timesteps (e.g., `save_path="./experiments/logs/rl_models/"`).

3. **Métricas Estructuradas (Pandas)**
   - Al realizar barridos de experimentos (ej. Comparar MO+RL vs. SABRE en 20 circuitos), exportar los resultados finales en formato tabular (`.csv` o `.parquet`) utilizando Pandas. Las columnas deben incluir al menos: `Circuito`, `Método`, `Profundidad`, `CNOTs`, `Fidelidad/Error`, `Tiempo_Ejecución`, `Seed`.

4. **Visualización Estándar (Matplotlib)**
   - Frentes de Pareto: Scatter plot 2D (ej. Profundidad vs. CNOTs). Destacar las soluciones no dominadas respecto al baseline (SABRE o Qiskit O3).
   - Curvas de Aprendizaje RL: Eje X (Timesteps), Eje Y (Recompensa Acumulada o Profundidad Final). Sombrear la desviación estándar si se evalúa en varias semillas.
   - Todas las gráficas deben llevar títulos, leyendas explícitas, ejes con sus unidades (ej. "Depth (Gates)") y guardarse en alta resolución (`.png` a 300 DPI) en `experiments/plots/`.
