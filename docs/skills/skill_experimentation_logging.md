# Skill: Experimentation Logging
**Cross-cutting context for analysis and benchmarking (`integration` / transversal).**

## Objective
Unify plot generation (Matplotlib), tabular data persistence (Pandas), and experiment reproducibility through strict seeds and TensorBoard logs.

## Implementation Rules

1. **Global Random Seeds**
   - It is **mandatory** to set seeds before any run that must be reproducible (including Optuna, pymoo, Gymnasium, and PyTorch scripts).
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
       # Avoid cuDNN non-determinism:
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
       # Gymnasium uses the `seed` parameter in env.reset(seed=...)
       # pymoo requires `pymoo.util.misc.set_random_seed(seed)`
   ```

2. **RL Logging (Stable-Baselines3)**
   - Always enable `tensorboard_log` when initializing PPO/DQN (e.g. `PPO(..., tensorboard_log="./experiments/logs/rl_logs/")`).
   - Use periodic `CheckpointCallback` to save model weights every N timesteps (e.g., `save_path="./experiments/logs/rl_models/"`).

3. **Structured Metrics (Pandas)**
   - During experiment sweeps (e.g. MO+RL vs. SABRE over 20 circuits), export final results to tabular format (`.csv` or `.parquet`) using Pandas. Columns should include at least: `Circuit`, `Method`, `Depth`, `CNOTs`, `Fidelity/Error`, `Execution_Time`, `Seed`.

4. **Standard Visualization (Matplotlib)**
   - Pareto fronts: 2D scatter plot (e.g. Depth vs. CNOTs). Highlight non-dominated solutions against the baseline (SABRE or Qiskit O3).
   - RL learning curves: X-axis (Timesteps), Y-axis (Cumulative Reward or Final Depth). Shade standard deviation when evaluated across seeds.
   - All plots must include titles, explicit legends, axes with units (e.g. "Depth (Gates)"), and be saved at high resolution (`.png` at 300 DPI) in `experiments/plots/`.
