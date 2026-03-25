---
name: experimentation-logging
description: 'Standardize experiment logging with reproducible seeds, TensorBoard, Pandas tabular exports, and Matplotlib plots. Use for benchmarking and cross-module analysis.'
argument-hint: 'Describe the experiment setup and the metrics you need to track.'
user-invocable: true
---

# Experimentation Logging

## When to Use
- Running benchmark suites or comparative studies across methods.
- Logging RL experiments with Stable-Baselines3.
- Exporting results for reproducibility and downstream analysis.

## Objective
Unify plotting (Matplotlib), tabular result storage (Pandas), and reproducibility through strict seed control and TensorBoard logging.

## Procedure
1. Set global seeds before any reproducible run (Optuna, pymoo, Gymnasium, and PyTorch).
2. Enable TensorBoard and periodic checkpoints for RL training in Stable-Baselines3.
3. Export final metrics to `.csv` or `.parquet` with a consistent schema.
4. Generate standard visualizations with labels, units, and high-resolution output.

## Implementation Rules

### 1) Global Random Seeds
- Set the seed at the beginning of each experiment.
- For Gymnasium, use `env.reset(seed=...)`.
- For pymoo, use `pymoo.util.misc.set_random_seed(seed)`.

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 2) RL Logging (SB3)
- Enable `tensorboard_log` when initializing PPO/DQN.
- Use `CheckpointCallback` to save model weights every N timesteps.

### 3) Structured Metrics (Pandas)
- Export results with at least these columns:
  - `Circuit`
  - `Method`
  - `Depth`
  - `CNOTs`
  - `Fidelity/Error`
  - `Execution_Time`
  - `Seed`

### 4) Standard Visualization (Matplotlib)
- Pareto fronts: 2D scatter (for example, depth vs CNOTs), highlighting non-dominated solutions against the baseline.
- RL learning curves: X as timesteps, Y as cumulative reward or final depth.
- Include titles, legends, axis units, and export at 300 DPI to `experiments/plots/`.

## Project References
- `docs/agents.md`
- `docs/ENVIRONMENT.md`
