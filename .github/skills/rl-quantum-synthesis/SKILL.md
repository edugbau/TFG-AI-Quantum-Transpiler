---
name: rl-quantum-synthesis
description: 'Guide for implementing a Gymnasium environment and SB3 training for quantum synthesis: state/action design, rewards, logging, checkpoints, and PyTorch CUDA usage.'
argument-hint: 'Describe your environment shape, action space, and current reward design issue.'
user-invocable: true
---

# RL Quantum Synthesis

## When to Use
- Implementing or modifying `src/rl_module`.
- Defining Gymnasium environment mechanics, rewards, or action strategy.
- Configuring SB3 training, evaluation, and logging.

## Objective
Standardize implementation of the reinforcement learning environment for Clifford quantum circuit synthesis.

## Main Libraries
- Gymnasium (no `gym` legacy).
- Stable-Baselines3.
- PyTorch with CUDA.

## Procedure
1. Define a `gymnasium.Env` with correct observation and action spaces.
2. Implement `step(action)` and `reset(seed=..., options=...)` following Gymnasium semantics.
3. Design dense/sparse rewards with penalties for invalid actions.
4. Train with PPO or DQN, using GPU when available.
5. Integrate `EvalCallback`, `CheckpointCallback`, and TensorBoard.

## Implementation Rules

### 1) Environment (`gymnasium.Env`)
- `__init__`: define `observation_space` and `action_space` with `gymnasium.spaces`.
- Observation: encode circuit state, qubit mapping, and when applicable, distance-to-goal or remaining connectivity constraints.
- Action: discrete or structured action design to represent gate/qubit choices (for example SWAP insertion or topology change).
- `step(action)`: return `observation, reward, terminated, truncated, info`.
- `reset(seed=seed, options=options)`: return `observation, info`.

### 2) Reward Function
- Dense: reward based on depth change or equivalent two-qubit gate change.
- Sparse: large bonus for exact synthesis or target fidelity attainment.
- Penalty for invalid transitions or redundant SWAP loops.

### 3) Agent and Training (SB3)
- Recommended algorithms: PPO or DQN.
- Hardware acceleration: `device="cuda"` when GPU is available.
- Evaluation and logging: `EvalCallback`, `CheckpointCallback`, and `tensorboard_log`.

## Project References
- `src/rl_module/docs/internal_documentation.md`
- `src/rl_module/docs/synthesis_mode_status.md`
