# Skill: RL Quantum Synthesis
**Context for Module 2 (`rl_module`).**

## Objective
Correct and standardized implementation of the reinforcement learning environment for Clifford quantum circuit synthesis.

## Main Libraries
- **Gymnasium** (not legacy OpenAI `gym`).
- **Stable-Baselines3** (SB3).
- **PyTorch** with CUDA (`torch`).

## Implementation Rules

1. **Environment Definition (`gymnasium.Env`)**
   - **`__init__`**: Define `observation_space` and `action_space` using `gymnasium.spaces` (for example `Discrete`, `Box`, `MultiDiscrete`).
   - **State (Observation):** Observation should encode current circuit state, qubit mapping, and when relevant, distance to target or remaining connectivity constraints (Coupling Map).
   - **Action:** Action should be discrete or structured to represent which gate to apply and on which logical/physical qubits. For example, insert a SWAP or change topology.
   - **`step(action)`**: Return `observation, reward, terminated, truncated, info`. Apply the step to the circuit/layout and compute metrics.
   - **`reset(seed=seed, options=options)`**: Return `observation, info`. Reset to a clean or predefined initial state (see Integration Module).

2. **Rewards (Reward Function)**
   - **Dense reward:** Positive/negative change in circuit depth or equivalent two-qubit gate count.
   - **Sparse reward:** Large bonus for achieving target fidelity or exact Clifford synthesis.
   - Penalize invalid transitions or redundant SWAP loops.

3. **Agent and Training (SB3)**
   - **Recommended algorithms:** PPO or DQN (typically PPO for more complex action spaces, though discrete actions fit both).
   - **Hardware acceleration:** Use `device="cuda"` when creating the SB3 model or internal MLP, if a GPU is available.
   - **Evaluation and logging:** Use SB3 `EvalCallback` and `CheckpointCallback`. Export to TensorBoard (`tensorboard_log` in the SB3 model).
