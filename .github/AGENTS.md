# Workspace Instructions

Hybrid Quantum Transpilation: Multi-Objective Layout Optimization and Reinforcement Learning Synthesis.

## Code Style & Quality

- **Language**: Python 3.10+
- **Format**: Follow PEP 8; use type hints where practical.
- **Qiskit Compatibility**: Enforce Qiskit >= 2.0. Load the `qiskit-2x-compliance` skill for any Qiskit code changes.
- **Reproducibility**: Set deterministic seeds (numpy, torch, random) in all experiments. Reference the `experimentation-logging` skill for seed-setting patterns.

## Architecture

The project has 4 interconnected modules:

1. **`src/qiskit_interface/`** — Qiskit backend abstraction, circuit metrics, and transpilation baseline.
2. **`src/mo_module/`** — Multi-objective evolutionary algorithms (NSGA-II) for quantum layout optimization. Use the `mo-optimization` skill when modifying fitness, operators, or Pareto analysis.
3. **`src/rl_module/`** — Gymnasium-based RL environment and Stable-Baselines3 agent for circuit synthesis. Use the `rl-quantum-synthesis` skill for environment and reward design.
4. **`src/integration/`** — Orchestration of handoff and benchmark scenarios across modules. Use the `mo-rl-pipeline` skill for handoff logic and benchmark scenarios (Baseline, MO_Only, RL_Only, MO+RL).

## Build, Test & Experiments

- **Verify environment**: `.venv/Scripts/Activate.ps1` (Windows) activates the virtual environment.
- **Install dependencies**: `pip install -r requirements.txt`.
- **Run tests**: `pytest tests/` (from workspace root).
- **Benchmarking**: Use the `experimentation-logging` skill to ensure seeds, TensorBoard logging, and tabular exports are coordinated.

See [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md) for detailed dependency versions (Qiskit 2.3.0, PyTorch 2.5.1, pymoo 0.6.1.6, etc.).

## Conventions & Skill Routing

Before modifying code, load the appropriate skill(s) based on the module or task:

### By Module
- **`src/qiskit_interface/`**: Always load `qiskit-2x-compliance`.
- **`src/mo_module/`**: Load `mo-optimization` for problem, fitness, operators, or tuning changes.
- **`src/rl_module/`**: Load `rl-quantum-synthesis` for environment, rewards, or SB3 agent changes.
- **`src/integration/`**: Load `mo-rl-pipeline` (plus `experimentation-logging` if benchmarking).

### By Task Type
- **Benchmarks or data exports**: Load `experimentation-logging`.
- **MO→RL handoff or benchmark orchestration**: Load `mo-rl-pipeline` first.
- **Combined MO + Transpilation**: Load both `mo-optimization` and `qiskit-2x-compliance`.
- **RL consuming MO layouts**: Load both `rl-quantum-synthesis` and `mo-rl-pipeline`.

## Quality Gates

1. ✅ Qiskit imports and APIs conform to 2.0+; no legacy `qiskit.terra`, `qiskit.aer`, `qiskit.execute()`, or `QuantumInstance`.
2. ✅ Global random seeds are set before experiment runs (numpy, torch, Optuna, pymoo, Gymnasium).
3. ✅ Experiment results are exported as `.csv` or `.parquet` with consistent column schema (Circuit, Method, Depth, CNOTs, Fidelity/Error, Execution_Time, Seed).
4. ✅ Benchmark comparisons include all four scenarios: Baseline, MO_Only, RL_Only, MO+RL.
5. ✅ Module boundaries are respected: integration owns orchestration and handoff scenarios across the other modules.

For in-depth guidance, see the project skills in [`./skills/`](./skills/).
