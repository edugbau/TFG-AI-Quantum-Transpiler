# Skill: MO Optimization (Multi-Objective)
**Context for Module 3 (`mo_module`).**

## Objective
Standardize the design and evaluation of multi-objective evolutionary algorithms for quantum layout optimization.

## Main Library
- **pymoo**: Use the `pymoo` API for problem definition and NSGA-II or MOEA/D execution.

## Implementation Rules

1. **Problem Definition (`pymoo.core.problem.Problem`)**
   - **Genotype (Representation):** The individual (chromosome) should represent an integer permutation or a direct logical-to-physical qubit mapping (`integer` or `permutation` variables).
   - **Evaluation (Fitness):** Multi-objective minimization. By default:
     - Transpiled circuit depth (`depth`).
     - CNOT-equivalent count (or native two-qubit gate count).

2. **Recommended Genetic Operators**
   - **Crossover:** Use **DPX (Dynastic Potential Crossover)** by default. It better preserves shared parent layout assignments (low epistasis).
   - **Mutation:** Position swap (SWAP mutator).

3. **Result Analysis and Pareto Front**
   - Extract non-dominated individuals from `pymoo` execution.
   - Use Hypervolume (HV) as the primary metric for front quality comparison.

4. **Hyperparameter Tuning (Optuna)**
   - Use `optuna` (TPE) to optimize population size, crossover rate, and mutation rate, maximizing mean Hypervolume (or minimizing penalties).
