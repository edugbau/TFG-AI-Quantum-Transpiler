---
name: mo-optimization
description: 'Standardize multi-objective optimization of quantum layouts with pymoo, including problem definition, genetic operators, Pareto analysis, and Optuna hyperparameter tuning.'
argument-hint: 'Specify circuit size, objectives, and whether you want NSGA-II or MOEA/D guidance.'
user-invocable: true
---

# MO Optimization

## When to Use
- Implementing or modifying `mo_module`.
- Adjusting representation, fitness functions, or evolutionary operators.
- Comparing Pareto front quality for quantum layout optimization.

## Objective
Standardize the design and evaluation of multi-objective evolutionary algorithms for quantum layout optimization.

## Main Library
- `pymoo` for problem modeling and NSGA-II or MOEA/D execution.

## Procedure
1. Define the problem with `pymoo.core.problem.Problem` and a valid layout encoding.
2. Evaluate multi-objective fitness by minimizing depth and CNOT-equivalent cost.
3. Use DPX as the default crossover and categorical mutation settings.
4. Extract non-dominated solutions and evaluate with Hypervolume (HV).
5. Tune hyperparameters with Optuna (TPE).

## Implementation Rules

### 1) Problem Definition
- Use a genotype as an integer permutation or direct logical-to-physical mapping.
- Evaluate by multi-objective minimization of:
  - Transpiled circuit `depth`.
  - CNOT-equivalent cost or native two-qubit gate count.

### 2) Recommended Genetic Operators
- Default crossover: DPX (Dynastic Potential Crossover).
- Mutation: combined swap and replace mutation, configured through categorical probability values reviewed at the config layer.

### 3) Pareto Front and Analysis
- Extract non-dominated individuals.
- Use Hypervolume (HV) as the primary quality metric.

### 4) Hyperparameter Tuning
- Optimize with Optuna (TPE):
  - Population size.
  - Crossover rate.
  - Categorical swap mutation probability.
  - Categorical replace mutation probability.
- Objective: maximize mean HV or minimize penalties.

## Project References
- `src/mo_module/docs/internal_documentation.md`
- `src/mo_module/docs/tuning.md`
