# Skill: MO Optimization (Multi-Objective)
**Contexto para el Módulo 3 (`mo_module`).**

## Objetivo
Estandarizar el diseño y evaluación de algoritmos evolutivos multiobjetivo para el Layout Cuántico.

## Librería Principal
- **pymoo**: Usar la API de `pymoo` para la definición de problemas y ejecución de NSGA-II o MOEA/D.

## Reglas de Implementación

1. **Definición del Problema (`pymoo.core.problem.Problem`)**
   - **Genotipo (Representación):** El individuo (cromosoma) debe representar una permutación de enteros o un mapeo directo de qubits lógicos a físicos (`integer` o `permutation` variables).
   - **Evaluación (Fitness):** Minimización de los múltiples objetivos. Por defecto:
     - Profundidad (`depth`) del circuito transpilado.
     - Equivalente de CNOTs (o recuento de puertas nativas bi-qubit).

2. **Operadores Genéticos Recomendados**
   - **Cruce (Crossover):** Usar **DPX (Dynastic Potential Crossover)** por defecto. Preserva mejor las asignaciones de layout comunes entre padres (baja epistasis).
   - **Mutación (Mutation):** Intercambio de posiciones (SWAP mutator).

3. **Análisis de Resultados y Frente de Pareto**
   - Extraer individuos no-dominados de la ejecución de `pymoo`.
   - Utilizar Hypervolume (HV) como métrica principal para comparar calidades del frente.

4. **Tuning de Hiperparámetros (Optuna)**
   - Utilizar el módulo `optuna` (TPE) para optimizar el tamaño de la población, tasa de cruce, y tasa de mutación, maximizando la media del Hipervolumen (o minimizando penalizaciones).
