# mo_module

`mo_module` implementa la busqueda multiobjetivo de layouts de qubits. Su funcion es producir candidatos de layout, analizarlos con Pareto y exponer una seleccion utilizable por `integration` y, de forma local, por tooling de benchmark.

## Estructura publica

| Archivo | Proposito | Simbolos clave |
| --- | --- | --- |
| `encoding.py` | Representacion del layout y operadores geneticos | `LayoutSearchSpace`, `LayoutSampling`, `LayoutCrossover`, `DPXCrossover`, `LayoutMutation`, `validate_layout`, `repair_layout` |
| `fitness.py` | Funciones objetivo y cache de transpilacion | `FitnessFunction`, `DepthFitness`, `CnotCountFitness`, `TranspilationCache`, `FitnessEvaluator`, `get_preset_objectives` |
| `optimizer.py` | Orquestacion del algoritmo evolutivo | `OptimizerConfig`, `LayoutOptimizationProblem`, `OptimizationResult`, `create_algorithm`, `optimize_layout`, `optimize_layout_quick`, `compare_layouts` |
| `pareto.py` | Analisis y seleccion de soluciones | `ParetoMetrics`, `compute_pareto_metrics`, `select_knee_point`, `select_weighted`, `select_min_objective`, `analyze_pareto_front` |
| `tuning.py` | Ajuste de hiperparametros con Optuna | `HyperparameterSpace`, `LayoutTuner` |
| `benchmark/` | Benchmarks reproducibles y analisis estadistico | `BenchmarkCircuit`, `BenchmarkRunner`, `BenchmarkResultSet`, `BenchmarkReport`, `run_benchmark`, `run_layout_selection_campaign`, `summarize_layout_campaign` |
| `benchmark/layout_campaigns.py` | Tooling experimental local para comparar layouts | `build_layout_campaign_spec`, `build_reference_layouts`, `run_layout_selection_campaign` |

## Flujo general

1. `qiskit_interface` aporta el backend, el circuito y las metricas iniciales.
2. `encoding.py` define el espacio de busqueda: un layout es una permutacion parcial logico -> fisico.
3. `fitness.py` transpila cada candidato y lo evalua sobre los objetivos activos.
4. `optimizer.py` ejecuta el algoritmo evolutivo y devuelve un `OptimizationResult`.
5. `pareto.py` interpreta el frente y selecciona un layout unico cuando hace falta.
6. `integration` consume esos layouts para `MO_Only` o para el camino hibrido `MO+RL`.

## Decisiones de diseno

- El layout se modela como permutacion parcial, no como asignacion libre, para mantener factibilidad y facilitar la reparacion.
- Los operadores geneticos son especificos para `pymoo` y trabajan con layouts parciales, no con cromosomas genericos.
- La evaluacion de fitness se apoya en `transpile_with_custom_layout`, de modo que los objetivos reflejan el efecto real del layout sobre Qiskit.
- `TranspilationCache` evita repetir la misma transpilacion cuando varios objetivos usan el mismo layout.
- `LayoutTuner` fija un `session_ref_point` para que el hipervolumen sea comparable dentro de toda la sesion.
- `run_layout_selection_campaign` y `benchmark/` son tooling local de evaluacion; no hacen el handoff MO -> RL.

## Limites y alcance

- `mo_module` no dirige RL ni define escenarios de `integration`.
- `layout_campaigns.py` compara layouts y referencias locales, pero no es un puente de orquestacion hacia `rl_module`.
- `benchmark` y `tuning` son soportes de experimentacion y analisis; la narrativa principal del modulo vive en este README y en `docs/internal_documentation.md`.

## Documentacion relacionada

- [docs/internal_documentation.md](docs/internal_documentation.md): doc tecnico canonic del modulo.
- [docs/tuning.md](docs/tuning.md): apunte de tuning con Optuna.
- [docs/benchmark_documentation.md](docs/benchmark_documentation.md): detalle del submodulo de benchmark.
- [docs/analisis_resultados.md](docs/analisis_resultados.md): analisis y resultados experimentales.
- [docs/generacion_soluciones.md](docs/generacion_soluciones.md): detalle de generacion de soluciones.

