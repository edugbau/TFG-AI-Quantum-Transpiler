# Documentación Interna del Módulo `mo_module`

Este documento detalla las funcionalidades, pipelines, patrones de diseño y funciones clave del módulo `mo_module`. Este módulo implementa la optimización multiobjetivo de layouts de qubits mediante algoritmos evolutivos (NSGA-II, MOEA/D), generando un frente de Pareto de soluciones que equilibran múltiples criterios de calidad.

## Funcionalidades Principales

### 1. Codificación del Layout (`encoding.py`)
Define cómo se representan los layouts como individuos dentro del algoritmo evolutivo.

- **Representación**: Un layout es un array de enteros de longitud `n_logical_qubits`, donde cada posición `i` contiene el qubit físico asignado al qubit lógico `i`. Es una *permutación parcial* (se seleccionan `n` qubits de entre los `N` del backend).
- **LayoutSearchSpace**: Dataclass que encapsula las restricciones del espacio de búsqueda (número de qubits lógicos/físicos, qubits disponibles, aristas del coupling map).
- **Operadores genéticos custom para pymoo**:
  - `LayoutSampling`: Genera individuos iniciales seleccionando subconjuntos aleatorios de qubits físicos.
  - `LayoutCrossover`: Order Crossover (OX) adaptado a permutaciones parciales.
  - `LayoutMutation`: Combina *swap mutation* (intercambio de posiciones) y *replace mutation* (reemplazo por qubit no usado).
- **Validación y reparación**: `validate_layout()` verifica factibilidad; `repair_layout()` corrige duplicados tras crossover.

### 2. Funciones de Fitness (`fitness.py`)
Sistema extensible de funciones objetivo siguiendo el **patrón Strategy**.

**Categoría 1 — Basadas en hardware (sin transpilación, rápidas):**
| Función | Métrica | Minimiza |
|:---|:---|:---|
| `ErrorRateFitness` | Error promedio de puertas 2Q en el layout | Sí |
| `MaxErrorRateFitness` | Error máximo de puertas 2Q | Sí |
| `DecoherenceFitness` | −T2 promedio (negado para minimizar) | Sí (= maximiza T2) |
| `ConnectivityFitness` | −nº aristas del coupling map (negado) | Sí (= maximiza conectividad) |

**Categoría 2 — Basadas en transpilación (requieren `qiskit.transpile`):**
| Función | Métrica | Minimiza |
|:---|:---|:---|
| `DepthFitness` | Profundidad del circuito transpilado | Sí |
| `TwoQubitGateFitness` | Nº de puertas 2Q tras transpilación | Sí |
| `TotalGateFitness` | Nº total de puertas tras transpilación | Sí |

- **FitnessEvaluator** (**patrón Composite**): Agrupa múltiples funciones de fitness y orquesta su evaluación. Gestiona automáticamente el `TranspilationCache` para evitar transpilaciones redundantes.
- **TranspilationCache**: Caché dict-based que almacena resultados de transpilación usando el layout (como tupla) como clave.
- **Presets**: Configuraciones predefinidas de objetivos (`hardware_only`, `transpilation_basic`, `balanced`, etc.).
- **Factory**: Registro de funciones de fitness por nombre (`get_fitness_function("depth")`).

### 3. Algoritmos Evolutivos (`optimizer.py`)
Orquesta la ejecución de la optimización con pymoo.

- **OptimizerConfig**: Dataclass centralizada con todos los hiperparámetros (algoritmo, población, generaciones, objetivos, probabilidades de operadores, seed).
- **LayoutOptimizationProblem**: Extiende `pymoo.core.problem.Problem` integrando el `FitnessEvaluator`.
- **Factory de algoritmos**: Crea NSGA-II o MOEA/D según configuración. MOEA/D genera automáticamente los vectores de referencia con `get_reference_directions("uniform", ...)`.
- **OptimizationResult**: Dataclass con el frente de Pareto completo (layouts, fitness), metadatos (tiempo, generaciones, caché) y métodos de selección (`get_best_layout`, `get_compromise_layout`).
- **compare_layouts()**: Compara múltiples layouts transpilando con cada uno (útil para contrastar MO vs SABRE vs heurísticas).

### 4. Análisis del Frente de Pareto (`pareto.py`)
Herramientas para evaluar la calidad del frente y seleccionar soluciones.

- **ParetoMetrics**: Hipervolumen (HV), spacing, spread, punto ideal/nadir.
- **Estrategias de selección**:
  - `select_knee_point()`: Punto de rodilla (mayor cambio en trade-off marginal).
  - `select_weighted()`: Suma ponderada escalarizada.
  - `select_min_objective()`: Mejor en un objetivo concreto.
- **Visualización**: Scatter 2D/3D del frente, coordenadas paralelas (matplotlib).

## Patrones de Diseño Aplicados

| Patrón | Dónde | Propósito |
|:---|:---|:---|
| **Strategy** | `FitnessFunction` (ABC) + implementaciones concretas | Funciones de fitness intercambiables y extensibles |
| **Composite** | `FitnessEvaluator` | Agrupa múltiples estrategias de fitness en un evaluador único |
| **Factory** | `get_fitness_function()`, `create_algorithm()` | Instanciación por nombre para configuración declarativa |
| **Template Method** | `LayoutOptimizationProblem._evaluate()` | pymoo invoca la evaluación de fitness como paso del algoritmo |
| **Dataclass como DTO** | `OptimizerConfig`, `OptimizationResult`, `ParetoMetrics` | Transportar datos estructurados entre capas |

## Pipelines (Flujos de Trabajo)

### A. Pipeline de Optimización Completo
Flujo principal ejecutado por `optimize_layout()`:

```
Circuito + Backend
       │
       ▼
┌─────────────────────┐
│ 1. extract_backend_  │  Obtener BackendInfo (topología, errores, T1/T2)
│    info()            │  [Usa: qiskit_interface.backend_info]
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. LayoutSearch-     │  Configurar espacio de búsqueda
│    Space.from_       │  (n_logical, n_physical, available_qubits)
│    backend_info()    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. FitnessEvaluator │  Crear evaluador compuesto con los objetivos
│    .from_names()     │  + TranspilationCache si hay obj. de transpilación
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. LayoutOptimiz-    │  Definir problema pymoo (n_var, n_obj, bounds)
│    ationProblem()    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. create_algorithm()│  NSGA-II o MOEA/D con operadores custom
│    (Factory)         │  (LayoutSampling, LayoutCrossover, LayoutMutation)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 6. pymoo_minimize()  │  Ejecutar algoritmo evolutivo
│    (generaciones)    │  Cada generación: sampling/crossover/mutation → eval
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 7. OptimizationResult│  Empaquetar frente de Pareto + metadatos
└─────────────────────┘
```

### B. Pipeline de Evaluación de un Individuo
Flujo ejecutado internamente por pymoo para cada individuo:

1. **Layout** (array de ints) pasa a `FitnessEvaluator.evaluate()`.
2. Si hay objetivos de transpilación → consultar `TranspilationCache`:
   - **Cache hit**: reusar resultado.
   - **Cache miss**: `transpile_with_custom_layout()` → almacenar.
3. Cada `FitnessFunction.evaluate()` recibe el layout, `BackendInfo`, y opcionalmente `TranspilationResult`.
4. Se retorna el vector `[f1, f2, ..., fn]` a pymoo.

### C. Pipeline de Análisis Post-Optimización
Flujo ejecutado por `analyze_pareto_front()`:

1. Recibir `OptimizationResult`.
2. Calcular `ParetoMetrics` (HV, spacing, spread, ideal/nadir).
3. Identificar soluciones clave (knee point, mejores por objetivo, compromiso).
4. Opcionalmente visualizar con `plot_pareto_front_2d()` / `plot_parallel_coordinates()`.

## Funciones Llamadas Extensamente

| Función | Archivo | Descripción | Uso |
|:---|:---|:---|:---|
| **`optimize_layout`** | `optimizer.py` | Función principal del módulo | Orquesta todo el pipeline de optimización |
| **`FitnessEvaluator.evaluate`** | `fitness.py` | Evalúa vector de fitness de un layout | Llamada por pymoo en cada evaluación de individuo |
| **`TranspilationCache.get`** | `fitness.py` | Caché de transpilación | Llamada internamente por FitnessEvaluator para cada layout con obj. de transpilación |
| **`validate_layout`** | `encoding.py` | Verifica factibilidad de un layout | Usada en operadores genéticos y en tests |
| **`repair_layout`** | `encoding.py` | Corrige layouts inválidos | Usada después del crossover si genera duplicados |
| **`compute_pareto_metrics`** | `pareto.py` | Calcula HV, spacing, etc. | Usada en análisis post-optimización |
| **`select_knee_point`** | `pareto.py` | Selecciona punto de rodilla | Usada para recomendar la solución de mayor trade-off |

## Integración con Otros Módulos

### Dependencias del módulo `qiskit_interface` (Módulo 1):
- `BackendInfo` / `extract_backend_info()`: información del backend (topología, errores).
- `get_error_for_layout()`: estadísticas de error para un layout (usado por fitness hardware).
- `transpile_with_custom_layout()`: transpilación con layout personalizado (usado por fitness de transpilación).
- `extract_metrics()` / `CircuitMetrics`: métricas del circuito transpilado.
- `get_backend()`: instanciar Fake Backends.

### Salida hacia el módulo `rl_module` (Módulo 2):
- Los layouts del frente de Pareto (`OptimizationResult.pareto_layouts`) se pasan como layouts iniciales al agente de RL.
- `get_compromise_layout()` o `get_best_layout()` proporcionan un layout único como semilla para el RL.

### Salida hacia el módulo `integration` (Módulo 4):
- `OptimizationResult.to_dict()` genera datos tabulares para análisis.
- `compare_layouts()` permite comparar MO vs baselines (SABRE, trivial).
- `ParetoMetrics` proporciona indicadores de calidad para reportes.
