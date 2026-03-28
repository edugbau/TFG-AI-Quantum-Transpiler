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
- `LayoutMutation`: Combina *swap mutation* (intercambio de posiciones) y *replace mutation* (reemplazo por qubit no usado). A nivel de configuración del módulo, ambas probabilidades se seleccionan de forma categórica para mejorar reproducibilidad y comparabilidad entre experimentos.
- **Validación y reparación**: `validate_layout()` verifica factibilidad; `repair_layout()` corrige duplicados tras crossover.

### 2. Funciones de Fitness (`fitness.py`)
Sistema extensible de funciones objetivo siguiendo el **patrón Strategy**.

**Objetivos activos (basados en transpilación):**
| Clase | Clave registro | Métrica | Minimiza |
|:---|:---|:---|:---|
| `DepthFitness` | `"depth"` | Profundidad del circuito transpilado | Sí |
| `CnotCountFitness` | `"cnot_count"` | Nº de puertas 2Q (CNOTs) tras transpilación | Sí |

Ambas requieren transpilar el circuito con el layout dado; el `TranspilationCache` garantiza que la transpilación se realiza una sola vez por layout aunque se evalúen varios objetivos.

**Extensibilidad — cómo añadir un nuevo objetivo:**
1. Crear una clase que herede de `FitnessFunction` e implemente `evaluate()`.
2. Establecer `name` (clave de log) y `requires_transpilation`.
3. Registrarla en `AVAILABLE_FITNESS_FUNCTIONS` con una clave de cadena.
4. Opcionalmente añadirla a un preset en `PRESET_OBJECTIVES`.

- **FitnessEvaluator** (**patrón Composite**): Agrupa múltiples funciones de fitness y orquesta su evaluación. Gestiona automáticamente el `TranspilationCache`.
- **TranspilationCache**: Caché dict-based que almacena resultados de transpilación usando el layout (como tupla) como clave.
- **Preset activo**: `"default"` → `["depth", "cnot_count"]`.
- **Factory**: Registro de funciones por nombre (`get_fitness_function("depth")`, `get_fitness_function("cnot_count")`).

### 3. Algoritmos Evolutivos (`optimizer.py`)
Orquesta la ejecución de la optimización con pymoo.

- **OptimizerConfig**: Dataclass centralizada con todos los hiperparámetros (algoritmo, población, generaciones, objetivos, probabilidades/categorías de operadores, seed).
- **LayoutOptimizationProblem**: Extiende `pymoo.core.problem.Problem` integrando el `FitnessEvaluator`.
- **Factory de algoritmos**: Crea NSGA-II o MOEA/D según configuración. MOEA/D genera automáticamente los vectores de referencia con `get_reference_directions("uniform", ...)`.
- **Mutación categórica**: `prob_swap_mutation` y `prob_replace_mutation` se validan contra un catálogo discreto de valores permitidos. Esto fija una rejilla experimental pequeña, reproducible y fácil de comparar entre benchmarks y tuning.
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
- `BackendInfo` / `extract_backend_info()`: información del backend (topología, coupling map).
- `transpile_with_custom_layout()`: transpilación con layout personalizado (usada por `TranspilationCache`).
- `extract_metrics()` / `CircuitMetrics`: métricas del circuito transpilado (profundidad, puertas 2Q).
- `get_error_for_layout()`: estadísticas de error hardware; usada en `compare_layouts()` para diagnóstico, no en las funciones de fitness.
- `get_backend()`: instanciar Fake Backends.

### Salida hacia el módulo `rl_module` (Módulo 2):
- Los layouts del frente de Pareto (`OptimizationResult.pareto_layouts`) se pasan como layouts iniciales al agente de RL.
- `get_compromise_layout()` o `get_best_layout()` proporcionan un layout único como semilla para el RL.

### Salida hacia el módulo `integration` (Módulo 4):
- `OptimizationResult.to_dict()` genera datos tabulares para análisis.
- `compare_layouts()` permite comparar MO vs baselines (SABRE, trivial).
- `ParetoMetrics` proporciona indicadores de calidad para reportes.
