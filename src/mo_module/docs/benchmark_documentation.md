# Documentación Interna del Submódulo `benchmark`

Este documento detalla las funcionalidades, pipelines, patrones de diseño y decisiones técnicas del submódulo `mo_module.benchmark`. El submódulo permite evaluar estadísticamente el algoritmo de optimización multiobjetivo ejecutándolo sobre múltiples circuitos y semillas, obteniendo métricas de calidad y estabilidad reproducibles.

## Funcionalidades Principales

### 1. Suite de Circuitos de Prueba (`circuits.py`)

Define un catálogo de circuitos de benchmark con características estructurales distintas, de forma que los resultados del optimizador se evalúen sobre casos variados y representativos.

- **`BenchmarkCircuit`**: Dataclass *frozen* que encapsula un nombre, descripción, etiquetas y un `factory` callable que genera el `QuantumCircuit` a demanda. La generación diferida evita tener instancias vivas en memoria hasta que se necesiten.
- **Suite por defecto** (`DEFAULT_BENCHMARK_CIRCUITS`):

| Nombre | Circuito | Qubits | Característica |
|:---|:---|:---:|:---|
| `ghz_5q` | GHZ de 5 qubits | 5 | Cadena lineal de CNOTs, muy sensible al routing |
| `qft_4q` | QFT de 4 qubits | 4 | O(n²) puertas 2Q densas; buen test de compromiso |
| `random_4q_d10` | Circuito aleatorio | 4 | Estructura impredecible; evalúa robustez |
| `clifford_4q` | Clifford aleatorio | 4 | Relevante para el módulo RL; puertas H/S/CX |

- **Filtrado por tag**: `get_circuits_by_tag("clifford")` permite obtener subconjuntos tematizados.
- **Circuitos custom**: `make_custom_circuit(name, qc)` convierte cualquier `QuantumCircuit` en un `BenchmarkCircuit` compatible con el runner.

### 2. Motor de Ejecución (`runner.py`)

Orquesta la ejecución del optimizador sobre el producto cartesiano circuitos × semillas.

- **`BenchmarkRun`**: Resultado atómico de una ejecución (circuito + seed). Contiene el `OptimizationResult` o un mensaje de error si la ejecución falló. La captura de excepciones garantiza que un fallo individual no interrumpe el benchmark.
- **`BenchmarkResultSet`**: Colección de todos los `BenchmarkRun`. Proporciona:
  - Filtros: `runs_for_circuit(name)`, `circuit_names`.
  - Extracción numérica: `fitness_matrix()`, `best_per_seed()`, `elapsed_per_seed()`, `pareto_sizes()`.
  - `summary()`: resumen estadístico en texto plano por circuito.
- **`BenchmarkRunner`**: Dataclass configurable con circuitos, semillas, backend y un `OptimizerConfig` base. La semilla del `OptimizerConfig` se sobreescribe en cada ejecución para garantizar que cada par (circuito, seed) es independiente. El backend se crea una sola vez y se reutiliza.

**Configuración por defecto** (orientada a exploración inicial):
```python
OptimizerConfig(
    algorithm="nsga2",
    population_size=30,
    n_generations=50,
    objectives=["depth", "cnot_count"],
    verbose=False,
)
```

### 3. Análisis Estadístico (`analysis.py`)

Transforma los resultados crudos del runner en estadísticas descriptivas estructuradas.

- **`ObjectiveStats`**: Estadísticas sobre el **mejor valor** alcanzado por objetivo a través de las semillas: media, std, mediana, IQR (rango intercuartílico), mín/máx, CV (coeficiente de variación en %). El CV permite comparar la variabilidad del algoritmo entre circuitos con escalas distintas.
- **`CircuitAnalysis`**: Análisis completo de un circuito: estadísticas por objetivo, estadísticas de tiempo de ejecución y tamaño del frente de Pareto. Incluye el resultado del test de Kruskal-Wallis para evaluar si las semillas producen distribuciones estadísticamente homogéneas.
- **`BenchmarkReport`**: Colección de `CircuitAnalysis` con métodos `to_text()` (para consola) y `to_dict()` (para JSON/pandas).
- **Test de Kruskal-Wallis** (`_seed_stability_test`): Compara las distribuciones del primer objetivo entre grupos de semillas. Un p-valor > 0.05 indica que el algoritmo es *estable* (semillas distintas producen resultados estadísticamente similares). Se usa Kruskal-Wallis en lugar de ANOVA porque no asume normalidad.

## Patrones de Diseño Aplicados

| Patrón | Dónde | Propósito |
|:---|:---|:---|
| **Factory (método `create`)** | `BenchmarkCircuit.create()` | Generación diferida del circuito; permite circuitos custom sin serializar `QuantumCircuit` |
| **Dataclass como DTO** | `BenchmarkRun`, `BenchmarkResultSet`, `BenchmarkReport` | Transportar datos entre capas sin dependencias circulares |
| **Facade** | `run_benchmark()` | Una sola función de entrada para el caso de uso habitual; oculta la composición `Runner → ResultSet → Report` |
| **Dataclass frozen** | `BenchmarkCircuit` | Inmutabilidad: el descriptor de un circuito no debe cambiar tras su definición |

## Pipelines (Flujos de Trabajo)

### A. Pipeline de Benchmark Completo

```
Configuración
(circuits, seeds, backend, config)
           │
           ▼
┌──────────────────────┐
│ BenchmarkRunner.run() │
│  ┌────────────────────┤
│  │  Para cada circuito│
│  │    Para cada seed  │
│  │      optimize_     │  optimize_layout(circuit, backend, config(seed=seed))
│  │      layout()      │  → OptimizationResult
│  │      └─ BenchmarkRun (ok / error)
│  └────────────────────┤
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  BenchmarkResultSet  │  Colección de BenchmarkRuns
│  .summary()          │  Resumen estadístico rápido
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  analyze_results()   │  Estadísticas descriptivas + Kruskal-Wallis
│  → BenchmarkReport   │  to_text() / to_dict()
└──────────────────────┘
```

### B. Pipeline de Análisis por Circuito

Para cada circuito en `BenchmarkResultSet`:

1. Filtrar ejecuciones exitosas con `runs_for_circuit(name)`.
2. Para cada objetivo, extraer el mejor valor de cada semilla (`best_per_seed()`).
3. Calcular `ObjectiveStats` (media, std, mediana, IQR, CV).
4. Calcular estadísticas de tiempo (`elapsed_per_seed()`) y tamaño del frente (`pareto_sizes()`).
5. Ejecutar `_seed_stability_test()`: grupos de valores del frente de Pareto → Kruskal-Wallis → p-valor.
6. Empaquetar en `CircuitAnalysis`.

## Decisiones de Diseño

### ¿Por qué el runner itera sobre semillas y no sobre circuitos en el bucle exterior?
El bucle exterior es circuitos y el interior semillas. Esto permite que el benchmark continúe produciendo resultados parciales útiles (todos los resultados del circuito 1) si se interrumpe, en lugar de obtener solo 1 semilla de cada circuito.

### ¿Por qué la semilla del `OptimizerConfig` base se sobreescribe en lugar de ignorarse?
Se crea un `OptimizerConfig` nuevo por cada ejecución con `seed=seed` para que sea un objeto independiente e inmutable. Esto evita efectos colaterales si el config base se modifica externamente durante la ejecución.

### ¿Por qué se usa la mediana del frente en `fitness_matrix()` en lugar del mínimo?
El mínimo favorece outliers extremos y no representa bien la calidad del frente completo. La mediana es más robusta y refleja la distribución típica de las soluciones del frente de Pareto para cada semilla.

### ¿Por qué Kruskal-Wallis y no ANOVA?
Los valores de fitness del frente de Pareto no tienen distribución normal garantizada (especialmente con frentes pequeños o circuitos pequeños). Kruskal-Wallis es no paramétrico y válido sin ese supuesto.

### ¿Por qué `BenchmarkCircuit` es frozen?
Un descriptor de circuito es conceptualmente constante: su nombre, descripción y factory no deben cambiar una vez definido. El frozen dataclass lo garantiza en tiempo de ejecución y facilita su uso como clave de diccionario si fuera necesario.

## Guía de Uso

### Exploración rápida (10 semillas)
```python
from src.mo_module.benchmark import run_benchmark, analyze_results

results = run_benchmark(n_seeds=10)
report  = analyze_results(results)
print(report.to_text())
```

### Análisis inicial completo (30 semillas)
```python
results = run_benchmark(n_seeds=30)
report  = analyze_results(results)
# Volcar a dict para pandas
import pandas as pd
df = pd.DataFrame(report.to_dict()["rows"])
print(df)
```

### Benchmark personalizado
```python
from src.mo_module.benchmark import BenchmarkRunner, get_default_circuits, make_custom_circuit
from src.mo_module import OptimizerConfig
from qiskit import QuantumCircuit

# Añadir un circuito propio
qc = QuantumCircuit(5, name="mi_circuito")
# ... construir qc ...

circuits = get_default_circuits()
circuits.append(make_custom_circuit("mi_circuito_5q", qc))

runner = BenchmarkRunner(
    circuits=circuits,
    seeds=list(range(30)),
    backend_name="fake_torino",
    config=OptimizerConfig(
        population_size=50,
        n_generations=100,
        verbose=False,
    ),
)
results = runner.run()
print(results.summary())
```

## Funciones Llamadas Extensamente

| Función | Archivo | Descripción | Uso |
|:---|:---|:---|:---|
| **`BenchmarkRunner.run`** | `runner.py` | Motor principal del benchmark | Punto de entrada para toda sesión de evaluación |
| **`optimize_layout`** | `optimizer.py` | Optimizador MO | Llamada una vez por par (circuito, seed) |
| **`BenchmarkResultSet.best_per_seed`** | `runner.py` | Extrae mejores valores por semilla | Usada internamente por `analyze_results` para cada objetivo |
| **`compute_objective_stats`** | `analysis.py` | Estadísticas descriptivas | Llamada por `analyze_results` para cada objetivo de cada circuito |
| **`analyze_results`** | `analysis.py` | Genera informe completo | Punto de entrada del análisis post-ejecución |
| **`run_benchmark`** | `__init__.py` | Facade de una sola llamada | Para uso interactivo y notebooks |

## Integración con el Resto del Módulo

- **Entrada desde `optimizer.py`**: `BenchmarkRunner` llama a `optimize_layout()` y consume `OptimizationResult` (layouts, `pareto_fitness`, `elapsed_time_s`, `objective_names`).
- **Entrada desde `qiskit_interface`**: Los circuitos de la suite usan `create_ghz_circuit`, `create_qft_circuit`, `create_random_circuit` y `create_clifford_circuit` de `circuit_utils.py`.
- **Sin salida directa hacia otros módulos**: El submódulo es tooling experimental local al módulo MO; sus resultados se consumen externamente (notebooks, scripts de análisis, módulo de integración).
- **Límite con RL**: Las layout campaigns no actúan como puente de orquestación hacia `rl_module`; cualquier handoff MO -> RL pertenece exclusivamente a `src/integration/`.
