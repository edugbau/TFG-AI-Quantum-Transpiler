# Tuning de Hiperparámetros del Optimizador MO

Documentación del módulo `tuning.py`, que implementa el ajuste automático de hiperparámetros de `OptimizerConfig` mediante **Optuna** (Bayesian Optimization con TPE).

---

## ¿Por qué hacer tuning?

`OptimizerConfig` contiene hiperparámetros que afectan directamente a la calidad del frente de Pareto generado por NSGA-II. Elegirlos manualmente es heurístico y subóptimo. Optuna realiza una búsqueda inteligente sobre el espacio de hiperparámetros buscando la configuración que **maximice el hipervolumen** del frente de Pareto.

---

## Hiperparámetros optimizados

| Hiperparámetro | Tipo | Rango por defecto | Descripción |
|---|---|---|---|
| `population_size` | `int` | [20, 100] | Tamaño de la población evolutiva |
| `n_generations` | `int` | [30, 150] | Número de generaciones (presupuesto evolutivo) |
| `prob_swap_mutation` | `categorical` | `[0.1, 0.3, 0.5, 0.7]` | Categoría de mutación por intercambio de qubits |
| `prob_replace_mutation` | `categorical` | `[0.1, 0.3, 0.5, 0.7, 0.9]` | Categoría de mutación por reemplazo de qubit |
| `crossover_operator` | `categorical` | `["dpx", "ox"]` | Operador de cruce (DPX o OX) |
| `algorithm` | `categorical` | `["nsga2"]` | Algoritmo evolutivo (ampliable a `moead`, etc.) |

Las categorías se configuran en `HyperparameterSpace`. Los valores por defecto están calibrados para circuitos de ≤ 10 qubits en backends de tamaño medio (ej. FakeTorino 133q).

Motivación del cambio a categórico:

- hace que el espacio de búsqueda de Optuna sea más estable y revisable;
- evita comparar valores casi idénticos sin significado experimental claro;
- alinea el tuning con un catálogo discreto de configuraciones reproducibles.

---

## Pipeline de tuning

```
┌─────────────────────────────────────────────────────┐
│  LayoutTuner.tune()                                 │
│                                                     │
│  Antes de los trials:                               │
│   - modo calibrated: warm-up/calibración automática │
│     para fijar un session_ref_point conservador     │
│   - modo manual: usa el ref_point dado por usuario  │
│                                                     │
│  Para cada trial (n_trials = 30 por defecto):       │
│   1. Optuna (TPE) sugiere una OptimizerConfig       │
│   2. Se ejecuta optimize_layout() con n_seeds       │
│      semillas distintas (n_seeds = 3 por defecto)   │
│   3. Se calcula el hipervolumen medio del frente    │
│      usando el mismo ref_point fijo de la sesión    │
│   4. Optuna recibe el score → actualiza modelo TPE  │
│                                                     │
│  Al finalizar: best_config() devuelve la config     │
│  con mayor HV medio encontrada.                     │
└─────────────────────────────────────────────────────┘
```

### Función objetivo: Hipervolumen

El **hipervolumen** (HV) mide el volumen del espacio objetivo dominado por el frente de Pareto respecto a un punto de referencia. Un HV mayor indica un frente de mejor calidad (soluciones mejores y más diversas). Optuna **maximiza** esta métrica.

Importante: durante una sesión de tuning ya no se recalcula un `ref_point` por trial. `LayoutTuner` fija un único `session_ref_point` y usa ese mismo valor para comparar **todos** los frentes evaluados en la sesión. Esto mantiene consistente la comparación de HV entre trials.

### Modos de `ref_point`

El tuning soporta dos modos explícitos:

- `calibrated`: antes de lanzar los trials, ejecuta un warm-up automático con varias configuraciones ancla del espacio de búsqueda y construye un `session_ref_point` conservador. Ese valor se muestra de forma explícita en la GUI y queda fijo para todos los trials Optuna de la sesión.
- `manual`: no ejecuta warm-up. El usuario proporciona manualmente `ref_point` al crear `LayoutTuner`, y ese valor fijo se usa en toda la sesión.

En modo `calibrated`, el `ref_point` se obtiene a partir del peor valor observado en las ejecuciones de calibración, con un margen adicional del 30 % (`1.3x + 1e-6`). La idea es que sea deliberadamente conservador para reducir el riesgo de invalidarlo durante los trials reales.

### Penalización por `ref_point` inválido

El `ref_point` fijo debe ser **estrictamente peor** que todo frente de Pareto evaluado en cada objetivo. En este proyecto, como los objetivos (`depth`, `cnot_count`) se minimizan, eso significa que cada coordenada del `ref_point` debe ser estrictamente mayor que el máximo de ese frente en la coordenada correspondiente.

Si durante la sesión aparece un frente cuya coordenada máxima alcanza o supera el `ref_point`, ese frente se penaliza con `HV=0.0` y se registra un warning en logs. Esto puede ocurrir si un `ref_point` manual se elige demasiado ajustado o si incluso el `ref_point` calibrado queda invalidado por un frente posterior peor que los usados en el warm-up. La sesión continúa y el trial conserva un score comparable porque el promedio incorpora ese `0.0` como contribución de la seed afectada.

### Score del trial: promedio por resultado/frente

Cada trial se ejecuta con varias seeds (`n_seeds`). Para cada seed, `optimize_layout()` produce un `OptimizationResult` con su frente de Pareto y se calcula un HV usando el mismo `session_ref_point` fijo. El score final del trial es la media de esas contribuciones individuales; no se selecciona un único frente "ganador" dentro del trial.

### Espacio categórico de mutación

Las probabilidades de mutación ya no se modelan como continuas durante el tuning. En su lugar, Optuna selecciona entre categorías discretas:

- `prob_swap_mutation`: `[0.1, 0.3, 0.5, 0.7]`
- `prob_replace_mutation`: `[0.1, 0.3, 0.5, 0.7, 0.9]`

Esto hace que cada trial sea más interpretable y que la `best_config()` resultante sea directamente reutilizable en benchmarking y producción sin redondeos adicionales.

---

## ⚠️ Valores reducidos durante el tuning (`DEFAULT_EVAL_*`)

Durante la evaluación de cada trial, **no se usan los valores completos** de `population_size` y `n_generations` sugeridos por Optuna. En su lugar, se aplican valores reducidos para que cada trial sea rápido:

| Constante | Valor | Aplicación |
|---|---|---|
| `DEFAULT_EVAL_POPULATION` | 30 | Máximo pop usado en cada trial durante tuning |
| `DEFAULT_EVAL_GENERATIONS` | 50 | Máximo gen usado en cada trial durante tuning |

Esto quiere decir que si Optuna sugiere `population_size=80` para un trial, ese trial se ejecuta con `population_size=min(80, 30)=30`. Sin embargo, el valor `80` queda registrado como parámetro del trial en el estudio Optuna.

**Al finalizar el tuning**, `best_config()` llama a `_trial_to_config()`, que reconstruye la `OptimizerConfig` usando los **valores reales sin reducción** sugeridos por Optuna (ej. `population_size=80`, `n_generations=120`). Esta es la configuración de producción.

```
                  ┌──────────────────────────┐
 Trial sugerido:  │ pop=80, gen=120, swap=0.5│
                  └──────────────┬───────────┘
                                 │
              ┌──────────────────▼────────────────────────┐
              │   _suggest_config()                        │
              │   Evaluación con valores REDUCIDOS:        │
              │   pop = min(80, 30) = 30                   │
              │   gen = min(120, 50) = 50                  │
              │   → rápido, para comparar relativo         │
              └──────────────────┬────────────────────────┘
                                 │ Score HV → Optuna
              ┌──────────────────▼────────────────────────┐
              │   _trial_to_config() (solo al final)       │
              │   Configuración REAL de producción:        │
              │   pop = 80, gen = 120  (sin reducción)     │
              └───────────────────────────────────────────┘
```

> **Implicación**: la comparación entre trials es relativa porque todos usan la misma reducción de presupuesto y el mismo `session_ref_point` fijo, por lo que el ranking de Optuna es válido. La `best_config()` final sí usa los valores completos para producción.

---

## Coste computacional

```
n_trials × n_seeds × NSGA-II(pop=30, gen=50)
    30    ×    3    × ~10s por run ≈ ~15 min
```

Para circuitos de ≤ 5 qubits en FakeTorino. A mayor tamaño del circuito o del backend, el tiempo aumenta proporcionalmente.

---

## Uso básico

```python
from src.mo_module.tuning import LayoutTuner, HyperparameterSpace
from src.qiskit_interface import create_ghz_circuit, get_backend

circuit = create_ghz_circuit(5)
backend = get_backend("fake_torino")

# Personalizar espacio de búsqueda (opcional)
space = HyperparameterSpace(
    population_size_range=(20, 80),
    n_generations_range=(30, 100),
    prob_swap_mutation_choices=(0.1, 0.3, 0.5),
    prob_replace_mutation_choices=(0.3, 0.7, 0.9),
)

tuner = LayoutTuner(
    circuit=circuit,
    backend=backend,
    n_trials=30,   # presupuesto de evaluaciones
    n_seeds=3,     # seeds por evaluación (robustez)
    space=space,
    ref_point_mode="calibrated",  # warm-up automático y ref_point fijo de sesión
)

tuner.tune()
best = tuner.best_config()
print(tuner.summary())
```

Modo manual:

```python
tuner = LayoutTuner(
    circuit=circuit,
    backend=backend,
    n_trials=30,
    n_seeds=3,
    ref_point_mode="manual",
    ref_point=[120.0, 240.0],
)
```

Tras `tune()`, `tuner.session_ref_point` devuelve el `ref_point` efectivo usado en toda la sesión: el manual introducido por el usuario o el calibrado automáticamente en el warm-up.

---

## Acceso al estudio Optuna

`tuner.study` expone el objeto `optuna.Study` completo para análisis avanzado:

```python
import optuna.visualization as vis

fig = vis.plot_optimization_history(tuner.study)
fig.show()

fig2 = vis.plot_param_importances(tuner.study)
fig2.show()
```

---

## GUI de tuning

La interfaz de `benchmark_gui.py` refleja el comportamiento actual del tuner:

- selector de modo `ref_point` entre `calibrated` y `manual`;
- progreso en vivo del tuning;
- fase actual (`preparando`, `warm-up`, `tuning`, `completado`);
- mejor HV observado hasta el momento;
- `ref_point` explícito usado en la sesión.

Los eventos del callback no comparten todos el mismo payload. En particular, `calibration_progress` incluye `current_step`, `total_steps`, `config` y `ref_point_candidate`, mientras que `trial_completed` incluye `score`, `best_score`, `params` y el `ref_point` fijo de la sesión.

En modo `calibrated`, la GUI muestra cuándo empieza la calibración, cuándo termina el warm-up y cuál fue el `ref_point` fijado automáticamente. En modo `manual`, muestra desde el inicio el `ref_point` introducido por el usuario y deja claro que no hay warm-up.

---

## Archivos relacionados

| Archivo | Descripción |
|---|---|
| [`tuning.py`](../tuning.py) | Implementación del `LayoutTuner` y `HyperparameterSpace` |
| [`optimizer.py`](../optimizer.py) | `OptimizerConfig` y `optimize_layout()` que el tuner evalúa |
| [`benchmark/benchmark_gui.py`](../benchmark/benchmark_gui.py) | GUI que integra el tuning |
