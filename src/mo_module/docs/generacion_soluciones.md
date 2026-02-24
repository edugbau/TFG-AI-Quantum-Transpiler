# Generación de Nuevas Soluciones en el Algoritmo Evolutivo

Este documento describe en detalle cómo el módulo `mo_module` genera nuevas soluciones (layouts) durante la ejecución del algoritmo evolutivo multiobjetivo.

Recuerda que un **layout** es una permutación parcial: un array de `n` enteros distintos donde `layout[i]` es el qubit físico asignado al qubit lógico `i`. En todo momento se garantiza que no hay enteros repetidos (ver `verificacion_unicidad.md`).

---

## 1. Generación Inicial — `LayoutSampling._do`

Al arrancar el algoritmo, pymoo invoca el operador de sampling para construir la **población inicial** de `pop_size` individuos. Cada layout se genera así:

```python
X[i] = rng.choice(pool, size=n, replace=False)
```

- `pool` = array con todos los qubits físicos disponibles del backend.
- `n` = número de qubits lógicos del circuito.
- `replace=False` garantiza que no se repite ningún qubit en el layout.

El resultado es una matriz `(pop_size, n)` con `pop_size` layouts independientes y aleatorios.

**Archivo**: `encoding.py` → clase `LayoutSampling`, método `_do`.

---

## 2. Generación por Crossover — `LayoutCrossover._do`

En cada generación, pymoo selecciona pares de padres y aplica el operador de cruce para producir hijos. Se usa **Order Crossover (OX)** adaptado a permutaciones parciales.

### Selección de padres
- **NSGA-II**: torneo binario (gestiona pymoo internamente).
- **MOEA/D**: vecindad por vectores de referencia (gestiona pymoo internamente).

### Procedimiento OX (`_ox_child`)

Dados el padre A (donante del segmento) y el padre B (donante del relleno):

1. Se eligen dos puntos de corte aleatorios `start` y `end`.
2. El hijo copia el segmento `parent_a[start:end]` y registra esos valores en `used`.
3. Se construye la lista de genes de relleno filtrando los del padre B:
   ```python
   fill_values = [g for g in parent_b if g not in used]
   ```
4. Las posiciones fuera del segmento se rellenan con `fill_values` en orden (`end → n-1`, luego `0 → start-1`).
5. Si quedan posiciones en `-1` (ambos padres comparten pocos genes comunes), se invoca `repair_layout()` como fallback.

Se genera el hijo 2 simétricamente intercambiando los roles de padre A y padre B.

```
Padre A:  [3, 1, 4, | 2, 0 | , 5]
Padre B:  [0, 5, 2, | 4, 1 | , 3]
                    ^start end^

Hijo 1:   [_, _, _, | 2, 0 | , _]  ← segmento de A
           rellenar con B sin {2,0}: [0, 5, 4, 1, 3]
Hijo 1:   [5, 4, 1, | 2, 0 | , 3]
```

**Archivo**: `encoding.py` → clase `LayoutCrossover`, métodos `_do` y `_ox_child`.

---

## 3. Generación por Mutación — `LayoutMutation._do`

Tras el crossover, cada hijo puede sufrir uno o ambos tipos de mutación, controlados por probabilidades independientes:

### 3a. Swap Mutation (`prob_swap`, por defecto 0.5)

Intercambia dos posiciones aleatorias del layout:

```python
i, j = rng.choice(n, size=2, replace=False)
result[i], result[j] = result[j], result[i]
```

- No altera el conjunto de qubits usados, solo cambia la asignación lógico→físico.
- Intensificación: explora asignaciones dentro del mismo subconjunto de qubits.

### 3b. Replace Mutation (`prob_replace`, por defecto 0.3)

Sustituye un qubit del layout por otro qubit físico no utilizado:

```python
unused = list(self.search_space.available_qubits - used)
pos = rng.integers(0, len(result))
new_qubit = rng.choice(unused)
result[pos] = new_qubit
```

- Cambia el subconjunto de qubits usado.
- Diversificación: explora regiones distintas del chip cuántico.

Ambas mutaciones pueden aplicarse al mismo individuo en la misma generación (son independientes).

**Archivo**: `encoding.py` → clase `LayoutMutation`, métodos `_do`, `_swap_mutate` y `_replace_mutate`.

---

## 4. Selección de Supervivientes

Pymoo evalúa el fitness de los hijos generados y decide qué individuos pasan a la siguiente generación:

- **NSGA-II**: ordena la población combinada (padres + hijos) por rango de no-dominancia y, dentro del mismo rango, por crowding distance.
- **MOEA/D**: cada subproblema (vector de referencia) mantiene al mejor individuo de su vecindad según la escalarización de Chebyshev.

Este paso lo gestiona pymoo internamente; no hay código propio del proyecto para ello.

---

## 5. Resumen del Ciclo por Generación

```
Población actual (pop_size individuos)
        │
        ▼
Selección de padres (torneo / vecindad)
        │
        ▼
LayoutCrossover._do → hijos con OX
        │
        ▼
LayoutMutation._do  → swap y/o replace
        │
        ▼
LayoutOptimizationProblem._evaluate → vector de fitness
        │
        ▼
Selección de supervivientes (NSGA-II / MOEA/D)
        │
        ▼
Nueva población (pop_size individuos)
```

Se repite durante `n_generations` generaciones. Al finalizar, pymoo extrae el frente de Pareto y el optimizador lo empaqueta en `OptimizationResult`.

---

## 6. Hiperparámetros Relevantes (`OptimizerConfig`)

| Parámetro | Valor por defecto | Efecto |
|:---|:---|:---|
| `population_size` | 50 | Número de individuos por generación |
| `n_generations` | 100 | Criterio de parada |
| `prob_crossover` | 0.9 | Probabilidad de que un par de padres haga crossover |
| `prob_swap_mutation` | 0.5 | Probabilidad de swap mutation por individuo |
| `prob_replace_mutation` | 0.3 | Probabilidad de replace mutation por individuo |
| `algorithm` | `"nsga2"` | Algoritmo evolutivo (`"nsga2"` o `"moead"`) |

**Archivo**: `optimizer.py` → dataclass `OptimizerConfig`.
