# Verificación de Unicidad en la Generación de Layouts

Un layout es una **permutación parcial**: `n` enteros distintos elegidos de los `N` qubits físicos del backend. Todos los operadores del módulo garantizan que los layouts generados nunca contienen enteros repetidos. A continuación se detalla el mecanismo de cada uno.

---

## `LayoutSampling._do` (`encoding.py`)
Usa `numpy.random.Generator.choice` con `replace=False`:
```python
X[i] = rng.choice(pool, size=n, replace=False)
```
El propio parámetro `replace=False` de numpy garantiza que no se repite ningún elemento del pool.

## `LayoutCrossover._ox_child` (`encoding.py`)
Implementa Order Crossover (OX) con seguimiento explícito de qubits usados:
1. Copia un segmento del padre A al hijo y registra los valores en el conjunto `used`.
2. Produce `fill_values` filtrando los genes del padre B que ya están en `used`:
   ```python
   fill_values = [g for g in parent_b if g not in used]
   ```
3. Si tras el OX quedan posiciones sin cubrir (`-1`), se invoca `repair_layout()` como fallback.

## `repair_layout` (`encoding.py`)
Fallback utilizado tras crossover. Recorre el layout, mantiene solo qubits *válidos y únicos* (mediante un conjunto `used`), y rellena huecos con qubits del pool `available_qubits - used`:
```python
available = list(search_space.available_qubits - used)
```
La resta de conjuntos garantiza que los qubits de relleno tampoco se repiten.

## `LayoutMutation._swap_mutate` (`encoding.py`)
Intercambia dos posiciones del array sin modificar el conjunto de valores. Al no añadir ni eliminar elementos, es imposible introducir duplicados.

## `LayoutMutation._replace_mutate` (`encoding.py`)
Calcula explícitamente los qubits no usados antes de elegir el sustituto:
```python
unused = list(self.search_space.available_qubits - used)
new_qubit = rng.choice(unused)
```
Solo se puede seleccionar un qubit que no esté ya presente en el layout.

## `random_layout` (`encoding.py`)
Igual que `LayoutSampling`, delega en `rng.choice(..., replace=False)`.

---

## Resumen

| Operador | Mecanismo de unicidad |
|:---|:---|
| `LayoutSampling._do` | `numpy.choice(..., replace=False)` |
| `LayoutCrossover._ox_child` | Conjunto `used` + filtrado de `fill_values` + `repair_layout` como fallback |
| `repair_layout` | Pool `available_qubits - used` para los huecos |
| `LayoutMutation._swap_mutate` | Intercambio in-place (no altera el conjunto de valores) |
| `LayoutMutation._replace_mutate` | Pool `available_qubits - used` para el qubit sustituto |
| `random_layout` | `numpy.choice(..., replace=False)` |
