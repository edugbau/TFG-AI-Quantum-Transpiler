"""
encoding.py — Codificación del layout como individuo evolutivo
===============================================================

Módulo 3 del TFG: Optimización Multiobjetivo.

Este fichero define la representación (codificación) de un layout de
qubits como un individuo dentro de un algoritmo evolutivo. También
proporciona los operadores genéticos personalizados (sampling, crossover,
mutación) adaptados a la naturaleza combinatoria del problema de layout.

Contexto del problema:
  Un **layout** es un mapeo de qubits lógicos (del circuito) a qubits
  físicos (del backend). Si el circuito tiene ``n`` qubits lógicos y el
  backend tiene ``N`` qubits físicos (con ``N >> n``), un layout es una
  selección *sin repetición* de ``n`` qubits de entre los ``N``
  disponibles, es decir, una permutación parcial.

  Representamos cada individuo como un array de enteros de longitud ``n``
  donde ``layout[i]`` = qubit físico asignado al qubit lógico ``i``.
  La restricción es que todos los valores deben ser **distintos** y
  estar en el rango ``[0, N)``.

Decisiones de diseño:
  1. **Permutación parcial** — No es una permutación completa (``n < N``).
     Los operadores genéticos deben respetar la restricción de
     unicidad de valores.

  2. **Operadores personalizados para pymoo** — pymoo soporta operadores
     custom extendiendo ``Sampling``, ``Crossover`` y ``Mutation``.
     Implementamos:
       - ``LayoutSampling``: muestreo aleatorio de subconjuntos de
         qubits físicos.
       - ``LayoutCrossover``: Order Crossover (OX) adaptado a
         permutaciones parciales.
       - ``LayoutMutation``: mutación por intercambio (swap) y por
         reemplazo con qubit físico no usado.

  3. **Validación** — Se incluyen funciones de validación para verificar
     que un layout cumple las restricciones del backend (qubits válidos,
     sin duplicados).

  4. **Compatibilidad con BackendInfo** — Las funciones aceptan un
     ``BackendInfo`` del módulo ``qiskit_interface`` para obtener el
     número de qubits del backend y las aristas del coupling map.

Autor: Eduardo González Bautista
Fecha: 2026-02-18
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

# ---------------------------------------------------------------------------
# Imports internos del proyecto
# ---------------------------------------------------------------------------
from src.qiskit_interface.backend_info import BackendInfo

# ---------------------------------------------------------------------------
# Logger del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
#  Dataclass de configuración del espacio de búsqueda
# ===========================================================================

@dataclass
class LayoutSearchSpace:
    """Define el espacio de búsqueda para la optimización de layouts.

    Encapsula la información necesaria para que los operadores genéticos
    generen y manipulen layouts válidos.

    Attributes:
        num_logical_qubits:
            Número de qubits lógicos del circuito a mapear (``n``).
        num_physical_qubits:
            Número total de qubits del backend (``N``).
        available_qubits:
            Conjunto de qubits físicos disponibles para el mapeo.
            Normalmente es ``{0, 1, ..., N-1}``, pero podría excluir
            qubits defectuosos en el futuro.
        coupling_edges:
            Lista de aristas del coupling map ``(qi, qj)``.
    """

    num_logical_qubits: int
    num_physical_qubits: int
    available_qubits: set[int] = field(default_factory=set)
    coupling_edges: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        """Inicializa available_qubits si no se proporcionó."""
        if not self.available_qubits:
            self.available_qubits = set(range(self.num_physical_qubits))

    @classmethod
    def from_backend_info(
        cls,
        backend_info: BackendInfo,
        num_logical_qubits: int,
    ) -> LayoutSearchSpace:
        """Crea un LayoutSearchSpace a partir de un BackendInfo.

        Args:
            backend_info: Información del backend destino.
            num_logical_qubits: Número de qubits lógicos del circuito.

        Returns:
            LayoutSearchSpace configurado para el backend dado.

        Raises:
            ValueError: Si el circuito tiene más qubits que el backend.
        """
        if num_logical_qubits > backend_info.num_qubits:
            raise ValueError(
                f"El circuito tiene {num_logical_qubits} qubits lógicos "
                f"pero el backend '{backend_info.name}' solo tiene "
                f"{backend_info.num_qubits} qubits físicos."
            )

        return cls(
            num_logical_qubits=num_logical_qubits,
            num_physical_qubits=backend_info.num_qubits,
            available_qubits=set(range(backend_info.num_qubits)),
            coupling_edges=backend_info.coupling_edges,
        )


# ===========================================================================
#  Funciones de validación de layouts
# ===========================================================================

def validate_layout(
    layout: list[int] | np.ndarray,
    search_space: LayoutSearchSpace,
) -> bool:
    """Valida que un layout sea factible en el espacio de búsqueda.

    Comprueba:
      1. El layout tiene la longitud correcta (``num_logical_qubits``).
      2. Todos los qubits físicos están en ``available_qubits``.
      3. No hay qubits físicos repetidos (inyectividad del mapeo).

    Args:
        layout: Array de qubits físicos.
        search_space: Espacio de búsqueda con las restricciones.

    Returns:
        True si el layout es válido, False en caso contrario.
    """
    arr = np.asarray(layout, dtype=int)

    # Longitud correcta
    if len(arr) != search_space.num_logical_qubits:
        logger.debug(
            "Layout inválido: longitud %d, esperada %d",
            len(arr), search_space.num_logical_qubits,
        )
        return False

    # Qubits dentro del rango válido
    for q in arr:
        if int(q) not in search_space.available_qubits:
            logger.debug("Layout inválido: qubit %d no disponible", q)
            return False

    # Sin duplicados
    if len(set(arr.tolist())) != len(arr):
        logger.debug("Layout inválido: qubits duplicados")
        return False

    return True


def repair_layout(
    layout: np.ndarray,
    search_space: LayoutSearchSpace,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Repara un layout que podría contener duplicados o qubits inválidos.

    Estrategia:
      1. Eliminar duplicados manteniendo la primera aparición.
      2. Reemplazar posiciones vacías por qubits del pool de disponibles.

    Esto es necesario tras operaciones de crossover que pueden generar
    individuos inválidos.

    Args:
        layout: Array de qubits físicos (puede contener duplicados).
        search_space: Espacio de búsqueda.
        rng: Generador de números aleatorios (para reproducibilidad).

    Returns:
        Array de qubits físicos válido (sin duplicados).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = search_space.num_logical_qubits
    result = np.full(n, -1, dtype=int)
    used: set[int] = set()

    # Paso 1: conservar qubits válidos y únicos
    for i in range(n):
        q = int(layout[i])
        if q in search_space.available_qubits and q not in used:
            result[i] = q
            used.add(q)

    # Paso 2: rellenar posiciones vacías (-1) con qubits no usados
    available = list(search_space.available_qubits - used)
    rng.shuffle(available)
    fill_idx = 0

    for i in range(n):
        if result[i] == -1:
            result[i] = available[fill_idx]
            fill_idx += 1

    return result


# ===========================================================================
#  Operadores genéticos para pymoo
# ===========================================================================

class LayoutSampling(Sampling):
    """Operador de muestreo: genera layouts iniciales aleatorios.

    Cada individuo es una selección aleatoria sin reemplazo de
    ``n`` qubits de entre los ``N`` disponibles.

    Decisión: se genera una permutación del pool de qubits disponibles
    y se toman los primeros ``n`` elementos. Esto es más eficiente que
    un muestreo iterativo y garantiza unicidad.
    """

    def __init__(self, search_space: LayoutSearchSpace):
        super().__init__()
        self.search_space = search_space

    def _do(self, problem, n_samples, **kwargs):
        """Genera ``n_samples`` layouts aleatorios.

        Args:
            problem: Instancia del problema pymoo.
            n_samples: Número de individuos a generar.

        Returns:
            ndarray de forma ``(n_samples, n_logical_qubits)`` con los
            layouts generados.
        """
        n = self.search_space.num_logical_qubits
        pool = np.array(list(self.search_space.available_qubits))
        rng = np.random.default_rng()

        X = np.empty((n_samples, n), dtype=int)
        for i in range(n_samples):
            X[i] = rng.choice(pool, size=n, replace=False)

        return X


class LayoutCrossover(Crossover):
    """Operador de cruce adaptado a permutaciones parciales.

    Implementa **Order Crossover (OX)** adaptado: dado que los
    individuos son subconjuntos (no permutaciones completas), el
    crossover copia un segmento del padre A al hijo y rellena las
    posiciones restantes con los qubits del padre B que no estén
    ya presentes. Si quedan conflictos, se reparan.

    El OX es la elección estándar para problemas de permutación
    (TSP, scheduling, etc.) porque preserva el orden relativo de los
    elementos.

    Parámetros pymoo:
      - ``n_parents=2``: cruce binario.
      - ``n_offsprings=2``: genera 2 hijos por cada par de padres.
    """

    def __init__(self, search_space: LayoutSearchSpace):
        super().__init__(n_parents=2, n_offsprings=2)
        self.search_space = search_space

    def _do(self, problem, X, **kwargs):
        """Ejecuta el cruce OX sobre los pares de padres.

        Args:
            problem: Instancia del problema pymoo.
            X: Array de padres con forma ``(n_matings, 2, n_vars)``.

        Returns:
            ndarray de hijos con forma ``(n_matings, 2, n_vars)``.
        """
        _, n_matings, n_vars = X.shape
        # pymoo espera forma (n_offsprings, n_matings, n_vars)
        Y = np.full((2, n_matings, n_vars), -1, dtype=int)
        rng = np.random.default_rng()

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            # Puntos de corte aleatorios
            start, end = sorted(rng.choice(n_vars, size=2, replace=False))

            # Hijo 1: segmento de p1, resto de p2
            Y[0, k] = self._ox_child(p1, p2, start, end, rng)
            # Hijo 2: segmento de p2, resto de p1
            Y[1, k] = self._ox_child(p2, p1, start, end, rng)

        return Y

    def _ox_child(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        start: int,
        end: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Genera un hijo con OX adaptado a permutaciones parciales.

        Args:
            parent_a: Padre donante del segmento.
            parent_b: Padre donante de los genes restantes.
            start: Inicio del segmento (inclusive).
            end: Fin del segmento (exclusive).
            rng: Generador aleatorio.

        Returns:
            Array del hijo.
        """
        n = len(parent_a)
        child = np.full(n, -1, dtype=int)

        # Copiar segmento del padre A
        child[start:end] = parent_a[start:end]
        used = set(child[start:end].tolist())

        # Rellenar con genes del padre B no presentes
        fill_values = [g for g in parent_b if g not in used]
        fill_idx = 0

        for i in list(range(end, n)) + list(range(0, start)):
            if fill_idx < len(fill_values):
                child[i] = fill_values[fill_idx]
                fill_idx += 1

        # Reparar si quedan posiciones sin llenar (puede ocurrir si
        # los padres comparten pocos genes)
        if -1 in child:
            child = repair_layout(child, self.search_space, rng)

        return child


class LayoutMutation(Mutation):
    """Operador de mutación para layouts.

    Combina dos tipos de mutación controlados por probabilidades:

      1. **Swap mutation** (prob_swap): intercambia dos posiciones
         aleatorias del layout. Preserva el conjunto de qubits usado
         pero cambia la asignación lógico→físico.

      2. **Replace mutation** (prob_replace): reemplaza un qubit del
         layout por otro qubit físico no utilizado actualmente. Esto
         explora regiones diferentes del chip cuántico.

    Decisión: ambos tipos se combinan para equilibrar *intensificación*
    (swap, explora asignaciones dentro del mismo subconjunto de qubits)
    y *diversificación* (replace, cambia el subconjunto de qubits).
    """

    def __init__(
        self,
        search_space: LayoutSearchSpace,
        prob_swap: float = 0.5,
        prob_replace: float = 0.3,
    ):
        """
        Args:
            search_space: Espacio de búsqueda.
            prob_swap: Probabilidad de aplicar swap mutation a un gen.
            prob_replace: Probabilidad de aplicar replace mutation a un gen.
        """
        super().__init__()
        self.search_space = search_space
        self.prob_swap = prob_swap
        self.prob_replace = prob_replace

    def _do(self, problem, X, **kwargs):
        """Aplica mutación a cada individuo de la población.

        Args:
            problem: Instancia del problema pymoo.
            X: Array de individuos con forma ``(n_individuals, n_vars)``.

        Returns:
            ndarray de individuos mutados.
        """
        rng = np.random.default_rng()
        Y = X.copy()

        for i in range(len(Y)):
            # Swap mutation
            if rng.random() < self.prob_swap:
                Y[i] = self._swap_mutate(Y[i], rng)

            # Replace mutation
            if rng.random() < self.prob_replace:
                Y[i] = self._replace_mutate(Y[i], rng)

        return Y

    @staticmethod
    def _swap_mutate(layout: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Intercambia dos posiciones aleatorias del layout.

        Args:
            layout: Layout a mutar.
            rng: Generador aleatorio.

        Returns:
            Layout mutado.
        """
        result = layout.copy()
        n = len(result)
        if n < 2:
            return result

        i, j = rng.choice(n, size=2, replace=False)
        result[i], result[j] = result[j], result[i]
        return result

    def _replace_mutate(
        self,
        layout: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Reemplaza un qubit por otro no utilizado del backend.

        Args:
            layout: Layout a mutar.
            rng: Generador aleatorio.

        Returns:
            Layout mutado.
        """
        result = layout.copy()
        used = set(result.tolist())
        unused = list(self.search_space.available_qubits - used)

        if not unused:
            return result  # El layout usa todos los qubits del backend

        # Seleccionar posición a reemplazar y nuevo qubit
        pos = rng.integers(0, len(result))
        new_qubit = rng.choice(unused)
        result[pos] = new_qubit

        return result


# ===========================================================================
#  Funciones de conversión y utilidades
# ===========================================================================

def layout_to_list(layout: np.ndarray) -> list[int]:
    """Convierte un layout (array numpy) a lista de enteros Python.

    Necesario para pasar el layout a funciones de Qiskit que esperan
    ``list[int]`` y no ``ndarray``.

    Args:
        layout: Array numpy con los qubits físicos.

    Returns:
        Lista de enteros con los qubits físicos.
    """
    return [int(q) for q in layout]


def layouts_from_population(population: np.ndarray) -> list[list[int]]:
    """Convierte una población completa de arrays a listas de layouts.

    Args:
        population: ndarray de forma ``(pop_size, n_logical_qubits)``.

    Returns:
        Lista de layouts (cada uno una ``list[int]``).
    """
    return [layout_to_list(ind) for ind in population]


def random_layout(
    search_space: LayoutSearchSpace,
    seed: int | None = None,
) -> list[int]:
    """Genera un layout aleatorio válido.

    Útil para inicialización y testing.

    Args:
        search_space: Espacio de búsqueda.
        seed: Semilla para reproducibilidad.

    Returns:
        Layout como lista de enteros.
    """
    rng = np.random.default_rng(seed)
    pool = np.array(list(search_space.available_qubits))
    selected = rng.choice(pool, size=search_space.num_logical_qubits, replace=False)
    return layout_to_list(selected)
