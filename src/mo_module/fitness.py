"""
fitness.py — Funciones de fitness para la optimización de layouts
=================================================================

Módulo 3 del TFG: Optimización Multiobjetivo.

Este fichero define las funciones de fitness (objetivos) que guían
al algoritmo evolutivo en la búsqueda de layouts óptimos. Cada
función de fitness evalúa un aspecto diferente de la calidad de un
layout y, combinadas, forman un problema de optimización multiobjetivo.

Objetivos activos:
  - **Profundidad** (``DepthFitness``): profundidad del circuito
    transpilado. Determina el tiempo de ejecución y la exposición
    a la decoherencia.
  - **Número de CNOTs** (``CnotCountFitness``): puertas de dos qubits
    tras transpilación. Refleja el coste de routing (cada SWAP = 3
    CNOTs) y es la principal fuente de error.

  Ambas métricas requieren transpilar el circuito con el layout dado.

Arquitectura (extensible):
  Se utiliza el patrón **Strategy** para las funciones de fitness,
  permitiendo que el usuario componga libremente los objetivos
  deseados. Cada función de fitness es una clase que implementa
  la interfaz ``FitnessFunction`` (ABC).

  El ``FitnessEvaluator`` actúa como **Composite**: agrupa múltiples
  funciones de fitness y las evalúa en bloque, devolviendo el vector
  de objetivos que pymoo necesita.

  Para añadir un nuevo objetivo:
    1. Crear una clase que herede de ``FitnessFunction`` e implemente
       ``evaluate()``.
    2. Registrarla en ``AVAILABLE_FITNESS_FUNCTIONS`` con una clave
       de cadena.
    3. Optionalmente añadirla a un preset en ``PRESET_OBJECTIVES``.

Decisiones de diseño:
  1. **Todas las funciones minimizan** — pymoo minimiza por defecto.
     Para métricas que se quieren maximizar en el futuro, devolver
     el valor negado.

  2. **Caché de transpilación** — Las funciones basadas en transpilación
     comparten un caché para evitar transpilar el mismo layout
     múltiples veces cuando hay varios objetivos activos.

  3. **Dependencia de qiskit_interface** — Las funciones utilizan
     ``extract_metrics``, ``transpile_with_custom_layout`` y
     ``BackendInfo`` del módulo 1.

Autor: Eduardo González Bautista
Fecha: 2026-02-18
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Protocol, Sequence

import numpy as np
from qiskit import QuantumCircuit

# ---------------------------------------------------------------------------
# Imports internos del proyecto
# ---------------------------------------------------------------------------
from src.qiskit_interface.backend_info import BackendInfo
from src.qiskit_interface.circuit_utils import CircuitMetrics, extract_metrics
from src.qiskit_interface.transpiler import (
    TranspilationResult,
    transpile_with_custom_layout,
)

# ---------------------------------------------------------------------------
# Logger del módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
#  Interfaz base — FitnessFunction (Strategy Pattern)
# ===========================================================================

class FitnessFunction(ABC):
    """Interfaz abstracta para funciones de fitness.

    Cada función de fitness evalúa un aspecto de la calidad de un
    layout y devuelve un valor escalar. **Siempre se minimiza**: valores
    menores son mejores.

    Attributes:
        name: Nombre descriptivo de la función (para logs y gráficas).
        requires_transpilation: Si True, necesita el circuito transpilado
            para evaluar. Si False, solo usa BackendInfo.
    """

    name: str = "base"
    requires_transpilation: bool = False

    @abstractmethod
    def evaluate(
        self,
        layout: list[int],
        backend_info: BackendInfo,
        transpilation_result: Optional[TranspilationResult] = None,
    ) -> float:
        """Evalúa la calidad de un layout.

        Args:
            layout: Mapeo de qubits lógicos → físicos.
            backend_info: Información del backend.
            transpilation_result: Resultado de transpilación (solo si
                ``requires_transpilation`` es True).

        Returns:
            Valor escalar del objetivo (a minimizar).
        """
        ...


# ===========================================================================
#  Funciones de fitness basadas en transpilación
# ===========================================================================
#
# Para añadir un nuevo objetivo:
#   1. Crear una clase que herede de FitnessFunction.
#   2. Definir name, requires_transpilation e implementar evaluate().
#   3. Registrarla en AVAILABLE_FITNESS_FUNCTIONS al final del fichero.
# ===========================================================================

class DepthFitness(FitnessFunction):
    """Minimiza la profundidad del circuito transpilado.

    Fundamento: la profundidad determina el tiempo de ejecución del
    circuito en el hardware. Una profundidad menor implica:
      - Menos exposición a la decoherencia.
      - Mayor probabilidad de obtener resultados correctos.

    Métrica: ``transpiled_circuit.depth()`` después de transpilar
    con el layout dado.

    Nota: Esta función **requiere transpilación** (``requires_transpilation=True``).
    El ``FitnessEvaluator`` se encarga de realizar la transpilación y
    pasar el resultado.
    """

    name = "depth"
    requires_transpilation = True

    def evaluate(
        self,
        layout: list[int],
        backend_info: BackendInfo,
        transpilation_result: Optional[TranspilationResult] = None,
    ) -> float:
        if transpilation_result is None or transpilation_result.transpiled_metrics is None:
            logger.warning("DepthFitness: no transpilation result available")
            return float("inf")
        return float(transpilation_result.transpiled_metrics.depth)


class CnotCountFitness(FitnessFunction):
    """Minimiza el número de CNOTs equivalentes tras transpilación.

    Cuenta los CNOTs reales obtenidos al descomponer el circuito
    transpilado a la base ``['cx', 'u']`` con ``optimization_level=0``.
    Convierte cualquier puerta nativa del backend a su coste en CNOTs:

      - CX / CNOT  → 1
      - CZ         → 1  (puerta nativa de Torino)
      - ECR        → 1  (puerta nativa de Sherbrooke/Brisbane)
      - SWAP       → 3
      - iSWAP      → 2
      - CRZ / CRX  → 2

    Métrica: ``transpiled_metrics.cnot_equivalent``.
    """

    name = "cnot_count"
    requires_transpilation = True

    def evaluate(
        self,
        layout: list[int],
        backend_info: BackendInfo,
        transpilation_result: Optional[TranspilationResult] = None,
    ) -> float:
        if transpilation_result is None or transpilation_result.transpiled_metrics is None:
            logger.warning("CnotCountFitness: no transpilation result available")
            return float("inf")
        return float(transpilation_result.transpiled_metrics.cnot_equivalent)


# ===========================================================================
#  Registro de funciones de fitness disponibles (Factory)
# ===========================================================================

# ---------------------------------------------------------------------------
# Registro de funciones de fitness por nombre.
# Para añadir un nuevo objetivo: registrar aquí la clase con su clave.
# ---------------------------------------------------------------------------
AVAILABLE_FITNESS_FUNCTIONS: dict[str, type[FitnessFunction]] = {
    "depth": DepthFitness,
    "cnot_count": CnotCountFitness,
    # --- Añadir aquí nuevos objetivos ---
    # "nombre_clave": NuevaClaseFitness,
}


def get_fitness_function(name: str) -> FitnessFunction:
    """Factory: obtiene una instancia de FitnessFunction por nombre.

    Args:
        name: Nombre de la función de fitness. Ver
              ``AVAILABLE_FITNESS_FUNCTIONS`` para las opciones.

    Returns:
        Instancia de la función de fitness.

    Raises:
        ValueError: Si el nombre no corresponde a una función registrada.
    """
    name_lower = name.lower().strip()
    if name_lower not in AVAILABLE_FITNESS_FUNCTIONS:
        available = ", ".join(AVAILABLE_FITNESS_FUNCTIONS.keys())
        raise ValueError(
            f"Función de fitness '{name}' no reconocida. "
            f"Opciones disponibles: {available}"
        )
    return AVAILABLE_FITNESS_FUNCTIONS[name_lower]()


def list_available_fitness_functions() -> list[str]:
    """Devuelve los nombres de todas las funciones de fitness registradas.

    Returns:
        Lista de nombres de funciones de fitness.
    """
    return list(AVAILABLE_FITNESS_FUNCTIONS.keys())


# ===========================================================================
#  Caché de transpilación
# ===========================================================================

class TranspilationCache:
    """Caché de resultados de transpilación para evitar recálculos.

    Cuando múltiples funciones de fitness basadas en transpilación
    comparten el mismo circuito, backend y nivel de optimización,
    la transpilación solo se realiza una vez por layout.

    El caché usa el layout (como tupla inmutable) como clave.

    Attributes:
        circuit: Circuito a transpilar.
        backend: Backend de destino.
        optimization_level: Nivel de optimización de Qiskit.
        seed: Semilla para reproducibilidad.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend,
        optimization_level: int = 1,
        seed: int = 42,
    ):
        self.circuit = circuit
        self.backend = backend
        self.optimization_level = optimization_level
        self.seed = seed
        self._cache: dict[tuple[int, ...], TranspilationResult] = {}
        self._hits = 0
        self._misses = 0

    def get(self, layout: list[int]) -> TranspilationResult:
        """Obtiene o calcula el resultado de transpilación para un layout.

        Args:
            layout: Layout a evaluar.

        Returns:
            TranspilationResult con las métricas del circuito transpilado.
        """
        key = tuple(layout)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        result = transpile_with_custom_layout(
            circuit=self.circuit,
            layout=layout,
            backend=self.backend,
            optimization_level=self.optimization_level,
            seed=self.seed,
        )
        self._cache[key] = result
        return result

    def clear(self) -> None:
        """Limpia el caché."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """Estadísticas del caché."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }


# ===========================================================================
#  Evaluador compuesto (Composite Pattern)
# ===========================================================================

@dataclass
class FitnessEvaluator:
    """Evaluador multiobjetivo que compone múltiples funciones de fitness.

    Actúa como el puente entre las funciones de fitness individuales y
    pymoo. Dado un layout, calcula el vector de objetivos
    ``[f1(layout), f2(layout), ..., fn(layout)]``.

    Gestiona automáticamente la transpilación: si alguna función la
    necesita, se transpila una sola vez (via ``TranspilationCache``)
    y se comparte el resultado.

    Attributes:
        objectives: Lista de funciones de fitness a evaluar.
        backend_info: Información del backend.
        transpilation_cache: Caché compartido de transpilación.
    """

    objectives: list[FitnessFunction]
    backend_info: BackendInfo
    transpilation_cache: Optional[TranspilationCache] = None

    @property
    def n_objectives(self) -> int:
        """Número de objetivos del problema."""
        return len(self.objectives)

    @property
    def objective_names(self) -> list[str]:
        """Nombres de los objetivos (para etiquetas de gráficos)."""
        return [obj.name for obj in self.objectives]

    @property
    def requires_transpilation(self) -> bool:
        """Indica si algún objetivo requiere transpilación."""
        return any(obj.requires_transpilation for obj in self.objectives)

    def evaluate(self, layout: list[int]) -> np.ndarray:
        """Evalúa todas las funciones de fitness para un layout.

        Args:
            layout: Mapeo de qubits lógicos → físicos.

        Returns:
            ndarray de forma ``(n_objectives,)`` con los valores de
            cada objetivo.
        """
        # Obtener resultado de transpilación si es necesario
        transpilation_result = None
        if self.requires_transpilation and self.transpilation_cache is not None:
            transpilation_result = self.transpilation_cache.get(layout)

        values = np.empty(self.n_objectives, dtype=float)
        for i, obj_fn in enumerate(self.objectives):
            values[i] = obj_fn.evaluate(
                layout=layout,
                backend_info=self.backend_info,
                transpilation_result=transpilation_result,
            )

        return values

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evalúa un vector de objetivos para toda una población.

        Args:
            population: ndarray de forma ``(pop_size, n_vars)`` con los
                layouts de cada individuo.

        Returns:
            ndarray de forma ``(pop_size, n_objectives)`` con los valores
            de fitness de cada individuo.
        """
        pop_size = population.shape[0]
        F = np.empty((pop_size, self.n_objectives), dtype=float)

        for i in range(pop_size):
            layout = [int(q) for q in population[i]]
            F[i] = self.evaluate(layout)

        return F

    @classmethod
    def from_names(
        cls,
        objective_names: Sequence[str],
        backend_info: BackendInfo,
        circuit: Optional[QuantumCircuit] = None,
        backend=None,
        optimization_level: int = 1,
        seed: int = 42,
    ) -> FitnessEvaluator:
        """Factory method: crea un evaluador a partir de nombres de objetivos.

        Ejemplo::

            evaluator = FitnessEvaluator.from_names(
                ["depth", "avg_error_2q"],
                backend_info=info,
                circuit=ghz,
                backend=backend,
            )

        Args:
            objective_names: Lista de nombres de funciones de fitness.
            backend_info: Información del backend.
            circuit: Circuito (necesario si hay objetivos de transpilación).
            backend: Backend (necesario si hay objetivos de transpilación).
            optimization_level: Nivel de optimización para transpilación.
            seed: Semilla para reproducibilidad.

        Returns:
            FitnessEvaluator configurado.

        Raises:
            ValueError: Si un nombre no es válido o falta el circuito
                para objetivos de transpilación.
        """
        objectives = [get_fitness_function(name) for name in objective_names]

        # Verificar que se proporcionan circuit/backend si hay objetivos
        # que requieren transpilación
        needs_transpilation = any(obj.requires_transpilation for obj in objectives)
        if needs_transpilation:
            if circuit is None or backend is None:
                raise ValueError(
                    "Se requieren 'circuit' y 'backend' cuando hay "
                    "objetivos que necesitan transpilación. "
                    f"Objetivos con transpilación: "
                    f"{[o.name for o in objectives if o.requires_transpilation]}"
                )

        # Crear caché de transpilación si es necesario
        cache = None
        if needs_transpilation and circuit is not None and backend is not None:
            cache = TranspilationCache(
                circuit=circuit,
                backend=backend,
                optimization_level=optimization_level,
                seed=seed,
            )

        return cls(
            objectives=objectives,
            backend_info=backend_info,
            transpilation_cache=cache,
        )


# ===========================================================================
#  Presets de configuración de objetivos
# ===========================================================================

# Configuraciones predefinidas comunes de objetivos para facilitar el uso.

PRESET_OBJECTIVES: dict[str, list[str]] = {
    # Preset por defecto: profundidad y número de CNOTs.
    "default": [
        "depth",
        "cnot_count",
    ],
    # --- Añadir aquí nuevos presets cuando se registren más objetivos ---
}


def get_preset_objectives(preset_name: str) -> list[str]:
    """Devuelve la lista de nombres de objetivos de un preset.

    Args:
        preset_name: Nombre del preset (ver ``PRESET_OBJECTIVES``).

    Returns:
        Lista de nombres de funciones de fitness.

    Raises:
        ValueError: Si el preset no existe.
    """
    if preset_name not in PRESET_OBJECTIVES:
        available = ", ".join(PRESET_OBJECTIVES.keys())
        raise ValueError(
            f"Preset '{preset_name}' no reconocido. "
            f"Presets disponibles: {available}"
        )
    return PRESET_OBJECTIVES[preset_name]
