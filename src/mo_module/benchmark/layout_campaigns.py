"""layout_campaigns.py — Campañas ligeras de selección de layouts.

Evalúa layouts candidatos del frente de Pareto junto a referencias
externas usando la infraestructura existente del módulo MO.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.qiskit_interface import get_backend, get_heaviest_hex_layout

from ..optimizer import OptimizerConfig, compare_layouts, optimize_layout
from ..pareto import analyze_pareto_front
from .circuits import BenchmarkCircuit


def _copy_config_with_seed(config: OptimizerConfig, seed: int) -> OptimizerConfig:
    """Crea una copia de la configuración cambiando solo la semilla."""
    return OptimizerConfig(
        algorithm=config.algorithm,
        population_size=config.population_size,
        n_generations=config.n_generations,
        objectives=list(config.objectives),
        optimization_level=config.optimization_level,
        crossover_operator=config.crossover_operator,
        prob_swap_mutation=config.prob_swap_mutation,
        prob_replace_mutation=config.prob_replace_mutation,
        seed=seed,
        verbose=False,
    )


def _build_reference_layouts(circuit_qubits: int, backend) -> dict[str, list[int]]:
    """Construye los layouts de referencia no evolutivos."""
    return {
        "trivial": list(range(circuit_qubits)),
        "heaviest_hex": get_heaviest_hex_layout(backend, circuit_qubits),
    }


def run_layout_selection_campaign(
    circuits: list[BenchmarkCircuit],
    seeds: list[int],
    config: OptimizerConfig,
    backend_name: str = "fake_torino",
) -> list[dict]:
    """Ejecuta una campaña comparando candidatos MO y referencias.

    Por cada circuito y semilla, optimiza el layout, analiza el frente de
    Pareto, selecciona cuatro candidatos del frente y los compara frente a
    dos referencias externas con ``compare_layouts``.
    """
    backend = get_backend(backend_name)
    rows: list[dict] = []

    for benchmark_circuit in circuits:
        circuit = benchmark_circuit.create()
        for seed in seeds:
            opt_result = optimize_layout(
                circuit=circuit,
                backend=backend,
                config=_copy_config_with_seed(config, seed),
            )
            analysis = analyze_pareto_front(opt_result)
            candidates = analysis.get("selection_candidates", {})

            layouts = {
                "compromise": candidates["compromise"]["layout"],
                "knee": candidates["knee"]["layout"],
                "best_depth": candidates["best_depth"]["layout"],
                "best_cnot_count": candidates["best_cnot_count"]["layout"],
                **_build_reference_layouts(circuit.num_qubits, backend),
            }
            evaluated_rows = compare_layouts(
                circuit=circuit,
                layouts=layouts,
                backend=backend,
                optimization_level=config.optimization_level,
                seed=seed,
            )

            for row in evaluated_rows:
                strategy = row["layout_name"]
                candidate = candidates.get(strategy)
                row_with_metadata = dict(row)
                row_with_metadata.update(
                    {
                        "circuit_name": benchmark_circuit.name,
                        "seed": seed,
                        "layout_family": (
                            "mo_candidate" if candidate is not None else "reference"
                        ),
                        "selection_strategy": strategy,
                        "pareto_index": (
                            int(candidate["index"]) if candidate is not None else None
                        ),
                    }
                )
                rows.append(row_with_metadata)

    return rows


def summarize_layout_campaign(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Resume una campaña agrupando por ``layout_name``."""
    if not rows:
        return {}

    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"depth": [], "cnot_equivalent": []}
    )

    for row in rows:
        grouped[row["layout_name"]]["depth"].append(float(row["depth"]))
        grouped[row["layout_name"]]["cnot_equivalent"].append(
            float(row["cnot_equivalent"])
        )

    summary: dict[str, dict[str, float]] = {}
    for layout_name, values in grouped.items():
        summary[layout_name] = {
            "count": len(values["depth"]),
            "depth_mean": float(np.mean(values["depth"])),
            "cnot_equivalent_mean": float(np.mean(values["cnot_equivalent"])),
        }

    return summary
