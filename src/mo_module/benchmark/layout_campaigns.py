"""layout_campaigns.py — Campañas ligeras de selección de layouts.

Evalúa layouts candidatos del frente de Pareto junto a referencias
externas usando la infraestructura existente del módulo MO.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from src.qiskit_interface import get_backend, get_heaviest_hex_layout

from ..optimizer import OptimizerConfig, compare_layouts, optimize_layout
from ..pareto import analyze_pareto_front
from .circuits import BenchmarkCircuit


def _copy_config_with_seed(config: OptimizerConfig, seed: int) -> OptimizerConfig:
    """Crea una copia de la configuración cambiando solo la semilla."""
    return replace(config, seed=seed)


def _build_reference_layouts(circuit_qubits: int, backend) -> dict[str, list[int]]:
    """Construye los layouts de referencia no evolutivos."""
    return {
        "trivial": list(range(circuit_qubits)),
        "heaviest_hex": get_heaviest_hex_layout(backend, circuit_qubits),
    }


def _extract_available_candidate_layouts(candidates: dict) -> dict[str, list[int]]:
    """Extrae solo los candidatos MO presentes en el análisis."""
    layouts: dict[str, list[int]] = {}
    for strategy, candidate in candidates.items():
        if strategy not in {"compromise", "knee"} and not strategy.startswith("best_"):
            continue
        if candidate is not None and candidate.get("layout") is not None:
            layouts[strategy] = candidate["layout"]
    return layouts


def _append_metric_if_present(values: list[float], row: dict, key: str) -> None:
    """Añade una métrica numérica si la fila la contiene."""
    value = row.get(key)
    if value is None:
        return
    values.append(float(value))


def run_layout_selection_campaign(
    *,
    circuits: list[BenchmarkCircuit],
    seeds: list[int],
    config: OptimizerConfig | None = None,
    backend_name: str = "fake_torino",
) -> list[dict]:
    """Ejecuta una campaña comparando candidatos MO y referencias.

    Por cada circuito y semilla, optimiza el layout, analiza el frente de
    Pareto, selecciona cuatro candidatos del frente y los compara frente a
    dos referencias externas con ``compare_layouts``.
    """
    if config is None:
        config = OptimizerConfig(
            algorithm="nsga2",
            population_size=30,
            n_generations=50,
            objectives=["depth", "cnot_count"],
            verbose=False,
        )

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
                **_extract_available_candidate_layouts(candidates),
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
    counts: dict[str, int] = defaultdict(int)

    for row in rows:
        layout_name = row["layout_name"]
        counts[layout_name] += 1
        _append_metric_if_present(grouped[layout_name]["depth"], row, "depth")
        _append_metric_if_present(
            grouped[layout_name]["cnot_equivalent"], row, "cnot_equivalent"
        )

    summary: dict[str, dict[str, float]] = {}
    for layout_name, count in counts.items():
        values = grouped[layout_name]
        summary[layout_name] = {
            "count": count,
            "depth_mean": (
                float(np.mean(values["depth"])) if values["depth"] else 0.0
            ),
            "cnot_equivalent_mean": (
                float(np.mean(values["cnot_equivalent"]))
                if values["cnot_equivalent"]
                else 0.0
            ),
        }

    return summary
