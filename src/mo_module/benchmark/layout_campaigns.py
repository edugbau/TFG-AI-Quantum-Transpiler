"""layout_campaigns.py — Campañas ligeras de selección de layouts.

Evalúa layouts candidatos del frente de Pareto junto a referencias
externas usando la infraestructura existente del módulo MO.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from src.qiskit_interface import get_backend, get_heaviest_hex_layout

from ..optimizer import OptimizerConfig, compare_layouts, optimize_layout
from ..pareto import analyze_pareto_front
from .circuits import BenchmarkCircuit


def _copy_config_with_seed(config: OptimizerConfig, seed: int) -> OptimizerConfig:
    """Crea una copia de la configuración cambiando solo la semilla."""
    return replace(config, seed=seed)


def build_reference_layouts(
    num_qubits: int,
    backend_num_qubits: int,
    *,
    backend=None,
) -> dict[str, list[int]]:
    """Construye los layouts de referencia no evolutivos de la campaña."""
    backend_for_heaviest_hex = backend
    if backend_for_heaviest_hex is None:
        backend_for_heaviest_hex = SimpleNamespace(num_qubits=backend_num_qubits)

    high_index_block = list(range(backend_num_qubits - num_qubits, backend_num_qubits))

    try:
        heaviest_hex = get_heaviest_hex_layout(backend_for_heaviest_hex, num_qubits)
    except (AttributeError, TypeError, ValueError):
        heaviest_hex = list(reversed(high_index_block))

    return {
        "trivial": list(range(num_qubits)),
        "reverse_trivial": list(reversed(range(num_qubits))),
        "high_index_block": high_index_block,
        "heaviest_hex": heaviest_hex,
    }


def build_layout_campaign_spec(*, preset: str = "balanced") -> dict[str, object]:
    """Devuelve la especificación del preset de campaña."""
    specs = {
        "quick": {
            "reference_names": ["trivial", "heaviest_hex"],
            "mo_candidate_names": ["compromise"],
            "include_knee": False,
            "include_best_per_objective": False,
        },
        "balanced": {
            "reference_names": ["trivial", "heaviest_hex"],
            "mo_candidate_names": ["compromise", "knee"],
            "include_knee": True,
            "include_best_per_objective": True,
        },
        "thorough": {
            "reference_names": [
                "trivial",
                "heaviest_hex",
                "reverse_trivial",
                "high_index_block",
            ],
            "mo_candidate_names": None,
            "include_knee": True,
            "include_best_per_objective": True,
        },
    }
    try:
        return specs[preset]
    except KeyError as exc:
        raise ValueError(f"Unknown layout campaign preset: {preset}") from exc


def _extract_available_candidate_layouts(candidates: dict) -> dict[str, list[int]]:
    """Extrae solo los candidatos MO presentes en el análisis."""
    layouts: dict[str, list[int]] = {}
    for strategy, candidate in candidates.items():
        if strategy not in {"compromise", "knee"} and not strategy.startswith("best_"):
            continue
        if candidate is not None and candidate.get("layout") is not None:
            layouts[strategy] = candidate["layout"]
    return layouts


def _select_candidate_layouts(
    candidates: dict,
    spec: dict[str, object],
) -> dict[str, list[int]]:
    """Filtra los candidatos MO según el preset activo."""
    available_layouts = _extract_available_candidate_layouts(candidates)
    selected: dict[str, list[int]] = {}

    mo_candidate_names = spec["mo_candidate_names"]
    if mo_candidate_names is None:
        return available_layouts

    for name in mo_candidate_names:
        layout = available_layouts.get(name)
        if layout is not None:
            selected[name] = layout

    if spec["include_best_per_objective"]:
        for name, layout in available_layouts.items():
            if name.startswith("best_"):
                selected[name] = layout

    return selected


def _select_reference_layouts(
    all_reference_layouts: dict[str, list[int]],
    spec: dict[str, object],
) -> dict[str, list[int]]:
    """Filtra las referencias según el preset activo."""
    return {
        name: all_reference_layouts[name]
        for name in spec["reference_names"]
        if name in all_reference_layouts
    }


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
    preset: str = "balanced",
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
    spec = build_layout_campaign_spec(preset=preset)
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

            reference_layouts = build_reference_layouts(
                circuit.num_qubits,
                backend.num_qubits,
                backend=backend,
            )
            layouts = {
                **_select_candidate_layouts(candidates, spec),
                **_select_reference_layouts(reference_layouts, spec),
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
