"""Manual microbenchmark for masked-routing v2/v3 ablations.

Run from the repository root:

    python notebooks/benchmark_routing_mask_v3.py --iterations 2000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.integration.synthetic_topology import SyntheticTopologySpec
from src.qiskit_interface.circuit_utils import create_qft_circuit
from src.rl_module.environment import QuantumTranspilationEnv
from src.rl_module.routing_mask import RoutingMaskConfig


def _build_env(*, semantics: str, sabre_top_k: int | None) -> QuantumTranspilationEnv:
    topology = SyntheticTopologySpec(shape="heavy_hex", distance=5)
    return QuantumTranspilationEnv(
        target_circuit=create_qft_circuit(8),
        coupling_map=list(topology.build_coupling_map().get_edges()),
        mode="routing",
        frontier_mode="dag",
        lookahead_window=16,
        max_steps=120,
        mask_semantics=semantics,
        routing_mask_config=(
            RoutingMaskConfig(sabre_top_k=sabre_top_k)
            if semantics == "frontier_restricted_edges.v3"
            else None
        ),
    )


def _measure_mask_calls(env: QuantumTranspilationEnv, *, iterations: int) -> float:
    env.reset(seed=42)
    started_at = perf_counter()
    for _ in range(iterations):
        env.action_masks()
    return perf_counter() - started_at


def _measure_transitions(env: QuantumTranspilationEnv, *, iterations: int) -> float:
    env.reset(seed=42)
    started_at = perf_counter()
    for _ in range(iterations):
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=42)
    return perf_counter() - started_at


def _run_case(*, semantics: str, sabre_top_k: int | None, iterations: int) -> dict[str, object]:
    env = _build_env(semantics=semantics, sabre_top_k=sabre_top_k)
    mask_elapsed = _measure_mask_calls(env, iterations=iterations)
    transition_elapsed = _measure_transitions(env, iterations=iterations)
    return {
        "semantics": semantics,
        "sabre_top_k": sabre_top_k,
        "iterations": iterations,
        "mask_us_per_call": 1e6 * mask_elapsed / iterations,
        "transition_us_per_call": 1e6 * transition_elapsed / iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--sabre-top-k", type=int, default=4)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero")
    if args.sabre_top_k <= 0:
        raise ValueError("--sabre-top-k must be greater than zero")

    results = [
        _run_case(
            semantics="frontier_restricted_edges.v2",
            sabre_top_k=None,
            iterations=args.iterations,
        ),
        _run_case(
            semantics="frontier_restricted_edges.v3",
            sabre_top_k=None,
            iterations=args.iterations,
        ),
        _run_case(
            semantics="frontier_restricted_edges.v3",
            sabre_top_k=args.sabre_top_k,
            iterations=args.iterations,
        ),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
