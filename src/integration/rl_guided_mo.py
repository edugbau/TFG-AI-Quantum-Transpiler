from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import src.mo_module as mo_module
import src.qiskit_interface as qiskit_interface
from src.integration.rl_model_contract import resolve_routing_model_contract
from src.integration.routing_evaluator import build_routed_circuit, evaluate_routing_episode


RL_GUIDED_MO_FILENAME = "rl_guided_mo.json"
RL_GUIDED_MO_ARTIFACT_VERSION = "rl_guided_mo.v1"
RL_GUIDED_MO_SCORE_FIELDS = ("trans_depth", "trans_cnot_equivalent")
RL_GUIDED_MO_INVALID_SCORE = (1e12, 1e12)


@dataclass(frozen=True, slots=True)
class RLGuidedMoAttempt:
    layout: list[int]
    status: str
    completed: bool
    truncated: bool
    truncation_reason: str | None
    termination_reason: str | None
    total_swaps: int | None
    score: list[float] | None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RLGuidedMoResult:
    artifact_version: str
    selector: str
    score_fields: list[str]
    checkpoint_source: str
    coupling_edges: list[list[int]]
    selected_layout: list[int]
    selected_score: list[float]
    valid_candidate_count: int
    invalid_candidate_count: int
    cache_stats: dict[str, int]
    pareto_layouts: list[list[int]]
    pareto_fitness: list[list[float]]
    attempts: list[dict[str, Any]]
    controls: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RLGuidedFitnessEvaluator:
    """Evaluate layouts with one frozen RL checkpoint on one fixed routing graph."""

    objective_names = list(RL_GUIDED_MO_SCORE_FIELDS)
    n_objectives = len(RL_GUIDED_MO_SCORE_FIELDS)
    transpilation_cache = None

    def __init__(
        self,
        *,
        circuit,
        backend_bundle,
        coupling_edges,
        model_path: Path | str,
        seed: int,
        agent=None,
        contract=None,
        progress_callback: Callable[[str], None] | None = None,
        progress_interval: int = 10,
    ) -> None:
        self.circuit = circuit
        self.backend_bundle = backend_bundle
        self.coupling_edges = [
            (int(left), int(right))
            for left, right in coupling_edges
        ]
        self.model_path = str(model_path)
        self.seed = seed
        self.contract = contract or resolve_routing_model_contract(model_path)
        if not self.contract.masked or self.contract.algorithm != "MaskablePPO":
            raise ValueError("rl_guided requires a masked MaskablePPO checkpoint")
        if agent is None:
            from src.rl_module import QuantumRLAgent

            agent = QuantumRLAgent.load(
                self.model_path,
                env=None,
                algorithm=self.contract.algorithm,
            )
        self.agent = agent
        self.progress_callback = progress_callback
        self.progress_interval = max(1, int(progress_interval))
        self._cache: dict[tuple[int, ...], RLGuidedMoAttempt] = {}
        self._hits = 0
        self._misses = 0

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    @property
    def attempts(self) -> list[RLGuidedMoAttempt]:
        return list(self._cache.values())

    def _emit_progress(self) -> None:
        if self.progress_callback is None:
            return
        attempts = self.attempts
        self.progress_callback(
            "RL-guided MO progress: "
            f"evaluated={self._misses}, "
            f"cache_hits={self._hits}, "
            f"valid={sum(attempt.score is not None for attempt in attempts)}, "
            f"incomplete={sum(attempt.status == 'incomplete_routing' for attempt in attempts)}, "
            f"errors={sum(attempt.status == 'evaluation_error' for attempt in attempts)}"
        )

    def _evaluate_uncached(self, layout: list[int]) -> RLGuidedMoAttempt:
        try:
            routing_summary = evaluate_routing_episode(
                circuit=self.circuit,
                coupling_edges=self.coupling_edges,
                agent=self.agent,
                seed=self.seed,
                initial_layout=list(layout),
                frontier_mode=self.contract.frontier_mode,
                max_steps=self.contract.max_steps,
                lookahead_window=self.contract.lookahead_window,
                masked=self.contract.masked,
                mask_semantics=self.contract.mask_semantics,
                routing_mask_config=self.contract.routing_mask_config,
            )
            if not routing_summary.completed or routing_summary.truncated:
                return RLGuidedMoAttempt(
                    layout=list(layout),
                    status="incomplete_routing",
                    completed=bool(routing_summary.completed),
                    truncated=bool(routing_summary.truncated),
                    truncation_reason=routing_summary.truncation_reason,
                    termination_reason=routing_summary.termination_reason,
                    total_swaps=routing_summary.total_swaps,
                    score=None,
                )

            routed_circuit, final_layout = build_routed_circuit(
                circuit=self.circuit,
                coupling_edges=self.coupling_edges,
                initial_layout=list(layout),
                swap_trace=routing_summary.swap_trace,
                frontier_mode=self.contract.frontier_mode,
                executed_gate_trace=routing_summary.executed_gate_trace,
            )
            transpilation_result = qiskit_interface.transpile_post_routing(
                routed_circuit,
                backend=self.backend_bundle.backend,
                backend_name=self.backend_bundle.backend_name,
                optimization_level=1,
                seed=self.seed,
                reference_circuit=self.circuit,
                initial_layout=list(layout),
                final_layout=final_layout,
            )
            metrics = transpilation_result.transpiled_metrics
            return RLGuidedMoAttempt(
                layout=list(layout),
                status="valid_candidate",
                completed=True,
                truncated=False,
                truncation_reason=None,
                termination_reason=None,
                total_swaps=routing_summary.total_swaps,
                score=[
                    float(metrics.depth),
                    float(metrics.cnot_equivalent),
                ],
            )
        except Exception as exc:
            return RLGuidedMoAttempt(
                layout=list(layout),
                status="evaluation_error",
                completed=False,
                truncated=False,
                truncation_reason=None,
                termination_reason=None,
                total_swaps=None,
                score=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def evaluate_attempt(self, layout: list[int]) -> RLGuidedMoAttempt:
        key = tuple(int(entry) for entry in layout)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        attempt = self._evaluate_uncached(list(key))
        self._cache[key] = attempt
        if self._misses == 1 or self._misses % self.progress_interval == 0:
            self._emit_progress()
        return attempt

    def evaluate(self, layout: list[int]) -> np.ndarray:
        attempt = self.evaluate_attempt(layout)
        score = attempt.score if attempt.score is not None else RL_GUIDED_MO_INVALID_SCORE
        return np.asarray(score, dtype=float)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                self.evaluate([int(entry) for entry in row])
                for row in population
            ],
            dtype=float,
        )


def _select_valid_compromise(
    mo_result,
    evaluator: RLGuidedFitnessEvaluator,
) -> tuple[list[int], list[float], list[list[int]], list[list[float]]]:
    valid_layouts: list[list[int]] = []
    valid_fitness: list[list[float]] = []
    if mo_result.pareto_fitness is None:
        raise ValueError("rl_guided MO did not produce a Pareto front")
    for layout, fitness in zip(mo_result.pareto_layouts, mo_result.pareto_fitness):
        attempt = evaluator.evaluate_attempt(layout)
        if attempt.score is None:
            continue
        valid_layouts.append([int(entry) for entry in layout])
        valid_fitness.append([float(entry) for entry in fitness])
    if not valid_layouts:
        raise ValueError("rl_guided MO did not produce any completed routing candidate")

    fitness_array = np.asarray(valid_fitness, dtype=float)
    minimum = fitness_array.min(axis=0)
    value_range = fitness_array.max(axis=0) - minimum
    value_range[value_range == 0] = 1.0
    distances = np.linalg.norm((fitness_array - minimum) / value_range, axis=1)
    selected_index = int(np.argmin(distances))
    return (
        valid_layouts[selected_index],
        valid_fitness[selected_index],
        valid_layouts,
        valid_fitness,
    )


def write_rl_guided_mo_artifact(path: Path | str, result: RLGuidedMoResult) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return artifact_path


def optimize_rl_guided_layouts(
    *,
    circuit,
    backend_bundle,
    coupling_edges,
    model_path: Path | str,
    qiskit_initial_layout: list[int],
    mo_only_layout: list[int],
    population_size: int,
    n_generations: int,
    seed: int,
    artifact_path: Path | str | None = None,
    verbose: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> RLGuidedMoResult:
    if progress_callback is None and verbose:
        progress_callback = lambda message: print(message, flush=True)
    if progress_callback is not None:
        progress_callback(
            "RL-guided MO started: "
            f"population_size={population_size}, "
            f"n_generations={n_generations}, "
            f"max_candidate_evaluations={population_size * n_generations}"
        )
    evaluator = RLGuidedFitnessEvaluator(
        circuit=circuit,
        backend_bundle=backend_bundle,
        coupling_edges=coupling_edges,
        model_path=model_path,
        seed=seed,
        progress_callback=progress_callback,
    )
    mo_result = mo_module.optimize_layout(
        circuit=circuit,
        backend=backend_bundle.backend,
        backend_name=backend_bundle.backend_name,
        config=mo_module.OptimizerConfig(
            algorithm="nsga2",
            population_size=population_size,
            n_generations=n_generations,
            objectives=list(RL_GUIDED_MO_SCORE_FIELDS),
            seed=seed,
            verbose=verbose,
        ),
        fitness_evaluator=evaluator,
    )
    selected_layout, selected_score, pareto_layouts, pareto_fitness = _select_valid_compromise(
        mo_result,
        evaluator,
    )
    candidate_attempts = evaluator.attempts
    controls = {
        "qiskit_initial_layout": evaluator.evaluate_attempt(qiskit_initial_layout).to_dict(),
        "mo_only_layout": evaluator.evaluate_attempt(mo_only_layout).to_dict(),
        "rl_guided_layout": evaluator.evaluate_attempt(selected_layout).to_dict(),
    }
    result = RLGuidedMoResult(
        artifact_version=RL_GUIDED_MO_ARTIFACT_VERSION,
        selector="rl_guided",
        score_fields=list(RL_GUIDED_MO_SCORE_FIELDS),
        checkpoint_source=str(model_path),
        coupling_edges=[
            [int(left), int(right)]
            for left, right in coupling_edges
        ],
        selected_layout=selected_layout,
        selected_score=selected_score,
        valid_candidate_count=sum(attempt.score is not None for attempt in candidate_attempts),
        invalid_candidate_count=sum(attempt.score is None for attempt in candidate_attempts),
        cache_stats=evaluator.cache_stats,
        pareto_layouts=pareto_layouts,
        pareto_fitness=pareto_fitness,
        attempts=[attempt.to_dict() for attempt in candidate_attempts],
        controls=controls,
    )
    if artifact_path is not None:
        write_rl_guided_mo_artifact(artifact_path, result)
    if progress_callback is not None:
        progress_callback(
            "RL-guided MO completed: "
            f"selected_layout={result.selected_layout}, "
            f"valid_candidates={result.valid_candidate_count}, "
            f"invalid_candidates={result.invalid_candidate_count}, "
            f"cache={result.cache_stats}"
        )
    return result
