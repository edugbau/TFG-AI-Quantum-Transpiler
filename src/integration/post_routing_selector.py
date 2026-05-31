from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

import src.qiskit_interface as qiskit_interface
from src.integration.routing_evaluator import build_routed_circuit, evaluate_routing_episode


logger = logging.getLogger(__name__)

POST_ROUTING_SELECTION_FILENAME = "post_routing_selection.json"
POST_ROUTING_SCORE_FIELDS = ("trans_cnot_equivalent", "trans_depth", "total_swaps")
DEFAULT_POST_ROUTING_MIN_EVALS = 50
DEFAULT_POST_ROUTING_MAX_NO_IMPROVEMENT_EVALS = 20
DEFAULT_POST_ROUTING_PATIENCE_FRACTION = 0.2


def resolve_post_routing_max_no_improvement_evals(
    *,
    total_timesteps: int,
    eval_freq: int,
) -> int:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be greater than zero")
    if eval_freq <= 0:
        raise ValueError("eval_freq must be greater than zero")
    proportional_patience = math.ceil(
        total_timesteps * DEFAULT_POST_ROUTING_PATIENCE_FRACTION / eval_freq
    )
    return max(DEFAULT_POST_ROUTING_MAX_NO_IMPROVEMENT_EVALS, proportional_patience)


class PostRoutingCheckpointSelector(BaseCallback):
    """Select campaign checkpoints using the materialized post-routing circuit."""

    def __init__(
        self,
        *,
        circuit,
        coupling_edges,
        backend,
        backend_name: str,
        seed: int,
        initial_layout,
        frontier_mode: str,
        max_steps: int,
        lookahead_window: int,
        masked: bool,
        mask_semantics: str | None,
        routing_mask_config,
        run_model_dir: str | Path,
        eval_freq: int = 5_000,
        min_evals: int = DEFAULT_POST_ROUTING_MIN_EVALS,
        max_no_improvement_evals: int = DEFAULT_POST_ROUTING_MAX_NO_IMPROVEMENT_EVALS,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.circuit = circuit
        self.coupling_edges = list(coupling_edges)
        self.backend = backend
        self.backend_name = backend_name
        self.seed = seed
        self.initial_layout = list(initial_layout) if initial_layout is not None else None
        self.frontier_mode = frontier_mode
        self.max_steps = max_steps
        self.lookahead_window = lookahead_window
        self.masked = masked
        self.mask_semantics = mask_semantics
        self.routing_mask_config = routing_mask_config
        self.run_model_dir = Path(run_model_dir)
        self.eval_freq = eval_freq
        self.min_evals = min_evals
        self.max_no_improvement_evals = max_no_improvement_evals
        self.best_model_path = self.run_model_dir / "best_model.zip"
        self.selection_path = self.run_model_dir / POST_ROUTING_SELECTION_FILENAME
        self.attempts: list[dict[str, Any]] = []
        self.best_score: tuple[float, int, int] | None = None
        self.first_solution_timestep: int | None = None
        self.no_improvement_evals = 0
        self.stop_reason: str | None = None

    @property
    def phase(self) -> str:
        return (
            "searching_first_solution"
            if self.first_solution_timestep is None
            else "optimizing_valid_solutions"
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        improved = False
        try:
            routing_summary = evaluate_routing_episode(
                circuit=self.circuit,
                coupling_edges=self.coupling_edges,
                agent=self.model,
                seed=self.seed,
                initial_layout=self.initial_layout,
                frontier_mode=self.frontier_mode,
                max_steps=self.max_steps,
                lookahead_window=self.lookahead_window,
                masked=self.masked,
                mask_semantics=self.mask_semantics,
                routing_mask_config=self.routing_mask_config,
            )
            if not routing_summary.completed or routing_summary.truncated:
                remaining_gates = max(
                    self._supported_gate_count() - routing_summary.gates_executed_count,
                    0,
                )
                attempt = self._base_attempt(
                    status="incomplete_routing",
                    reward=routing_summary.total_reward,
                    swaps=routing_summary.total_swaps,
                    remaining_gates=remaining_gates,
                    truncation_reason=routing_summary.truncation_reason,
                )
            else:
                attempt, improved = self._evaluate_completed_routing(routing_summary)
        except Exception as exc:
            logger.exception("Post-routing checkpoint evaluation failed at timestep %s", self.num_timesteps)
            attempt = self._base_attempt(
                status="evaluation_error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        self.attempts.append(attempt)
        self._update_patience(improved=improved)
        self._persist()
        self._record_scalars(attempt)
        if self._should_stop():
            self.stop_reason = "post_routing_no_improvement"
            self._persist()
            return False
        return True

    def _supported_gate_count(self) -> int:
        return sum(1 for instruction in self.circuit.data if len(instruction.qubits) in {1, 2})

    def _base_attempt(
        self,
        *,
        status: str,
        reward: float | None = None,
        swaps: int | None = None,
        remaining_gates: int | None = None,
        truncation_reason: str | None = None,
        score: tuple[float, int, int] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        return {
            "timestep": int(self.num_timesteps),
            "status": status,
            "reward": reward,
            "swaps": swaps,
            "remaining_gates": remaining_gates,
            "truncation_reason": truncation_reason,
            "score": list(score) if score is not None else None,
            "error_type": error_type,
            "error_message": error_message,
        }

    def _evaluate_completed_routing(self, routing_summary) -> tuple[dict[str, Any], bool]:
        routed_circuit, final_layout = build_routed_circuit(
            circuit=self.circuit,
            coupling_edges=self.coupling_edges,
            initial_layout=self.initial_layout,
            swap_trace=routing_summary.swap_trace,
            frontier_mode=self.frontier_mode,
            executed_gate_trace=routing_summary.executed_gate_trace,
        )
        transpilation_result = qiskit_interface.transpile_post_routing(
            routed_circuit,
            backend=self.backend,
            backend_name=self.backend_name,
            optimization_level=1,
            seed=self.seed,
            reference_circuit=self.circuit,
            initial_layout=self.initial_layout,
            final_layout=final_layout,
        )
        metrics = transpilation_result.transpiled_metrics
        score = (
            float(metrics.cnot_equivalent),
            int(metrics.depth),
            int(routing_summary.total_swaps),
        )
        improved = self.best_score is None or score < self.best_score
        if improved:
            self.model.save(str(self.best_model_path))
            self.best_score = score
            if self.first_solution_timestep is None:
                self.first_solution_timestep = int(self.num_timesteps)
        return (
            self._base_attempt(
                status="valid_candidate",
                reward=routing_summary.total_reward,
                swaps=routing_summary.total_swaps,
                remaining_gates=0,
                score=score,
            ),
            improved,
        )

    def _update_patience(self, *, improved: bool) -> None:
        if self.first_solution_timestep is None:
            return
        if improved:
            self.no_improvement_evals = 0
        else:
            self.no_improvement_evals += 1

    def _should_stop(self) -> bool:
        return (
            self.first_solution_timestep is not None
            and len(self.attempts) >= self.min_evals
            and self.no_improvement_evals >= self.max_no_improvement_evals
        )

    def _record_scalars(self, attempt: dict[str, Any]) -> None:
        self.logger.record("post_routing/has_valid_solution", float(self.first_solution_timestep is not None))
        self.logger.record("post_routing/no_improvement_evals", float(self.no_improvement_evals))
        if attempt["score"] is not None:
            self.logger.record("post_routing/cnot_equivalent", float(attempt["score"][0]))
            self.logger.record("post_routing/depth", float(attempt["score"][1]))
            self.logger.record("post_routing/swaps", float(attempt["score"][2]))

    def finalize(self, *, actual_timesteps: int) -> None:
        if self.stop_reason is None:
            self.stop_reason = "training_budget_exhausted"
        self._persist(actual_timesteps=actual_timesteps)

    def to_summary(self, *, actual_timesteps: int | None = None) -> dict[str, Any]:
        return {
            "selector": "post_routing_checkpoint",
            "score_fields": list(POST_ROUTING_SCORE_FIELDS),
            "phase": self.phase,
            "has_valid_solution": self.first_solution_timestep is not None,
            "first_solution_timestep": self.first_solution_timestep,
            "best_score": list(self.best_score) if self.best_score is not None else None,
            "evaluation_count": len(self.attempts),
            "no_improvement_evals": self.no_improvement_evals,
            "min_evals": self.min_evals,
            "max_no_improvement_evals": self.max_no_improvement_evals,
            "actual_timesteps": actual_timesteps,
            "stop_reason": self.stop_reason,
            "attempts": list(self.attempts),
        }

    def _persist(self, *, actual_timesteps: int | None = None) -> None:
        self.run_model_dir.mkdir(parents=True, exist_ok=True)
        self.selection_path.write_text(
            json.dumps(self.to_summary(actual_timesteps=actual_timesteps), indent=2),
            encoding="utf-8",
        )
