from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from qiskit import QuantumCircuit

from src.integration.campaign_contracts import CampaignCase, CampaignConfig
from src.integration.post_routing_selector import (
    DEFAULT_POST_ROUTING_MIN_EVALS,
    DEFAULT_POST_ROUTING_PATIENCE_FRACTION,
    PostRoutingCheckpointSelector,
    resolve_post_routing_max_no_improvement_evals,
)
from src.rl_module.training import DEFAULT_EVAL_FREQ, setup_training_pipeline
from src.rl_module.routing_mask import DEFAULT_NEW_MASK_SEMANTICS, RoutingMaskConfig


@dataclass(frozen=True, slots=True)
class TrainingConfigSummary:
    algorithm: str
    total_timesteps: int
    frontier_mode: str
    lookahead_window: int
    max_steps: int
    seed: int
    learning_rate: float = 1e-4
    clip_range: float = 0.1
    target_kl: float = 0.03
    n_eval_episodes: int = 1
    cycle_window: int = 8
    stagnation_patience: int | None = None
    sabre_top_k: int | None = None
    sabre_decay_increment: float = 0.001
    sabre_decay_reset_interval: int = 5
    routing_depth_penalty_weight: float = 0.5


@dataclass(frozen=True, slots=True)
class TrainingBridgeResult:
    status: str
    selected_artifact_path: Path | None
    best_model_path: Path | None
    final_model_path: Path | None
    run_model_dir: Path
    run_log_dir: Path
    effective_training_config: TrainingConfigSummary
    actual_timesteps: int | None = None
    post_routing_selection: dict | None = None


def _build_training_config_summary(campaign_config: CampaignConfig) -> TrainingConfigSummary:
    return TrainingConfigSummary(
        algorithm=campaign_config.rl_algorithm,
        total_timesteps=campaign_config.rl_total_timesteps,
        frontier_mode=campaign_config.rl_frontier_mode,
        lookahead_window=campaign_config.rl_lookahead_window,
        max_steps=campaign_config.rl_max_steps,
        seed=campaign_config.seed,
        learning_rate=campaign_config.rl_learning_rate,
        clip_range=campaign_config.rl_clip_range,
        target_kl=campaign_config.rl_target_kl,
        n_eval_episodes=campaign_config.rl_n_eval_episodes,
        cycle_window=campaign_config.rl_cycle_window,
        stagnation_patience=campaign_config.rl_stagnation_patience,
        sabre_top_k=campaign_config.rl_sabre_top_k,
    )


def _build_training_hyperparams(campaign_config: CampaignConfig) -> dict[str, float]:
    if campaign_config.rl_algorithm not in {"PPO", "MaskablePPO"}:
        return {}
    return {
        "learning_rate": campaign_config.rl_learning_rate,
        "clip_range": campaign_config.rl_clip_range,
        "target_kl": campaign_config.rl_target_kl,
    }


def _normalize_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return Path(value)


def _select_training_artifact(best_model_path: Path | None, final_model_path: Path | None) -> Path | None:
    if best_model_path is not None and best_model_path.exists():
        return best_model_path
    if final_model_path is not None and final_model_path.exists():
        return final_model_path
    return None


def _path_belongs_to_dir(path: Path | None, parent_dir: Path) -> bool:
    if path is None:
        return False
    try:
        path.resolve().relative_to(parent_dir.resolve())
    except ValueError:
        return False
    return True


def train_case(
    *,
    campaign_case: CampaignCase,
    campaign_config: CampaignConfig,
    target_circuit: QuantumCircuit,
    coupling_map: Sequence[tuple[int, int]],
    case_output_dir: Path | str,
    initial_layout: Sequence[int] | None = None,
    backend_bundle=None,
    verbose: bool = False,
) -> TrainingBridgeResult:
    del campaign_case

    case_output_path = Path(case_output_dir)
    run_log_base_dir = case_output_path / "training" / "logs"
    run_model_base_dir = case_output_path / "training" / "models"
    config_summary = _build_training_config_summary(campaign_config)
    routing_mask_config = RoutingMaskConfig(
        cycle_window=campaign_config.rl_cycle_window,
        stagnation_patience=campaign_config.rl_stagnation_patience,
        sabre_top_k=campaign_config.rl_sabre_top_k,
    )
    selector_enabled = (
        backend_bundle is not None
        and initial_layout is not None
    )
    masked_routing = campaign_config.rl_algorithm == "MaskablePPO"
    post_routing_max_no_improvement_evals = resolve_post_routing_max_no_improvement_evals(
        total_timesteps=campaign_config.rl_total_timesteps,
        eval_freq=DEFAULT_EVAL_FREQ,
    )
    extra_callback_factories = []
    if selector_enabled:
        def _build_post_routing_selector(*, run_model_dir, run_log_dir):
            del run_log_dir
            return PostRoutingCheckpointSelector(
                circuit=target_circuit,
                coupling_edges=coupling_map,
                backend=backend_bundle.backend,
                backend_name=backend_bundle.backend_name,
                seed=campaign_config.seed,
                initial_layout=initial_layout,
                frontier_mode=campaign_config.rl_frontier_mode,
                max_steps=campaign_config.rl_max_steps,
                lookahead_window=campaign_config.rl_lookahead_window,
                masked=masked_routing,
                mask_semantics=DEFAULT_NEW_MASK_SEMANTICS if masked_routing else None,
                routing_mask_config=(
                    routing_mask_config.resolve(num_qubits=target_circuit.num_qubits)
                    if masked_routing
                    else None
                ),
                run_model_dir=run_model_dir,
                eval_freq=DEFAULT_EVAL_FREQ,
                min_evals=DEFAULT_POST_ROUTING_MIN_EVALS,
                max_no_improvement_evals=post_routing_max_no_improvement_evals,
            )

        extra_callback_factories.append(_build_post_routing_selector)

    try:
        agent = setup_training_pipeline(
            target_circuit=target_circuit,
            coupling_map=list(coupling_map),
            mode="routing",
            frontier_mode=campaign_config.rl_frontier_mode,
            algorithm=campaign_config.rl_algorithm,
            total_timesteps=campaign_config.rl_total_timesteps,
            seed=campaign_config.seed,
            log_dir=str(run_log_base_dir),
            model_save_dir=str(run_model_base_dir),
            lookahead_window=campaign_config.rl_lookahead_window,
            max_steps=campaign_config.rl_max_steps,
            hyperparams=_build_training_hyperparams(campaign_config),
            initial_layout=list(initial_layout) if initial_layout is not None else None,
            routing_mask_config=routing_mask_config,
            n_eval_episodes=campaign_config.rl_n_eval_episodes,
            extra_callback_factories=extra_callback_factories,
            use_reward_early_stopping=not selector_enabled,
            reward_best_model_subdir="reward_best" if selector_enabled else None,
            evaluation_metadata=(
                {
                    "checkpoint_selector": {
                        "enabled": True,
                        "callback": "PostRoutingCheckpointSelector",
                        "score_fields": [
                            "trans_cnot_equivalent",
                            "trans_depth",
                            "total_swaps",
                        ],
                        "early_stopping": {
                            "min_evals": DEFAULT_POST_ROUTING_MIN_EVALS,
                            "max_no_improvement_evals": post_routing_max_no_improvement_evals,
                            "patience_fraction": DEFAULT_POST_ROUTING_PATIENCE_FRACTION,
                            "starts_after_first_valid_solution": True,
                        },
                    }
                }
                if selector_enabled
                else None
            ),
            verbose=verbose,
        )
    except Exception:
        return TrainingBridgeResult(
            status="failed",
            selected_artifact_path=None,
            best_model_path=None,
            final_model_path=None,
            run_model_dir=run_model_base_dir,
            run_log_dir=run_log_base_dir,
            effective_training_config=config_summary,
            actual_timesteps=None,
            post_routing_selection=None,
        )

    run_model_dir = _normalize_optional_path(getattr(agent, "run_model_dir", None)) or run_model_base_dir
    run_log_dir = _normalize_optional_path(getattr(agent, "run_log_dir", None)) or run_log_base_dir
    if not _path_belongs_to_dir(run_model_dir, run_model_base_dir):
        run_model_dir = run_model_base_dir
    if not _path_belongs_to_dir(run_log_dir, run_log_base_dir):
        run_log_dir = run_log_base_dir
    best_model_path = _normalize_optional_path(getattr(agent, "best_model_path", None))
    final_model_path = _normalize_optional_path(getattr(agent, "last_model_path", None))
    if not _path_belongs_to_dir(best_model_path, run_model_dir):
        best_model_path = None
    if not _path_belongs_to_dir(final_model_path, run_model_dir):
        final_model_path = None
    if best_model_path is not None and not best_model_path.exists():
        best_model_path = None
    if final_model_path is not None and not final_model_path.exists():
        final_model_path = None
    selected_artifact_path = _select_training_artifact(best_model_path, final_model_path)
    status = "completed" if selected_artifact_path is not None else "failed"

    return TrainingBridgeResult(
        status=status,
        selected_artifact_path=selected_artifact_path,
        best_model_path=best_model_path,
        final_model_path=final_model_path,
        run_model_dir=run_model_dir,
        run_log_dir=run_log_dir,
        effective_training_config=config_summary,
        actual_timesteps=getattr(agent, "actual_timesteps", None),
        post_routing_selection=getattr(agent, "post_routing_selection", None),
    )
