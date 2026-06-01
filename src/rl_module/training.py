"""
Script de Entrenamiento del Agente de RL.

Proporciona la orquestación principal para inicializar el entorno, configurar
las semillas globales, configurar los callbacks (TensorBoard y Checkpoints)
y lanzar el proceso de entrenamiento.
"""

import os
import logging
import random
from inspect import Parameter, signature
from datetime import datetime
import numpy as np
import torch
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from .agent import QuantumRLAgent
from .environment import QuantumTranspilationEnv
from .model_metadata import build_run_metadata, save_run_metadata
from .routing_mask import (
    DEFAULT_NEW_MASK_SEMANTICS,
    FRONTIER_RESTRICTED_EDGES_V3,
    FRONTIER_RESTRICTED_EDGES_V4,
    FRONTIER_RESTRICTED_EDGES_V5,
    RoutingMaskConfig,
    resolve_routing_mask_config,
)
from qiskit import QuantumCircuit
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PPO_STABILITY_HYPERPARAMS = {
    "learning_rate": 1e-4,
    "clip_range": 0.1,
    "target_kl": 0.03,
}
DEFAULT_N_EVAL_EPISODES = 1
DEFAULT_EVAL_FREQ = 5_000
DEFAULT_EARLY_STOPPING_MIN_EVALS = 50
DEFAULT_EARLY_STOPPING_MAX_NO_IMPROVEMENT_EVALS = 20
_PPO_LIKE_ALGORITHMS = {"PPO", "MaskablePPO"}

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
except ModuleNotFoundError:
    MaskableEvalCallback = None


class _StickyResetOptionsWrapper(gym.Wrapper):
    """Reapplies reset options on every downstream environment reset."""

    def __init__(self, env: gym.Env, *, reset_options: Optional[dict] = None):
        super().__init__(env)
        self._reset_options = dict(reset_options) if reset_options is not None else None

    def reset(self, *, seed=None, options=None):
        merged_options = options
        if self._reset_options is not None:
            merged_options = dict(self._reset_options)
            if options is not None:
                merged_options.update(options)

        if merged_options is None:
            return self.env.reset(seed=seed)
        return self.env.reset(seed=seed, options=merged_options)

def _make_run_dir(base_dir: str, prefix: str = "run") -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(base_dir, f"{prefix}_{run_id}")

def set_global_seeds(seed: int = 42):
    """
    Fija las semillas aleatorias globales para garantizar reproducibilidad.
    (Basado en `skill_experimentation_logging`).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Semillas globales fijadas a %d.", seed)


def _build_effective_hyperparams(algorithm: str, hyperparams: Optional[dict]) -> dict:
    effective_hyperparams = dict(hyperparams or {})
    if algorithm in _PPO_LIKE_ALGORITHMS:
        for key, value in DEFAULT_PPO_STABILITY_HYPERPARAMS.items():
            effective_hyperparams.setdefault(key, value)
    return effective_hyperparams


def _callable_accepts_kwarg(callable_obj, kwarg_name: str) -> bool:
    try:
        parameters = signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (parameter.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD) and parameter.name == kwarg_name)
        or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _train_agent(agent: QuantumRLAgent, *, total_timesteps: int, callbacks: list, progress_bar: bool) -> None:
    if _callable_accepts_kwarg(agent.train, "progress_bar"):
        agent.train(total_timesteps=total_timesteps, callbacks=callbacks, progress_bar=progress_bar)
        return
    agent.train(total_timesteps=total_timesteps, callbacks=callbacks)


def setup_training_pipeline(
    target_circuit: QuantumCircuit,
    coupling_map: List[Tuple[int, int]],
    mode: str = "routing",
    frontier_mode: str = "sequential",
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    seed: int = 42,
    log_dir: str = "./experiments/logs/rl_logs",
    model_save_dir: str = "./experiments/models/rl_models",
    lookahead_window: int = 10,
    max_steps: int = 1000,
    hyperparams: Optional[dict] = None,
    basis_gates: Optional[List[str]] = None,
    initial_layout: Optional[List[int]] = None,
    routing_mask_config: RoutingMaskConfig | dict | None = None,
    n_eval_episodes: int = DEFAULT_N_EVAL_EPISODES,
    eval_freq: int = DEFAULT_EVAL_FREQ,
    extra_callbacks: Optional[list] = None,
    extra_callback_factories: Optional[list] = None,
    use_reward_early_stopping: bool = True,
    reward_best_model_subdir: Optional[str] = None,
    evaluation_metadata: Optional[dict] = None,
    verbose: bool = False,
) -> QuantumRLAgent:
    """
    Configura y lanza el pipeline completo de entrenamiento del agente.
    
    Args:
        target_circuit: Circuito cuántico a transpilar.
        coupling_map: Mapa de acoplamiento físico (lista de aristas).
        mode: "routing" o "synthesis".
        frontier_mode: Estrategia publica de frontera expuesta al pipeline.
        algorithm: "PPO" o "DQN".
        total_timesteps: Pasos de entrenamiento.
        seed: Semilla de reproducibilidad.
        log_dir: Directorio para logs de TensorBoard.
        model_save_dir: Directorio para guardar los checkpoints.
        lookahead_window: Número de puertas futuras visibles para el agente.
        max_steps: Máximo de pasos por episodio antes de truncar.
        hyperparams: Diccionario con hiperparámetros para PPO/DQN.
        basis_gates: Base nativa explícita requerida por ``mode="synthesis"``.
        initial_layout: Layout inicial opcional inyectado en train/eval reset.
        
    Returns:
        El agente entrenado listo para evaluación.
    """

    if algorithm == "MaskablePPO" and mode != "routing":
        raise ValueError("MaskablePPO solo esta soportado para mode='routing'.")
    if n_eval_episodes <= 0:
        raise ValueError("n_eval_episodes must be greater than zero")
    if eval_freq <= 0:
        raise ValueError("eval_freq must be greater than zero")

    effective_hyperparams = _build_effective_hyperparams(algorithm, hyperparams)
    mask_semantics = (
        DEFAULT_NEW_MASK_SEMANTICS
        if algorithm == "MaskablePPO" and mode == "routing"
        else None
    )
    resolved_routing_mask_config = (
        resolve_routing_mask_config(
            routing_mask_config,
            num_qubits=target_circuit.num_qubits,
        )
        if mask_semantics in {FRONTIER_RESTRICTED_EDGES_V3, FRONTIER_RESTRICTED_EDGES_V4, FRONTIER_RESTRICTED_EDGES_V5}
        else None
    )
    
    # 1. Semillas
    set_global_seeds(seed)
    
    # 2. Configurar Entorno
    # Usamos Monitor para que SB3 registre las métricas automáticas (rewards, episode lengths)
    reset_options = None
    if initial_layout is not None:
        reset_options = {"initial_layout": list(initial_layout)}

    raw_env = QuantumTranspilationEnv(
        target_circuit=target_circuit,
        coupling_map=coupling_map,
        mode=mode,
        frontier_mode=frontier_mode,
        lookahead_window=lookahead_window,
        max_steps=max_steps,
        basis_gates=basis_gates,
        mask_semantics=mask_semantics,
        routing_mask_config=resolved_routing_mask_config,
    )
    if reset_options is None:
        _, train_reset_info = raw_env.reset(seed=seed)
        env = Monitor(raw_env)
    else:
        _, train_reset_info = raw_env.reset(seed=seed, options=reset_options)
        env = Monitor(_StickyResetOptionsWrapper(raw_env, reset_options=reset_options))
    routing_completed_at_reset = mode == "routing" and bool(
        train_reset_info.get(
            "already_completed_at_reset",
            getattr(raw_env, "was_completed_at_reset", False),
        )
    )
    
    # Entorno de Evaluación independiente
    eval_raw_env = QuantumTranspilationEnv(
        target_circuit=target_circuit,
        coupling_map=coupling_map,
        mode=mode,
        frontier_mode=frontier_mode,
        lookahead_window=lookahead_window,
        max_steps=max_steps,
        basis_gates=basis_gates,
        mask_semantics=mask_semantics,
        routing_mask_config=resolved_routing_mask_config,
    )
    if reset_options is None:
        eval_raw_env.reset(seed=seed)
        eval_env = Monitor(eval_raw_env)
    else:
        eval_raw_env.reset(seed=seed, options=reset_options)
        eval_env = Monitor(_StickyResetOptionsWrapper(eval_raw_env, reset_options=reset_options))
    
    # 3. Callbacks (Logs y Checkpoints)
    run_log_dir = _make_run_dir(log_dir, prefix="rl")
    run_model_dir = _make_run_dir(model_save_dir, prefix="rl")
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)
    metadata_basis_gates = basis_gates if mode == "synthesis" else None
    reward_function = getattr(raw_env, "reward_function", None)
    run_metadata = build_run_metadata(
            mode=mode,
            algorithm=algorithm,
            seed=seed,
            frontier_mode=frontier_mode,
            lookahead_window=lookahead_window,
            max_steps=max_steps,
            basis_gates=metadata_basis_gates,
            mask_semantics=mask_semantics,
            routing_mask_config=resolved_routing_mask_config,
            reward_config=(
                reward_function.to_dict()
                if hasattr(reward_function, "to_dict")
                else None
            ),
            training_hyperparams=effective_hyperparams,
            evaluation_config={
                "eval_freq": eval_freq,
                "n_eval_episodes": n_eval_episodes,
                "deterministic": True,
                **dict(evaluation_metadata or {}),
                **(
                    {
                        "early_stopping": {
                            "enabled": True,
                            "callback": "StopTrainingOnNoModelImprovement",
                            "min_evals": DEFAULT_EARLY_STOPPING_MIN_EVALS,
                            "max_no_improvement_evals": DEFAULT_EARLY_STOPPING_MAX_NO_IMPROVEMENT_EVALS,
                        }
                    }
                    if algorithm == "MaskablePPO" and mode == "routing" and use_reward_early_stopping
                    else {}
                ),
            },
    )
    save_run_metadata(run_model_dir, run_metadata)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=run_model_dir,
        name_prefix=f"rl_model_{mode}_{algorithm}"
    )

    eval_callback_cls = EvalCallback
    eval_callback_kwargs = {}
    if algorithm == "MaskablePPO" and mode == "routing":
        if MaskableEvalCallback is None:
            raise ModuleNotFoundError(
                "MaskablePPO routing requiere instalar sb3-contrib."
            )
        eval_callback_cls = MaskableEvalCallback
        if use_reward_early_stopping:
            eval_callback_kwargs["callback_after_eval"] = StopTrainingOnNoModelImprovement(
                min_evals=DEFAULT_EARLY_STOPPING_MIN_EVALS,
                max_no_improvement_evals=DEFAULT_EARLY_STOPPING_MAX_NO_IMPROVEMENT_EVALS,
                verbose=1 if verbose else 0,
            )
    
    eval_callback = eval_callback_cls(
        eval_env, 
        best_model_save_path=(
            os.path.join(run_model_dir, reward_best_model_subdir)
            if reward_best_model_subdir is not None
            else run_model_dir
        ),
        log_path=run_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        **eval_callback_kwargs,
    )
    
    callbacks = [checkpoint_callback, eval_callback, *(extra_callbacks or [])]
    callbacks.extend(
        factory(run_model_dir=run_model_dir, run_log_dir=run_log_dir)
        for factory in (extra_callback_factories or [])
    )
    
    # 4. Inicializar y entrenar Agente
    agent = QuantumRLAgent(
        env=env,
        algorithm=algorithm,
        tensorboard_log=run_log_dir,
        seed=seed,
        verbose=1 if verbose else 0,
        **effective_hyperparams
    )

    if routing_completed_at_reset:
        logger.info(
            "Routing training skipped because the target circuit is already executable on the initial layout."
        )
        final_path = os.path.join(run_model_dir, f"final_{mode}_{algorithm}.zip")
        agent.save(final_path)
        agent.last_model_path = final_path
        agent.run_model_dir = run_model_dir
        agent.run_log_dir = run_log_dir
        agent.best_model_path = None
        agent.actual_timesteps = 0
        agent.post_routing_selection = None
        agent.training_skipped_reason = "routing_completed_at_reset"
        run_metadata["training_runtime"] = {
            "actual_timesteps": 0,
            "stop_reason": "routing_completed_at_reset",
            "post_routing_selection": None,
        }
        save_run_metadata(run_model_dir, run_metadata)
        return agent
    
    _train_agent(agent, total_timesteps=total_timesteps, callbacks=callbacks, progress_bar=verbose)
    actual_timesteps = int(getattr(getattr(agent, "model", None), "num_timesteps", total_timesteps))
    for callback in callbacks:
        finalize = getattr(callback, "finalize", None)
        if callable(finalize):
            finalize(actual_timesteps=actual_timesteps)
    
    # 5. Guardar modelo final
    final_path = os.path.join(run_model_dir, f"final_{mode}_{algorithm}.zip")
    agent.save(final_path)
    agent.last_model_path = final_path
    agent.run_model_dir = run_model_dir
    agent.run_log_dir = run_log_dir
    agent.actual_timesteps = actual_timesteps
    agent.post_routing_selection = next(
        (
            callback.to_summary(actual_timesteps=actual_timesteps)
            for callback in callbacks
            if callable(getattr(callback, "to_summary", None))
        ),
        None,
    )
    run_metadata["training_runtime"] = {
        "actual_timesteps": actual_timesteps,
        "stop_reason": (
            agent.post_routing_selection.get("stop_reason")
            if agent.post_routing_selection is not None
            else "training_finished"
        ),
        "post_routing_selection": agent.post_routing_selection,
    }
    save_run_metadata(run_model_dir, run_metadata)

    best_model_path = os.path.join(run_model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        logger.info("Mejor artefacto evaluado disponible en %s", best_model_path)
        agent.best_model_path = best_model_path
    else:
        agent.best_model_path = None

    return agent
