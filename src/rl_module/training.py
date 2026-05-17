"""
Script de Entrenamiento del Agente de RL.

Proporciona la orquestación principal para inicializar el entorno, configurar
las semillas globales, configurar los callbacks (TensorBoard y Checkpoints)
y lanzar el proceso de entrenamiento.
"""

import os
import logging
import random
from datetime import datetime
import numpy as np
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from .agent import QuantumRLAgent
from .environment import QuantumTranspilationEnv
from .model_metadata import build_run_metadata, save_run_metadata
from qiskit import QuantumCircuit
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PPO_STABILITY_HYPERPARAMS = {
    "learning_rate": 1e-4,
    "clip_range": 0.1,
    "target_kl": 0.03,
}
DEFAULT_N_EVAL_EPISODES = 5
DEFAULT_EVAL_FREQ = 5_000
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
    n_eval_episodes: int = DEFAULT_N_EVAL_EPISODES,
    eval_freq: int = DEFAULT_EVAL_FREQ,
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
    save_run_metadata(
        run_model_dir,
        build_run_metadata(
            mode=mode,
            algorithm=algorithm,
            seed=seed,
            frontier_mode=frontier_mode,
            lookahead_window=lookahead_window,
            max_steps=max_steps,
            basis_gates=metadata_basis_gates,
            training_hyperparams=effective_hyperparams,
            evaluation_config={
                "eval_freq": eval_freq,
                "n_eval_episodes": n_eval_episodes,
                "deterministic": True,
            },
        ),
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=run_model_dir,
        name_prefix=f"rl_model_{mode}_{algorithm}"
    )

    eval_callback_cls = EvalCallback
    if algorithm == "MaskablePPO" and mode == "routing":
        if MaskableEvalCallback is None:
            raise ModuleNotFoundError(
                "MaskablePPO routing requiere instalar sb3-contrib."
            )
        eval_callback_cls = MaskableEvalCallback
    
    eval_callback = eval_callback_cls(
        eval_env, 
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # 4. Inicializar y entrenar Agente
    agent = QuantumRLAgent(
        env=env,
        algorithm=algorithm,
        tensorboard_log=run_log_dir,
        seed=seed,
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
        agent.training_skipped_reason = "routing_completed_at_reset"
        return agent
    
    agent.train(total_timesteps=total_timesteps, callbacks=callbacks)
    
    # 5. Guardar modelo final
    final_path = os.path.join(run_model_dir, f"final_{mode}_{algorithm}.zip")
    agent.save(final_path)
    agent.last_model_path = final_path
    agent.run_model_dir = run_model_dir
    agent.run_log_dir = run_log_dir

    best_model_path = os.path.join(run_model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        logger.info("Mejor artefacto evaluado disponible en %s", best_model_path)
        agent.best_model_path = best_model_path
    else:
        agent.best_model_path = None

    return agent
