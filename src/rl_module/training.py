"""
Script de Entrenamiento del Agente de RL.

Proporciona la orquestación principal para inicializar el entorno, configurar
las semillas globales, configurar los callbacks (TensorBoard y Checkpoints)
y lanzar el proceso de entrenamiento.
"""

import os
import random
import numpy as np
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from .agent import QuantumRLAgent
from .environment import QuantumTranspilationEnv
from qiskit import QuantumCircuit
from typing import Tuple, List, Optional

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
    print(f"Semillas globales fijadas a {seed}.")


def setup_training_pipeline(
    target_circuit: QuantumCircuit,
    coupling_map: List[Tuple[int, int]],
    mode: str = "routing",
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    seed: int = 42,
    log_dir: str = "./experiments/logs/rl_logs",
    model_save_dir: str = "./experiments/models/rl_models",
    hyperparams: Optional[dict] = None
) -> QuantumRLAgent:
    """
    Configura y lanza el pipeline completo de entrenamiento del agente.
    
    Args:
        target_circuit: Circuito cuántico a transpilar.
        coupling_map: Mapa de acoplamiento físico (lista de aristas).
        mode: "routing" o "synthesis".
        algorithm: "PPO" o "DQN".
        total_timesteps: Pasos de entrenamiento.
        seed: Semilla de reproducibilidad.
        log_dir: Directorio para logs de TensorBoard.
        model_save_dir: Directorio para guardar los checkpoints.
        hyperparams: Diccionario con hiperparámetros para PPO/DQN.
        
    Returns:
        El agente entrenado listo para evaluación.
    """
    
    # 1. Semillas
    set_global_seeds(seed)
    
    # 2. Configurar Entorno
    # Usamos Monitor para que SB3 registre las métricas automáticas (rewards, episode lengths)
    raw_env = QuantumTranspilationEnv(
        target_circuit=target_circuit,
        coupling_map=coupling_map,
        mode=mode
    )
    env = Monitor(raw_env)
    
    # Entorno de Evaluación independiente
    eval_raw_env = QuantumTranspilationEnv(
        target_circuit=target_circuit,
        coupling_map=coupling_map,
        mode=mode
    )
    eval_env = Monitor(eval_raw_env)
    
    # 3. Callbacks (Logs y Checkpoints)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=model_save_dir,
        name_prefix=f"rl_model_{mode}_{algorithm}"
    )
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=model_save_dir,
        log_path=log_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # 4. Inicializar y entrenar Agente
    if hyperparams is None:
        hyperparams = {}
        
    agent = QuantumRLAgent(
        env=env,
        algorithm=algorithm,
        tensorboard_log=log_dir,
        seed=seed,
        **hyperparams
    )
    
    agent.train(total_timesteps=total_timesteps, callbacks=callbacks)
    
    # 5. Guardar modelo final
    final_path = os.path.join(model_save_dir, f"final_{mode}_{algorithm}.zip")
    agent.save(final_path)
    
    return agent
