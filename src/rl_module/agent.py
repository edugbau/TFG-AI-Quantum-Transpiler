"""
Wrapper para el Agente de Aprendizaje por Refuerzo utilizando Stable-Baselines3.

Este módulo encapsula la inicialización, entrenamiento y predicción de los 
modelos PPO/DQN, gestionando el uso de GPU (PyTorch) automáticamente.
"""

import os
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
import gymnasium as gym
from typing import Any, Dict, Optional, Type

class QuantumRLAgent:
    """
    Agente de RL para el entorno de transpilación cuántica.
    Actúa como un wrapper de configuración y ejecución para modelos de Stable-Baselines3.
    """
    
    ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
        "PPO": PPO,
        "DQN": DQN
    }

    def __init__(
        self, 
        env: gym.Env, 
        algorithm: str = "PPO", 
        policy: str = "MultiInputPolicy", 
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        **kwargs: Any
    ):
        """
        Inicializa el agente RL.
        
        Args:
            env: Entorno Gymnasium a utilizar.
            algorithm: 'PPO' o 'DQN'.
            policy: Política de la red neuronal (por defecto MultiInputPolicy al usar Dict spaces).
            tensorboard_log: Ruta para guardar logs de TensorBoard.
            verbose: Nivel de verbosidad (0, 1 o 2).
            **kwargs: Hiperparámetros adicionales para el modelo SB3.
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Algoritmo {algorithm} no soportado. Usar: {list(self.ALGORITHMS.keys())}")
            
        self.algorithm_name = algorithm
        self.env = env
        
        # Detección automática de dispositivo (CUDA si está disponible)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{algorithm}] Inicializando agente en dispositivo: {self.device.upper()}")
        
        AlgorithmClass = self.ALGORITHMS[algorithm]
        
        # Instanciar modelo de SB3
        self.model = AlgorithmClass(
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=self.device,
            **kwargs
        )

    def train(self, total_timesteps: int, callbacks: Optional[list] = None, progress_bar: bool = True) -> BaseAlgorithm:
        """
        Entrena el agente en el entorno proporcionado.
        
        Args:
            total_timesteps: Número de pasos de entrenamiento.
            callbacks: Lista de callbacks (ej. CheckpointCallback, EvalCallback).
            progress_bar: Si mostrar barra de progreso (requiere tqdm y rich).
            
        Returns:
            El modelo entrenado.
        """
        print(f"Iniciando entrenamiento por {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=progress_bar)
        return self.model

    def predict(self, observation: Dict[str, Any], deterministic: bool = True) -> tuple:
        """
        Pide al agente que prediga una acción basada en una observación.
        
        Args:
            observation: La observación actual del entorno.
            deterministic: Si usar una política determinista.
            
        Returns:
            Acción predicha por el modelo y el estado interno.
        """
        action, states = self.model.predict(observation, deterministic=deterministic)
        return action, states

    def save(self, path: str):
        """Guarda los pesos del modelo en disco."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Modelo guardado en {path}")

    @classmethod
    def load(cls, path: str, env: gym.Env, algorithm: str = "PPO", **kwargs) -> 'QuantumRLAgent':
        """
        Carga un modelo entrenado desde el disco.
        """
        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"Algoritmo {algorithm} no soportado.")
            
        AlgorithmClass = cls.ALGORITHMS[algorithm]
        agent = cls(env=env, algorithm=algorithm, **kwargs)
        
        print(f"Cargando modelo desde {path}")
        agent.model = AlgorithmClass.load(path, env=env, device=agent.device)
        return agent
