"""
Wrapper para el Agente de Aprendizaje por Refuerzo utilizando Stable-Baselines3.

Este módulo encapsula la inicialización, entrenamiento y predicción de los 
modelos PPO/DQN, gestionando el uso de GPU (PyTorch) automáticamente.
"""

import os
import importlib
import logging
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
import gymnasium as gym
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)  # FIX #5: logging en vez de print

class QuantumRLAgent:
    """
    Agente de RL para el entorno de transpilación cuántica.
    Actúa como un wrapper de configuración y ejecución para modelos de Stable-Baselines3.
    """
    
    ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
        "PPO": PPO,
        "DQN": DQN
    }

    @classmethod
    def _get_algorithm_class(cls, algorithm: str) -> Type[BaseAlgorithm]:
        if algorithm in cls.ALGORITHMS:
            return cls.ALGORITHMS[algorithm]
        if algorithm == "MaskablePPO":
            try:
                sb3_contrib = importlib.import_module("sb3_contrib")
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "MaskablePPO requiere instalar sb3-contrib."
                ) from exc
            return sb3_contrib.MaskablePPO
        raise ValueError(f"Algoritmo {algorithm} no soportado. Usar: {list(cls.ALGORITHMS.keys()) + ['MaskablePPO']}")

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
        AlgorithmClass = self._get_algorithm_class(algorithm)

        self.algorithm_name = algorithm
        self.env = env
        
        # Detección automática de dispositivo (CUDA si está disponible)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[%s] Inicializando agente en dispositivo: %s", algorithm, self.device.upper())
        
        # Inyectar hiperparámetros por defecto para DQN si no se proporcionan,
        # para estabilizar el entrenamiento en un entorno de penalizaciones densas.
        if algorithm == "DQN":
            kwargs.setdefault("exploration_fraction", 0.5)
            kwargs.setdefault("tau", 0.05)
            kwargs.setdefault("learning_starts", 1000)

        kwargs.setdefault("seed", None)
            
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
        logger.info("Iniciando entrenamiento por %d timesteps...", total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=progress_bar)
        return self.model

    def predict(
        self,
        observation: Dict[str, Any],
        deterministic: bool = True,
        **kwargs: Any,
    ) -> tuple:
        """
        Pide al agente que prediga una acción basada en una observación.
        
        Args:
            observation: La observación actual del entorno.
            deterministic: Si usar una política determinista.
            
        Returns:
            Acción predicha por el modelo y el estado interno.
        """
        action, states = self.model.predict(observation, deterministic=deterministic, **kwargs)
        return action, states

    def save(self, path: str):
        """Guarda los pesos del modelo en disco."""
        # FIX #7: Guard clause para rutas sin directorio padre
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.model.save(path)
        logger.info("Modelo guardado en %s", path)

    @classmethod
    def load(cls, path: str, env: gym.Env, algorithm: str = "PPO", **kwargs) -> 'QuantumRLAgent':
        """
        Carga un modelo entrenado desde el disco.
        
        FIX #6: Evita crear un modelo que se descarta inmediatamente.
        Usa object.__new__ para construir sin pasar por __init__ completo.
        """
        AlgorithmClass = cls._get_algorithm_class(algorithm)
        
        # Crear instancia sin __init__ para evitar instanciar un modelo descartable
        agent = object.__new__(cls)
        agent.algorithm_name = algorithm
        agent.env = env
        agent.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("Cargando modelo desde %s", path)
        agent.model = AlgorithmClass.load(path, env=env, device=agent.device)
        return agent
