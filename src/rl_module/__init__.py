"""
Módulo de Aprendizaje por Refuerzo (RL Module).

Este módulo provee el entorno Gymnasium y los agentes necesarios para la transpilación
cuántica híbrida, incluyendo enrutamiento (routing) y síntesis completa (synthesis).
"""

from importlib import import_module


_EXPORTS = {
    "QuantumTranspilationEnv": (".environment", "QuantumTranspilationEnv"),
    "RoutingStrategy": (".env_strategies", "RoutingStrategy"),
    "SynthesisStrategy": (".env_strategies", "SynthesisStrategy"),
    "LookaheadEntry": (".frontier", "LookaheadEntry"),
    "DagFrontier": (".frontier", "DagFrontier"),
    "SequentialFrontier": (".frontier", "SequentialFrontier"),
    "RoutingReward": (".rewards", "RoutingReward"),
    "SynthesisReward": (".rewards", "SynthesisReward"),
    "QuantumRLAgent": (".agent", "QuantumRLAgent"),
    "setup_training_pipeline": (".training", "setup_training_pipeline"),
    "set_global_seeds": (".training", "set_global_seeds"),
}

_SUBMODULES = {
    "training": ".training",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attribute_name = _EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute_name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        value = import_module(_SUBMODULES[name], __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(__all__) + list(_SUBMODULES))
