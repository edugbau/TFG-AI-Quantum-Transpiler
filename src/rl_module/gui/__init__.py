"""
gui — Sub-módulo de interfaz gráfica para pruebas del módulo RL
================================================================

Proporciona una GUI con CustomTkinter para configurar, entrenar
y evaluar el agente RL de forma aislada.

Uso::

    python -m src.rl_module.gui.rl_gui

Autor: Eduardo González Bautista
Fecha: 2026-03-05
"""

from .rl_gui import RLBenchmarkGUI

__all__ = ["RLBenchmarkGUI"]
