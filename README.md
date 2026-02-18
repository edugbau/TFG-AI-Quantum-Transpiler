# TFG — Transpilación Cuántica Híbrida

**Optimización de Layout Multiobjetivo y Síntesis mediante Aprendizaje por Refuerzo**

> Hybrid Quantum Transpilation: Multi-Objective Layout Optimization and Reinforcement Learning Synthesis.

---

## Descripción

Pipeline híbrido de transpilación de circuitos cuánticos que combina:

1. **Optimización Multiobjetivo (MO)** — Algoritmos evolutivos (NSGA-II) para generar layouts iniciales óptimos considerando múltiples métricas de calidad simultáneamente
2. **Aprendizaje por Refuerzo (RL)** — Agente de síntesis de circuitos Clifford (PPO/DQN) que recibe los layouts optimizados como entrada

El objetivo es superar las limitaciones de heurísticas como SABRE, que utilizan inicializaciones aleatorias y mono-objetivo.

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd TFG-Quantum-Transpiler

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Estructura del Proyecto

```
src/
├── qiskit_interface/   # Módulo 1: Interfaz con Qiskit
├── rl_module/          # Módulo 2: Aprendizaje por Refuerzo
├── mo_module/          # Módulo 3: Optimización Multiobjetivo
└── integration/        # Módulo 4: Pipeline híbrido y experimentación
```

Para más detalles sobre la arquitectura, ver [agents.md](docs/agents.md).

## Entorno Tecnológico

| Componente | Versión |
|---|---|
| Python | 3.10.8 |
| Qiskit | 2.3.0 |
| PyTorch | 2.5.1+cu121 |
| Gymnasium | 1.2.3 |
| pymoo | 0.6.1.6 |
| GPU | NVIDIA RTX 3060 (CUDA 12.1) |

Ver detalles completos en [ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Referencias

- [Qiskit Documentation](https://docs.quantum.ibm.com/)
- [AI-driven circuit synthesis (arXiv:2405.13196)](https://arxiv.org/abs/2405.13196)
- [pymoo (IEEE Access, 2020)](https://ieeexplore.ieee.org/document/9078759)
- [Gymnasium (Farama Foundation)](https://gymnasium.farama.org/)
- [Stable Baselines3 (JMLR, 2021)](http://jmlr.org/papers/v22/20-1364.html)

## Autor

**Eduardo González Bautista** — Universidad de Málaga, E.T.S. de Ingeniería Informática  
Tutores: Gabriel Jesús Luque Polo, Zakaria Abdelmoiz Dahi
