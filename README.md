# TFG — Transpilación Cuántica Híbrida

**Optimización de Layout Multiobjetivo y Síntesis mediante Aprendizaje por Refuerzo**

> Hybrid Quantum Transpilation: Multi-Objective Layout Optimization and Reinforcement Learning Synthesis.

---

## Descripción

Proyecto de transpilación cuántica organizado en cuatro módulos: `qiskit_interface`, `mo_module`, `rl_module` e `integration`.

- **Optimización Multiobjetivo (MO)** — Algoritmos evolutivos (NSGA-II) para explorar layouts iniciales según múltiples métricas de calidad.
- **Aprendizaje por Refuerzo (RL)** — Entorno y agentes para routing/síntesis, con soporte para consumir un `initial_layout` genérico sin acoplarse a un productor concreto.
- **Integración** — `integration` posee el handoff MO -> RL y la orquestación de escenarios de evaluación `Baseline`, `MO_Only`, `RL_Only` y `MO+RL` en `src/integration/`.

El objetivo es superar las limitaciones de heurísticas como SABRE. MO y RL evolucionan como módulos separados mientras la integración define y evalua los flujos `Baseline`, `MO_Only`, `RL_Only` y `MO+RL`.

En el estado actual de integration v1, los escenarios basados en RL devuelven `episode summaries`, not final circuits. Reconstructing/exporting the final circuit from RL is left for a future iteration. QASM input is available for `Baseline` and `MO_Only` through `qasm_file`, mientras que los escenarios basados en RL todavía no exponen una entrada QASM equivalente en su superficie pública.

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
└── integration/        # Módulo 4: Orquestación MO->RL y evaluación de routing v1
```

Para más detalles sobre la arquitectura y los contratos entre módulos, ver [agents.md](docs/agents.md).

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
