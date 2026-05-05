# TFG — Transpilación Cuántica Híbrida

**Optimización de Layout Multiobjetivo y Síntesis mediante Aprendizaje por Refuerzo**

> Hybrid Quantum Transpilation: Multi-Objective Layout Optimization and Reinforcement Learning Synthesis.

---

## Descripción

Proyecto de transpilación cuántica organizado en cuatro módulos: `qiskit_interface`, `mo_module`, `rl_module` e `integration`.

- **Optimización Multiobjetivo (MO)** — Algoritmos evolutivos (NSGA-II) para explorar layouts iniciales según múltiples métricas de calidad.
- **Aprendizaje por Refuerzo (RL)** — Entorno y agentes para routing/síntesis, con soporte para consumir un `initial_layout` genérico sin acoplarse a un productor concreto.
- **Integración** — `integration` posee el handoff MO -> RL, la orquestación de Scenarios `Baseline`, `MO_Only`, `RL_Only` y `MO+RL`, y la orquestación de Campaigns `train+eval` en `src/integration/`.

El objetivo es superar las limitaciones de heurísticas como SABRE. MO y RL evolucionan como módulos separados mientras la integración define y compara los Scenarios `Baseline`, `MO_Only`, `RL_Only` y `MO+RL`, y además ejecuta Campaigns reproducibles de `train+eval` sobre Campaign Cases `circuit x backend`.

En routing, `rl_module` mantiene un espacio de acción discreto fijo (`fixed`) sobre las aristas del coupling map y añade un nuevo régimen de **masked routing**. Este régimen no cambia el catálogo base de acciones: aplica `action_masks()` como una hard mask determinista y frontier-aware para restringir, al estilo SABRE, qué swaps candidatos puede muestrear la política en cada estado.

Los modelos de routing guardados por `rl_module` pueden incluir un sidecar `run_metadata.json` junto al modelo para describir el contrato de evaluación consumible por `integration`, manteniendo desacoplados `mo_module` y `rl_module`. Cuando ese sidecar contiene metadata versionada de masked routing, `integration` la consume; si no está presente, se conserva el fallback legacy/default para checkpoints previos.

`MaskablePPO` pasa a ser el estándar para checkpoints nuevos de masked routing. Los checkpoints legacy de `PPO` y `DQN` siguen soportados mediante contratos legacy/default o flujos unmasked, por lo que ambos regímenes coexisten durante la transición.

En el estado actual de integration v1, `RL_Only` sigue devolviendo `episode summaries`, not final circuits. `MO+RL`, en cambio, ya reconstruye el circuito ruteado desde la traza RL: usa `executed_gate_trace` cuando está disponible para reproducir exactamente las puertas ejecutadas y `swap_trace` para materializar los swaps físicos, con `total_swaps == len(swap_trace)` como contador de swaps realmente materializados, y después ejecuta las fases post-routing de Qiskit cuando el episodio RL completa el routing; si no completa, devuelve un resultado controlado sin artefacto de transpilación. QASM input is available for `Baseline` and `MO_Only` through `qasm_file`, mientras que los escenarios basados en RL todavía no exponen una entrada QASM equivalente en su superficie pública. Para comparar layouts dispersos en los artefactos/resultados de Qiskit, `trans_num_qubits`/`trans_width` siguen representando anchura física materializada, mientras que `trans_active_qubits` refleja los qubits físicamente activos del circuito transpìlado.

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

### Arquitectura y Contratos entre Módulos

| Módulo | Responsabilidad | No Responsable De |
| --- | --- | --- |
| `src/qiskit_interface/` | Backends, transpilación, métricas y baselines | Orquestación MO -> RL |
| `src/rl_module/` | Entorno Gymnasium, recompensas, entrenamiento del agente e ingestión genérica de `initial_layout` | Producir layouts u orquestar Campaigns |
| `src/mo_module/` | Búsqueda multiobjetivo de layouts, frentes de Pareto y evaluación de layouts | Dirigir RL directamente |
| `src/integration/` | Orquestación de Scenarios, Campaign orchestration, scenario comparison, persistence, Summary Document y handoff MO -> RL | Reimplementar internos de los módulos |

**Convención de layout compartido:**

```python
layout[i] = physical_qubit_for_logical_qubit_i
```

- `qiskit_interface` puede evaluar este layout mediante helpers de transpilación.
- `rl_module` puede ingerirlo a través de `env.reset(options={"initial_layout": layout})`.
- `integration` posee el proceso que conecta productor y consumidor.

### Campaigns de Integración

`src/integration/` ya no se limita a Scenarios unitarios. Ahora también soporta una **Train+Eval Campaign** reproducible compuesta por uno o más **Campaign Cases**. Cada Campaign Case corresponde a una combinación `circuit x backend` y ejecuta la comparación canónica `Baseline`, `MO_Only` y `MO+RL`.

Dentro de esa comparación guiada, `MO_Only` es el Scenario que selecciona el layout. El training de Campaign para `MO+RL` arranca desde ese layout exacto y la evaluación posterior de `MO+RL` reutiliza ese mismo layout junto con el Training Artifact producido para el mismo Campaign Case.

`RL_Only` sigue existiendo como Scenario, pero queda fuera del flujo guiado principal de Campaign.

La guided CLI ofrece dos caminos:

- **Default Campaign**: usa valores canónicos compartidos para RL y MO, con un flujo breve y reproducible.
- **Advanced Campaign**: permite ajustar explícitamente backend(s), configuración RL, parámetros MO y política de selección de layout.

La capa subyacente de `integration` puede trabajar con el catálogo actual de fake backends publicado por `qiskit_interface`, pero la guided Campaign CLI hoy expone un subconjunto más estrecho: `fake_torino` y `fake_brisbane`.

Cada Campaign persiste al menos:

- `summary.md` como Summary Document con metadata, aggregate comparison, per-case detail, training notes e incidents;
- `campaign.json` como salida estructurada de la Campaign;
- `cases/<case>/result.json` como persistencia por caso.

Para el estado de comparación, las secciones agregadas del Summary Document y los incidents son la referencia principal para detectar Campaign Cases no comparables. En la implementación actual, el Campaign status superior puede seguir siendo `completed` y un case status puede seguir siendo `completed` aunque falte un bundle comparable completo de métricas.

Los límites de ownership se mantienen explícitos:

- `integration` owns Campaign orchestration, scenario comparison, persistence, Summary Document y el handoff MO -> RL.
- `rl_module` owns how RL training is implemented and how checkpoints are produced.
- `mo_module` owns layout generation/selection inputs.
- En el camino híbrido de Campaign, `MO_Only` selecciona el layout, `integration` lo reenvía como `initial_layout` al training RL y la evaluación `MO+RL` reutiliza ese mismo layout junto con el Training Artifact del caso.

**Estado actual:**
- `src/integration/` implementa la orquestación v1 de `Baseline`, `MO_Only`, `RL_Only` y `MO+RL` para evaluación de routing.
- `src/integration/` implementa Campaign contracts, training bridge, campaign reporting/summary rendering, sequential campaign runner y guided campaign CLI.
- En la comparación guiada de Campaign, `MO_Only` selecciona el layout y `MO+RL` entrena/evalúa desde ese layout exacto para el mismo Campaign Case usando el Training Artifact resultante.
- `src/rl_module/` soporta routing y un primer modo de `synthesis` entrenable restringido a circuitos Clifford.
- `mo_module` y `rl_module` deben permanecer testeables de forma independiente.
- La comparación canónica de Campaign usa `Baseline`, `MO_Only` y `MO+RL`; `RL_Only` queda fuera del flujo guiado principal.
- `RL_Only` devuelve `episode summaries`, no circuitos finales.
- `MO+RL` reconstruye el circuito ruteado desde la traza RL (`executed_gate_trace` + `swap_trace`) y ejecuta post-routing de Qiskit cuando el episodio completa el routing; si no, devuelve un resultado controlado sin transpilación final.
- En `RoutingEpisodeSummary`, `total_swaps == len(swap_trace)` y representa swaps realmente materializados/reproducibles.
- En métricas Qiskit, `trans_num_qubits`/`trans_width` siguen siendo anchura física materializada; usar `trans_active_qubits` para comparar ocupación física real cuando el layout es disperso.
- QASM input está disponible para `Baseline` y `MO_Only` mediante `qasm_file`; los escenarios RL aún no exponen una entrada QASM equivalente.

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
