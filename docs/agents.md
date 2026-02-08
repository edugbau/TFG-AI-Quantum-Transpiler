# agents.md — Guía para Agentes IA / Copilot

## Descripción del Proyecto

**Transpilación Cuántica Híbrida: Optimización de Layout Multiobjetivo y Síntesis mediante Aprendizaje por Refuerzo**

> Hybrid Quantum Transpilation: Multi-Objective Layout Optimization and Reinforcement Learning Synthesis.

**Alumno:** Eduardo González Bautista  
**Tutores:** Gabriel Jesús Luque Polo, Zakaria Abdelmoiz Dahi  
**Titulación:** Grado en Ingeniería del Software — E.T.S. de Ingeniería Informática, Universidad de Málaga

---

## Contexto del Problema

La transpilación de circuitos cuánticos adapta algoritmos abstractos a las restricciones físicas del hardware real (conectividad entre qubits, conjunto de puertas nativas, etc.). Actualmente, enfoques como el de IBM usan **Aprendizaje por Refuerzo (RL)** para la síntesis de circuitos, pero dependen de heurísticas estáticas y aleatorias (como **SABRE**) para la fase de layout inicial (mapeo de qubits lógicos a físicos). Esta inicialización subóptima limita el rendimiento del agente RL y la calidad del circuito final transpilado.

Este TFG propone un **enfoque híbrido** que sustituye la inicialización heurística por una fase de **Optimización Multiobjetivo (MO)** que considera simultáneamente múltiples métricas de calidad (profundidad del circuito, número de CNOTs, tasas de error, decoherencia, etc.), generando layouts iniciales de alta calidad que alimentan al agente de RL.

---

## Arquitectura Modular del Proyecto

El proyecto se organiza en **4 módulos principales** que interactúan entre sí en un pipeline secuencial:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   1. MÓDULO      │     │   3. MÓDULO MO    │     │   2. MÓDULO RL   │
│   QISKIT         │────▶│   (Multiobjetivo) │────▶│   (Refuerzo)     │
│   (Interfaz)     │     │                   │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │   4. MÓDULO INTEGRACIÓN   │
                    │   (Experimentación)       │
                    └──────────────────────────┘
```

### Módulo 1 — Interfaz Qiskit (`qiskit_interface/`)

Módulo sencillo para interactuar con Qiskit. Se encarga de:

- **Carga y representación** de circuitos cuánticos (QuantumCircuit)
- **Información del backend**: topología del dispositivo, puertas nativas, mapa de acoplamiento (coupling map)
- **Transpilación estándar** de Qiskit como baseline (niveles 0–3)
- **Extracción de métricas** del circuito: profundidad (depth), conteo de puertas CNOT/CX, número total de operaciones
- **Utilidades**: conversión entre representaciones, visualización de circuitos, manejo de Clifford circuits

**Dependencias principales:** `qiskit >= 2.0`, `qiskit-aer`, `qiskit-ibm-runtime`

> ⚠️ **CRÍTICO: Protocolo de Versión Qiskit 2.3.0 (Qiskit 1.0+)**
> 
> Este proyecto opera bajo **Qiskit 2.3.0**. Las versiones 1.0+ introdujeron "breaking changes" masivos. El código generado por conocimiento previo a 2024 suele fallar.
>
> **REGLAS ABSOLUTAS E INVIOLABLES DE QISKIT:**
> 1. **Estructura de Paquetes (The Monolith is Gone):** 
>    - ❌ NUNCA importar: `qiskit.terra`, `qiskit.aer` (paquete legacy), `qiskit.ignis`, `qiskit.aqua`, `qiskit.finance`, `qiskit.optimization`, `qiskit.ml`.
>    - ✅ IMPORTAR ASÍ: `import qiskit`, `from qiskit import QuantumCircuit`, `from qiskit.transpiler import ...`.
>    - ✅ Simulador: `import qiskit_aer` (¡con guion bajo!).
> 
> 2. **Ejecución y Backend:**
>    - ❌ NUNCA usar: `qiskit.execute()`, `QuantumInstance`, `qiskit.providers.ibmq`.
>    - ✅ USAR: `backend.run(transpiled_circuit)` o Primitivas (`SamplerV2`, `EstimatorV2`).
> 
> 3. **Circuitos y Operadores:**
>    - ❌ Métodos obsoletos: `circuit.qasm()` (usar `qasm2` module), `bind_parameters` (ciertos contextos), `snapshot`.
>    - ✅ Parametrización: Usar `assign_parameters()`.
>
> 4. **Algoritmos:**
>    - Todo `qiskit.algorithms` está **DEPRECADO/ELIMINADO**. Usar librerías específicas (o `scipy` + primitivas si es necesario reimplementar).
>
> **Si dudas, consulta la [Guía de Migración a Qiskit 1.0](https://docs.quantum.ibm.com/migration-guides). El código que no cumpla esto será rechazado sistemáticamente.**

#### Política de Backends (Sin API Keys)
**IMPORTANTE:** No se utilizarán backends reales ni claves de API (IBM Quantum Token). 
- Se utilizarán exclusivamente **Fake Backends** (e.g., `FakeTorino`, `FakeSherbrooke`, `FakeBrisbane`) disponibles en `qiskit_ibm_runtime.fake_provider` o generados artificialmente.
- El objetivo es simular la topología, conectividad y conjunto de puertas de un dispositivo real para la transpilación.
- **No se envía ningún circuito a ejecución remota**. Solo buscamos los **resultados de la transpilación** y sus **métricas** (profundidad, CNOTs, etc.).

### Módulo 2 — Aprendizaje por Refuerzo (`rl_module/`)

Entorno y agente de RL para la síntesis de circuitos Clifford. Se encarga de:

- **Entorno Gymnasium**: definición del espacio de estados, acciones y recompensas para la síntesis de circuitos
- **Agente de síntesis**: implementación basada en algoritmos como PPO o DQN
- **Entrenamiento y evaluación**: loops de entrenamiento con logging, callbacks y checkpointing
- **Línea base funcional**: comparable al estado del arte (enfoque PPO de IBM, ref. arXiv:2405.13196)

**Dependencias principales:** `gymnasium`, `stable-baselines3`, `torch` (PyTorch con CUDA)

### Módulo 3 — Optimización Multiobjetivo (`mo_module/`)

Módulo de optimización de layouts basado en algoritmos evolutivos. Se encarga de:

- **Algoritmos evolutivos**: implementación/uso de NSGA-II, MOEA/D u otros
- **Funciones de fitness extensibles**: profundidad del circuito, número de CNOTs, tasas de error, decoherencia
- **Representación**: codificación del layout (mapeo de qubits lógicos → físicos) como individuo del algoritmo
- **Frente de Pareto**: generación de un conjunto de soluciones no-dominadas que representan trade-offs entre objetivos
- **Calibración de hiperparámetros**: tamaño de población, operadores de cruce/mutación, criterios de parada

**Dependencias principales:** `pymoo`, `numpy`, `scipy`

### Módulo 4 — Integración y Experimentación (`integration/`)

Módulo orquestador del pipeline híbrido MO → RL. Se encarga de:

- **Pipeline híbrido**: el módulo MO genera layouts optimizados que alimentan la entrada del agente RL
- **Benchmarking**: comparación sistemática del enfoque híbrido (MO+RL) frente a heurísticas estándar (SABRE, etc.)
- **Métricas de evaluación**: calidad del circuito final transpilado (profundidad, CNOTs, fidelidad)
- **Análisis estadístico**: tests de significancia, gráficas de rendimiento, logs de entrenamiento
- **Reproducibilidad**: seeds, configuraciones exportables, scripts de ejecución

**Dependencias principales:** todas las anteriores + `matplotlib`, `pandas`

---

## Estructura de Directorios Propuesta

```
TFG-Quantum-Transpiler/
├── agents.md                  # Este archivo — guía para agentes IA
├── README.md                  # Documentación general del proyecto
├── ENVIRONMENT.md             # Detalle del entorno tecnológico y versiones
├── requirements.txt           # Dependencias Python del proyecto
├── setup.py / pyproject.toml  # Configuración del paquete (futuro)
│
├── src/
│   ├── __init__.py
│   ├── qiskit_interface/      # Módulo 1: Interfaz con Qiskit
│   │   ├── __init__.py
│   │   ├── circuit_utils.py   # Carga, conversión, métricas de circuitos
│   │   ├── backend_info.py    # Topología, coupling maps, puertas nativas
│   │   └── transpiler.py      # Transpilación estándar de Qiskit (baseline)
│   │
│   ├── rl_module/             # Módulo 2: Aprendizaje por Refuerzo
│   │   ├── __init__.py
│   │   ├── environment.py     # Entorno Gymnasium para síntesis de circuitos
│   │   ├── agent.py           # Agente RL (PPO/DQN via Stable Baselines3)
│   │   ├── rewards.py         # Funciones de recompensa
│   │   └── training.py        # Loop de entrenamiento, callbacks, checkpoints
│   │
│   ├── mo_module/             # Módulo 3: Optimización Multiobjetivo
│   │   ├── __init__.py
│   │   ├── optimizer.py       # Algoritmo evolutivo (NSGA-II, MOEA/D)
│   │   ├── fitness.py         # Funciones de fitness (profundidad, CNOTs, error)
│   │   ├── encoding.py        # Codificación del layout como individuo
│   │   └── pareto.py          # Análisis del frente de Pareto
│   │
│   └── integration/           # Módulo 4: Integración y Experimentación
│       ├── __init__.py
│       ├── pipeline.py        # Pipeline híbrido MO → RL
│       ├── benchmark.py       # Suite de benchmarks comparativos
│       ├── analysis.py        # Análisis estadístico y visualización
│       └── config.py          # Configuración centralizada de experimentos
│
├── tests/                     # Tests unitarios y de integración
│   ├── test_qiskit_interface/
│   ├── test_rl_module/
│   ├── test_mo_module/
│   └── test_integration/
│
├── notebooks/                 # Jupyter notebooks (exploración y demos)
│   └── descargados/
│       └── chsh-inequality.ipynb
│
├── experiments/               # Resultados experimentales
│   ├── configs/               # Configuraciones de experimentos
│   ├── logs/                  # Logs de entrenamiento (TensorBoard, etc.)
│   ├── results/               # Métricas y resultados
│   └── plots/                 # Gráficas generadas
│
├── benchmarks/                # Circuitos cuánticos de prueba
│   └── circuits/              # Archivos QASM o QuantumCircuit serializados
│
└── docs/                      # Documentación adicional
    └── references/            # Papers y referencias bibliográficas
```

---

## Entorno Tecnológico Verificado

| Componente | Versión | Notas |
|---|---|---|
| **Python** | 3.10.8 | Entorno virtual (`.venv/`) |
| **Qiskit** | **2.3.0** | ⚠️ Arquitectura nueva (post 1.0). NO usar imports legacy |
| **Qiskit Aer** | 0.17.2 | Simulador de circuitos cuánticos |
| **Qiskit IBM Runtime** | 0.43.1 | Acceso a hardware IBM Quantum |
| **PyTorch** | 2.5.1+cu121 | Con soporte CUDA 12.1 |
| **Gymnasium** | 1.2.3 | Framework de entornos RL (sucesor de OpenAI Gym) |
| **pymoo** | 0.6.1.6 | Optimización multiobjetivo en Python |
| **Stable Baselines3** | 2.7.1 | Algoritmos RL (PPO, DQN, A2C, SAC, etc.) |
| **NumPy** | 2.2.6 | Computación numérica |
| **SciPy** | 1.15.3 | Computación científica |
| **Matplotlib** | 3.10.8 | Visualización |
| **Pandas** | 2.3.3 | Análisis de datos |
| **GPU** | NVIDIA RTX 3060 Laptop | CUDA 12.1 disponible |

### Librerías adicionales instaladas

| Componente | Versión | Notas |
|---|---|---|
| **Qiskit IBM Transpiler** | 0.16.0 | AI transpiler passes de IBM |
| **TensorBoard** | 2.20.0 | Logging de entrenamiento RL |


---

## Pipeline Conceptual

```
Circuito Cuántico Abstracto
         │
         ▼
 ┌───────────────────┐
 │ Módulo 1: Qiskit  │  Extraer info e info del backend (FakeBackend)
 │ (Interfaz)        │  (coupling map, puertas nativas, métricas iniciales)
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │ Módulo 3: MO      │  Optimizar layout inicial con algoritmo evolutivo
 │ (Multiobjetivo)   │  multiobjetivo (NSGA-II). Genera frente de Pareto
 │                   │  con layouts que minimizan profundidad, CNOTs, error...
 └────────┬──────────┘
          │  Layout(s) optimizado(s)
          ▼
 ┌───────────────────┐
 │ Módulo 2: RL      │  Agente de síntesis de circuitos Clifford
 │ (Refuerzo)        │  recibe layout optimizado como estado inicial.
 │                   │  Entrena/ejecuta política PPO/DQN para sintetizar
 │                   │  el circuito final transpilado.
 └────────┬──────────┘
          │  Circuito transpilado
          ▼
 ┌───────────────────┐
 │ Módulo 4:         │  Comparar MO+RL vs. SABRE+RL vs. Qiskit estándar
 │ Integración       │  Métricas: profundidad, CNOTs, fidelidad
 │ (Experimentación) │  Análisis estadístico y visualización
 └───────────────────┘
```

---

## Convenciones para Agentes IA

- **Lenguaje de código**: Python 3.10+
- **Gestor de paquetes**: pip con virtualenv (`.venv/`)
- **Ejecutar Python**: usar `C:/Users/Eduardo/Desktop/universidad/TFG-Quantum-Transpiler/.venv/Scripts/python.exe`
- **Qiskit 2.x ENFORCEMENT (Estricto):**
    - **Prohibido**: `qiskit.terra`, `qiskit.ibmq` (usar `qiskit_ibm_runtime`), `qiskit.execute()`, `QuantumInstance`.
    - **Aer**: Usar `import qiskit_aer` (paquete separado).
    - **Transpilador**: Estructura modular en `qiskit.transpiler`. Uso intensivo de `PassManager`.
    - **Validación**: Verificar siempre compatibilidad con Qiskit 2.0 antes de sugerir código.
- **Entornos RL**: usar `gymnasium` (NO `gym` antiguo de OpenAI)
- **NO API KEYS**: Todo el desarrollo y pruebas debe ser local usando Fake Backends (ej. `FakeTorino`). Simulación de hardware sin conexión.
- **Metas**: Obtener métricas de transpilación (calidad del circuito), no ejecutar algoritmos cuánticos reales.
- **GPU**: disponible (CUDA 12.1, RTX 3060). Usar `torch.device('cuda')` cuando sea beneficioso
- **Estilo de código**: PEP 8, type hints recomendados, docstrings en español o inglés
- **Tests**: pytest como framework de testing
