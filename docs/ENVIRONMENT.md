# ENVIRONMENT.md — Entorno Tecnológico

Este documento detalla la configuración del entorno, versiones de dependencias y capacidades de hardware disponibles para el proyecto.

## Entorno Virtual

- **Directorio**: `.venv/`
- **Intérprete Python**: `C:/Users/Eduardo/Desktop/universidad/TFG-Quantum-Transpiler/.venv/Scripts/python.exe`
- **Versión Python**: 3.10.8

## Librerías y Dependencias (Estado Actual)

Las versiones se han verificado en el entorno actual.

### Core Cuántico (Qiskit Ecosystem)

| Librería | Versión | Descripción |
|---|---|---|
| `qiskit` | **2.3.0** | SDK principal (Unified 2.0+ architecture) |
| `qiskit-aer` | 0.17.2 | Simulador de alto rendimiento |
| `qiskit-ibm-runtime` | 0.43.1 | Conexión con servicios IBM Quantum (y Fake Providers) |
| `qiskit-ibm-transpiler` | 0.16.0 | Transpilation services/passes basados en AI de IBM |

### Aprendizaje por Refuerzo (RL Module)

| Librería | Versión | Descripción |
|---|---|---|
| `gymnasium` | 1.2.3 | API estándar para entornos de RL (sustituye a `gym`) |
| `stable-baselines3` | 2.7.1 | Implementaciones fiables de algoritmos RL (PPO, DQN, etc.) |
| `tensorboard` | 2.20.0 | Visualización y logging de métricas de entrenamiento |

### Aprendizaje Profundo (Deep Learning)

| Librería | Versión | CUDA | Descripción |
|---|---|---|---|
| `torch` | 2.5.1+cu121 | **Sí (12.1)** | Framework de DL principal |
| `torchaudio` | 2.5.1+cu121 | - | Audio (dependencia opcional de torch) |
| `torchvision` | 0.20.1+cu121 | - | Visión (dependencia opcional de torch) |

### Optimización Multiobjetivo (MO Module)

| Librería | Versión | Descripción |
|---|---|---|
| `pymoo` | 0.6.1.6 | Algoritmos evolutivos multiobjetivo (NSGA-II, MOEA/D) |

### Computación Científica y Análisis

| Librería | Versión | Descripción |
|---|---|---|
| `numpy` | 2.2.6 | Arrays y computación numérica |
| `scipy` | 1.15.3 | Algoritmos científicos |
| `pandas` | 2.3.3 | Estructuras de datos y análisis |
| `matplotlib` | 3.10.8 | Visualización y gráficas |

### Utilidades

| Librería | Versión | Descripción |
|---|---|---|
| `pypdf2` | 3.0.1 | Manejo de archivos PDF |
| `ipython` | 8.32.0 | Shell interactivo |

---

## Hardware Detectado

- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU
- **CUDA Version**: 12.1
- **Disponibilidad para PyTorch**: `torch.cuda.is_available() == True`

---

## Notas de Compatibilidad

### Política de Fake Backends
Dado que no se usan credenciales de IBM Quantum, se debe utilizar `qiskit_ibm_runtime.fake_provider` para obtener backends simulados (ej. `FakeTorino`, `FakeSherbrooke`).
Esto permite acceder a:
- Coupling Map (mapa de conectividad)
- Basis Gates (puertas nativas)
- Properties (tiempos de coherencia, error rates - simulados)

Sin necesidad de conexión a internet o token de autenticación.

### Gymnasium vs Gym
El proyecto utiliza estrictamente `gymnasium`.
- Importar como: `import gymnasium as gym`
- Espacios: `gymnasium.spaces`
- Wrappers: `gymnasium.wrappers`

Evitar mezclar con `gym` legacy para prevenir conflictos de tipos.

### Qiskit 2.x
El código debe ser compatible con Qiskit 2.0+.
- `qiskit.circuit.QuantumCircuit` en lugar de constructores antiguos.
- `qiskit.transpiler.PassManager` para gestión de pases.
- Evitar `qiskit.execute` en favor de `Sampler`/`Estimator` primitives o transpilación explícita seguida de simulación en Aer si es necesario ejecutar.
