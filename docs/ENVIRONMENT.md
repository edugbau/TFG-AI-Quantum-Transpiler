# Entorno Tecnológico — Detalle de Versiones

> Última verificación: 8 de febrero de 2026 (actualizado tras instalación completa)

## Python

- **Versión**: 3.10.8
- **Tipo de entorno**: Virtual Environment (`.venv/`)
- **Ruta del intérprete**: `C:/Users/Eduardo/Desktop/universidad/TFG-Quantum-Transpiler/.venv/Scripts/python.exe`
- **Sistema operativo**: Windows (MSC v.1933 64 bit, AMD64)

## Librerías Principales

### Qiskit (Framework de Computación Cuántica)

| Paquete | Versión | Estado |
|---|---|---|
| **qiskit** | **2.3.0** | ✅ Instalado |
| qiskit-aer | 0.17.2 | ✅ Instalado |
| qiskit-ibm-runtime | 0.43.1 | ✅ Instalado |
| qiskit-ibm-transpiler | 0.16.0 | ✅ Instalado |

> ⚠️ **IMPORTANTE**: Se usa **Qiskit 2.3.0**, que sigue la nueva arquitectura unificada introducida en Qiskit 1.0 (enero 2024). Los paquetes legacy (`qiskit-terra`, `qiskit-ignis`, `qiskit-aqua`) ya no existen como paquetes separados. Todo está consolidado bajo el paquete `qiskit`.
>
> **Implicaciones**:
> - NO usar: `from qiskit.terra import ...`, `from qiskit.ignis import ...`
> - SÍ usar: `from qiskit import QuantumCircuit`, `from qiskit.transpiler import ...`
> - La clase `PassManager` y los transpiler passes se importan desde `qiskit.transpiler`
> - Para simulación: `from qiskit_aer import AerSimulator`
> - Para hardware IBM: `from qiskit_ibm_runtime import QiskitRuntimeService`

### PyTorch (Deep Learning)

| Paquete | Versión | Estado |
|---|---|---|
| **torch** | **2.5.1+cu121** | ✅ Instalado con CUDA |

- Build: CUDA 12.1
- GPU disponible: **NVIDIA GeForce RTX 3060 Laptop GPU**
- Soporte CUDA: ✅ Activo (`torch.cuda.is_available() == True`)

### Aprendizaje por Refuerzo

| Paquete | Versión | Estado |
|---|---|---|
| **gymnasium** | **1.2.3** | ✅ Instalado |
| **stable-baselines3** | **2.7.1** | ✅ Instalado |
| tensorboard | 2.20.0 | ✅ Instalado |

> Nota: usar `gymnasium` (Farama Foundation), NO el antiguo `gym` de OpenAI.

### Optimización Multiobjetivo

| Paquete | Versión | Estado |
|---|---|---|
| **pymoo** | **0.6.1.6** | ✅ Instalado |

### Librerías de Soporte

| Paquete | Versión | Estado |
|---|---|---|
| numpy | 2.2.6 | ✅ Instalado |
| scipy | 1.15.3 | ✅ Instalado |
| matplotlib | 3.10.8 | ✅ Instalado |
| pandas | 2.3.3 | ✅ Instalado |

## Hardware

| Componente | Detalle |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| CUDA | 12.1 |
| OS | Windows 10/11 (64-bit) |

## Instalación de Dependencias

Todas las dependencias están instaladas. Para replicar el entorno:

```bash
# Activar el entorno virtual
.venv\Scripts\activate

# Instalar todas las dependencias
pip install -r requirements.txt
```

## requirements.txt sugerido

```
qiskit>=2.0
qiskit-aer>=0.17
qiskit-ibm-runtime>=0.40
torch>=2.0
gymnasium>=1.0
pymoo>=0.6
stable-baselines3>=2.0
numpy>=2.0
scipy>=1.10
matplotlib>=3.8
pandas>=2.0
tensorboard>=2.15
```
