# Módulo 1: Interfaz con Qiskit (`qiskit_interface`)

Este módulo implementa la capa de abstracción sobre Qiskit 2.3.0+, gestionando la carga de circuitos, la interacción con backends simulados y la transpilación estándar (baseline).

## Estructura del Módulo

El paquete se divide en tres componentes principales:

1.  **`circuit_utils.py`**: Utilidades para circuitos (carga, creación, métricas).
2.  **`backend_info.py`**: Extracción unificada de información de hardware.
3.  **`transpiler.py`**: Pipeline de transpilación estándar y benchmarking.

---

## 1. Utilidades de Circuitos (`circuit_utils`)

Funciones para manejar `QuantumCircuit`, generar benchmarks y extraer métricas para los algoritmos de optimización.

### Funciones Clave

#### `create_ghz_circuit(num_qubits: int) -> QuantumCircuit`
Genera un circuito con estado GHZ (Greenberger–Horne–Zeilinger).
- **Entrada**: Número de qubits ($n \ge 2$).
- **Salida**: `QuantumCircuit` con profundidad $n$ y $n-1$ puertas CNOT.

#### `create_qft_circuit(num_qubits: int, inverse: bool = False) -> QuantumCircuit`
Genera un circuito de Transformada Cuántica de Fourier.
- **Entrada**: Quibits y flag para inversa.
- **Salida**: `QuantumCircuit` (usando `synth_qft_full` de Qiskit 2.1+).

#### `extract_metrics(circuit: QuantumCircuit) -> CircuitMetrics`
Extrae características cuantitativas del circuito.
- **Salida**: Objeto `CircuitMetrics` con:
    - `depth`: Profundidad crítica.
    - `two_qubit_gates`: Conteo exacto de puertas CX/CZ/ECR.
    - `total_gates`: Tamaño total.
    - `nonlocal_gates`: Puertas que generan entrelazamiento.

#### IO QASM
- **`load_circuit_from_qasm2(source)`**: Carga desde archivo o string QASM 2.0.
- **`export_circuit_to_qasm2(circuit)`**: Exporta a texto QASM 2.0.

---

## 2. Información de Backends (`backend_info`)

Abstracción para consultar propiedades de dispositivos simulados (Fake Backends) sin credenciales.

### Funciones Clave

#### `get_backend(name: str) -> Backend`
Obtiene una instancia de backend simulado.
- **Backends soportados**: `"fake_torino"` (133q, CZ), `"fake_sherbrooke"` (127q, ECR), `"fake_brisbane"` (127q, ECR).

#### `extract_backend_info(backend) -> BackendInfo`
Recopila toda la información relevante para la optimización en una sola estructura.
- **Salida**: Objeto `BackendInfo` que contiene:
    - `coupling_map`: Topología de conexiones.
    - `gate_errors_2q`: Diccionario `{(q1, q2): error_rate}` con las tasas de error de puertas de 2 qubits.
    - `qubit_t1`, `qubit_t2`: Tiempos de coherencia por qubit.
    - `basis_gates`: Puertas nativas detectadas dinámicamente.

#### `get_heaviest_hex_layout(backend, num_qubits) -> list[int]`
Genera un layout inicial heurístico seleccionando los qubits con mayor conectividad (grado).

---

## 3. Transpilación Baseline (`transpiler`)

Ejecuta el pipeline de transpilación estándar de IBM para establecer comparativas.

### Funciones Clave

#### `transpile_circuit(...) -> TranspilationResult`
Transpila un circuito individual con control total de parámetros.
- **Entrada**:
    - `circuit`: Circuito cuántico.
    - `backend` / `backend_name`: Objetivo de hardware.
    - `optimization_level`: 0 (nada) a 3 (máximo).
    - `initial_layout`: Lista opcional de qubits físicos (para inyectar layouts del módulo MO).
    - `seed`: Semilla para reproducibilidad.
- **Salida**: `TranspilationResult` con métricas pre/post transpilación, tiempos y reducción de profundidad.

#### `run_baseline(circuit, ...)`
Ejecuta transpilaciones masivas en varios backends y niveles de optimización.
- **Uso**: Generar datos tabulados para comparar con el enfoque híbrido.

#### `transpile_with_custom_layout(circuit, layout, ...)`
Función puente para el Módulo MO. Fuerza al transpilador a usar un layout específico optimizado evolutivamente.

---

## Aspectos a Considerar

### Protocolo Qiskit 2.3.0+
Este módulo cumple estrictamente con la API moderna de Qiskit:
- **NO usa**: `qiskit.execute`, `QuantumInstance`, `qiskit.circuit.library.QFT` (class).
- **SÍ usa**: `backend.target` (para propiedades), `qiskit.qasm2`, `qiskit.synthesis.qft`.
- **Detección de Puertas**: No asume que la puerta de dos qubits se llama "cx". Inspecciona el backend para encontrar "cz", "ecr", etc.

### Política de Excepciones
- **QASM**: `load_circuit_from_qasm2` distingue inteligentemente entre rutas de archivo y contenido de texto.
- **Validación**: Las funciones de creación lanzan `ValueError` si los parámetros (como qubits < 2) son inválidos.

## Guía de Uso Rápida

```python
from src.qiskit_interface import (
    create_ghz_circuit,
    get_backend,
    transpile_circuit,
    extract_backend_info
)

# 1. Crear circuito y definir backend
qc = create_ghz_circuit(5)
backend = get_backend("fake_torino")

# 2. (Opcional) Inspeccionar backend antes de optimizar
info = extract_backend_info(backend)
print(f"Error promedio 2Q: {sum(info.gate_errors_2q.values()) / len(info.gate_errors_2q)}")

# 3. Transpilar (Baseline modo experto)
result = transpile_circuit(
    qc,
    backend=backend,
    optimization_level=3,
    seed=42
)

# 4. Ver resultados
print(result.summary())
# Salida esperada:
#   Backend: fake_torino
#   Nivel optim.: 3
#   ...
#   Reducción depth: +20.5%
```
