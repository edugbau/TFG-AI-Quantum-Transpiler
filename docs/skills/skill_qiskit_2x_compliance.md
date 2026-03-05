# Skill: Qiskit 2.x Compliance
**Contexto transversal para todo el proyecto.**

## Objetivo
Garantizar la correcta compatibilidad del código generado con **Qiskit >= 2.0** (actualmente 2.3.0). En esta versión principal, la API ha sufrido cambios disruptivos ("breaking changes") masivos frente a la serie 0.x.

## Reglas Inviolables (The Monolith is Gone)

1. **Importaciones (Packages):**
   - ❌ **PROHIBIDO:** `qiskit.terra`, `qiskit.aer`, `qiskit.ignis`, `qiskit.aqua`, `qiskit.providers.ibmq`.
   - ✅ **CORRECTO:** `import qiskit`, `from qiskit import QuantumCircuit`.
   - ✅ **Simulador:** `import qiskit_aer` (como paquete independiente).
   - ✅ **Hardware / Fake Backends:** `from qiskit_ibm_runtime.fake_provider import FakeTorino` (o similares).

2. **Ejecución y Simulación:**
   - ❌ **PROHIBIDO:** `qiskit.execute()`, `QuantumInstance`.
   - ✅ **CORRECTO:** `backend.run(circuit, **kwargs)`.
   - ✅ **CORRECTO:** Primitivas V2 (`SamplerV2`, `EstimatorV2`) para extracción de valores de expectación.

3. **Operaciones sobre Circuitos:**
   - ❌ **PROHIBIDO:** `circuit.qasm()` (método string antiguo).
   - ✅ **CORRECTO:** Usar el módulo `qiskit.qasm2` o `qiskit.qasm3` para exportar/importar.
   - ✅ **Parametrización:** Usar `assign_parameters()` en lugar de `bind_parameters()` en contextos deprecados.

4. **Transpilación:**
   - ✅ **CORRECTO:** Construir pipelines con `PassManager` (`from qiskit.transpiler import PassManager`). Evitar llamar iterativamente a `transpile` si se pueden agrupar los pases.
   - ✅ Extraer propiedades físicas desde el `Target` o el `CouplingMap` del backend.

5. **Entorno de Pruebas:**
   - No se envían circuitos a hardware real. Siempre usar `FakeBackend` para simular la topología, el conjunto de puertas nativas y el mapa de acoplamiento.
