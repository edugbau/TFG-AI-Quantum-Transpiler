# Skill: MO -> RL Pipeline (Integration)
**Contexto para el Módulo 4 (`integration`).**

## Objetivo
El traspaso de información genotípica desde el Módulo de Optimización (MO) al Agente de Refuerzo (RL) y su orquestación.

## Reglas de Implementación

1. **El Handshake (Intercambio de Estados)**
   - **Salida MO:** El proceso `pymoo` finaliza y retorna un conjunto de individuos Pareto-óptimos (el Frente de Pareto). Cada individuo es un layout inicial válido (ej. `[1, 0, 3, 2]`).
   - **Ingesta RL:** El entorno Gymnasium debe instanciarse (o resetearse) aceptando **directamente** uno de estos layouts óptimos como su estado inicial o `initial_layout`. El método `env.reset(options={"initial_layout": layout})` es el lugar idóneo para esto.

2. **Benchmarking Estructural**
   - El pipeline debe correr múltiples experimentos secuencialmente:
     - `Baseline`: Qiskit (SABRE o Default Transpilation Nivel 3).
     - `MO_Only`: Qiskit usando **sólo** el layout obtenido del MO (sin RL extra).
     - `RL_Only`: Qiskit con un layout aleatorio (o trivial) + Síntesis RL.
     - `MO+RL`: El layout obtenido del MO se pasa al agente RL. Comparativa final.

3. **Manejo de Qubits (Físicos vs. Lógicos)**
   - Al traspasar el layout, asegurar que el array devuelto por el MO representa el mapeo correcto (`logical_qubit -> physical_qubit` o viceversa, dependiendo de la convención interna del proyecto). **Documentar siempre** qué representa cada índice.
   - El RL debe iniciar su `state` (observación) considerando que las puertas del circuito abstracto ya están mapeadas a esa asignación inicial, y sus acciones consisten en insertar SWAPs para resolver el enrutamiento (routing) restante si la conectividad no se satisface, o en sintetizar el circuito directamente sobre ese mapa.
