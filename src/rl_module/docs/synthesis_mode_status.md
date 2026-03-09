# Estado del Modo Synthesis — Pendiente de Implementación

> **Última actualización:** 2026-03-09  
> **Estado:** 🟡 Placeholder — pendiente de decisión de diseño con tutores.

## Problema Actual

El modo `synthesis` del entorno RL (`QuantumTranspilationEnv`) **no es entrenable** en su estado actual:

1. Las acciones de tipo `"gate"` **nunca modifican** `remaining_gates`, así que `terminated` nunca será `True`. Todo episodio se trunca por `max_steps`.
2. `info["gate_matched_target"]` está **hardcodeado a `False`**, por lo que `SynthesisReward` siempre aplica `incorrect_gate_penalty`. El agente no puede aprender nada útil.

El modo routing **funciona correctamente** y no se ve afectado.

## Opciones de Diseño (Pendientes de Decisión)

| Opción | Descripción | Pros | Contras |
|--------|-------------|------|---------|
| **Gate-by-gate** | Comparar acciones con la secuencia de puertas del target en orden | Simple, testeable | Limita al agente a una sola secuencia |
| **Unitaria acumulada** | Construir la unitaria del circuito sintetizado y comparar con target | Flexible (circuitos equivalentes) | O(2ⁿ), no escalable |
| **Tableau estabilizador** | Usar `StabilizerState` de Qiskit para comparar (solo Clifford) | Eficiente, alineado con skill doc | Solo circuitos Clifford |

## Qué Se Necesita Implementar

Cuando se elija una opción, en `environment.py` → `step()` → bloque `elif action_info["type"] == "gate":` se debe:

1. **Aplicar la puerta** al circuito sintetizado (nuevo atributo del entorno).
2. **Comparar** con el circuito target según la opción elegida.
3. **Actualizar** `remaining_gates` o un indicador de fidelidad.
4. **Asignar** `info["gate_matched_target"]` según corresponda.

## Archivos Afectados

- `environment.py` — Bloque de acción `"gate"` en `step()`
- `env_strategies.py` — `SynthesisStrategy` (posible cambio de observación)
- `rewards.py` — `SynthesisReward` (ajustar lógica de recompensa)
