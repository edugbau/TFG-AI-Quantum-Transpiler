# Estado del Modo Synthesis

> **Última actualización:** 2026-04-01  
> **Estado:** primer alcance funcional para síntesis Clifford con layout fijo e inspector de evaluación residual-céntrico.

## Encaje dentro de la GUI RL

La GUI RL ahora debe leerse como **una sola aplicación con dos vistas especializadas**:

- `routing`: orientada a frontera visible, SWAPs y ejecución de puertas desbloqueadas.
- `synthesis`: orientada a primitivas nativas y reducción del residual.

`synthesis` no reutiliza la semántica visual principal de routing. Comparte la infraestructura de evaluación e inspección de episodios, pero prioriza métricas de residual Clifford frente al lenguaje de desbloqueo de puertas propio de routing.

## Estado actual

`synthesis` ya no es un placeholder, pero sigue siendo una primera fase acotada:

- soporta solo circuitos Clifford;
- requiere `basis_gates` explícitas además de `coupling_map`;
- usa equivalencia Clifford en espacio físico, no coincidencia secuencial con una lista de puertas target;
- mantiene layout fijo durante el episodio;
- evalúa el progreso mediante reducción del residual, no mediante consumo de una frontera de routing.

## Semántica del inspector de episodios

Cada paso de evaluación en `synthesis` registra y presenta:

- la primitiva elegida (`primitive_name`);
- sus qubits físicos (`primitive_physical_qargs`);
- su coste (`primitive_cost`);
- la progresión del residual (`residual_distance_before -> residual_distance_after`);
- la mejora neta (`residual_distance_delta`).

La interpretación correcta es: una acción aplica o intenta aplicar una primitiva del catálogo y el panel muestra cuánto acerca esa acción al residual identidad. El inspector puede compartir estructura con routing, pero en `synthesis` el foco es la distancia residual y no el desbloqueo de puertas futuras.

## Futuro trabajo

- síntesis general no-Clifford;
- espacios de acción parametrizados o continuos;
- integración con `swap` dinámico dentro del episodio;
- criterios de equivalencia o aproximación más generales que el residual Clifford.
