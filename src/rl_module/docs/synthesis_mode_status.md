# Estado del Modo Synthesis

> **Última actualización:** 2026-04-01  
> **Estado:** primer alcance funcional para síntesis Clifford con layout fijo.

## Estado actual

`synthesis` ya no es un placeholder, pero sigue siendo una primera fase acotada:

- soporta solo circuitos Clifford;
- requiere `basis_gates` explícitas además de `coupling_map`;
- usa equivalencia Clifford, no coincidencia secuencial con la lista de puertas target;
- mantiene layout fijo durante el episodio.

## Futuro trabajo

- síntesis general no-Clifford;
- espacios de acción parametrizados o continuos;
- integración con `swap` dinámico dentro del episodio;
- criterios de equivalencia o aproximación más generales que el residual Clifford.
