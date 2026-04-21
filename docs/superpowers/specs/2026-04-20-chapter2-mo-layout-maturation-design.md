# Chapter 2 MO Layout Maturation Design

**Goal:** madurar `src/mo_module/` para seleccionar, explicar y comparar de forma sistematica layouts del frente de Pareto actual, manteniendo `depth` y `cnot_count` como unicos objetivos de optimizacion y dejando `src/integration/` fuera de esta iteracion.

## Contexto

El repositorio ya dispone de una base solida para este capitulo:

- `src/mo_module/optimizer.py` produce un `OptimizationResult` estable con el frente de Pareto, metadatos y dos metodos publicos de seleccion (`get_compromise_layout()` y `get_best_layout()`).
- `src/mo_module/pareto.py` ya contiene primitivas utiles para este capitulo: `select_knee_point()`, `select_weighted()`, `select_min_objective()` y `analyze_pareto_front()`.
- `src/mo_module/benchmark/` ya permite ejecutar benchmarks reproducibles sobre `circuitos x seeds` y agregar estadisticas basicas.
- `src/qiskit_interface/backend_info.py` ya expone `get_heaviest_hex_layout()`, que puede reutilizarse como referencia heuristica externa sin acoplar la iteracion a `integration`.

Tambien existe una restriccion importante: la iteracion no debe ampliar el espacio objetivo. El modulo MO seguira trabajando exclusivamente con `depth` y `cnot_count`, por lo que el valor de este capitulo no estara en optimizar "mas cosas", sino en explotar mejor el frente ya generado, justificar mejor la seleccion de una solucion unica y sistematizar la comparacion frente a referencias externas.

## Alcance aprobado

Este diseño cubre una iteracion por fases dentro de `src/mo_module/`:

- **Fase 1**: enriquecer la seleccion y la interpretacion del frente de Pareto sin cambiar el contrato actual de `OptimizationResult`.
- **Fase 2**: añadir una capa de campañas comparativas de layouts para evaluar de forma homogenea candidatos MO y referencias externas.
- **Fase 3**: documentada en este diseño como cierre previsto de la iteracion, pero no incluida en el primer lote de implementacion. Esta fase consolidara presets experimentales ligeros, referencias adversariales controladas y la documentacion final de uso sobre la infraestructura creada en las fases 1 y 2.

La secuencia aprobada para implementacion es:

1. implementar primero las fases 1 y 2;
2. ejecutar y estabilizar la verificacion de esas dos fases;
3. abordar la fase 3 como paso final de la iteracion, no como parte del arranque.

## No objetivos

Quedan fuera de esta iteracion:

- añadir nuevos objetivos de fitness distintos de `depth` y `cnot_count`;
- cambiar la representacion compartida del layout (`layout[i] = physical_qubit_for_logical_qubit_i`);
- modificar la semantica de `cnot_count`, que debe seguir leyendo `cnot_equivalent`;
- alterar el contrato publico actual de `OptimizationResult`;
- modificar `src/integration/`, `LayoutSelectionPolicy` o el handoff MO -> RL;
- rediseñar `LayoutTuner`, su `session_ref_point` o su contrato de eventos;
- introducir backends nuevos o ampliar el catalogo de objetivos/tuning mas alla de lo necesario para las campañas de comparacion.

## Restricciones de diseño

1. `src/mo_module/` sigue siendo dueño de la busqueda multiobjetivo, del analisis del frente y de la comparacion de layouts.
2. `src/integration/` no cambia en esta iteracion y sigue consumiendo unicamente `get_compromise_layout()` y `get_best_layout()`.
3. La convencion de layout no cambia y todos los layouts comparados deben seguir siendo listas de `int` validas.
4. `depth` y `cnot_count` siguen siendo los unicos objetivos de optimizacion activos.
5. `cnot_count` debe seguir representando `cnot_equivalent`, no `two_qubit_gates`.
6. Las nuevas capacidades deben ser aditivas: enriquecer analisis, filas experimentales y campañas, pero no romper tests existentes.
7. La seleccion y las campañas deben ser deterministas para una misma `seed`; en caso de empate entre candidatos, la resolucion debe favorecer el menor indice del frente para no introducir no determinismo oculto.
8. La fase 3 debe construirse sobre estructuras ya validadas en fases 1 y 2, no rediseñarlas al final de la iteracion.
9. `src/mo_module/` no debe comunicarse directamente con `src/rl_module/`: no debe importar codigo de RL, no debe declarar APIs cuyo consumidor sea `rl_module` y no debe presentarse en docs o comentarios como dueño del handoff MO -> RL.
10. El unico punto autorizado para orquestar MO -> RL es `src/integration/`. En esta iteracion, `mo_module` solo puede producir layouts, analisis y filas experimentales genericas que un consumidor externo podria usar despues.

## Diseño propuesto

### 1. Fase 1: seleccion e interpretacion del frente

La primera fase añadira una capa explicativa encima del frente de Pareto ya existente. El objetivo no es reemplazar `OptimizationResult`, sino dar una superficie mas expresiva para responder preguntas como:

- que layout escoger si quiero una solucion equilibrada;
- que layout representa el punto de mayor trade-off;
- que diferencia real hay entre el mejor en `depth`, el mejor en `cnot_count` y el layout de compromiso.

La pieza central seguira siendo `analyze_pareto_front()` en `src/mo_module/pareto.py`, pero su salida se ampliara de forma aditiva. Debe mantener las claves actuales (`metrics`, `knee_point_idx`, `knee_point_layout`, `best_per_objective`, `compromise_layout`) y añadir una estructura nueva tipo `selection_candidates`.

Forma conceptual:

```python
{
    "metrics": ParetoMetrics(...),
    "knee_point_idx": 2,
    "knee_point_layout": [10, 14, 18, 22, 26],
    "best_per_objective": {
        "depth": {...},
        "cnot_count": {...},
    },
    "compromise_layout": [10, 14, 18, 22, 26],
    "selection_candidates": {
        "compromise": {
            "index": 2,
            "layout": [...],
            "objective_values": {"depth": 17.0, "cnot_count": 11.0},
            "normalized_objective_values": {"depth": 0.40, "cnot_count": 0.35},
            "distance_to_ideal": 0.53,
            "reason": "closest_to_normalized_ideal",
        },
        "knee": {
            "index": 2,
            "layout": [...],
            "objective_values": {...},
            "normalized_objective_values": {...},
            "distance_to_ideal": 0.53,
            "reason": "max_tradeoff_change",
        },
        "best_depth": {
            "index": 0,
            "layout": [...],
            "objective_values": {...},
            "normalized_objective_values": {...},
            "distance_to_ideal": 0.71,
            "reason": "min_depth",
        },
        "best_cnot_count": {
            "index": 4,
            "layout": [...],
            "objective_values": {...},
            "normalized_objective_values": {...},
            "distance_to_ideal": 0.66,
            "reason": "min_cnot_count",
        },
    },
    "tradeoff_table": [
        {
            "index": 0,
            "layout": [...],
            "depth": 15.0,
            "cnot_count": 13.0,
            "depth_norm": 0.00,
            "cnot_count_norm": 0.62,
            "distance_to_ideal": 0.62,
        },
        ...
    ],
}
```

Claves del diseño:

- `compromise`, `knee`, `best_depth` y `best_cnot_count` seran candidatos por defecto.
- `weighted` seguira existiendo como estrategia disponible en `pareto.py`, pero no se incorporara por defecto al conjunto de candidatos porque esta iteracion no necesita una policy adicional dependiente de pesos externos.
- `tradeoff_table` sera una vista tabular y serializable del frente, orientada a notebooks, pandas y reporting experimental.
- los candidatos usaran siempre los nombres de objetivo reales (`depth`, `cnot_count`) para evitar alias ambiguos.

Esta fase no cambia como `integration` selecciona layouts, pero si prepara una API analitica mucho mas util dentro de `mo_module`.

### 2. Fase 2: campañas comparativas de layouts

La segunda fase trasladara el trabajo desde la optimizacion aislada hacia campañas experimentales sistematicas. La idea es tomar el frente ya optimizado, seleccionar varios candidatos con la fase 1, compararlos contra referencias externas y devolver filas homogeneas listas para analisis.

Se añadira un helper especifico en `src/mo_module/benchmark/` para campañas de layouts. La nueva capa no reemplazara a `BenchmarkRunner`, porque ese runner actual esta centrado en optimizar `circuitos x seeds`. La nueva capa se situara encima del resultado MO y reutilizara el evaluador de layouts ya existente.

Responsabilidades esperadas de esta capa:

- ejecutar `optimize_layout()` o `optimize_layout_quick()` para un circuito y `seed` dados;
- llamar a `analyze_pareto_front()` para extraer candidatos MO;
- construir un conjunto pequeno de referencias externas validas;
- evaluar todos los layouts bajo el mismo pipeline de transpilacion;
- devolver filas comparables y agregables.

Referencias externas incluidas en fase 2:

- `trivial`: `list(range(num_qubits))`;
- `heaviest_hex`: `get_heaviest_hex_layout(backend, num_qubits)`;
- candidatos MO de fase 1: `compromise`, `knee`, `best_depth`, `best_cnot_count`.

La funcion `compare_layouts()` en `src/mo_module/optimizer.py` se mantendra como helper publico, pero se ampliara para alinear mejor la comparacion con los objetivos reales del modulo. En concreto, las filas deben incluir `cnot_equivalent` ademas de `two_qubit_gates`, porque el segundo es diagnostico y el primero es el que realmente representa `cnot_count`.

Forma conceptual de cada fila de campaña:

```python
{
    "circuit_name": "ghz_5q",
    "seed": 7,
    "layout_name": "knee",
    "layout_family": "mo_candidate",
    "selection_strategy": "knee",
    "pareto_index": 2,
    "layout": [10, 14, 18, 22, 26],
    "depth": 17,
    "two_qubit_gates": 9,
    "cnot_equivalent": 11,
    "total_gates": 41,
    "elapsed_time_s": 0.18,
    "avg_error_2q": 0.012,
    "max_error_2q": 0.018,
    "avg_t2": 0.00009,
    "num_edges": 5,
}
```

Esta fase debe producir dos salidas principales:

- filas crudas listas para `pd.DataFrame`;
- un resumen textual o tabular ligero que permita contestar preguntas como:
  - con que frecuencia `compromise` supera a `heaviest_hex`;
  - cuando `best_depth` empeora demasiado el `cnot_count`;
  - si `knee` y `compromise` convergen o divergen segun el circuito.

La campaña no debe depender de `integration`. Es una herramienta experimental del modulo MO.

### 3. Fase 3: cierre de iteracion con presets y referencias controladas

La fase 3 queda documentada en esta spec como paso final de la iteracion. Su objetivo no es abrir nuevas capacidades nucleares, sino consolidar y empaquetar mejor lo construido en fases 1 y 2.

Esta fase se ejecutara solo despues de verificar que:

- `selection_candidates` es estable;
- la campaña comparativa devuelve filas utiles y consistentes;
- las metricas publicadas reflejan correctamente `depth` y `cnot_count`.

Elementos previstos para esta fase final:

1. **Presets experimentales ligeros**

Se añadiran presets de campaña, no presets nuevos de fitness. La intencion es reducir friccion de uso y dejar configuraciones defendibles en el TFG.

Presets previstos:

- `quick`: pocas seeds, poblacion/generaciones reducidas, referencias esenciales (`trivial`, `heaviest_hex`, `compromise`);
- `balanced`: presupuesto medio y conjunto completo de candidatos MO de fase 1;
- `thorough`: mas seeds, conjunto completo de candidatos y tablas mas ricas para analisis final.

2. **Referencias adversariales controladas**

Se añadiran referencias deterministas adicionales para tensionar la narrativa comparativa sin introducir aleatoriedad extra. Estas referencias no se usan como objetivos de optimizacion, solo como comparadores deliberadamente pobres o al menos ingenuos.

Referencias previstas:

- `reverse_trivial`: `list(reversed(range(num_qubits)))`;
- `high_index_block`: bloque contiguo anclado al extremo alto del backend, `list(range(backend.num_qubits - num_qubits, backend.num_qubits))`.

No se promete que estas referencias sean siempre las peores en todos los backends, pero si que sean validas, deterministas y suficientemente distintas de las referencias base para enriquecer la comparacion.

3. **Consolidacion documental**

Se actualizaran los documentos internos del modulo para reflejar:

- la nueva superficie de seleccion del frente;
- la diferencia entre comparacion puntual (`compare_layouts`) y campaña sistematica;
- los presets experimentales disponibles y sus limites;
- el hecho de que el capitulo sigue operando solo sobre `depth` y `cnot_count`.

Esta fase final sigue sin tocar `integration`.

## Flujo de datos

### Flujo principal de fases 1 y 2

1. `optimize_layout()` genera `OptimizationResult`.
2. `analyze_pareto_front()` calcula metricas, candidatos y tabla de trade-offs.
3. La capa de campaña construye referencias externas validas (`trivial`, `heaviest_hex`).
4. `compare_layouts()` evalua candidatos MO y referencias bajo el mismo pipeline de transpilacion.
5. Las filas resultantes se agregan por `circuit_name`, `seed`, `layout_family` y `selection_strategy`.
6. Un resumen ligero permite inspeccionar resultados sin depender de notebooks externos para todo.

### Flujo de cierre de fase 3

1. Se selecciona un preset de campaña (`quick`, `balanced`, `thorough`).
2. El preset decide seeds, presupuesto y referencias activas.
3. Se ejecuta la campaña comparativa sobre la infraestructura ya creada.
4. Se documenta el preset y se dejan ejemplos reproducibles de uso.

## Errores esperados

- frente vacio o `pareto_fitness=None`: el analisis debe seguir devolviendo una estructura vacia y las campañas deben registrar filas vacias o HV/campos nulos sin crashear;
- estrategia desconocida en la capa de seleccion: error temprano y legible;
- pesos invalidos si se usa `weighted`: longitud incoherente o suma no interpretable;
- referencia externa invalida o fuera de rango: error temprano antes de transpilacion;
- inconsistencia entre `num_qubits` del circuito y longitud del layout de referencia;
- intento de usar un preset de fase 3 antes de disponer de la infraestructura de campaña subyacente.

## Estrategia de pruebas

### Fase 1

- ampliar `tests/test_mo_module/test_mo_module.py` para verificar que `analyze_pareto_front()` mantiene sus claves actuales y añade `selection_candidates` y `tradeoff_table` sin romper compatibilidad;
- validar que `compromise`, `knee`, `best_depth` y `best_cnot_count` son deterministas;
- verificar que empates se resuelven por indice estable.

### Fase 2

- ampliar `tests/test_mo_module/test_benchmark.py` para cubrir la nueva capa de campañas;
- verificar que las filas incluyen `cnot_equivalent` y siguen incluyendo `two_qubit_gates`;
- validar que `trivial` y `heaviest_hex` se evalúan bajo el mismo contrato que los candidatos MO;
- comprobar que el resumen agregado no depende de resultados completos de notebooks o scripts externos.

### Fase 3

- tests de presets experimentales para asegurar que activan el conjunto esperado de referencias y presupuestos;
- tests de referencias adversariales controladas para asegurar validez, determinismo y longitud correcta;
- actualización de tests documentales si el modulo MO ya tiene comprobaciones sobre docs o API publica reexportada.

### Regresion transversal

- `tests/test_mo_module/test_tuning.py` no debe cambiar semantica;
- `tests/test_integration/test_layout_policy.py` debe seguir pasando sin cambios, confirmando que `integration` permanece ajeno a esta iteracion.

## Riesgos y mitigaciones

- Riesgo: inflar demasiado `analyze_pareto_front()` y convertirlo en un punto de acoplamiento excesivo.
  - Mitigacion: ampliar la salida solo con claves serializables y derivadas, sin mover logica del optimizador al analisis.

- Riesgo: mezclar metricas diagnosticas con metricas objetivo y acabar comparando layouts sobre `two_qubit_gates` en vez de `cnot_equivalent`.
  - Mitigacion: toda fila comparativa debe incluir ambos campos y dejar claro que `cnot_equivalent` es el contrato del objetivo `cnot_count`.

- Riesgo: que la fase 3 reabra decisiones de alcance y convierta la iteracion en una expansion encubierta del espacio objetivo.
  - Mitigacion: la spec deja fijado que la fase 3 solo consolida presets, referencias y documentacion; no añade objetivos nuevos.

- Riesgo: que la narrativa comparativa se disperse entre `mo_module` e `integration`.
  - Mitigacion: toda la infraestructura nueva vive en `src/mo_module/` y se mantiene como herramienta experimental propia del modulo.

- Riesgo: reintroducir acoplamiento conceptual o tecnico entre `mo_module` y `rl_module` al documentar campañas o futuras extensiones.
  - Mitigacion: cualquier referencia a MO -> RL debe formularse siempre como consumo futuro via `src/integration/`, y la verificacion final debe incluir los tests de contratos modulares que ya prohíben imports directos y ownership ambiguo.

## Resultado esperado de la iteracion

La iteracion quedara bien cerrada cuando el modulo MO pueda:

- producir un frente de Pareto como hoy;
- explicar de forma mas rica por que escoger `compromise`, `knee`, `best_depth` o `best_cnot_count`;
- comparar sistematicamente esos candidatos frente a referencias externas validas;
- ejecutar campañas reproducibles con presets ligeros y referencias controladas;
- sostener la narrativa del TFG sin haber aumentado el espacio objetivo ni alterado `integration`.
