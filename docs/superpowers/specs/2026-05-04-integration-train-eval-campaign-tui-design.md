# Integration Train+Eval Campaign TUI Design

**Goal:** dejar `src/integration/` listo para ejecutar campañas reproducibles de `train+eval` desde terminal, comparando `Baseline`, `MO_Only` y `MO+RL` sobre uno o varios circuitos de librería y uno o varios backends, con persistencia por campaña y un resumen Markdown final.

## Contexto

El estado actual de `integration` ya cubre bien la comparación unitaria de escenarios de routing:

- `Baseline` usa la referencia canónica de Qiskit `qiskit_level_1`;
- `MO_Only` selecciona un layout MO y lo evalúa con Qiskit;
- `MO+RL` ya reconstruye el circuito ruteado y obtiene métricas comparables con Qiskit cuando RL completa el routing;
- `RL_Only` sigue devolviendo `episode summaries` y no encaja todavía en la comparación principal homogénea.

Sin embargo, `integration` sigue siendo una capa `eval-only` con un `runner.py` fino para un único `ScenarioRequest`. No existe todavía una unidad de trabajo tipo **Campaign**, no existe un flujo `train+eval` orquestado por `integration`, no existe persistencia agrupada por campaña y no existe una interfaz de terminal guiada para seleccionar circuitos, configuraciones, ejecución y resumen final.

La nueva necesidad cambia el alcance del módulo de forma deliberada: `integration` deja de limitarse a evaluar checkpoints externos y pasa a orquestar una **Train+Eval Campaign** por `Campaign Case`, manteniendo que `rl_module` sigue siendo dueño del training en sí mismo y de la producción del checkpoint.

## Alcance aprobado

Este trabajo cubre:

1. introducir el concepto de **Campaign** en `integration`;
2. permitir campañas sobre `1..N` circuitos de librería interna seleccionados por tipo y tamaños;
3. permitir campañas sobre uno o varios fake backends, con un único backend en el flujo default y multi-backend en avanzado;
4. ejecutar por cada `Campaign Case` el flujo secuencial:
   - `Baseline`
   - `MO_Only`
   - training RL específico para ese `circuito x backend`
   - `MO+RL` evaluado con el `Training Artifact` recién producido;
5. introducir un modo `default` y un modo `advanced` en la interfaz interactiva de terminal;
6. persistir artefactos, resultados estructurados e informe final dentro de una carpeta por campaña;
7. generar un **Summary Document** en Markdown con resumen agregado, detalle por caso, incidencias y resumen breve del training RL.

## No objetivos

Quedan fuera de esta iteración:

- incluir `RL_Only` en la comparación principal de la TUI;
- soportar `qasm_file` de forma completa en flujos RL dentro de la campaña;
- implementar una TUI fullscreen con nueva librería de terminal;
- ejecutar `Campaign Cases` en paralelo;
- introducir resumibilidad completa de campañas interrumpidas;
- exponer hiperparámetros internos finos de RL más allá de los knobs principales;
- exponer toda `OptimizerConfig` de MO o rediseñar su API.

## Decisiones aprobadas

### 1. Unidad principal: Campaign

La unidad visible para el usuario pasa a ser una **Campaign**, no un escenario suelto.

Una Campaign:

- selecciona uno o varios circuitos;
- selecciona uno o varios backends;
- fija una configuración global compartida;
- ejecuta los `Campaign Cases` secuencialmente;
- produce una carpeta propia con artefactos y Summary Document.

### 2. Comparación canónica por caso

La comparación principal por `Campaign Case` será:

- `Baseline`
- `MO_Only`
- `MO+RL`

`RL_Only` queda explícitamente fuera de la TUI v1 porque no produce todavía métricas finales homogéneas con Qiskit.

### 3. Significado canónico de Baseline

En esta Campaign, **Baseline** significa exactamente el escenario `qiskit_level_1`:

- transpilación estándar de Qiskit;
- `optimization_level=1`;
- sin layout inicial MO;
- sin routing RL.

La TUI v1 no abre selector de optimization level de Qiskit para no mezclar el eje de comparación del pipeline con otro eje distinto de experimentación.

### 4. El alcance de integration pasa a ser train+eval orchestration

`integration` ya no será solo `eval-only`.

La nueva responsabilidad aprobada es:

- orquestar una **Train+Eval Campaign** por `Campaign Case`;
- lanzar training RL específico por `circuito x backend`;
- usar el `Training Artifact` resultante para evaluar `MO+RL`;
- seguir delegando en `rl_module` la implementación del training, del agente y de la persistencia del checkpoint.

Este cambio es de alcance, no de ownership algorítmico.

### 5. Granularidad de training RL

El training RL será específico por `circuito x backend`.

No se entrenará una sola vez por circuito para reutilizar el modelo en varios backends, porque el training actual de `rl_module` depende directamente de:

- `target_circuit`
- `coupling_map`

Cada `Campaign Case` produce, por tanto, su propio `Training Artifact`.

### 6. Relación entre MO y training RL

MO entra solo en la evaluación `MO+RL`, no en el training RL.

El flujo acordado es:

1. entrenar RL de forma layout-agnostic para ese `Campaign Case`;
2. ejecutar MO;
3. seleccionar layout MO;
4. inyectarlo en la evaluación `MO+RL` como `initial_layout`.

Esto preserva mejor la separación modular actual y evita rediseñar la superficie del training RL en esta iteración.

### 7. Modelo canónico a evaluar tras el training

Cuando el training RL produzca varios checkpoints, la evaluación usará:

1. `best_model.zip` si existe;
2. `final_model.zip` en caso contrario.

Esta decisión sigue la convención ya presente en `rl_module` y evita que `integration` redefina la semántica del mejor artefacto.

### 8. Estilo de TUI v1

Aunque el término inicial fue “TUI”, la forma aprobada para la v1 es una CLI interactiva guiada en terminal.

Esto implica:

- prompts secuenciales;
- menús de selección;
- confirmaciones explícitas;
- progreso textual por caso y por campaña;
- cancelación segura básica.

No implica:

- pantalla fullscreen;
- layout de paneles;
- navegación compleja por teclado;
- dependencia nueva tipo `textual` en esta iteración.

### 9. Configuración global de Campaign

La Campaign tendrá una única configuración global compartida por todos los casos.

No habrá overrides por circuito en la v1. Esto mantiene:

- comparabilidad más limpia;
- TUI más simple;
- Summary Document más legible.

### 10. Ejecución secuencial total

Todos los `Campaign Cases` se ejecutarán de forma secuencial.

Cada caso hará:

1. training RL;
2. evaluación de escenarios;
3. persistencia local;
4. actualización del progreso.

No habrá concurrencia en v1 porque el training actual consume recursos de forma intensa y no está modelado como scheduler multi-worker.

### 11. Política de fallos

Si un `Campaign Case` falla durante el training RL o `MO+RL` queda incompleto:

- la Campaign continúa con los siguientes casos;
- el caso queda registrado como fallido o incompleto;
- el Summary Document incluye una sección separada de incidencias.

Los agregados globales usarán solo casos comparables completos. Los fallos e incompletos se contarán aparte.

### 12. Cancelación segura básica

La v1 permitirá cancelación segura básica.

El objetivo no es reanudar campañas incompletas, sino:

- detener la Campaign de forma controlada;
- persistir lo ya completado;
- marcar la Campaign como interrumpida;
- dejar trazabilidad en el Summary Document y/o resultado estructurado.

### 13. Circuitos soportados en la TUI v1

La Campaign v1 usará solo circuitos de librería interna.

La selección se hará por:

- tipo de circuito (`ghz`, `qft`, `qft_inv`, `random_shallow`, `random_deep`, `clifford`);
- uno o varios tamaños de qubits.

La TUI construirá internamente la lista final de circuitos.

### 14. Política para circuitos aleatorios

Si la Campaign usa `random_shallow` o `random_deep`, la instancia del circuito aleatorio quedará fijada por Campaign.

Esto evita que cambie el circuito objetivo entre escenarios o entre backends dentro de una misma Campaign y preserva la comparabilidad.

### 15. Default Campaign

La Campaign tendrá un camino `default` canónico con estos valores iniciales:

- backend default: `fake_torino`;
- MO default: `mo_use_quick=True`, `layout_policy=compromise`;
- MO quick default: `population_size=30`, `n_generations=50`;
- RL default:
  - `algorithm=MaskablePPO`
  - `timesteps=5000`
  - `frontier_mode` según el valor default vigente del training/evaluación
  - `lookahead=10`
  - `max_steps=200`
  - `seed=42`

El flujo default existe para hacer la Campaign ejecutable casi sin decisiones manuales.

### 16. Advanced Campaign

El modo `advanced` permitirá:

- elegir varios backends a la vez;
- configurar knobs principales de RL:
  - `algorithm`
  - `timesteps`
  - `frontier_mode`
  - `lookahead`
  - `max_steps`
  - `seed`
- configurar MO de forma parecida a la GUI actual:
  - `population_size`
  - `n_generations`
  - `layout_policy`

No abrirá todavía hiperparámetros internos finos de RL ni toda `OptimizerConfig`.

### 17. Política avanzada de selección de layout MO

La TUI v1 debe permitir estas políticas de selección de layout:

- `compromise`
- `best_on_objective`

Si el usuario elige `best_on_objective`, la TUI debe pedir también el objetivo concreto. Con el preset default actual de MO, esos objetivos nominales son:

- `depth`
- `cnot_count`

No se aceptará que esta política quede implícita sobre el primer objetivo sin decirlo.

### 18. Summary Document canónico

La salida humana principal de la Campaign será un Summary Document en Markdown.

Ese documento incluirá:

1. metadatos de Campaign;
2. configuración global aprobada;
3. circuitos y backends seleccionados;
4. tabla agregada principal de comparación;
5. detalle por `Campaign Case`;
6. sección de incidencias;
7. resumen breve del training RL por caso;
8. estado final de la Campaign.

### 19. Métricas principales de comparación

La comparación principal entre `Baseline`, `MO_Only` y `MO+RL` se centrará en:

- `trans_depth`
- `trans_two_qubit_gates`
- `trans_cnot_equivalent`
- `elapsed_time_s`

Estas métricas son suficientemente compactas para el resumen principal y ya están alineadas con la semántica comparativa del proyecto.

Otras métricas pueden seguir disponibles en resultados estructurados o detalle por caso, pero no deben dominar la tabla principal de la Campaign.

## Arquitectura propuesta

## 1. Mantener escenarios unitarios y añadir una capa Campaign

`scenarios.py` se mantiene como capa de escenarios unitarios.

La nueva capa de Campaign se apoya en esos escenarios en lugar de mezclarse con ellos. La Campaign no debe reimplementar la lógica de:

- `Baseline`
- `MO_Only`
- `MO+RL`

Debe consumirla.

### 2. Nuevos seams en integration

La propuesta introduce cuatro áreas nuevas dentro de `src/integration/`.

#### A. Campaign contracts

Un archivo de contratos de campaña para modelar:

- configuración global;
- lista de circuitos seleccionados;
- lista de backends seleccionados;
- combinación `circuito x backend` como `Campaign Case`;
- resultado detallado por caso;
- resumen agregado de Campaign.

Estos contratos deben mantener separadas tres capas de información:

- configuración aprobada;
- resultado detallado por caso;
- resumen agregado de Campaign.

#### B. Training bridge

Un seam pequeño y explícito entre `integration` y `rl_module.training.setup_training_pipeline(...)`.

Objetivo:

- evitar que el orquestador de Campaign conozca demasiados detalles internos del training RL;
- encapsular cómo se resuelve el `Training Artifact` final;
- devolver un resultado estable para reporting.

#### C. Campaign runner

Responsable de:

- construir la lista de `Campaign Cases`;
- iterar secuencialmente;
- congelar la instancia del circuito por caso;
- lanzar training RL;
- lanzar `Baseline`, `MO_Only` y `MO+RL`;
- registrar incidencias;
- persistir resultados parciales;
- devolver el estado global de Campaign.

#### D. Campaign reporting

Responsable de:

- producir el resultado estructurado en JSON;
- construir agregados de métricas principales;
- renderizar `summary.md`;
- separar claramente casos completos, fallidos e incompletos.

### 3. Interactive CLI

La nueva TUI v1 debe implementarse como CLI interactiva guiada.

La CLI necesita estas fases:

1. seleccionar tipo(s) de circuito;
2. seleccionar tamaño(s);
3. elegir `default` o `advanced`;
4. confirmar backend(s);
5. confirmar configuración MO/RL;
6. mostrar resumen previo de Campaign;
7. ejecutar;
8. mostrar ruta final del Summary Document y estado de Campaign.

## Flujo funcional detallado

## 1. Construcción de la Campaign

La Campaign arranca con una configuración global y una selección de circuitos y backends.

La lista final de `Campaign Cases` se construye como producto cartesiano:

- circuitos seleccionados;
- backends seleccionados.

Cada caso recibe:

- un identificador estable;
- el circuito ya resuelto para esa Campaign;
- el backend;
- la configuración global;
- el estado de ejecución.

## 2. Ejecución de un Campaign Case

Para cada caso:

1. cargar o crear el circuito de librería correspondiente;
2. resolver backend y topología;
3. ejecutar `Baseline`;
4. ejecutar `MO_Only`;
5. entrenar RL específico con la configuración global aprobada;
6. resolver `Training Artifact` (`best_model.zip` o fallback a `final_model.zip`);
7. ejecutar `MO+RL` usando ese artifact;
8. registrar resultados, incidencias, paths de artefactos y metadatos resumidos del training.

## 3. Política sobre resultados incompletos

Si `MO+RL` no completa el routing:

- el caso no aporta métricas comparables a los agregados principales;
- el caso sí aparece en el detalle por caso;
- el caso sí alimenta la sección de incidencias;
- el caso debe preservar su `routing_summary` y notas relevantes.

## 4. Política sobre fallos de training

Si falla el training RL:

- `Baseline` y `MO_Only` pueden seguir conservándose si ya se habían ejecutado;
- `MO+RL` quedará marcado como no ejecutado o fallido;
- el caso no entra en agregados principales;
- la Campaign continúa con el siguiente caso.

## Persistencia propuesta

## 1. Carpeta por Campaign

Cada Campaign crea su propio directorio bajo una ruta canónica de `integration`, por ejemplo:

- `campaigns/<campaign_id>/`

Dentro de esa carpeta vivirán:

- `summary.md`
- `campaign.json`
- `cases/<case_id>/result.json`
- `cases/<case_id>/...` para artefactos adicionales y referencias de training

El objetivo es aislar ejecuciones y hacerlas rastreables sin depender de rutas planas compartidas.

## 2. Relación con las rutas actuales de RL

El training RL ya sabe escribir en `experiments/models/rl_models` y `experiments/logs/rl_logs`.

La v1 puede seguir reutilizando ese comportamiento interno del training, pero la Campaign debe:

- registrar en sus resultados los paths producidos;
- copiar o referenciar claramente los artefactos relevantes dentro de la estructura de la Campaign si hace falta para trazabilidad;
- no dejar que el Summary Document dependa de buscar manualmente en directorios planos globales.

La regla importante es que la Campaign tenga su propio índice canónico, aunque el training conserve sus rutas internas actuales.

## Reporting propuesto

## 1. Resumen agregado

El resumen agregado debe responder de forma inmediata:

- cuántos casos fueron completos, fallidos o incompletos;
- cómo se comportó `MO_Only` frente a `Baseline`;
- cómo se comportó `MO+RL` frente a `Baseline` y frente a `MO_Only`;
- en qué métricas principales hubo mejora o empeoramiento.

No debe mezclar con el agregado casos sin métricas comparables.

## 2. Detalle por caso

Cada `Campaign Case` debe incluir:

- circuito y backend;
- configuración efectiva resumida;
- métricas de `Baseline`;
- métricas de `MO_Only`;
- métricas de `MO+RL` si existen;
- layout seleccionado por MO;
- resumen breve del training RL;
- incidencias o notas.

## 3. Resumen breve del training RL

El Summary Document debe incluir, por caso, un bloque breve separado de la comparación Qiskit con:

- algoritmo;
- timesteps;
- frontier mode;
- lookahead;
- max steps;
- seed;
- artifact usado (`best_model.zip` o `final_model.zip`);
- estado del training.

No debe convertirse en un log exhaustivo de entrenamiento.

## Testing propuesto

La implementación debe quedar cubierta con tests en estas capas:

1. contratos de Campaign;
2. construcción del producto `circuitos x backends`;
3. ejecución secuencial por caso con dobles de training y escenarios;
4. selección del `Training Artifact` preferido;
5. tratamiento de fallos e incompletos;
6. generación del Summary Document;
7. prompts y validación básica de la CLI interactiva.

La mayor parte del testing debe apoyarse en stubs y monkeypatching, no en campañas reales pesadas de training.

## Plan de implementación sugerido

### Fase 1. Contratos y reporting

- introducir DTOs de Campaign;
- introducir agregación y render de Markdown;
- introducir persistencia por Campaign.

### Fase 2. Training bridge y runner

- encapsular el training RL dentro de `integration`;
- resolver el `Training Artifact` canónico;
- ejecutar `Campaign Cases` secuencialmente.

### Fase 3. Interactive CLI

- construir prompts guiados;
- soportar `default` y `advanced`;
- soportar selección de circuitos por tipo y tamaños;
- soportar cancelación segura básica.

### Fase 4. Documentación

- actualizar `src/integration/README.md`;
- actualizar `src/integration/docs/internal_documentation.md`;
- alinear el README raíz con el nuevo alcance `train+eval`.

## Riesgos y mitigaciones

### 1. Cambio de alcance de integration

Riesgo:
el módulo deja de ser puramente `eval-only`.

Mitigación:
mantener un seam de training explícito y no mezclar detalles de SB3 por todo `integration`.

### 2. Duración de las Campaigns

Riesgo:
el training por caso puede volver lentas las campañas.

Mitigación:
defaults modestos, ejecución secuencial clara y cancelación segura básica.

### 3. Comparabilidad parcial

Riesgo:
no todos los casos producirán métricas `MO+RL` comparables.

Mitigación:
separar agregados válidos de incidencias y no forzar a los casos incompletos dentro del agregado principal.

### 4. Ambigüedad entre Campaign y runner unitario

Riesgo:
mezclar la Campaign con el `runner.py` actual y complicar ambos caminos.

Mitigación:
mantener el runner unitario como seam pequeño y añadir una capa Campaign explícita por encima.

## Recomendación final

La mejor forma de dejar `integration` listo en esta iteración es construir una Campaign `train+eval` guiada por terminal, apoyada sobre los escenarios unitarios ya existentes, con configuración global, ejecución secuencial por `circuito x backend`, persistencia por Campaign y Summary Document Markdown como salida principal.

Ese camino maximiza reutilización del código ya sólido (`Baseline`, `MO_Only`, `MO+RL`) y minimiza el riesgo de abrir a la vez demasiadas fronteras nuevas como TUI fullscreen, multi-seed, `RL_Only` homogéneo o QASM completo para RL.
