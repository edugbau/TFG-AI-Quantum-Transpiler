# OpenCode Module Skills Migration Design

**Goal:** migrar las cinco skills activas del repositorio al sistema nativo de OpenCode en `.opencode/skills/`, reescribiendolas con formato superpowers compatible con las constraints reales de discovery, frontmatter y routing del proyecto.

## Contexto

El repositorio ya contiene cinco skills activas bajo `.github/skills/`:

- `mo-optimization`
- `rl-quantum-synthesis`
- `mo-rl-pipeline`
- `qiskit-2x-compliance`
- `experimentation-logging`

Estas skills ya cumplen una funcion util de routing desde `.github/AGENTS.md`, pero no siguen todavia el formato esperado para skills nativas de OpenCode:

- no viven en la ruta de discovery de OpenCode para proyecto local;
- usan campos de frontmatter que OpenCode ignora (`argument-hint`, `user-invocable`);
- varias `description` mezclan condiciones de uso con resumenes de workflow;
- la estructura del cuerpo es mas breve y menos uniforme que la recomendada por las constraints documentadas de authoring.

La documentacion actual de OpenCode confirma que las skills del proyecto deben vivir en `.opencode/skills/<name>/SKILL.md` y que solo reconoce estos campos de frontmatter:

- `name`
- `description`
- `license`
- `compatibility`
- `metadata`

Ademas, el skill `writing-skills` fija constraints de calidad que tambien aplican aqui:

- `name` debe ser estable, searchable y compatible con la carpeta;
- `description` debe empezar con `Use when...`;
- `description` debe describir cuando cargar la skill, no resumir su proceso;
- el contenido debe ser reusable, con keywords de descubrimiento y sin narrativa historica;
- las referencias del proyecto deben apuntar a rutas reales y activas.

## Alcance aprobado

Esta migracion cubre las cinco skills activas actuales y su enrutado documental asociado:

1. migrar `mo-optimization`;
2. migrar `rl-quantum-synthesis`;
3. migrar `mo-rl-pipeline`;
4. migrar `qiskit-2x-compliance`;
5. migrar `experimentation-logging`;
6. alinear `.github/AGENTS.md` y cualquier referencia activa para que la ubicacion canonica pase a ser `.opencode/skills/`.

## No objetivos

Quedan fuera de este trabajo:

- cambiar el nombre publico de las cinco skills;
- rediseñar la arquitectura de los modulos del proyecto;
- introducir nuevas skills distintas de las cinco ya activas;
- convertir esta migracion documental en una refactorizacion funcional de `src/`;
- mantener una solucion dual permanente con `.github/skills/` como ruta activa en paralelo.

## Decisiones aprobadas

### 1. Ruta canonica: `.opencode/skills/`

Las skills activas del proyecto pasaran a vivir en:

- `.opencode/skills/mo-optimization/SKILL.md`
- `.opencode/skills/rl-quantum-synthesis/SKILL.md`
- `.opencode/skills/mo-rl-pipeline/SKILL.md`
- `.opencode/skills/qiskit-2x-compliance/SKILL.md`
- `.opencode/skills/experimentation-logging/SKILL.md`

Esta decision sigue la discovery nativa de OpenCode para skills locales de proyecto. `.github/skills/` deja de ser la ubicacion activa.

### 2. Nombres estables

Los nombres de skill no cambian. Esto evita drift con `.github/AGENTS.md`, conserva continuidad semantica y mantiene el routing ya documentado por modulo y por tipo de tarea.

### 3. Formato superpowers, pero compatible con OpenCode real

Cada `SKILL.md` se reescribira con estructura tipo superpowers, pero respetando las constraints reales de OpenCode en frontmatter.

Frontmatter permitido:

```yaml
---
name: mo-optimization
description: Use when modifying src/mo_module fitness, encoding, operators, Pareto analysis, or tuning workflows for quantum layout optimization.
compatibility: opencode
metadata:
  module: mo_module
  scope: project
---
```

Campos como `argument-hint` y `user-invocable` no se conservaran, porque no forman parte del esquema reconocido por OpenCode y solo anaden ruido.

### 4. Descriptions optimizadas para discovery

Todas las `description` deben:

- empezar por `Use when...`;
- describir triggers concretos de carga;
- incluir vocabulario que OpenCode pueda usar para discovery;
- evitar resumenes de procedimiento del estilo "does X by doing Y and Z".

### 5. Estructura canonica del cuerpo

Cada skill seguira una estructura comun suficiente para discovery, lectura rapida y aplicacion practica:

1. `# <Skill Title>`
2. `## Overview`
3. `## When to Use`
4. `## Quick Reference`
5. `## Core Pattern` o `## Implementation`
6. `## Common Mistakes`
7. `## Project References`

No todas las skills necesitan flowchart. Solo se anadira uno si existe una decision no obvia entre rutas de actuacion. En esta migracion, la expectativa por defecto es no usar flowcharts.

## Constraints de implementacion

### Constraints de OpenCode

1. La carpeta debe coincidir con `name`.
2. `name` debe seguir el patron lowercase-hyphenated.
3. `description` debe estar dentro del limite valido y ser suficientemente especifica.
4. El archivo debe llamarse exactamente `SKILL.md`.
5. Las skills deben vivir bajo `.opencode/skills/` para ser descubiertas localmente por OpenCode.

### Constraints del repositorio

1. `.github/AGENTS.md` sigue siendo el punto de routing por modulo y por tipo de tarea.
2. El proyecto mantiene cuatro modulos canonicos:
   - `src/qiskit_interface/`
   - `src/mo_module/`
   - `src/rl_module/`
   - `src/integration/`
3. Las skills deben reforzar, no difuminar, los limites modulares ya documentados.
4. `integration` sigue siendo el unico dueno del handoff MO -> RL y de la orquestacion de escenarios.

### Constraints de authoring derivadas de `writing-skills`

1. Sin narrativa historica ni ejemplos de "una vez hicimos...".
2. Keywords concretas para discovery: librerias, APIs, errores, conceptos y nombres de modulo.
3. Referencias solo a archivos reales y utiles del repo.
4. Contenido centrado en guidance reusable, no en checklist de una sola sesion.
5. Las secciones deben ser escaneables y mantener una convencion comun entre las cinco skills.

## Diseno por skill

### 1. `qiskit-2x-compliance`

Tipo de skill: reglas rigidas de compatibilidad.

Debe cubrir:

- imports validos de Qiskit >= 2.x;
- APIs prohibidas (`qiskit.execute`, `QuantumInstance`, paquetes legacy);
- pautas correctas para `backend.run`, primitives V2 y export/import de QASM;
- testing local con fake backends;
- referencias al estado real de `src/qiskit_interface/`.

Keywords esperadas para discovery:

- `qiskit.terra`
- `qiskit.aer`
- `QuantumInstance`
- `execute`
- `SamplerV2`
- `EstimatorV2`
- `FakeBackend`

### 2. `mo-optimization`

Tipo de skill: tecnica y patron para `src/mo_module/`.

Debe cubrir:

- representacion de layout y encoding;
- fitness multiobjetivo con `depth` y `cnot_equivalent`;
- operadores evolutivos y defaults razonables;
- analisis de Pareto e hypervolume;
- tuning con Optuna;
- recordatorio explicito de que `mo_module` no orquesta MO -> RL directamente.

### 3. `rl-quantum-synthesis`

Tipo de skill: tecnica para `src/rl_module/`.

Debe cubrir:

- `gymnasium.Env` y su contrato actual;
- `reset(seed=..., options=...)`;
- diseno de observacion, accion y recompensa;
- entrenamiento con Stable-Baselines3;
- callbacks, checkpoints y TensorBoard;
- separacion entre capacidades actuales de routing y alcance limitado de synthesis.

### 4. `mo-rl-pipeline`

Tipo de skill: skill de integracion con ownership rigido.

Debe cubrir:

- handoff mediante `initial_layout`;
- escenarios `Baseline`, `MO_Only`, `RL_Only`, `MO+RL`;
- ownership de `src/integration/` sobre la orquestacion;
- contrato de layout compartido;
- necesidad de mantener desacoplados `mo_module` y `rl_module`.

### 5. `experimentation-logging`

Tipo de skill: skill transversal de benchmarking y reproducibilidad.

Debe cubrir:

- seeds globales reproducibles;
- logging para SB3;
- export tabular consistente;
- plots comparables;
- columnas minimas del esquema experimental compartido.

## Cambios documentales asociados

### 1. Alinear `.github/AGENTS.md`

`.github/AGENTS.md` debe seguir usando los mismos nombres de skill, pero dejar claro que las skills activas del proyecto viven en `.opencode/skills/`.

La referencia final no debe seguir apuntando a `./skills/` si esa carpeta no existe como ubicacion activa.

### 2. Eliminar dependencia activa de `.github/skills/`

La migracion debe evitar que el repositorio mantenga dos ubicaciones activas en paralelo. Si se decide conservar `.github/skills/` temporalmente durante la implementacion, esa situacion debe entenderse como transitoria y cerrarse en el mismo lote de trabajo.

## Riesgos

### 1. Riesgo de compatibilidad documental

Si se cambian nombres de skill o se deja `.github/AGENTS.md` desalineado, el routing del repositorio quedara roto aunque los archivos nuevos existan.

Mitigacion:

- mantener nombres estables;
- actualizar referencias activas en el mismo cambio.

### 2. Riesgo de falsa compatibilidad con OpenCode

Si se conserva frontmatter heredado no reconocido como parte del diseno canonico, el repo parecera migrado pero seguira documentando un contrato incorrecto.

Mitigacion:

- usar solo frontmatter reconocido por OpenCode;
- documentar `compatibility: opencode`.

### 3. Riesgo de skills demasiado genericas

Si las skills se vuelven demasiado abstractas, perderan valor como guias modulares y como superficie de discovery.

Mitigacion:

- incluir referencias concretas al repo;
- incluir vocabulario especifico del dominio y del modulo.

## Verificacion

Antes de considerar cerrada la migracion:

1. las cinco skills deben existir en `.opencode/skills/` con `SKILL.md` valido;
2. cada `name` debe coincidir con su carpeta;
3. cada `description` debe empezar con `Use when...`;
4. ninguna skill migrada debe depender de `argument-hint` o `user-invocable`;
5. `.github/AGENTS.md` debe seguir ruteando correctamente a las cinco skills por nombre;
6. no debe quedar ninguna referencia activa que presente `.github/skills/` como ubicacion canonica.

## Resultado esperado

Tras esta migracion:

- OpenCode podra descubrir las skills del proyecto desde su ruta nativa local;
- las cinco skills conservaran sus nombres y su semantica de routing;
- el repositorio tendra skills mas uniformes, mas buscables y alineadas con las constraints reales de OpenCode y con las reglas de authoring de superpowers;
- la documentacion activa dejara de tratar `.github/skills/` como ubicacion canonica.
