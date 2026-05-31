# TFG - Transpilacion Cuantica Hibrida

**Optimizacion de layout multiobjetivo y sintesis mediante aprendizaje por refuerzo**

Este repositorio implementa un flujo de transpilacion cuantica dividido en cuatro modulos: `qiskit_interface`, `mo_module`, `rl_module` e `integration`.

La idea central es separar responsabilidades:

- `qiskit_interface` aporta circuitos, backends, metricas y baselines de Qiskit.
- `mo_module` busca layouts iniciales con optimizacion multiobjetivo.
- `rl_module` resuelve routing y sintesis con aprendizaje por refuerzo.
- `integration` conecta los modulos, compara escenarios y ejecuta Campaigns reproducibles.

## Como leer el proyecto

1. Empieza por [src/qiskit_interface/README.md](src/qiskit_interface/README.md) para entender los contratos de circuitos, backends y transpilacion.
2. Sigue con [src/mo_module/README.md](src/mo_module/README.md) para ver como se generan y seleccionan layouts.
3. Continua con [src/rl_module/README.md](src/rl_module/README.md) para el entorno RL, el entrenamiento y la metadata de checkpoints.
4. Termina con [src/integration/README.md](src/integration/README.md) para ver como se conectan los escenarios y las Campaigns.

## Mapa del sistema

| Modulo | Responsabilidad principal | No se encarga de |
| --- | --- | --- |
| `src/qiskit_interface/` | Circuitos, backend metadata, metricas y transpilacion baseline | Orquestar MO o RL |
| `src/mo_module/` | Busqueda multiobjetivo de layouts y analisis de Pareto | Ejecutar entrenamiento RL |
| `src/rl_module/` | Entorno Gymnasium, recompensas, agente SB3, training, GUI y synthesis v1 | Generar layouts o orquestar Campaigns |
| `src/integration/` | Scenarios, handoff MO -> RL, Campaigns train+eval y persistencia publica | Reimplementar la logica interna de los otros modulos |

## Contratos compartidos

La representacion de layout compartida es:

```python
layout[i] = physical_qubit_for_logical_qubit_i
```

Ese contrato permite que:

- `qiskit_interface` lo evale con helpers de transpilacion.
- `mo_module` lo produzca como salida de optimizacion.
- `rl_module` lo reciba como `initial_layout` externo.
- `integration` actue como puente entre productor y consumidor.

## Scenarios y Campaigns

`integration` distingue dos niveles:

- **Scenario**: evaluacion puntual de `Baseline`, `MO_Only`, `RL_Only` o `MO+RL`.
- **Campaign**: ejecucion reproducible `train+eval` compuesta por uno o varios `Campaign Case`, donde cada case es una combinacion `circuit x backend`.

En el camino hibrido, `MO_Only` selecciona el layout, `integration` lo reusa como `initial_layout` al entrenar RL y `MO+RL` evalua el mismo layout con el Training Artifact del case. Si el routing deriva un subgrafo path-expanded, se usa ese grafo; si no, la Campaign cae al coupling map completo y deja trazabilidad del fallback.

## Estado actual

- `Baseline` y `MO_Only` aceptan entrada `qasm_file`; los escenarios basados en RL siguen orientados a trazas y no exponen esa entrada publica.
- `rl_module` soporta routing con action space fijo y `masked routing`, ademas de un primer modo de `synthesis` restringido a Clifford.
- Los checkpoints RL pueden incluir `run_metadata.json` con metadata versionada de masked routing para que `integration` mantenga compatibilidad hacia atras.
- Los checkpoints nuevos de routing enmascarado usan `MaskablePPO` con `frontier_restricted_edges.v3`: bloquean undo-SWAPs, ciclos cortos y episodios estancados, con poda SABRE top-k opcional. Los checkpoints `v1`, `v2`, `PPO` y `DQN` siguen siendo evaluables con su contrato historico.
- `integration` puede reconstruir el circuito ruteado cuando el episodio RL completa y usa `trans_active_qubits` para comparar layouts dispersos cuando la anchura materializada no cuenta toda la historia.

## Documentacion de apoyo

- [src/mo_module/docs/internal_documentation.md](src/mo_module/docs/internal_documentation.md) y sus apendices: tuning, benchmark y analisis de resultados.
- [src/rl_module/docs/internal_documentation.md](src/rl_module/docs/internal_documentation.md) y sus apendices: frontier, estabilidad y notas de defensa.
- [src/integration/docs/internal_documentation.md](src/integration/docs/internal_documentation.md) para contratos, CLI, Campaigns y compatibilidad.

## Entorno tecnico

| Componente | Version |
|---|---|
| Python | 3.10.8 |
| Qiskit | 2.3.0 |
| PyTorch | 2.5.1+cu121 |
| Gymnasium | 1.2.3 |
| pymoo | 0.6.1.6 |

Consulta [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) para la lista completa de dependencias y versiones.

## Autor

**Eduardo Gonzalez Bautista** - Universidad de Malaga, E.T.S. de Ingenieria Informatica  
Tutores: Gabriel Jesus Luque Polo, Zakaria Abdelmoiz Dahi
