# RL Clifford Synthesis Design

## Goal

Implement the first trainable `mode="synthesis"` in `src/rl_module/` as a hardware-aware Clifford synthesis environment while preserving the current functional behavior of `routing`.

## Scope

This design covers:

- a first trainable synthesis mode restricted to Clifford targets;
- fixed-layout episodes, where `initial_layout` is chosen at reset and does not change during the episode;
- hardware-aware action semantics driven by both `coupling_map` and explicit `basis_gates`;
- environment, strategy, reward, training, GUI safety, docs, and tests inside `src/rl_module/`;
- support for the fake-backend native two-qubit gates already used in the repository, especially `cz` and `ecr`.

## Non-Goals

The following are explicitly out of scope for this iteration:

- general synthesis for non-Clifford targets;
- continuous or high-resolution parameterized action spaces;
- dynamic `swap` actions inside `synthesis` episodes;
- redesigning `routing` behavior, reward semantics, or frontier logic;
- implementing `src/integration/` or changing MO -> RL orchestration ownership.

## Current State Confirmed In The Repository

The repository already exposes `mode="synthesis"` in the public API, but the implementation is still a placeholder.

- `src/rl_module/environment.py` creates `SynthesisStrategy` and `SynthesisReward`, but the `step()` branch for `gate` actions does not apply any gate, does not update synthesis state, and always returns `gate_matched_target = False`.
- `src/rl_module/env_strategies.py` currently models synthesis as `MultiDiscrete([gate, q1, q2])` with a fixed basis `['cx', 'sx', 'rz', 'x']`.
- `tests/test_rl_module/test_rl_module.py` only checks smoke-level behavior for `synthesis`: the mode does not crash and returns a `gate_matched_target` key.
- `src/rl_module/frontier.py` and the current shared observation scaffolding are routing-oriented: they describe a target gate frontier, not equivalence to a synthesized circuit.

This means the documentation that calls `synthesis` a placeholder is correct in the main point. The optimistic parts of the documentation are the action model and the idea that the current frontier-based state would naturally extend to synthesis by equivalence.

## Confirmed Technical Constraints

Several implementation constraints were verified against the installed environment and current codebase.

- Qiskit 2.3.0 is installed and supports `Clifford.from_circuit()` and `Clifford.to_circuit()`.
- `Clifford.from_circuit()` accepts Clifford-native or Clifford-compatible gates such as `x`, `sx`, `cz`, `ecr`, `swap`, and `rz(pi/2)`.
- `Clifford.from_circuit()` rejects non-Clifford rotations such as `rz(pi/4)`.
- The fake backends already used in the repository expose native basis sets compatible with a Clifford-only first phase:
- `fake_torino`: `['cz', 'id', 'rz', 'sx', 'x']`
- `fake_sherbrooke`: `['ecr', 'id', 'rz', 'sx', 'x']`
- `fake_brisbane`: `['ecr', 'id', 'rz', 'sx', 'x']`

The most important additional constraint is architectural, not algebraic:

- `coupling_map` alone is not enough to make `synthesis` hardware-aware.
- To choose valid native two-qubit primitives, the environment must also know the backend basis or an equivalent `basis_gates` contract.

This requirement does not currently exist in `QuantumTranspilationEnv`, so the design must add it explicitly.

## Design Summary

The first trainable synthesis mode will be a fixed-layout Clifford synthesis environment over physical qubits.

- The target circuit is accepted only if it is Clifford.
- The target is remapped onto physical qubits using the episode layout chosen at `reset()`.
- The agent applies discrete, hardware-aware primitives on physical qubits.
- Success is defined by Clifford equivalence to the remapped physical target, not by reproducing the target gate sequence.
- `routing` keeps its current frontier-driven semantics and observation contract.

## External API Contract

`QuantumTranspilationEnv` will gain a new optional constructor parameter:

```python
basis_gates: Optional[list[str]] = None
```

The intended contract is:

- in `routing`, `basis_gates` is optional and ignored by the routing transition logic;
- in `synthesis`, `basis_gates` is required, because coupling topology alone does not determine the native primitive set;
- if `mode="synthesis"` and `basis_gates` is missing, the environment raises `ValueError` with a clear explanation.

`setup_training_pipeline()` will thread `basis_gates` through to both training and evaluation environments.

The GUI cannot silently guess native basis gates from a generic coupling preset, so it must expose an explicit synthesis basis profile instead of inferring one from the topology.

## Synthesis State Model

The synthesis state will be tracked in physical-qubit space.

At reset time the environment will build:

- `current_layout[logical] = physical`, as today;
- `_inverse_layout[physical] = logical`, as today;
- a physical target Clifford obtained by remapping the logical target circuit onto the current physical layout;
- a current physical Clifford initialized as identity on `num_physical_qubits`.

The residual is defined as:

```python
residual = current_physical_clifford.adjoint().compose(target_physical_clifford)
```

This orientation is consistent with Qiskit 2.3 `Clifford.compose()` semantics and yields identity exactly when the synthesized circuit matches the target.

The episode is complete when the residual is identity.

This design intentionally works on physical-qubit space instead of only logical-qubit space, because actions are taken on physical qubits and unused physical qubits must remain identity.

## Primitive Catalog

`synthesis` will no longer use a generic `MultiDiscrete([gate, q1, q2])` action space. It will use a fixed discrete catalog of hardware-aware primitives.

The catalog will be built from:

- `num_physical_qubits`;
- the undirected edge set derived from `coupling_map`;
- explicit `basis_gates`.

The first-phase catalog contains:

- single-qubit Clifford-compatible primitives on every physical qubit:
- `x`
- `sx`
- `rz(pi/2)`
- `rz(pi)`
- `rz(3*pi/2)`
- native two-qubit Clifford primitives on every physical edge:
- `cz` if the backend basis contains `cz`
- `ecr` if the backend basis contains `ecr`
- `cx` only if explicitly present in a future or test basis profile

The catalog deliberately excludes:

- `id`, because it rewards stalling;
- `swap`, because layout changes are out of scope in this first synthesis phase;
- arbitrary `rz(theta)`, because that would escape the Clifford group.

Actions are chosen by a single discrete index. The catalog entry itself defines the gate, qubits, and parameter tuple.

## Validity Rules

Not every catalog entry is valid at every step.

- a single-qubit primitive is invalid if its physical qubit is empty under the current layout;
- a two-qubit primitive is invalid if either physical qubit is empty;
- a two-qubit primitive is invalid if its edge is not present in the coupling map;
- a target circuit is invalid for this mode if `Clifford.from_circuit(target_circuit)` fails.

Invalid actions do not terminate the episode. They only produce a penalty and leave the current Clifford unchanged.

## Observation Contract

`routing` keeps its current observation contract.

`synthesis` gets a new observation tailored to equivalence-based progress:

- `layout`: logical -> physical, shape `(num_qubits,)`
- `physical_to_logical`: physical -> logical, shape `(num_physical_qubits,)`
- `residual_symplectic`: flattened symplectic matrix of the residual Clifford, shape `(4 * num_physical_qubits * num_physical_qubits,)`
- `residual_phase`: residual phase bits, shape `(2 * num_physical_qubits,)`
- `step_progress`: normalized scalar in `[0, 1]`

The residual arrays are derived from the physical-space residual Clifford. This gives the agent a direct state signal tied to correctness instead of a proxy based on a target frontier.

## Reward Contract

`SynthesisReward` will be redefined around residual progress and primitive cost.

The environment will populate at least these `info` fields for `synthesis` steps:

- `is_valid_action`
- `primitive_name`
- `primitive_cost`
- `residual_distance_before`
- `residual_distance_after`
- `residual_distance_delta`
- `is_completed`
- `is_truncated`

The reward should combine:

- invalid-action penalty;
- per-step penalty;
- primitive-cost penalty, so two-qubit primitives are naturally more expensive than cheap one-qubit primitives;
- positive shaping proportional to `residual_distance_delta`;
- strong completion bonus;
- truncation penalty when `max_steps` is reached without success.

The first distance metric does not need to be a shortest-path metric over Clifford space. It only needs to be deterministic, cheap, and monotonic enough to serve as a shaping signal. A bitwise distance between the residual tableau and identity is sufficient for v1.

## Episode Semantics

`routing` termination remains driven by frontier exhaustion.

`synthesis` uses mode-specific termination:

- `terminated = residual_is_identity`
- `truncated = current_step >= max_steps`

The fixed-layout design has one important consequence:

- `initial_layout` matters to synthesis quality, but layout changes do not occur inside the episode.

This keeps the first environment trainable and isolates the problem to synthesis under a chosen placement. Dynamic routing+synthesis is future work.

## Routing Compatibility Rule

`routing` is a protected compatibility target.

- No functional change is allowed in routing action semantics, frontier behavior, reward logic, or observation keys.
- Shared files may be refactored only to separate synthesis-specific logic from routing-specific logic.
- Any refactor in `environment.py` must keep the routing branch behaviorally equivalent to the current implementation.

## File Boundaries

To keep the synthesis work from spilling into routing, the implementation should introduce two new modules.

- `src/rl_module/synthesis_primitives.py`: discrete hardware-aware primitive catalog and primitive-to-circuit helpers.
- `src/rl_module/synthesis_clifford.py`: physical target remapping, identity helpers, residual computation, observation extraction, and distance metric.

Existing files should then consume those modules:

- `src/rl_module/env_strategies.py`: action and observation contract.
- `src/rl_module/environment.py`: reset orchestration and mode-specific step logic.
- `src/rl_module/rewards.py`: synthesis reward shaping.
- `src/rl_module/training.py`: basis-gate threading.
- `src/rl_module/gui/rl_gui.py`: explicit synthesis basis-profile selection or equivalent guard.

## Risks And Mitigations

- Risk: accidentally regressing `routing` while editing `environment.py`.
- Mitigation: keep synthesis logic in new helper modules and protect routing with regression tests.

- Risk: using the wrong Clifford composition order and training an inconsistent environment.
- Mitigation: freeze the residual convention with unit tests on tiny one-qubit and two-qubit examples before integrating the environment.

- Risk: making the discrete catalog too large for early training.
- Mitigation: start with the minimal Clifford-capable primitive set and no `swap`.

- Risk: pretending hardware-awareness is encoded by topology alone.
- Mitigation: require explicit `basis_gates` in synthesis mode.

- Risk: leaving the GUI in a broken state once `basis_gates` becomes required.
- Mitigation: add an explicit synthesis basis-profile selector or block synthesis until that input is present.

## Path To General Synthesis

The design must prepare for future general synthesis without implementing it now.

The extension points are:

- the primitive catalog, which can later expose parameterized or continuous actions;
- the equivalence engine, which can later move from Clifford residuals to more general fidelity, unitary-distance, or approximate-synthesis criteria.

The first phase therefore solves the right problem for Clifford circuits without baking Clifford-only assumptions into every file.

## Success Criteria

The design is successful when all of the following are true:

- `mode="synthesis"` is trainable for Clifford targets on hardware-aware primitive catalogs;
- `routing` behavior is unchanged from the caller point of view;
- synthesis actions are discrete catalog indices instead of free-form gate/qubit tuples;
- the environment computes completion by residual identity instead of gate-sequence matching;
- synthesis mode requires explicit `basis_gates` and does not guess native hardware from topology alone;
- docs clearly state that general synthesis is still future work.

## Verification Plan

- add unit tests for primitive catalog construction and primitive-to-circuit generation;
- add unit tests for logical-to-physical target remapping, residual identity, and residual distance;
- add environment tests for synthesis reset, valid completion on tiny Clifford targets, and invalid actions on empty physical qubits;
- re-run `tests/test_rl_module/` to confirm routing remains stable;
- optionally run the full repository suite after the RL module is green.
