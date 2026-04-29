# Integration v1

`src/integration/` owns the routing evaluation scenarios that connect the project modules at integration level.

## Current v1 scope

Integration v1 covers these routing evaluation scenarios:

- `Baseline`
- `MO_Only`
- `RL_Only`
- `MO+RL`

The current implementation focuses on scenario orchestration and shared contracts for routing evaluation.

When an RL routing model has a neighboring `run_metadata.json`, integration uses the saved routing contract from that sidecar when available before falling back to legacy defaults.

When present, that sidecar can carry versioned masked routing metadata for newer checkpoints. `integration` consumes that versioned masked routing metadata when available; otherwise the legacy fallback remains so previously saved PPO/DQN or other unmasked checkpoints stay evaluable.

QASM input is available for the Qiskit-facing scenarios in this v1 scope. Concretely, `Baseline` and `MO_Only` can load circuits from `qasm_file`, while `RL_Only` and `MO+RL` still do not expose `qasm_file` publicly.

The backend catalog is intentionally limited to the current fake backends exposed by `qiskit_interface` so the integration scenarios stay reproducible and credential-free.

Internal implementation details, contracts and pipelines are documented in `docs/internal_documentation.md`.

## RL scenario semantics

`RL_Only` returns episode summaries, not final circuits.

`MO+RL` now attempts to reconstruct the routed circuit from the RL trace, preferring the exact `executed_gate_trace` when available and using `swap_trace` to replay the physical swaps, and then runs the post-routing Qiskit stages to produce comparable transpilation metrics and an artifact.

In those routing summaries, `total_swaps == len(swap_trace)` and counts only the swaps that were actually materialized and can be replayed into the reconstructed circuit.

For transpilation metrics, `trans_num_qubits`/`trans_width` still represent materialized physical width, while `trans_active_qubits` is the better comparator for sparse-layout physical occupancy.

If the RL episode does not complete routing, `MO+RL` returns a controlled result with the routing summary preserved and skips routed-circuit reconstruction and post-routing transpilation.

If the metadata sidecar is missing, integration reports that condition through an extra note and falls back to legacy defaults so previously saved routing checkpoints remain evaluable.

This does not change module ownership or public scope: `integration` still owns only scenario orchestration, while `rl_module` owns how masked routing or unmasked routing checkpoints are produced.

## Deferred work

- Final circuit reconstruction/export remains deferred for `RL_Only`.
- QASM input for RL-based scenarios is still deferred until those flows can consume circuit artifacts beyond episode summaries.
