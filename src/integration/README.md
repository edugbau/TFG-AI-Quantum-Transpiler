# Integration v1

`src/integration/` owns the integration-level orchestration that connects the project modules through both single-scenario evaluation and Campaign-level `train+eval` execution.

## Current v1 scope

Integration v1 covers these routing evaluation scenarios:

- `Baseline`
- `MO_Only`
- `RL_Only`
- `MO+RL`

The scenario layer still preserves `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL` as explicit Scenarios.

Integration v1 also owns the Campaign layer used for reproducible `train+eval` comparison across one or more Campaign Cases. The canonical Campaign comparison set is `Baseline`, `MO_Only`, and `MO+RL`; `RL_Only` remains available as a standalone Scenario outside the primary guided Campaign flow.

Within that guided Campaign comparison, `MO_Only` selects the layout for the Campaign Case. Campaign training for `MO+RL` starts from that exact layout, and `MO+RL` evaluation reuses the same layout together with the resulting Training Artifact for the same Campaign Case.

The current implementation now includes campaign contracts, the training bridge, campaign reporting and Summary Document rendering, the sequential Campaign runner, and the guided Campaign CLI, while preserving the shared contracts for routing evaluation.

When an RL routing model has a neighboring `run_metadata.json`, integration uses the saved routing contract from that sidecar when available before falling back to legacy defaults.

When present, that sidecar can carry versioned masked routing metadata for newer checkpoints. `integration` consumes that versioned masked routing metadata when available; otherwise the legacy fallback remains so previously saved PPO/DQN or other unmasked checkpoints stay evaluable.

QASM input is available for the Qiskit-facing scenarios in this v1 scope. Concretely, `Baseline` and `MO_Only` can load circuits from `qasm_file`, while `RL_Only` and `MO+RL` still do not expose `qasm_file` publicly.

The backend catalog is intentionally limited so the integration scenarios stay reproducible and credential-free. The underlying integration/backend layer works with the current fake backends exposed by `qiskit_interface`, while the guided Campaign CLI currently exposes the narrower backend set implemented in `campaign_cli.py`: `fake_torino` and `fake_brisbane`.

Internal implementation details, contracts and pipelines are documented in `docs/internal_documentation.md`.

## Campaign ownership and boundaries

- `integration` owns Campaign orchestration, Scenario comparison, Campaign persistence, Summary Document generation, and the MO -> RL handoff contract.
- `rl_module` owns how RL training is implemented and how Training Artifacts are produced.
- `mo_module` owns layout generation and selection inputs.
- In the hybrid Campaign path, `MO_Only` selects the layout, `integration` forwards it through `initial_layout` into RL training, and `MO+RL` evaluation reuses that same layout and the resulting Training Artifact; `rl_module` consumes the contract but does not own the handoff.

## Campaign semantics

`integration` now supports a Train+Eval Campaign made of one or more Campaign Cases, where each Campaign Case is one `circuit x backend` combination.

`Default Campaign` is the canonical guided path. It uses the shared default configuration values exposed by `campaign_cli.py`, keeps a single default backend, and favors a concise reproducible setup instead of per-field tuning.

`Advanced Campaign` exposes explicit configuration choices for backend selection, RL algorithm and training settings, MO sizing, and layout policy before execution.

Each Campaign persists a public root with at least:

- `summary.md` as the Summary Document;
- `campaign.json` as the structured Campaign output;
- `cases/<case>/result.json` for per-case persistence.

The Summary Document records Campaign metadata, aggregate comparison across `Baseline`, `MO_Only`, and `MO+RL`, per-case detail, RL training notes, and recorded incidents. Cases that fail before comparison, or complete without a comparable metric bundle, are reported explicitly through the aggregate comparison and incidents sections. In the current implementation, those sections are the authoritative signal for non-comparable cases; top-level Campaign status can still remain `completed`, and a per-case status can still remain `completed`, when comparability is missing.

For the Campaign hybrid path, the sequence is explicit: `MO_Only` selects the layout, Campaign training produces the Training Artifact starting from that exact layout, and `MO+RL` evaluation uses the same layout and that artifact when it runs the routed comparison.

## RL scenario semantics

`RL_Only` returns episode summaries, not final circuits.

`MO+RL` now attempts to reconstruct the routed circuit from the RL trace, preferring the exact `executed_gate_trace` when available and using `swap_trace` to replay the physical swaps, and then runs the post-routing Qiskit stages to produce comparable transpilation metrics and an artifact.

In those routing summaries, `total_swaps == len(swap_trace)` and counts only the swaps that were actually materialized and can be replayed into the reconstructed circuit.

For transpilation metrics, `trans_num_qubits`/`trans_width` still represent materialized physical width, while `trans_active_qubits` is the better comparator for sparse-layout physical occupancy.

If the RL episode does not complete routing, `MO+RL` returns a controlled result with the routing summary preserved and skips routed-circuit reconstruction and post-routing transpilation.

If the metadata sidecar is missing, integration reports that condition through an extra note and falls back to legacy defaults so previously saved routing checkpoints remain evaluable.

This does not change module ownership at Scenario level: `integration` owns Scenario orchestration and Campaign comparison, while `rl_module` owns how masked or unmasked routing checkpoints are produced.

## Deferred work

- Final circuit reconstruction/export remains deferred for `RL_Only`.
- QASM input for RL-based scenarios is still deferred until those flows can consume circuit artifacts beyond episode summaries.
- Final circuit materialization for RL-focused flows remains a future iteration beyond the current `RL_Only` episode-summary scope.
