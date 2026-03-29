# Module Contract Cleanup Design

## Goal

Realign the repository with its intended four-module architecture so that:
- `src/qiskit_interface/` stays responsible for Qiskit-facing transpilation and metrics only;
- `src/mo_module/` produces and evaluates candidate layouts without owning RL handoff;
- `src/rl_module/` accepts a generic `initial_layout` input without knowing who produced it;
- `src/integration/` is the only future owner of MO -> RL orchestration and benchmark scenarios.

The immediate objective is to remove misleading coupling in code comments, docstrings, docs, and public narratives while preserving the ability to test MO and RL independently.

## Scope

This design covers the following areas:
- `src/qiskit_interface/transpiler.py` and related docs that currently describe MO+RL pipeline ownership too broadly;
- `src/mo_module/docs/internal_documentation.md` and MO-facing documentation that currently claims direct output to `rl_module`;
- `src/rl_module/environment.py` and RL-facing docs that currently describe `initial_layout` as coming specifically from MO;
- repository-level architecture docs and broken references, including `README.md`, `.github/AGENTS.md`, and the missing `docs/agents.md` reference target.

This design does not include implementation of the integration module itself.

## Non-Goals

The following are explicitly out of scope for this change:
- implementing `src/integration/` orchestration logic;
- implementing the `MO+RL` benchmark scenario end-to-end;
- making `synthesis` trainable in `src/rl_module/`;
- redesigning MO fitness, RL rewards, or benchmark methodology beyond what is needed to restore module boundaries and public contracts.

## Current Problems

The repository currently mixes two different stories about ownership:
- the architecture docs and project skills describe a four-module system where `src/integration/` owns MO -> RL handoff;
- several module-local docs and comments describe direct MO -> RL interaction as if it were already a responsibility of `mo_module` or `rl_module`.

This creates three concrete problems:
- public contracts are ambiguous, so future work can easily reintroduce cross-module imports and hidden coupling;
- function names and docstrings suggest behavior that belongs to future orchestration rather than current local module responsibilities;
- the repo root points users to `docs/agents.md`, but that file does not exist, leaving the canonical architecture undocumented in the workspace.

## Design

### Canonical Module Ownership

The repository should present exactly one ownership model:
- `qiskit_interface` owns backend access, transpilation, metrics extraction, and baseline evaluation;
- `mo_module` owns layout search and objective evaluation;
- `rl_module` owns environment state, action/observation design, and generic layout ingestion through `initial_layout`;
- `integration` owns cross-module orchestration, MO -> RL handoff, and scenario comparison (`Baseline`, `MO_Only`, `RL_Only`, `MO+RL`).

No module other than `integration` should describe itself as the bridge between MO and RL.

### Shared Layout Contract

The shared layout representation remains:
- `layout[i] = physical_qubit_for_logical_qubit_i`

This convention is already compatible with:
- `src/qiskit_interface/transpiler.py`, where `initial_layout` is passed to Qiskit-facing transpilation;
- `src/rl_module/environment.py`, where `reset(options={"initial_layout": layout})` injects a starting assignment.

The important design change is not the data shape, but ownership of the handoff:
- `mo_module` may produce a layout;
- `rl_module` may consume a layout;
- only `integration` is allowed to define the process that connects those two facts.

### `transpile_with_custom_layout()` Contract

`transpile_with_custom_layout()` should be documented as a local evaluation helper for Qiskit-facing transpilation with a caller-supplied initial layout.

Its contract is:
- input: circuit, backend, and a layout using the repository's logical -> physical convention;
- output: a `TranspilationResult` describing the transpiled circuit and metrics;
- responsibility: evaluate a layout under Qiskit/backend constraints;
- non-responsibility: perform RL handoff, own hybrid pipeline logic, or describe future orchestration.

This keeps the function useful for MO scoring without turning it into an integration API.

### `initial_layout` in RL

`rl_module` should continue to expose `initial_layout` support because it is the correct generic environment input. However, the public contract must stop saying that the layout comes specifically from MO.

The correct wording is:
- `initial_layout` is an optional external starting assignment;
- the producer may be a test, a benchmark harness, a future integration module, or another caller;
- `rl_module` validates and applies the layout, but does not care where it came from.

This preserves testability and future integration readiness without introducing module coupling.

### Documentation Strategy

The cleanup should establish one canonical architecture document inside the repo:
- create `docs/agents.md` as the stable reference for module boundaries, responsibilities, and shared conventions;
- keep `README.md` and `.github/AGENTS.md` aligned with that document;
- update module-local docs so they reference `integration` for future handoff instead of describing direct MO -> RL ownership.

Module-local docs may still mention future consumers, but only as examples, not as ownership claims.

### Testing Strategy

This work is mostly contract cleanup, so regression protection should focus on module-local behavior that must remain true after the wording and API cleanup.

Tests should ensure:
- `qiskit_interface` continues to accept and report `initial_layout` consistently;
- `rl_module` continues to accept and validate `initial_layout` as a generic input;
- module-level test suites still pass independently in the absence of `integration`;
- no new direct dependency from `mo_module` to `rl_module` is introduced while doing the cleanup.

Documentation-only assertions do not need dedicated tests, but the touched module suites should be re-run after the cleanup.

## Risks And Mitigations

- Risk: over-correcting by removing useful local APIs.
  - Mitigation: keep `transpile_with_custom_layout()` and `initial_layout`; only narrow their documented responsibility.

- Risk: prematurely implementing `integration` behavior inside existing modules.
  - Mitigation: treat every cross-module orchestration concern as explicitly out of scope.

- Risk: making end-to-end benchmarking seem closer than it is.
  - Mitigation: state clearly that `src/integration/` is still a stub and that `synthesis` remains placeholder work outside this change.

## Success Criteria

The change is successful when all of the following are true:
- `README.md` no longer points to a missing architecture document;
- `docs/agents.md` exists and matches the four-module architecture already described in repository instructions;
- `src/qiskit_interface/` no longer presents itself as the MO+RL bridge;
- `src/mo_module/` no longer claims direct output ownership toward `rl_module`;
- `src/rl_module/` describes `initial_layout` as a generic external input rather than a MO-specific handoff;
- `tests/test_qiskit_interface`, `tests/test_mo_module`, and `tests/test_rl_module` still pass independently.

## Verification Plan

- Re-run `tests/test_qiskit_interface` after Qiskit-facing contract cleanup.
- Re-run `tests/test_mo_module` after MO docs and local narratives are updated.
- Re-run `tests/test_rl_module` after RL contract wording is updated.
- Optionally run the full `pytest tests/` suite once all touched files are in place.

The expected outcome is unchanged module-local functionality with clearer boundaries and a cleaner path toward a future `integration` module.
