# Branch 1 Closure And Canonical Doc Cleanup

## Context

The repository currently documents Chapter 3 Branch 1 as the active worktree direction: persisted RL routing metadata sidecars, routing-only integration v1 orchestration, and explicit legacy fallback behavior for older checkpoints.

The codebase already reflects most of that state:

- `src/integration/` owns `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL` scenario orchestration.
- RL routing model evaluation reuses `run_metadata.json` through `resolve_routing_model_contract()`.
- missing sidecars fall back to legacy defaults and surface a public note.
- repository and integration docs already explain that RL scenarios return episode summaries instead of final circuits.

The remaining problem is documentation drift. `docs/agents.md` still says that `src/integration/` is a stub and that `synthesis` is placeholder work, while the rest of the repository already documents a more advanced state. The user also wants `docs/agents.md` removed entirely.

## Goals

This work has two goals:

1. remove `docs/agents.md` cleanly;
2. close Chapter 3 Branch 1 only if no additional public-facing inconsistencies remain.

## Non-Goals

This work will not:

- add new RL functionality;
- change the public scope of integration v1;
- reconstruct final circuits for RL scenarios;
- broaden synthesis beyond its current first Clifford-capable scope;
- design or plan Branch 2.

## Recommended Approach

Use the repository root `README.md` as the single canonical architecture reference and remove `docs/agents.md`.

This keeps public module ownership and shared layout conventions in one place, reduces duplication, and makes the repository landing page the canonical explanation of current system boundaries.

## Branch 1 Closure Criteria

Branch 1 will be considered closed only if all of these are true after the documentation cleanup:

1. public docs consistently state that `src/integration/` owns MO-to-RL orchestration for `Baseline`, `MO_Only`, `RL_Only`, and `MO+RL`;
2. public docs consistently state that RL scenarios return routing episode summaries, not final circuits;
3. public docs consistently state that persisted `run_metadata.json` sidecars are reused when present and legacy defaults are used when missing;
4. no public-facing canonical document still describes `src/integration/` as a stub;
5. no public-facing canonical document still describes current `synthesis` as pure placeholder work.

Historical design and plan documents under `docs/superpowers/` may remain unchanged when they are clearly archival and do not act as current user-facing references.

## Documentation Changes

### 1. Promote `README.md` to canonical architecture reference

Expand `README.md` so it directly contains the essential content currently duplicated in `docs/agents.md`:

- module responsibilities and boundaries;
- shared layout convention `layout[i] = physical_qubit_for_logical_qubit_i`;
- current status summary aligned with integration v1 and limited synthesis v1;
- current routing-only limitation for RL-backed integration scenarios.

The `README.md` should no longer send readers to `docs/agents.md` for the canonical explanation.

### 2. Remove `docs/agents.md`

Delete the file once all live references are updated.

### 3. Redirect live references

Update all current live references that treat `docs/agents.md` as canonical, especially:

- `README.md`;
- `.github/AGENTS.md`.

If internal project skills mention `docs/agents.md` only as a suggested reading path, update those references too if they are part of the active repository instructions rather than external archived notes.

### 4. Fix any remaining active doc drift relevant to Branch 1 closure

If an active, user-facing document still claims `integration` is a stub or `synthesis` is still placeholder-only, align it to the real state. Archival implementation plans and historical specs are not part of this closure gate unless they are currently presented as the live source of truth.

## Code And Test Changes

No production behavior changes are expected.

Tests should be updated only where they encode the old canonical documentation location or wording. Existing tests that already validate integration v1 scope should continue to pass without semantic weakening.

## Risks

### Documentation cleanup risks

- deleting `docs/agents.md` without moving its canonical content would leave the repository without a stable architecture reference;
- stale references in `README.md`, `.github/AGENTS.md`, or tests could leave broken links or contradictory instructions.

## Verification

Before considering this work complete:

1. search for active references to `docs/agents.md` and replace or justify them;
2. read the updated `README.md` and `.github/AGENTS.md` to confirm they are aligned;
3. run the targeted documentation and integration tests that validate public scope.

## Expected Outcome

After this work:

- the repository has one canonical architecture reference in `README.md`;
- `docs/agents.md` is removed without leaving live reference drift;
- Chapter 3 Branch 1 can be treated as closed if no other active inconsistencies remain.
