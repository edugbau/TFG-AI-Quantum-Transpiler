# RL GUI Mode Split and Episode Inspector Design

## Goal

Separate the RL GUI into one application with two specialized mode views (`routing` and `synthesis`) and make episode evaluation understandable without relying on a raw text transcript.

## Scope

This design covers:

- a shared RL GUI shell with mode-specific subviews instead of one monolithic mixed screen;
- a structured episode inspector for both `routing` and `synthesis`;
- visualization of step-by-step actions, layout changes, executed gates, and synthesis primitives;
- environment and frontier trace plumbing needed by the GUI to render evaluation details;
- tests and documentation updates inside `src/rl_module/`.

## Non-Goals

The following are explicitly out of scope for this iteration:

- splitting the project into two standalone desktop applications;
- redesigning RL training semantics or changing policy behavior;
- changing the observation contract consumed by the agent unless strictly required for GUI rendering;
- implementing a full interactive DAG viewer;
- adding dynamic routing inside `mode="synthesis"`.

## Current State Confirmed In The Repository

The current GUI is implemented as a single `RLBenchmarkGUI` class in `src/rl_module/gui/rl_gui.py`.

- Training and evaluation orchestration are shared for both modes.
- Mode differences are handled mostly with inline branching.
- The sidebar mixes controls that are meaningful for `routing` with controls that are meaningful for `synthesis`.
- The "Evaluación Episodio" tab is currently a text box populated from `_evaluation_thread()`.
- `routing` evaluation shows only a compact row per step even though the environment already exposes richer signals such as `repeated_layout`, `undo_swap`, and `routing_progress_delta`.
- `synthesis` evaluation still inherits routing-oriented framing such as remaining gates, even though synthesis progress is residual-based.

This means the main user problem is visual and semantic clarity, not lack of a training/evaluation pipeline.

## User Experience Direction

The user priority for this work is clarity of use, not maximum code separation. The approved direction is:

- one application window;
- two mode-specific views;
- one shared evaluation inspector with mode-specific detail rendering.

This keeps the operational workflow stable while making the UI match the real differences between `routing` and `synthesis`.

## Architecture Summary

The GUI will be reorganized into:

- a shared `RLBenchmarkGUI` shell responsible for app lifecycle, threading, training/evaluation orchestration, model loading, progress, terminal logs, and plot tabs;
- a `RoutingView` responsible for routing-specific controls and routing-specific evaluation detail rendering;
- a `SynthesisView` responsible for synthesis-specific controls and synthesis-specific evaluation detail rendering;
- an `EvaluationPanel` responsible for a shared table/timeline of episode steps, a shared summary area, and a mode-specific detail pane.

This is intentionally not two separate GUI entrypoints. The current codebase shares too much orchestration logic for that to be a good first split.

## Mode Separation Rules

### Routing view

The routing view should foreground:

- frontier mode selection;
- lookahead window;
- layout movement;
- SWAP actions and their effect on the visible frontier;
- gates unlocked and executed by each SWAP.

### Synthesis view

The synthesis view should foreground:

- basis profile selection;
- fixed-layout synthesis semantics;
- applied primitive sequence;
- residual progress toward Clifford equivalence.

The synthesis view should not inherit routing-specific language such as "SWAPs insertados" or "puertas restantes" as primary progress indicators.

## Evaluation Model

The current evaluation transcript will be replaced by a structured step model stored in memory and rendered by the GUI.

Each step record should include at least:

- `step`;
- `reward`;
- `action_type`;
- `is_valid_action`;
- `layout_before` and `layout_after`;
- mode-specific `info` fields;
- a pre-step snapshot of the visible frontier or synthesis state.

The text terminal may remain as a secondary channel for headings, compatibility, and error messages, but it will no longer be the primary evaluation representation.

## Routing Evaluation Semantics

For `routing`, the GUI should present:

- chosen swap edge;
- whether the action was invalid;
- number and identities of gates executed in cascade;
- `routing_progress_delta`;
- loop-related flags such as `repeated_layout` and `undo_swap`;
- visible frontier entries with gate name, logical qubits, physical qubits, and executability;
- layout before and after the action.

The easiest useful visual form is a step table plus a detail panel. A full DAG renderer is not required in this iteration.

## Synthesis Evaluation Semantics

For `synthesis`, the GUI should present:

- primitive name;
- primitive physical qubits;
- primitive cost;
- residual distance before and after the action;
- residual delta;
- current `physical_to_logical` mapping;
- the accumulated primitive sequence applied so far.

Synthesis progress should be explained as residual reduction, not as consumption of target gates in order.

## Data Plumbing Requirements

The environment already exposes most scalar fields needed for the inspector, but the GUI still lacks a reliable structured trace source.

This design adds read-oriented trace plumbing:

- `frontier.py` should optionally capture the exact `GateTuple` values executed during a cascade;
- `environment.py` should expose those executed gates in `info["executed_gates"]` for routing;
- `environment.py` should expose `swap_edge` and synthesis primitive physical qargs in `info`;
- `environment.py` should expose a helper for visible frontier entries so the GUI can render details without depending on internals.

These additions must not change the observation contract consumed by the RL agent.

## Compatibility Rules

- The training flow, deterministic evaluation policy, and best-model selection logic must remain behaviorally equivalent.
- Existing routing behavior is a protected compatibility target.
- Synthesis should keep its current fixed-layout Clifford scope.
- Existing tests in `tests/test_rl_module/test_rl_module.py` must be preserved or upgraded rather than removed.

## File Boundaries

- `src/rl_module/gui/rl_gui.py` keeps the app shell and thread orchestration.
- `src/rl_module/gui/routing_view.py` contains routing-specific controls and detail rendering.
- `src/rl_module/gui/synthesis_view.py` contains synthesis-specific controls and detail rendering.
- `src/rl_module/gui/evaluation_panel.py` contains the shared episode inspector widgets and the step record model.
- `src/rl_module/environment.py` and `src/rl_module/frontier.py` expose GUI-facing trace data without taking on presentation responsibilities.

## Risks

- Over-refactoring `rl_gui.py` and mixing visual cleanup with functional changes.
- Preserving routing-only labels inside synthesis screens and summaries.
- Making the GUI depend on private environment state instead of stable helper methods.
- Breaking tests that currently assert textual evaluation output.

## Incremental Delivery

The implementation should proceed in this order:

1. add trace plumbing in `frontier.py` and `environment.py`;
2. split the GUI into shared shell plus subviews;
3. replace text-only evaluation with structured step records and a shared inspector;
4. add mode-specific detail rendering;
5. update docs and run RL-focused verification.

## Success Criteria

This work is successful when:

- a user can immediately tell whether they are in `routing` or `synthesis` mode from the screen structure alone;
- evaluating an episode makes it obvious which action was chosen and what changed because of it;
- routing clearly shows SWAP effects and executed gates;
- synthesis clearly shows primitive application and residual progress;
- the RL training/evaluation backend remains stable and the focused RL test suite passes.
