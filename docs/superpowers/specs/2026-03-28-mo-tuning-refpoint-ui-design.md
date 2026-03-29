# MO Tuning Ref Point and Live Status Design

## Goal

Make MO tuning easier to trust and inspect by:
- using a session-wide hypervolume reference point instead of a per-trial point;
- exposing live tuning progress in the GUI;
- allowing either calibrated warm-up reference point discovery or a manual reference point.

## Scope

The change affects three areas:
- `src/mo_module/tuning.py` for tuning control flow, scoring, validation, and progress events;
- `src/mo_module/benchmark/benchmark_gui.py` for live tuning visibility and manual/calibrated reference-point controls;
- `tests/test_mo_module/test_tuning.py` and tuning docs for regression coverage and user-facing explanation.

## Design

### Reference Point Strategy

The current tuning logic computes hypervolume with a reference point derived from the current Pareto front. That makes scores hard to compare across trials. The new design fixes this by selecting exactly one reference point for the full tuning session.

Two modes are supported:
- `calibrated`: run a short warm-up phase, collect Pareto fronts, and compute the session reference point from the worst observed values plus a configurable margin;
- `manual`: skip warm-up entirely and use a user-provided reference point.

The calibrated reference point is stored on the tuner instance, added to study metadata, printed in summaries, and shown explicitly in the GUI.

### Progress and Observability

`LayoutTuner.tune()` emits structured progress events through an optional callback. Events cover warm-up start and end, trial start and end, per-seed completion, best-trial updates, and final completion or failure.

The GUI consumes those events from the tuning worker thread through `self.after(...)` and updates:
- a progress bar;
- a phase/status label;
- a compact best-so-far label;
- the tuning terminal log.

This removes the current dependence on `stdout` capture and makes the state visible while Optuna is running.

### Risk Mitigations

- Validate tuning ranges and budgets early in `HyperparameterSpace` and `LayoutTuner`.
- Keep warm-up explicitly separate from optimization trials so that all optimized trials share the same fixed reference point.
- Record the reference point mode and explicit coordinates in `summary()` and study user attributes.
- Test both calibrated and manual modes directly so the GUI depends on already-verified tuning behavior instead of encoding tuning logic itself.

## Verification Plan

- Extend `tests/test_mo_module/test_tuning.py` with failing tests first for fixed reference-point scoring, manual reference-point validation, calibrated reference-point derivation, progress events, and new validation paths.
- Run the targeted tuning test module first, then the full MO test suite.
