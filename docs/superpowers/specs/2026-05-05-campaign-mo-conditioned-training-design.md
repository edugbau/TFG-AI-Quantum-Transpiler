# Campaign MO-Conditioned Training Design

## Goal

Adjust the Campaign-only `MO+RL` flow so RL training for each Campaign Case starts from the exact MO-selected layout that will later be evaluated, without changing the public behavior of `RL_Only`, the single-scenario CLI, or non-Campaign training call sites.

## Context

The current Campaign implementation trains RL for each `circuit x backend` case in a layout-agnostic way, then evaluates `MO+RL` by injecting the MO-selected layout as `initial_layout` only at evaluation time.

Recent manual diagnosis showed a stable failure mode:

- `RL_Only` on `qft_5` + `fake_torino` completes from the default layout.
- The same trained model fails when evaluated from the MO-selected `initial_layout`.
- The same failure reproduces in `RL_Only` whenever a non-trivial `initial_layout` is injected, even outside the Campaign runner.

This means the current Campaign semantics produce a mismatch between how RL is trained and how `MO+RL` is actually evaluated.

## Approved decision

For Campaign `MO+RL`, RL training must become layout-conditioned.

That means each Campaign Case will now follow this sequence:

1. run `Baseline`;
2. run `MO_Only`;
3. take the exact `selected_layout` produced by `MO_Only`;
4. train RL for that Campaign Case starting from that exact layout;
5. evaluate `MO+RL` using the same layout and the produced Training Artifact.

This behavior is mandatory for Campaign `MO+RL`; it is not a user-facing toggle in this iteration.

## Scope

This work covers:

- adding an optional `initial_layout` input to the RL training pipeline;
- wiring Campaign training so it always consumes the exact MO-selected layout for `MO+RL` cases;
- keeping Campaign ownership of the MO -> RL handoff;
- keeping the single-scenario `RL_Only` and `MO+RL` public interfaces unchanged;
- documenting the new Campaign semantics.

## Out of scope

This work does not cover:

- changing the public CLI for unitary scenarios;
- exposing a new public training mode outside Campaign;
- training over multiple candidate layouts per case;
- adding a user-facing flag to disable this behavior;
- redesigning MO objective selection.

## Ownership and boundaries

`integration` remains the owner of Campaign orchestration and of the MO -> RL handoff.

`mo_module` still produces layouts.

`rl_module` still owns training implementation and checkpoint production. It only receives an optional caller-provided `initial_layout` when asked to train from one.

This preserves the current ownership model:

- `integration` decides when a Campaign Case should be conditioned on a layout;
- `rl_module` executes training from the provided starting layout;
- `RL_Only` remains a standalone Scenario and does not change its public request surface.

## Functional design

### 1. Campaign runner semantics

The Campaign runner must treat the `MO_Only` selected layout as an input to both:

- Campaign training for that case;
- `MO+RL` evaluation for that case.

If `MO_Only` does not produce a successful `selected_layout`, training for the hybrid path must not start.

### 2. Training pipeline semantics

`src/rl_module/training.setup_training_pipeline(...)` will accept a new optional `initial_layout` parameter.

Behavior:

- if `initial_layout is None`, training behaves exactly as it does today;
- if `initial_layout` is provided, both the training environment and the internal evaluation environment created by the training pipeline must reset using that same layout.

This keeps the new behavior opt-in at the training API level while making it mandatory at the Campaign orchestration level.

### 3. Training bridge semantics

`src/integration/training_bridge.py` will accept an optional `initial_layout` from the Campaign runner and forward it to `setup_training_pipeline(...)`.

The bridge remains responsible for:

- case-local output directories;
- artifact selection (`best_model.zip` first, `final_model.zip` fallback);
- stable reporting of paths and effective config.

The bridge does not decide whether a layout should be used; that decision remains in the Campaign runner.

### 4. Backward compatibility

The following behavior must remain unchanged:

- `RL_Only` public scenario execution;
- `MO+RL` public scenario execution outside Campaign;
- the single-scenario `src/integration/runner.py` CLI;
- any existing training call site that does not pass `initial_layout`.

This iteration adds capability; it does not replace the default training API semantics.

## Error handling

- If `MO_Only` succeeds but `selected_layout` is missing, the Campaign Case should be treated as failed before hybrid training/evaluation.
- If the training pipeline rejects the provided `initial_layout`, the Training Bridge should surface a failed training result and the Campaign should continue to the next case.
- If training succeeds but `MO+RL` still fails to complete, the case remains `incomplete` exactly as today.

## Testing

Required coverage:

1. `setup_training_pipeline(...)` uses the provided `initial_layout` when resetting both training and eval environments.
2. The default training path remains unchanged when no layout is provided.
3. `train_case(...)` forwards the Campaign-provided `initial_layout` to the training pipeline.
4. `run_campaign(...)` trains hybrid Campaign Cases using the `MO_Only.selected_layout`.
5. Existing Campaign and scenario tests continue to pass.

## Documentation impact

The Campaign documentation must explicitly state that Campaign `MO+RL` now uses MO-conditioned training:

- `MO_Only` selects the layout;
- RL training starts from that same layout for the Campaign Case;
- `MO+RL` evaluation uses the same layout and the resulting Training Artifact.

This replaces the earlier documented assumption that Campaign training remains layout-agnostic while only evaluation is MO-conditioned.
