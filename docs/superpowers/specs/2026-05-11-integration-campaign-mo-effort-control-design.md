# Integration Campaign MO Effort Control Design

## Goal

Improve the guided `integration` Campaign CLI for large circuits by making MO effort controllable and by ensuring those controls actually affect the MO execution path used by Campaign `MO_Only` and Campaign `MO+RL`.

## Problem

The current guided Campaign CLI already exposes MO sizing knobs such as `MO population size` and `MO generations`, but the current `integration` flow does not propagate those values into `ScenarioRequest` or into `scenarios._run_mo(...)`.

As a result:

- the guided CLI gives the impression that Campaign MO effort is configurable;
- larger circuits, especially beyond roughly 7 qubits, can still run with the same fixed MO behavior;
- the user cannot reliably trade runtime for better MO layouts inside the guided Campaign surface.

## Desired Outcome

The guided Campaign CLI should support both:

1. real manual MO control, where the user chooses the MO effort explicitly;
2. an explicit adaptive mode that automatically increases MO effort for larger Campaign Cases.

This must remain an `integration` concern. `mo_module` continues to execute layout optimization, but it should not own Campaign-level policy for how much MO effort to spend.

## Scope

In scope:

- add `MO effort mode = auto/custom` to the guided Campaign CLI;
- make `default` Campaign use `auto`;
- make `advanced` Campaign expose `auto/custom` clearly;
- resolve effective MO settings per Campaign Case using `num_qubits`;
- propagate effective MO settings through `CampaignConfig`, `campaign_runner.py`, `ScenarioRequest`, and `scenarios.py`;
- ensure `optimize_layout_quick(...)` and `optimize_layout(...)` consume the effective MO knobs;
- render the selected MO effort mode in confirmation output and Campaign reporting.

Out of scope:

- moving MO tuning logic into `mo_module`;
- introducing Optuna or dynamic search-time tuning;
- changing scenario ownership boundaries between `integration`, `mo_module`, and `rl_module`;
- changing the existing single-scenario CLI in `src/integration/runner.py`.

## UX

### Default Campaign

`Default Campaign` remains short and reproducible, but it should no longer imply a single fixed MO effort for every circuit size.

Behavior:

- the CLI still only asks for circuit families and qubit sizes before building the default config;
- the resulting `CampaignConfig` uses `mo_effort_mode="auto"`;
- the confirmation summary shows that the Campaign is using `auto` MO effort;
- the confirmation summary also shows the auto-resolved MO preview for each distinct selected qubit size.

### Advanced Campaign

`Advanced Campaign` should explicitly ask for `MO effort mode (auto/custom)` before collecting manual MO knobs.

Behavior:

- if the user chooses `auto`, the CLI does not ask for `MO quick`, `MO population size`, or `MO generations` manually;
- if the user chooses `custom`, the CLI asks for all three manual MO knobs;
- the CLI still asks for `layout_policy` and, when needed, `mo_objective_name` after the MO effort choice.

### Confirmation Summary

The confirmation summary should always tell the truth about the selected behavior.

Required summary lines:

- `MO Effort Mode`;
- if `auto`, one preview line per distinct qubit size showing effective `quick`, `population_size`, and `n_generations`;
- if `custom`, the explicit manual `quick`, `population_size`, and `n_generations` values.

## Effective MO Resolution

`integration` owns the adaptive policy through a small focused helper.

Recommended helper module:

- `src/integration/mo_effort.py`

Recommended public helper:

- `resolve_effective_mo_settings(...)`

The effective settings are resolved per Campaign Case, not once per whole Campaign. This avoids a mixed Campaign letting a large case force expensive MO settings onto smaller cases or vice versa.

## Auto Heuristic

Use a simple stable heuristic keyed only by `num_qubits`:

- `<= 7`: `quick=True`, `population_size=30`, `n_generations=50`
- `8-10`: `quick=False`, `population_size=60`, `n_generations=120`
- `11-14`: `quick=False`, `population_size=80`, `n_generations=160`
- `>= 15`: `quick=False`, `population_size=100`, `n_generations=220`

This heuristic is intentionally conservative and reviewable. It is not a search-based tuner.

## Contracts

### CampaignConfig

Add a new field:

- `mo_effort_mode: "auto" | "custom"`

Keep these fields:

- `mo_use_quick`
- `mo_population_size`
- `mo_n_generations`

Rationale:

- `CampaignConfig` needs to preserve both the chosen mode and the manual defaults used in `custom` mode;
- in `auto`, those stored manual values remain the canonical base/default values, while effective values are resolved per Campaign Case.

### ScenarioRequest

Add the effective MO sizing fields directly to `ScenarioRequest`:

- `mo_population_size`
- `mo_n_generations`

`ScenarioRequest` should represent the effective MO configuration that a given scenario run will actually use, not just the high-level Campaign preference.

### Baseline and RL-Only Safety

`Baseline` and `RL_Only` continue to reject non-default MO selection inputs.

That safety should extend to the new effective MO sizing fields so these scenarios cannot accidentally accept MO-specific non-default values.

## Execution Flow

### Campaign Runner

`campaign_runner.py` should resolve effective MO settings before building the `ScenarioRequest` for `MO_Only` and `MO+RL`.

Behavior:

- `Baseline` continues using safe defaults;
- `MO_Only` gets the effective MO settings for that Campaign Case;
- `MO+RL` gets the same effective MO settings for that Campaign Case, although injected layout reuse still means it will usually skip rerunning MO in the Campaign hybrid path.

### Scenario Layer

`scenarios._run_mo(...)` should consume the effective settings from `ScenarioRequest`:

- `optimize_layout_quick(...)` receives `population_size` and `n_generations` explicitly;
- `optimize_layout(...)` receives an `OptimizerConfig` built from the effective settings and `request.seed`.

This is the critical fix that turns the existing CLI knobs from cosmetic inputs into real execution controls.

## Reporting

`campaign_reporting.py` should render:

- `MO Effort Mode` in the Summary Document;
- auto preview lines when the Campaign uses `auto`;
- manual MO fields when the Campaign uses `custom`.

This keeps public Campaign output aligned with the real execution behavior.

## Boundaries

- `integration` owns Campaign-level MO effort policy.
- `mo_module` owns MO execution and optimizer semantics.
- `rl_module` remains unaffected.
- The MO -> RL handoff remains owned by `integration`.

No direct MO policy logic should move into `mo_module`.

## Testing

Add or update tests in:

- `tests/test_integration/test_mo_effort.py`
- `tests/test_integration/test_campaign_contracts.py`
- `tests/test_integration/test_contracts.py`
- `tests/test_integration/test_campaign_runner.py`
- `tests/test_integration/test_scenarios.py`
- `tests/test_integration/test_campaign_cli.py`
- `tests/test_integration/test_campaign_reporting.py`

Test focus:

- auto heuristic tiers;
- `CampaignConfig.mo_effort_mode` validation;
- `ScenarioRequest` effective MO sizing defaults and restrictions;
- per-Case MO effort resolution in `campaign_runner.py`;
- propagation into quick and full MO execution paths;
- advanced CLI `auto/custom` branching;
- truthful confirmation and reporting output.

## Expected Result

After this change:

- the guided Campaign CLI exposes a clear `auto/custom` MO effort choice;
- manual MO controls actually change execution behavior;
- larger circuits get more MO effort when `auto` is selected;
- Campaign reporting reflects the selected MO effort policy faithfully;
- `integration` keeps ownership of Campaign orchestration and MO effort policy.
