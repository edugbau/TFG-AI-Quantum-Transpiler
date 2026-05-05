# Integration Campaign CLI Design

## Goal

Add a guided interactive CLI for **Campaign** execution without disturbing the existing single-scenario runner in `src/integration/runner.py`.

## Entrypoint

- Create `src/integration/campaign_cli.py`.
- Keep `src/integration/runner.py` as the stable unitary-scenario entrypoint.
- The new CLI owns guided terminal input for building a `CampaignConfig`, confirming it, executing a **Train+Eval Campaign**, and printing final output paths.

## Scope

The CLI will support the agreed v1 Campaign flow only:

- select one or more library circuits;
- select one or more qubit sizes;
- choose `default` or `advanced` mode;
- select backend(s);
- run `Baseline`, `MO_Only`, and `MO+RL` through `run_campaign(...)`;
- print Campaign completion status and output paths.

Out of scope for this task:

- changing `runner.py` into a shared subcommand CLI;
- fullscreen TUI behavior;
- `RL_Only` in the Campaign flow;
- public `qasm_file` support for Campaign creation.

## UX Shape

The flow is step-by-step and reprompts on invalid input at the same step.

Required order:

1. choose circuit family or families;
2. choose qubit sizes for those circuit families;
3. choose `default` or `advanced` Campaign mode;
4. choose backend or backends;
5. collect RL knobs;
6. collect MO knobs;
7. print a confirmation summary;
8. execute the Campaign;
9. print final status and output paths.

## Config Building

The CLI should separate prompt/validation from config assembly.

Recommended internal split:

- prompt helpers: typed input, menu selection, comma-separated selection, yes/no confirmation, reprompt loops;
- config builders: build `CampaignConfig` for `default` and `advanced` paths;
- execution wrapper: construct `Campaign`, call `run_campaign(...)`, print final result.

This split keeps config-building testable without depending on full stdin integration.

## Default Mode

The CLI should build the canonical default Campaign config:

- backend: `fake_torino`
- RL algorithm: `MaskablePPO`
- RL timesteps: `5000`
- RL frontier mode: existing campaign default
- RL lookahead window: `10`
- RL max steps: `200`
- seed: `42`
- MO quick enabled
- MO population size: `30`
- MO generations: `50`
- layout policy: `compromise`

Default mode should ask only for the inputs still intended to vary in v1: selected circuits and qubit sizes.

## Advanced Mode

Advanced mode should expose:

- multiple backends;
- RL knobs: `algorithm`, `timesteps`, `frontier_mode`, `lookahead`, `max_steps`, `seed`;
- MO knobs: `population_size`, `n_generations`, `layout_policy`;
- explicit objective selection when `layout_policy=best_on_objective`.

The CLI should map the selected objective name into the `CampaignConfig.mo_objective_name` field and let downstream orchestration keep ownership of MO-to-RL behavior.

## Execution

After confirmation, the CLI should:

- create a `Campaign` from the built config;
- call `run_campaign(...)`;
- print at least:
  - Campaign id
  - final status
  - path to `summary.md`
  - path to structured campaign output

The CLI should not absorb failures silently. If config building fails unexpectedly after validation, surface the error. If campaign execution returns a failed, cancelled, or interrupted Campaign, print the final status and output paths anyway.

## Testing

Create `tests/test_integration/test_campaign_cli.py`.

Initial test focus:

- `test_default_campaign_cli_builds_canonical_defaults(...)`
- `test_advanced_campaign_cli_accepts_multiple_backends(...)`
- `test_advanced_campaign_cli_collects_mo_policy_and_objective(...)`
- reprompt behavior for invalid selections

Keep `tests/test_integration/test_runner.py` focused on the existing single-scenario CLI.

## Boundaries

- `campaign_cli.py` owns guided terminal interaction.
- `campaign_runner.py` owns Campaign execution order and persistence.
- `campaign_reporting.py` owns Markdown and structured output writing.
- `runner.py` remains the unitary-scenario CLI.

This keeps the new Campaign UX additive and minimizes regression risk in the existing scenario entrypoint.
