# Integration Train+Eval Campaign TUI Implementation Plan

> **For agentic workers:** Use a test-first workflow for each new seam (`campaign contracts`, `training bridge`, `campaign runner`, `campaign reporting`, `interactive CLI`). Run commands with `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe`.

**Goal:** implementar en `src/integration/` una Campaign `train+eval` guiada por terminal que permita seleccionar `1..N` circuitos de librería, ejecutar `Baseline`, `MO_Only` y `MO+RL` por cada `circuito x backend`, entrenar RL por caso, persistir resultados por Campaign y generar un Summary Document en Markdown.

**Architecture:** `src/integration/` pasa a ser dueño de la orquestación de **Train+Eval Campaigns** además de los escenarios unitarios existentes. `src/integration/scenarios.py` seguirá siendo la capa de escenarios unitarios (`Baseline`, `MO_Only`, `MO+RL`). La nueva capa de Campaign se apoyará en esos escenarios y en un seam explícito hacia `src.rl_module.training`, sin repartir lógica de training RL por todo el módulo. `src/rl_module/` seguirá siendo dueño del training y de la producción de checkpoints; `src/integration/` será dueño del producto `circuito x backend`, de la persistencia de Campaign, del resumen agregado y del Summary Document.

**Tech Stack:** Python 3.10, pytest, argparse/dataclasses/pathlib, Qiskit 2.x, fake backends de Qiskit, RL training pipeline actual de `src.rl_module.training`.

---

## File Map

- Create: `src/integration/campaign_contracts.py` - DTOs y validaciones para Campaign, Campaign Case, resultados por caso y resumen agregado.
- Create: `src/integration/training_bridge.py` - seam entre Campaign y `src.rl_module.training.setup_training_pipeline(...)`.
- Create: `src/integration/campaign_reporting.py` - agregación de resultados y render del Summary Document Markdown.
- Create: `src/integration/campaign_runner.py` - orquestación secuencial `train+eval` por `circuito x backend`.
- Modify: `src/integration/__init__.py` - reexportar nuevas entradas públicas si procede.
- Modify: `src/integration/runner.py` or create `src/integration/campaign_cli.py` - CLI interactiva guiada para Campaign.
- Modify: `src/integration/contracts.py` - ampliar si hace falta algún contrato unitario reutilizable por Campaign.
- Modify: `src/integration/scenarios.py` - solo si hace falta un seam más limpio para consumo desde Campaign, sin mezclar responsabilidades.
- Modify: `src/integration/README.md` - documentar el nuevo alcance `train+eval`.
- Modify: `src/integration/docs/internal_documentation.md` - alinear ownership, pipelines y límites de la v2.
- Modify: `README.md` - actualizar el alcance visible del módulo `integration`.
- Test: `tests/test_integration/test_campaign_contracts.py`
- Test: `tests/test_integration/test_training_bridge.py`
- Test: `tests/test_integration/test_campaign_reporting.py`
- Test: `tests/test_integration/test_campaign_runner.py`
- Test: `tests/test_integration/test_runner.py` and/or `tests/test_integration/test_campaign_cli.py`
- Modify: `tests/test_integration/test_docs.py` - verificar que la documentación pública/interna refleja el nuevo alcance.

---

## Task 1: Introduce Campaign Contracts

**Files:**
- Create: `src/integration/campaign_contracts.py`
- Test: `tests/test_integration/test_campaign_contracts.py`

- [ ] **Step 1: Write failing tests for Campaign DTOs**

Cover at least:

```python
def test_campaign_config_builds_global_default_campaign(...):
    ...

def test_campaign_case_id_is_stable_from_circuit_and_backend(...):
    ...

def test_campaign_result_distinguishes_completed_failed_and_interrupted_cases(...):
    ...

def test_campaign_config_rejects_empty_circuit_selection(...):
    ...
```

Required contract surface:

- `CampaignConfig`
- `CampaignCircuitSpec` or equivalent
- `CampaignCase`
- `CampaignCaseResult`
- `CampaignSummary` or equivalent aggregate DTO

- [ ] **Step 2: Implement minimal dataclasses and validation**

Implementation requirements:

- model a global Campaign config shared by all cases;
- distinguish `default` vs `advanced` Campaign mode;
- store circuit family + qubit count explicitly rather than only a display string;
- model a case status that can represent at least `completed`, `failed`, `incomplete`, `cancelled` or equivalent;
- keep unitary scenario results separate from aggregate summary fields.

- [ ] **Step 3: Add helpers to enumerate Campaign Cases**

Implementation requirements:

- build the product `selected_circuits x selected_backends`;
- generate stable case identifiers from resolved circuit/backend names;
- keep ordering deterministic.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_contracts.py -q
```

---

## Task 2: Add a Training Bridge Owned by Integration

**Files:**
- Create: `src/integration/training_bridge.py`
- Test: `tests/test_integration/test_training_bridge.py`

- [ ] **Step 1: Write failing tests for the bridge contract**

Cover at least:

```python
def test_train_case_returns_best_model_when_available(...):
    ...

def test_train_case_falls_back_to_final_model_when_best_model_is_missing(...):
    ...

def test_train_case_surfaces_training_failure_with_paths_and_status(...):
    ...
```

- [ ] **Step 2: Implement a bridge result contract**

Implementation requirements:

- encapsulate the call to `setup_training_pipeline(...)`;
- return a small stable result object with:
  - training status
  - selected artifact path
  - `best_model_path`
  - `final_model_path`
  - `run_model_dir`
  - `run_log_dir`
  - effective training configuration summary
- prefer `best_model.zip` when present, fallback to final.

- [ ] **Step 3: Make the bridge Campaign-aware without making RL aware of Campaign**

Recommended approach:

- let Campaign provide a base output directory for the case;
- let the bridge adapt that into the training call if possible;
- if the current RL API cannot fully honor a case-local directory yet, record and surface the actual run paths explicitly.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_training_bridge.py -q
```

---

## Task 3: Build Campaign Reporting and Summary Rendering

**Files:**
- Create: `src/integration/campaign_reporting.py`
- Test: `tests/test_integration/test_campaign_reporting.py`

- [ ] **Step 1: Write failing tests for aggregation semantics**

Cover at least:

```python
def test_aggregate_summary_uses_only_comparable_completed_cases(...):
    ...

def test_aggregate_summary_counts_failed_and_incomplete_cases_separately(...):
    ...

def test_summary_markdown_includes_config_aggregate_case_detail_and_incidents(...):
    ...
```

- [ ] **Step 2: Implement aggregate metric helpers**

Implementation requirements:

- aggregate only cases with comparable `Baseline`, `MO_Only`, `MO+RL` metrics;
- keep separate counters for completed comparable cases, failed cases, incomplete cases, and cancelled campaigns if applicable;
- focus the main metric bundle on:
  - `trans_depth`
  - `trans_two_qubit_gates`
  - `trans_cnot_equivalent`
  - `elapsed_time_s`

- [ ] **Step 3: Implement Markdown rendering**

The Summary Document should include:

- Campaign metadata
- global configuration
- selected circuits and backends
- aggregate comparison table
- per-case detail section
- incidents section
- short RL training summary per case
- final campaign status

- [ ] **Step 4: Add persistence helpers**

Implementation requirements:

- write `summary.md`;
- write a structured `campaign.json` or equivalent;
- support writing per-case `result.json` into a case subdirectory.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_reporting.py -q
```

---

## Task 4: Implement the Sequential Campaign Runner

**Files:**
- Create: `src/integration/campaign_runner.py`
- Test: `tests/test_integration/test_campaign_runner.py`

- [ ] **Step 1: Write failing tests for Campaign orchestration**

Cover at least:

```python
def test_run_campaign_executes_cases_in_stable_sequential_order(...):
    ...

def test_run_campaign_trains_then_runs_baseline_mo_only_and_mo_rl(...):
    ...

def test_run_campaign_preserves_baseline_and_mo_only_when_training_fails(...):
    ...

def test_run_campaign_records_incomplete_mo_rl_without_aborting_remaining_cases(...):
    ...
```

- [ ] **Step 2: Implement case execution flow**

Implementation requirements per case:

1. resolve the circuit instance for the case;
2. run `Baseline`;
3. run `MO_Only`;
4. run training through the bridge;
5. run `MO+RL` with the produced artifact;
6. persist per-case outputs;
7. update in-memory Campaign summary.

- [ ] **Step 3: Freeze random circuits per Campaign**

Implementation requirements:

- if the user selected `random_shallow` or `random_deep`, generate the random circuit once per Campaign case definition using the campaign seed;
- reuse that same circuit across the scenario executions for that case.

- [ ] **Step 4: Add interruption support**

Implementation requirements:

- catch `KeyboardInterrupt` or explicit cancellation signal;
- stop after the current safe boundary;
- preserve completed results;
- mark the Campaign final status as interrupted/cancelled.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_runner.py -q
```

---

## Task 5: Add the Interactive Terminal Campaign CLI

**Files:**
- Modify: `src/integration/runner.py` or create `src/integration/campaign_cli.py`
- Test: `tests/test_integration/test_runner.py` and/or `tests/test_integration/test_campaign_cli.py`

- [ ] **Step 1: Decide the public entrypoint shape**

Recommended option:

- keep `runner.py` for unitary scenario execution;
- add a dedicated `campaign_cli.py` for guided Campaign execution.

Acceptable alternative:

- extend `runner.py` with a top-level subcommand split such as `scenario` vs `campaign`.

Prefer the option that keeps the existing unitary runner stable and minimally disturbed.

- [ ] **Step 2: Write failing interaction tests around config building**

Test the config-building layer with monkeypatched inputs rather than full stdin integration first.

Cover at least:

```python
def test_default_campaign_cli_builds_canonical_defaults(...):
    ...

def test_advanced_campaign_cli_accepts_multiple_backends(...):
    ...

def test_advanced_campaign_cli_collects_mo_policy_and_objective(...):
    ...
```

- [ ] **Step 3: Implement the guided flow**

Required flow:

1. choose circuit family/families;
2. choose qubit sizes;
3. choose `default` or `advanced`;
4. choose backend(s);
5. collect RL knobs;
6. collect MO knobs;
7. print a confirmation summary;
8. execute the Campaign;
9. print final status and output paths.

- [ ] **Step 4: Implement default and advanced rules**

Default mode:

- `fake_torino`
- `MaskablePPO`
- `timesteps=5000`
- `lookahead=10`
- `max_steps=200`
- `seed=42`
- MO quick + compromise
- MO quick `population_size=30`, `n_generations=50`

Advanced mode:

- support multiple backends;
- expose RL knobs: `algorithm`, `timesteps`, `frontier_mode`, `lookahead`, `max_steps`, `seed`;
- expose MO knobs: `population_size`, `n_generations`, `layout_policy`;
- if `best_on_objective`, require explicit objective selection.

- [ ] **Step 5: Run targeted CLI tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_runner.py tests/test_integration/test_campaign_cli.py -q
```

---

## Task 6: Integrate Persistence Layout and Public Paths

**Files:**
- Modify: `src/integration/campaign_runner.py`
- Modify: `src/integration/campaign_reporting.py`
- Test: `tests/test_integration/test_campaign_runner.py`
- Test: `tests/test_integration/test_campaign_reporting.py`

- [ ] **Step 1: Introduce a canonical campaign directory layout**

Recommended structure:

```text
campaigns/<campaign_id>/
  summary.md
  campaign.json
  cases/<case_id>/result.json
```

- [ ] **Step 2: Ensure every case has a local output anchor**

Implementation requirements:

- every case gets a directory under `cases/<case_id>/`;
- the structured case result lands there;
- the Summary Document references case ids and relevant training artifact paths.

- [ ] **Step 3: Make final console output stable and useful**

At Campaign completion, print at least:

- Campaign id
- final status
- path to `summary.md`
- path to structured campaign output

- [ ] **Step 4: Re-run persistence-related tests**

---

## Task 7: Update Documentation To Match the New Scope

**Files:**
- Modify: `src/integration/README.md`
- Modify: `src/integration/docs/internal_documentation.md`
- Modify: `README.md`
- Test: `tests/test_integration/test_docs.py`

- [ ] **Step 1: Update the public module scope**

Document that `integration` now supports a Campaign-level `train+eval` flow in addition to unitary scenario orchestration.

- [ ] **Step 2: Keep boundaries explicit**

The docs should still state:

- `integration` owns Campaign orchestration and scenario comparison;
- `rl_module` owns how RL training is implemented and how checkpoints are produced;
- MO still enters the hybrid path through `initial_layout` at evaluation time.

- [ ] **Step 3: Document default vs advanced campaign paths**

Cover:

- default campaign semantics;
- advanced campaign semantics;
- Summary Document contents;
- failure/incomplete-case reporting.

- [ ] **Step 4: Run doc tests**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_docs.py -q
```

---

## Task 8: Full Verification Sweep

**Files:**
- Modify or create the files above

- [ ] **Step 1: Run targeted integration suite**

Run:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration -q
```

- [ ] **Step 2: Run any adjacent tests affected by public contract changes**

At minimum consider:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_rl_module/test_rl_module.py -k "setup_training_pipeline or model_metadata" -q
```

and, if docs or public README assertions changed:

```bash
C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_module_contracts.py -q
```

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Confirm that only intended source, test, and documentation files are modified.

---

## Sequencing Notes

- Start with contracts and reporting before the interactive CLI. The CLI should build on stable DTOs, not invent its own config shape.
- Keep `scenarios.py` mostly unchanged unless a genuinely reusable seam is missing.
- Do not entangle Campaign persistence with RL internal storage more than necessary in v1.
- Prefer a dedicated campaign entrypoint over overloading the existing single-scenario runner unless the split stays very clean.

## Self-Review Notes

- The plan keeps the scenario layer and the campaign layer separate.
- The plan preserves the agreed semantics: `Baseline`, `MO_Only`, `MO+RL` only.
- The plan keeps `integration` as owner of orchestration, not of RL internals.
- The plan defers fullscreen TUI, parallel execution, full resume support, `RL_Only` in TUI, and complete RL-side QASM support.
