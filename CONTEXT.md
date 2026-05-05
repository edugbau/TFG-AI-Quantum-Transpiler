# Hybrid Quantum Transpilation

This context describes the language used to talk about cross-module transpilation evaluation in this repository. It focuses on the `integration` layer as the owner of campaign orchestration, scenario comparison, and MO-to-RL evaluation flow.

## Language

**Campaign**:
A reproducible train-and-evaluate execution that runs the selected comparison flow across one or more circuit-backend cases.
_Avoid_: batch, session, experiment run

**Campaign Case**:
One concrete `circuit x backend` combination executed inside a Campaign.
_Avoid_: item, job, task

**Scenario**:
One named evaluation path inside integration, such as **Baseline**, **MO_Only**, or **MO+RL**.
_Avoid_: mode, variant, pipeline step

**Baseline**:
The canonical Qiskit reference scenario using `qiskit_level_1` without MO-selected layout or RL routing.
_Avoid_: default Qiskit, plain transpilation

**MO_Only**:
The scenario that selects a layout with multi-objective optimization and evaluates that layout with Qiskit.
_Avoid_: MO baseline, layout-only run

**MO+RL**:
The scenario that injects an MO-selected layout into RL routing and, when routing completes, compares the reconstructed routed circuit through Qiskit post-routing metrics.
_Avoid_: hybrid baseline, full pipeline

**Train+Eval Campaign**:
An Integration Campaign where RL is trained per Campaign Case before the comparison scenarios are evaluated.
_Avoid_: evaluation-only campaign, checkpoint replay

**Training Artifact**:
The RL checkpoint produced during a Campaign Case and later used for evaluation, preferring `best_model.zip` and falling back to `final_model.zip`.
_Avoid_: saved state, model file

**Default Campaign**:
The canonical campaign path using shared default configuration values instead of sequential manual tuning.
_Avoid_: quick run, easy mode

**Advanced Campaign**:
The campaign path where the user adjusts MO and RL configuration explicitly before execution.
_Avoid_: expert run, custom mode

**Summary Document**:
The Markdown report produced for a Campaign, with aggregate comparisons, per-case detail, training notes, and recorded incidents.
_Avoid_: export, log dump, notebook

## Relationships

- A **Campaign** contains one or more **Campaign Cases**
- A **Campaign Case** runs the comparison **Scenarios** selected for that Campaign
- The canonical comparison set is **Baseline**, **MO_Only**, and **MO+RL**
- A **Train+Eval Campaign** produces one **Training Artifact** per **Campaign Case** before evaluating **MO+RL**
- A **Summary Document** belongs to exactly one **Campaign**

## Example dialogue

> **Dev:** "If a Campaign selects three circuits and two backends, how many Campaign Cases do we run?"
> **Domain expert:** "Six — one per `circuit x backend` combination, each with its own RL training artifact and comparison results."

## Flagged ambiguities

- "TUI" was used to mean a fullscreen terminal UI at first, but resolved here as a guided interactive CLI in the terminal.
- "Baseline" could have meant any Qiskit default; resolved here as the specific **Baseline** Scenario backed by `qiskit_level_1`.
- "integration flow" originally referred to evaluation only, but resolved here so **Train+Eval Campaign** explicitly includes RL training orchestration per Campaign Case.
