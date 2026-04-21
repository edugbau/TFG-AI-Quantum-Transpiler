# Chapter 3 Branch 1 Train/Eval Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make the Chapter 3 branch-1 worktree complete by documenting the implemented train/eval contract around persisted RL routing metadata, `integration`-owned contract reuse, and the public fallback path for legacy checkpoints.

**Architecture:** `src/integration/` remains the only owner of MO-to-RL orchestration and routing-evaluation v1 scenario composition. `rl_module` may persist routing-evaluation metadata next to saved checkpoints, but `RL_Only` and `MO+RL` continue to return episode summaries, not final circuits. When the `run_metadata.json` sidecar is missing, `integration` falls back to legacy defaults and reports that fallback through `ScenarioResult.notes`.

**Tech Stack:** Markdown documentation, repository contract vocabulary, git

---

## File Map

- Create: `docs/future-iterations/README.md` - map the five Chapter 3 branches, mark branch 1 as the active plan, and recommend execution order.
- Create: `docs/future-iterations/chapter3-branch1-train-eval-contract.md` - describe the implemented branch-1 focus and completion criteria.
- Create: `docs/future-iterations/chapter3-branch2-frontier-observation-and-feasible-actions.md` - record the deferred observation/action branch.
- Create: `docs/future-iterations/chapter3-branch3-generalization-and-curriculum.md` - record the deferred generalization branch.
- Create: `docs/future-iterations/chapter3-branch4-benchmarking-and-comparability.md` - record the deferred benchmarking branch.
- Create: `docs/future-iterations/chapter3-branch5-target-aware-cost-signals.md` - record the deferred reward-shaping branch.
- Create: `docs/superpowers/plans/2026-04-21-chapter3-branch1-train-eval-contract.md` - this implementation plan, copied into the branch worktree.

## Task 1: Add the future-iteration branch map

**Files:**
- Create: `docs/future-iterations/README.md`

- [ ] **Step 1: Document the branch sequence**

Add a short overview that maps all five Chapter 3 branches and recommends the implementation order.

- [ ] **Step 2: Mark branch 1 as active**

Point explicitly to `docs/superpowers/plans/2026-04-21-chapter3-branch1-train-eval-contract.md` as the active technical plan.

- [ ] **Step 3: Keep wording aligned with implemented scope**

Ensure the overview says:

- `integration` is routing-only v1;
- RL scenarios return episode summaries, not final circuits;
- later branches are deferred and not implemented here.

## Task 2: Document the implemented branch-1 direction

**Files:**
- Create: `docs/future-iterations/chapter3-branch1-train-eval-contract.md`

- [ ] **Step 1: Describe the implemented focus**

Cover:

- persisted RL metadata sidecar support;
- `integration` reusing the saved routing contract;
- fallback to legacy defaults when the sidecar is missing;
- public fallback reporting through `ScenarioResult.notes`.

- [ ] **Step 2: Record branch-1 completion criteria**

Make the success conditions explicit so later documentation does not reinterpret the branch as final-circuit work or broader RL orchestration.

- [ ] **Step 3: Reassert module boundaries**

Keep all MO-to-RL orchestration language routed through `src/integration/` and avoid introducing direct MO/RL coupling anywhere else.

## Task 3: Document deferred branches 2 through 5

**Files:**
- Create: `docs/future-iterations/chapter3-branch2-frontier-observation-and-feasible-actions.md`
- Create: `docs/future-iterations/chapter3-branch3-generalization-and-curriculum.md`
- Create: `docs/future-iterations/chapter3-branch4-benchmarking-and-comparability.md`
- Create: `docs/future-iterations/chapter3-branch5-target-aware-cost-signals.md`

- [ ] **Step 1: Give each branch a narrow deferred focus**

Each file should say what that branch is for without claiming the work is implemented in the current branch.

- [ ] **Step 2: Explain why each branch comes later**

Tie the deferral back to branch 1: persisted train/eval metadata and routing-contract reuse are foundational, while richer RL episode design, benchmarking comparability, and reward shaping are later layers.

- [ ] **Step 3: Preserve ownership language**

Do not introduce any MO/RL coupling language outside `integration`.

## Task 4: Verify docs and create one commit

**Files:**
- Create: `docs/future-iterations/README.md`
- Create: `docs/future-iterations/chapter3-branch1-train-eval-contract.md`
- Create: `docs/future-iterations/chapter3-branch2-frontier-observation-and-feasible-actions.md`
- Create: `docs/future-iterations/chapter3-branch3-generalization-and-curriculum.md`
- Create: `docs/future-iterations/chapter3-branch4-benchmarking-and-comparability.md`
- Create: `docs/future-iterations/chapter3-branch5-target-aware-cost-signals.md`
- Create: `docs/superpowers/plans/2026-04-21-chapter3-branch1-train-eval-contract.md`

- [ ] **Step 1: Read back the created files**

Use file reads to confirm all files exist and that the wording is coherent with the implemented branch-1 contract.

- [ ] **Step 2: Check git status**

Run: `git status --short`

Expected: the seven documentation files appear as new files; unrelated `.pyc` changes may still exist but remain untouched.

- [ ] **Step 3: Create the commit**

Run: `git add docs/future-iterations/README.md docs/future-iterations/chapter3-branch1-train-eval-contract.md docs/future-iterations/chapter3-branch2-frontier-observation-and-feasible-actions.md docs/future-iterations/chapter3-branch3-generalization-and-curriculum.md docs/future-iterations/chapter3-branch4-benchmarking-and-comparability.md docs/future-iterations/chapter3-branch5-target-aware-cost-signals.md docs/superpowers/plans/2026-04-21-chapter3-branch1-train-eval-contract.md`

Run: `git commit -m "docs: add chapter 3 future iteration map"`

Expected: one documentation-only commit with no staged runtime changes and no amend.

## Self-Review Notes

- Scope stays documentation-only.
- Branch 1 is documented as the implemented contract foundation, not as broader RL feature work.
- Deferred branches are described as future layers that build on the current routing-evaluation v1 contract.
