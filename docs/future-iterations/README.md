# Chapter 3 Future Iterations

This directory maps the five Chapter 3 branches that complete the current integration roadmap.

## Recommended order

1. Branch 1: train/eval contract
2. Branch 2: frontier observation and feasible actions
3. Branch 3: generalization and curriculum
4. Branch 4: benchmarking and comparability
5. Branch 5: target-aware cost signals

That order keeps the work incremental: branch 1 locks the persisted routing-evaluation contract first, branches 2 and 3 improve the RL-side episode interface, branch 4 makes scenario comparisons cleaner, and branch 5 adds richer reward shaping after the evaluation surface is stable.

## Branch map

| Branch | Status | Focus | Why it sits here |
| --- | --- | --- | --- |
| [Branch 1](chapter3-branch1-train-eval-contract.md) | Active plan | Persisted RL train/eval metadata, routing contract reuse, and legacy fallback behavior | This is the contract foundation already implemented in the current branch. |
| [Branch 2](chapter3-branch2-frontier-observation-and-feasible-actions.md) | Deferred | Richer routing observations and feasible-action masking | Depends on the branch-1 contract being stable before expanding RL episode inputs. |
| [Branch 3](chapter3-branch3-generalization-and-curriculum.md) | Deferred | Training breadth, curriculum structure, and transfer-oriented evaluation | Best layered after branch 2 clarifies the episode interface used by routing agents. |
| [Branch 4](chapter3-branch4-benchmarking-and-comparability.md) | Deferred | More comparable reporting across Baseline, MO_Only, RL_Only, and MO+RL | Comes later because branch 1 still exposes episode summaries, not final circuits, for RL scenarios. |
| [Branch 5](chapter3-branch5-target-aware-cost-signals.md) | Deferred | Reward and cost signals that account for backend-aware targets | Should build on the more stable contracts and evaluation surface from branches 1-4. |

## Active branch-1 plan

The branch-1 technical plan is tracked in `docs/superpowers/plans/2026-04-21-chapter3-branch1-train-eval-contract.md`.

Use that plan as the implementation reference for the current branch. The other branch documents in this directory are future-iteration notes only and are not implemented here.
