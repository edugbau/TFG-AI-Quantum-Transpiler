# Campaign Summary
## Campaign Metadata
Campaign ID: `test-matrix-ghz-3__seed_41__best_cnot_count`
Campaign Mode: `advanced`
Final Campaign Status: `completed`

## Global Configuration
RL Algorithm: `MaskablePPO`
RL Timesteps: `500`
RL Frontier Mode: `dag`
RL Lookahead Window: `4`
RL Max Steps: `80`
RL Learning Rate: `0.0001`
RL Clip Range: `0.1`
RL Target KL: `0.03`
RL Eval Episodes: `1`
Seed: `41`
Topology Source: `backend`
MO Effort Mode: `custom`
Layout Policy: `best_on_objective`
MO Quick: `True`
MO Population Size: `30`
MO Generations: `10`
MO Objective: `cnot_count`

## Selected Circuits
- ghz (3 qubits)

## Selected Backends
- fake_torino

## Aggregate Comparison
Comparable Completed Cases: `1`
Failed Cases: `0`
Incomplete Cases: `0`
Cancelled Cases: `0`

| Metric | Baseline Mean | MO_Only Mean | RL_Only Mean | MO+RL Mean |
| --- | ---: | ---: | ---: | ---: |
| trans_depth | 11.00 | 11.00 | 11.00 | 11.00 |
| trans_two_qubit_gates | 2.00 | 2.00 | 2.00 | 2.00 |
| trans_cnot_equivalent | 2.00 | 2.00 | 2.00 | 2.00 |
| elapsed_time_s | 0.01 | 0.01 | 0.02 | 0.02 |

## Per-Case Detail

## Case `ghz_3__fake_torino`
- Status: `completed`
- Circuit: `ghz` (3 qubits)
- Backend: `fake_torino`
- Effective Config: rl_algorithm=MaskablePPO, rl_total_timesteps=500, rl_frontier_mode=dag, rl_lookahead_window=4, rl_max_steps=80, rl_learning_rate=0.0001, rl_clip_range=0.1, rl_target_kl=0.03, rl_n_eval_episodes=1, seed=41
- Selected Layout: [110, 97, 96]
- Baseline: depth=11, two_qubit=2, cnot_equivalent=2, elapsed_time_s=0.013419700000667945
- MO_Only: depth=11, two_qubit=2, cnot_equivalent=2, elapsed_time_s=0.010100899999088142
- RL_Only: depth=11, two_qubit=2, cnot_equivalent=2, elapsed_time_s=0.016272199998638825
- MO+RL: depth=11, two_qubit=2, cnot_equivalent=2, elapsed_time_s=0.0180294000019785
- RL_Only Notes: RL_Only rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages. | Routing graph: path_expanded_subgraph with 3 nodes, 2 edges, 0 added intermediate qubits, 2 interacting pairs.
- MO+RL Notes: MO+RL rebuilds the routed circuit from the RL swap trace before running Qiskit post-routing stages. | Routing graph: path_expanded_subgraph with 3 nodes, 2 edges, 0 added intermediate qubits, 2 interacting pairs.
### RL_Only Training Summary
- Status: `completed`
- Algorithm: `MaskablePPO`
- Timesteps: `500`
- Frontier Mode: `dag`
- Lookahead Window: `4`
- Max Steps: `80`
- Learning Rate: `0.0001`
- Clip Range: `0.1`
- Target KL: `0.03`
- Eval Episodes: `1`
- Seed: `41`
- Selected Artifact: `runs/test-matrix-ghz-3__seed_41__best_cnot_count/cases/ghz_3__fake_torino/rl_only/training/models/rl_20260526_203528_012417/final_routing_MaskablePPO.zip`
### MO+RL Training Summary
- Status: `completed`
- Algorithm: `MaskablePPO`
- Timesteps: `500`
- Frontier Mode: `dag`
- Lookahead Window: `4`
- Max Steps: `80`
- Learning Rate: `0.0001`
- Clip Range: `0.1`
- Target KL: `0.03`
- Eval Episodes: `1`
- Seed: `41`
- Selected Artifact: `runs/test-matrix-ghz-3__seed_41__best_cnot_count/cases/ghz_3__fake_torino/training/models/rl_20260526_203534_089141/final_routing_MaskablePPO.zip`

## Incidents
- None
