# Campaign Matrix Summary
## Campaign Matrix
Campaign ID: `test-matrix-ghz-3`
Final Matrix Status: `completed`
Seeds: `41, 42`
MO Selection Modes: `compromise, best_depth, best_cnot_count`
Parallel Workers: `2`

## Aggregate Comparison by MO Mode

### `trans_depth`
| MO Mode | Seeds | Comparable Cases | Failed | Incomplete | Cancelled | Baseline Mean | RL_Only Mean | MO_Only Mean | MO+RL Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compromise | 2 | 2 | 0 | 0 | 0 | 11.00 | 11.00 | 14.50 | 19.00 |
| best_depth | 2 | 2 | 0 | 0 | 0 | 11.00 | 11.00 | 13.50 | 16.50 |
| best_cnot_count | 2 | 2 | 0 | 0 | 0 | 11.00 | 11.00 | 11.00 | 11.00 |

### `trans_two_qubit_gates`
| MO Mode | Seeds | Comparable Cases | Failed | Incomplete | Cancelled | Baseline Mean | RL_Only Mean | MO_Only Mean | MO+RL Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compromise | 2 | 2 | 0 | 0 | 0 | 2.00 | 2.00 | 5.00 | 6.50 |
| best_depth | 2 | 2 | 0 | 0 | 0 | 2.00 | 2.00 | 5.00 | 5.00 |
| best_cnot_count | 2 | 2 | 0 | 0 | 0 | 2.00 | 2.00 | 2.00 | 2.00 |

### `trans_cnot_equivalent`
| MO Mode | Seeds | Comparable Cases | Failed | Incomplete | Cancelled | Baseline Mean | RL_Only Mean | MO_Only Mean | MO+RL Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compromise | 2 | 2 | 0 | 0 | 0 | 2.00 | 2.00 | 5.00 | 6.50 |
| best_depth | 2 | 2 | 0 | 0 | 0 | 2.00 | 2.00 | 5.00 | 5.00 |
| best_cnot_count | 2 | 2 | 0 | 0 | 0 | 2.00 | 2.00 | 2.00 | 2.00 |

### `elapsed_time_s`
| MO Mode | Seeds | Comparable Cases | Failed | Incomplete | Cancelled | Baseline Mean | RL_Only Mean | MO_Only Mean | MO+RL Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compromise | 2 | 2 | 0 | 0 | 0 | 0.01 | 0.01 | 0.01 | 0.01 |
| best_depth | 2 | 2 | 0 | 0 | 0 | 0.01 | 0.01 | 0.01 | 0.01 |
| best_cnot_count | 2 | 2 | 0 | 0 | 0 | 0.01 | 0.01 | 0.01 | 0.10 |

## Child Campaigns
- `test-matrix-ghz-3__seed_41__compromise` seed=`41` mode=`compromise` status=`completed` summary=`campaigns\test-matrix-ghz-3\runs\test-matrix-ghz-3__seed_41__compromise\summary.md`
- `test-matrix-ghz-3__seed_41__best_depth` seed=`41` mode=`best_depth` status=`completed` summary=`campaigns\test-matrix-ghz-3\runs\test-matrix-ghz-3__seed_41__best_depth\summary.md`
- `test-matrix-ghz-3__seed_41__best_cnot_count` seed=`41` mode=`best_cnot_count` status=`completed` summary=`campaigns\test-matrix-ghz-3\runs\test-matrix-ghz-3__seed_41__best_cnot_count\summary.md`
- `test-matrix-ghz-3__seed_42__compromise` seed=`42` mode=`compromise` status=`completed` summary=`campaigns\test-matrix-ghz-3\runs\test-matrix-ghz-3__seed_42__compromise\summary.md`
- `test-matrix-ghz-3__seed_42__best_depth` seed=`42` mode=`best_depth` status=`completed` summary=`campaigns\test-matrix-ghz-3\runs\test-matrix-ghz-3__seed_42__best_depth\summary.md`
- `test-matrix-ghz-3__seed_42__best_cnot_count` seed=`42` mode=`best_cnot_count` status=`completed` summary=`campaigns\test-matrix-ghz-3\runs\test-matrix-ghz-3__seed_42__best_cnot_count\summary.md`

## Incidents
- None
