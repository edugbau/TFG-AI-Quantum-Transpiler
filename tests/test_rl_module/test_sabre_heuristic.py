import numpy as np
from qiskit import QuantumCircuit

from src.rl_module.environment import QuantumTranspilationEnv
from src.rl_module.routing_mask import FRONTIER_RESTRICTED_EDGES_V4, RoutingMaskConfig


def _env() -> QuantumTranspilationEnv:
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    env = QuantumTranspilationEnv(
        target_circuit=circuit,
        coupling_map=[(0, 1), (1, 2)],
        mode="routing",
        frontier_mode="dag",
        mask_semantics=FRONTIER_RESTRICTED_EDGES_V4,
        routing_mask_config=RoutingMaskConfig(),
    )
    env.reset(options={"initial_layout": [0, 1, 2]})
    env.action_masks = lambda: np.array([True, True], dtype=bool)
    return env


def test_sabre_heuristic_prioritizes_actions_that_unlock_frontier() -> None:
    env = _env()
    env._candidate_unlocks_frontier = lambda layout: tuple(layout) == (0, 2, 1)
    env._score_sabre_candidate = lambda layout, **kwargs: 0.0 if tuple(layout) == (1, 0, 2) else 99.0

    assert env.select_sabre_heuristic_action() == 1


def test_sabre_heuristic_breaks_equal_scores_by_action_index() -> None:
    env = _env()
    env._candidate_unlocks_frontier = lambda layout: False
    env._score_sabre_candidate = lambda layout, **kwargs: 5.0

    assert env.select_sabre_heuristic_action() == 0
