from src.qiskit_interface import create_qft_circuit
from src.rl_module.environment import QuantumTranspilationEnv
from src.rl_module.routing_mask import RoutingMaskConfig


_SYNTHETIC_T_EDGES = [(0, 1), (1, 2), (2, 3), (2, 5), (3, 4), (5, 6), (6, 7)]
_QISKIT_INITIAL_LAYOUT = [7, 6, 4, 0, 3, 5, 1, 2]
_QISKIT_TRACE_PREFIX = [(1, 2), (1, 2), (2, 3), (1, 2), (0, 1), (2, 3), (2, 5)]
_PREPARATORY_EDGE = (0, 1)


def _replay_trace_prefix(mask_semantics: str) -> QuantumTranspilationEnv:
    env = QuantumTranspilationEnv(
        target_circuit=create_qft_circuit(8),
        coupling_map=_SYNTHETIC_T_EDGES,
        mode="routing",
        frontier_mode="dag",
        lookahead_window=10,
        max_steps=100,
        mask_semantics=mask_semantics,
        routing_mask_config=RoutingMaskConfig(sabre_top_k=3),
    )
    env.reset(seed=42, options={"initial_layout": _QISKIT_INITIAL_LAYOUT})
    for edge in _QISKIT_TRACE_PREFIX:
        env.step(env.strategy.edges.index(edge))
    return env


def test_v4_rejects_campaign_qiskit_preparatory_edge() -> None:
    env = _replay_trace_prefix("frontier_restricted_edges.v4")

    assert not env.action_masks()[env.strategy.edges.index(_PREPARATORY_EDGE)]


def test_v5_keeps_campaign_qiskit_preparatory_edge_after_top_k() -> None:
    env = _replay_trace_prefix("frontier_restricted_edges.v5")

    assert env.action_masks()[env.strategy.edges.index(_PREPARATORY_EDGE)]
