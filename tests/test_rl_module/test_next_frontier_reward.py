from src.integration.synthetic_topology import SyntheticTopologySpec
from src.qiskit_interface import create_ghz_circuit
from src.rl_module.environment import QuantumTranspilationEnv
from src.rl_module.routing_mask import RoutingMaskConfig


def _build_campaign_env() -> QuantumTranspilationEnv:
    topology = SyntheticTopologySpec(shape="t", num_qubits=11)
    env = QuantumTranspilationEnv(
        target_circuit=create_ghz_circuit(11),
        coupling_map=list(topology.build_coupling_map().get_edges()),
        mode="routing",
        frontier_mode="dag",
        lookahead_window=10,
        max_steps=20,
        mask_semantics="frontier_restricted_edges.v5",
        routing_mask_config=RoutingMaskConfig(sabre_top_k=4),
    )
    env.reset(seed=42, options={"initial_layout": [0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10]})
    env.step(env.strategy.edges.index((4, 5)))
    return env


def test_new_frontier_shaping_prefers_campaign_shorter_route() -> None:
    better_env = _build_campaign_env()
    _, better_reward, _, _, better_info = better_env.step(
        better_env.strategy.edges.index((5, 6))
    )

    worse_env = _build_campaign_env()
    _, worse_reward, _, _, worse_info = worse_env.step(
        worse_env.strategy.edges.index((4, 5))
    )

    assert better_info["gates_executed"] == 1
    assert worse_info["gates_executed"] == 1
    assert better_info["next_frontier_routing_signal"] == 1.0
    assert worse_info["next_frontier_routing_signal"] == 2.0
    assert better_reward > worse_reward
