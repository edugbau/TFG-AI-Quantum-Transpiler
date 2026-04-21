from src.integration.contracts import RoutingEpisodeSummary


def _create_routing_env(*, circuit, coupling_edges, frontier_mode, max_steps, lookahead_window):
    from src.rl_module.environment import QuantumTranspilationEnv

    return QuantumTranspilationEnv(
        target_circuit=circuit,
        coupling_map=coupling_edges,
        mode="routing",
        frontier_mode=frontier_mode,
        max_steps=max_steps,
        lookahead_window=lookahead_window,
    )


def _normalize_layout(layout):
    if layout is None:
        return None
    return [int(entry) for entry in layout]


def _count_gates_executed_at_reset(env, info) -> int:
    total_gates = int(info.get("total_gates", 0))
    remaining_gates = getattr(env, "remaining_gates", None)

    if remaining_gates is not None:
        return max(total_gates - len(remaining_gates), 0)

    if info.get("already_completed_at_reset"):
        return total_gates

    return 0


def evaluate_routing_episode(
    *,
    circuit,
    coupling_edges,
    agent,
    seed,
    initial_layout,
    frontier_mode,
    max_steps,
    lookahead_window,
) -> RoutingEpisodeSummary:
    env = _create_routing_env(
        circuit=circuit,
        coupling_edges=coupling_edges,
        frontier_mode=frontier_mode,
        max_steps=max_steps,
        lookahead_window=lookahead_window,
    )

    try:
        reset_options = None
        if initial_layout is not None:
            reset_options = {"initial_layout": list(initial_layout)}

        obs, info = env.reset(seed=seed, options=reset_options)

        steps_executed = 0
        total_reward = 0.0
        total_gates_executed = _count_gates_executed_at_reset(env, info)
        terminated = bool(info.get("already_completed_at_reset", False))
        truncated = False

        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            steps_executed += 1
            total_reward += float(reward)
            total_gates_executed += int(step_info.get("gates_executed", 0))

        final_layout = _normalize_layout(env.current_layout)
        return RoutingEpisodeSummary(
            initial_layout=_normalize_layout(initial_layout),
            final_layout=final_layout,
            steps_executed=steps_executed,
            total_reward=total_reward,
            completed=bool(terminated),
            truncated=bool(truncated),
            total_swaps=int(env.total_swaps),
            gates_executed_count=total_gates_executed,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
