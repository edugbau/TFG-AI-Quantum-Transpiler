from __future__ import annotations

from collections import defaultdict, deque

from qiskit import QuantumCircuit
import numpy as np

from src.integration.contracts import RoutingEpisodeSummary
from src.rl_module.frontier import DagFrontier, GateTuple, SequentialFrontier


def _create_routing_env(
    *,
    circuit,
    coupling_edges,
    frontier_mode,
    max_steps,
    lookahead_window,
    mask_semantics=None,
):
    from src.rl_module.environment import QuantumTranspilationEnv

    return QuantumTranspilationEnv(
        target_circuit=circuit,
        coupling_map=coupling_edges,
        mode="routing",
        frontier_mode=frontier_mode,
        max_steps=max_steps,
        lookahead_window=lookahead_window,
        mask_semantics=mask_semantics,
    )


def _normalize_layout(layout):
    if layout is None:
        return None
    return [int(entry) for entry in layout]


def _normalize_swap_trace(swap_trace) -> list[tuple[int, int]]:
    return [tuple(int(entry) for entry in edge) for edge in swap_trace]


def _ensure_reconstructable_circuit(circuit) -> None:
    for instruction in circuit.data:
        if len(instruction.qubits) > 2:
            raise ValueError("build_routed_circuit does not support operations acting on >2 qubits")


def _build_frontier(circuit, frontier_mode):
    if frontier_mode == "dag":
        return DagFrontier.from_circuit(circuit)
    extracted_gates: list[GateTuple] = []
    for instruction in circuit.data:
        gate_name = instruction.operation.name
        qargs = [circuit.find_bit(q).index for q in instruction.qubits]
        if len(qargs) == 2:
            extracted_gates.append((gate_name, qargs[0], qargs[1]))
        elif len(qargs) == 1:
            extracted_gates.append((gate_name, qargs[0], qargs[0]))
    return SequentialFrontier(extracted_gates)


def _infer_num_physical_qubits(circuit, coupling_edges, initial_layout, swap_trace) -> int:
    candidates = [circuit.num_qubits]
    if initial_layout:
        candidates.append(max(initial_layout) + 1)
    if coupling_edges:
        candidates.append(max(max(int(left), int(right)) for left, right in coupling_edges) + 1)
    if swap_trace:
        candidates.append(max(max(int(left), int(right)) for left, right in swap_trace) + 1)
    return max(candidates)


def _apply_swap_to_layout(current_layout: list[int], swap_edge: tuple[int, int]) -> None:
    pq1, pq2 = swap_edge
    lq1 = current_layout.index(pq1) if pq1 in current_layout else None
    lq2 = current_layout.index(pq2) if pq2 in current_layout else None

    if lq1 is not None:
        current_layout[lq1] = pq2
    if lq2 is not None:
        current_layout[lq2] = pq1


def _is_connected(coupling_set: set[tuple[int, int]], physical_q1: int, physical_q2: int) -> bool:
    return (physical_q1, physical_q2) in coupling_set


def _append_gate_on_physical_layout(
    routed_circuit: QuantumCircuit,
    gate: GateTuple,
    current_layout: list[int],
    instruction_pools,
) -> None:
    gate_name, logical_q1, logical_q2 = gate
    pool = instruction_pools.get(gate)
    if not pool:
        raise ValueError(f"Unable to replay routed gate {gate_name!r} on logical qubits {(logical_q1, logical_q2)}")

    instruction = pool.popleft()
    physical_qargs = [routed_circuit.qubits[current_layout[index]] for index in instruction[1]]
    physical_cargs = [routed_circuit.clbits[index] for index in instruction[2]]
    routed_circuit.append(instruction[0].copy(), physical_qargs, physical_cargs)


def _build_instruction_pools(source_circuit: QuantumCircuit):
    pools = defaultdict(deque)
    for instruction in source_circuit.data:
        logical_qargs = [source_circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if len(logical_qargs) == 1:
            gate = (instruction.operation.name, logical_qargs[0], logical_qargs[0])
        elif len(logical_qargs) == 2:
            gate = (instruction.operation.name, logical_qargs[0], logical_qargs[1])
        else:
            continue
        logical_cargs = [source_circuit.find_bit(clbit).index for clbit in instruction.clbits]
        pools[gate].append((instruction.operation, logical_qargs, logical_cargs))
    return pools


def _append_executed_gates(
    routed_circuit: QuantumCircuit,
    executed_gates: list[GateTuple],
    current_layout: list[int],
    instruction_pools,
) -> None:
    for gate in executed_gates:
        _append_gate_on_physical_layout(routed_circuit, gate, current_layout, instruction_pools)


def _is_gate_executable_on_current_layout(
    gate: GateTuple,
    current_layout: list[int],
    coupling_set: set[tuple[int, int]],
) -> bool:
    _, logical_q1, logical_q2 = gate
    if logical_q1 == logical_q2:
        return True
    return _is_connected(coupling_set, current_layout[logical_q1], current_layout[logical_q2])


def _reconstruct_from_exact_gate_trace(
    *,
    routed_circuit: QuantumCircuit,
    current_layout: list[int],
    instruction_pools,
    coupling_set: set[tuple[int, int]],
    swap_trace: list[tuple[int, int]],
    executed_gate_trace: list[GateTuple],
) -> list[int]:
    swap_edges_remaining = list(swap_trace)
    for gate in executed_gate_trace:
        while not _is_gate_executable_on_current_layout(gate, current_layout, coupling_set):
            if not swap_edges_remaining:
                raise ValueError("executed_gate_trace requires more swaps than swap_trace provides")
            swap_edge = swap_edges_remaining.pop(0)
            routed_circuit.swap(*swap_edge)
            _apply_swap_to_layout(current_layout, swap_edge)

        _append_gate_on_physical_layout(routed_circuit, gate, current_layout, instruction_pools)

    for swap_edge in swap_edges_remaining:
        routed_circuit.swap(*swap_edge)
        _apply_swap_to_layout(current_layout, swap_edge)

    return current_layout


def build_routed_circuit(
    *,
    circuit,
    coupling_edges,
    initial_layout,
    swap_trace,
    frontier_mode,
    executed_gate_trace=None,
):
    _ensure_reconstructable_circuit(circuit)
    initial_layout = _normalize_layout(initial_layout)
    if initial_layout is None:
        raise ValueError("initial_layout is required to reconstruct a routed circuit")

    normalized_swap_trace = _normalize_swap_trace(swap_trace)
    current_layout = list(initial_layout)
    instruction_pools = _build_instruction_pools(circuit)
    coupling_set = {
        (int(left), int(right)) for left, right in coupling_edges
    } | {
        (int(right), int(left)) for left, right in coupling_edges
    }
    routed_circuit = QuantumCircuit(
        _infer_num_physical_qubits(circuit, coupling_edges, initial_layout, normalized_swap_trace),
        circuit.num_clbits,
        name=f"{circuit.name or 'circuit'}_mo_rl_routed",
    )
    routed_circuit.metadata = dict(getattr(circuit, "metadata", None) or {})

    if executed_gate_trace is not None:
        current_layout = _reconstruct_from_exact_gate_trace(
            routed_circuit=routed_circuit,
            current_layout=current_layout,
            instruction_pools=instruction_pools,
            coupling_set=coupling_set,
            swap_trace=normalized_swap_trace,
            executed_gate_trace=_normalize_executed_gates(executed_gate_trace),
        )
        return routed_circuit, list(current_layout)

    frontier = _build_frontier(circuit, frontier_mode)
    layout_array = np.array(current_layout, dtype=np.int32)

    executed_gates: list[GateTuple] = []
    frontier.execute_ready_cascade(
        current_layout=layout_array,
        is_connected=lambda left, right: _is_connected(coupling_set, left, right),
        cascade_successors=True,
        executed_gates=executed_gates,
    )
    _append_executed_gates(routed_circuit, executed_gates, current_layout, instruction_pools)

    for swap_edge in normalized_swap_trace:
        pq1, pq2 = swap_edge
        routed_circuit.swap(pq1, pq2)
        _apply_swap_to_layout(current_layout, swap_edge)
        layout_array = np.array(current_layout, dtype=np.int32)

        executed_gates = []
        frontier.execute_ready_cascade(
            current_layout=layout_array,
            is_connected=lambda left, right: _is_connected(coupling_set, left, right),
            cascade_successors=True,
            executed_gates=executed_gates,
        )
        _append_executed_gates(routed_circuit, executed_gates, current_layout, instruction_pools)

    if frontier.remaining_gate_count != 0:
        raise ValueError("swap_trace did not complete routing for the target circuit")

    return routed_circuit, list(current_layout)


def _count_gates_executed_at_reset(env, info) -> int:
    total_gates = int(info.get("total_gates", 0))
    remaining_gates = getattr(env, "remaining_gates", None)

    if remaining_gates is not None:
        return max(total_gates - len(remaining_gates), 0)

    if info.get("already_completed_at_reset"):
        return total_gates

    return 0


def _normalize_executed_gates(gates) -> list[tuple[str, int, int]]:
    return [
        (str(gate_name), int(logical_q1), int(logical_q2))
        for gate_name, logical_q1, logical_q2 in gates
    ]


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
    masked=False,
    mask_semantics=None,
) -> RoutingEpisodeSummary:
    env = _create_routing_env(
        circuit=circuit,
        coupling_edges=coupling_edges,
        frontier_mode=frontier_mode,
        max_steps=max_steps,
        lookahead_window=lookahead_window,
        mask_semantics=mask_semantics,
    )

    try:
        swap_trace: list[tuple[int, int]] = []
        reset_options = None
        if initial_layout is not None:
            reset_options = {"initial_layout": list(initial_layout)}

        obs, info = env.reset(seed=seed, options=reset_options)

        steps_executed = 0
        total_reward = 0.0
        total_gates_executed = _count_gates_executed_at_reset(env, info)
        executed_gate_trace = _normalize_executed_gates(info.get("executed_gates", []))
        terminated = bool(info.get("already_completed_at_reset", False))
        truncated = False

        while not (terminated or truncated):
            if masked:
                action, _ = agent.predict(
                    obs,
                    action_masks=env.action_masks(),
                    deterministic=True,
                )
            else:
                action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            steps_executed += 1
            total_reward += float(reward)
            total_gates_executed += int(step_info.get("gates_executed", 0))
            executed_gate_trace.extend(_normalize_executed_gates(step_info.get("executed_gates", [])))
            if step_info.get("action_type") == "swap" and step_info.get("is_valid_action"):
                swap_edge = step_info.get("swap_edge")
                if swap_edge is not None:
                    swap_trace.append(tuple(int(entry) for entry in swap_edge))

        final_layout = _normalize_layout(env.current_layout)
        return RoutingEpisodeSummary(
            initial_layout=_normalize_layout(initial_layout),
            final_layout=final_layout,
            steps_executed=steps_executed,
            total_reward=total_reward,
            completed=bool(terminated),
            truncated=bool(truncated),
            total_swaps=len(swap_trace),
            gates_executed_count=total_gates_executed,
            swap_trace=swap_trace,
            executed_gate_trace=executed_gate_trace,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
