from types import SimpleNamespace

from qiskit import QuantumCircuit

from src.integration.contracts import RoutingEpisodeSummary


class FakeMoResult:
    pareto_layouts = [[0, 1, 2], [2, 1, 0], [0, 1, 2]]

    def get_compromise_layout(self):
        return [2, 1, 0]


def _completed_summary(layout, *, swaps):
    return RoutingEpisodeSummary(
        initial_layout=list(layout),
        final_layout=list(layout),
        steps_executed=swaps,
        total_reward=0.0,
        completed=True,
        truncated=False,
        total_swaps=swaps,
        gates_executed_count=1,
        swap_trace=[(0, 1)] * swaps,
    )


def _incomplete_summary(layout):
    return RoutingEpisodeSummary(
        initial_layout=list(layout),
        final_layout=list(layout),
        steps_executed=5,
        total_reward=-10.0,
        completed=False,
        truncated=True,
        truncation_reason="max_steps",
        total_swaps=0,
        gates_executed_count=0,
    )


def test_hybrid_probe_deduplicates_candidates_ranks_lexicographically_and_excludes_control(monkeypatch) -> None:
    from src.integration import hybrid_layout_probe

    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    score_by_layout = {
        (0, 1, 2): (8.0, 20),
        (2, 1, 0): (7.0, 99),
        (1, 0, 2): (1.0, 1),
    }

    monkeypatch.setattr(
        hybrid_layout_probe,
        "evaluate_routing_episode",
        lambda **kwargs: _completed_summary(kwargs["initial_layout"], swaps=sum(kwargs["initial_layout"]) + 1),
    )
    monkeypatch.setattr(
        hybrid_layout_probe,
        "build_routed_circuit",
        lambda **kwargs: (QuantumCircuit(3), list(kwargs["initial_layout"])),
    )
    monkeypatch.setattr(
        hybrid_layout_probe.qiskit_interface,
        "transpile_post_routing",
        lambda routed_circuit, *, initial_layout, **kwargs: SimpleNamespace(
            transpiled_metrics=SimpleNamespace(
                cnot_equivalent=score_by_layout[tuple(initial_layout)][0],
                depth=score_by_layout[tuple(initial_layout)][1],
            )
        ),
    )

    result = hybrid_layout_probe.select_hybrid_probe_layout(
        circuit=circuit,
        mo_result=FakeMoResult(),
        backend_bundle=SimpleNamespace(
            backend=object(),
            backend_name="fake_backend",
            coupling_edges=[(0, 1), (1, 2)],
        ),
        qiskit_initial_layout=[1, 0, 2],
        seed=42,
        frontier_mode="dag",
        max_steps=20,
        lookahead_window=4,
    )

    assert len(result.candidates) == 2
    assert result.selected_layout == [2, 1, 0]
    assert result.selected_score == [7.0, 99, 4]
    assert result.qiskit_control.score == [1.0, 1, 4]
    assert result.valid_candidate_count == 2
    assert result.fallback_reason is None


def test_hybrid_probe_falls_back_to_compromise_when_no_mo_candidate_completes(monkeypatch, tmp_path) -> None:
    from src.integration import hybrid_layout_probe

    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    monkeypatch.setattr(
        hybrid_layout_probe,
        "evaluate_routing_episode",
        lambda **kwargs: _incomplete_summary(kwargs["initial_layout"]),
    )

    artifact_path = tmp_path / "hybrid_layout_probe.json"
    result = hybrid_layout_probe.select_hybrid_probe_layout(
        circuit=circuit,
        mo_result=FakeMoResult(),
        backend_bundle=SimpleNamespace(
            backend=object(),
            backend_name="fake_backend",
            coupling_edges=[(0, 1), (1, 2)],
        ),
        qiskit_initial_layout=[1, 0, 2],
        seed=42,
        frontier_mode="dag",
        max_steps=5,
        lookahead_window=4,
        artifact_path=artifact_path,
    )

    assert result.selected_layout == [2, 1, 0]
    assert result.selected_score is None
    assert result.valid_candidate_count == 0
    assert result.fallback_reason == "no_completed_mo_probe_candidates"
    assert artifact_path.exists()
