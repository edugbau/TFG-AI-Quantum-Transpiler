from types import SimpleNamespace

from qiskit import QuantumCircuit

from src.integration.contracts import RoutingEpisodeSummary
from src.integration.post_routing_selector import PostRoutingCheckpointSelector


class DummyModel:
    def __init__(self) -> None:
        self.saved_paths = []

    def save(self, path: str) -> None:
        self.saved_paths.append(path)


def _selector(tmp_path, *, min_evals=50, max_no_improvement_evals=20):
    selector = PostRoutingCheckpointSelector(
        circuit=QuantumCircuit(2),
        coupling_edges=[(0, 1)],
        backend=object(),
        backend_name="fake_backend",
        seed=42,
        initial_layout=[0, 1],
        frontier_mode="dag",
        max_steps=10,
        lookahead_window=4,
        masked=True,
        mask_semantics="frontier_restricted_edges.v4",
        routing_mask_config=None,
        run_model_dir=tmp_path,
        eval_freq=1,
        min_evals=min_evals,
        max_no_improvement_evals=max_no_improvement_evals,
    )
    selector.model = DummyModel()
    selector._record_scalars = lambda attempt: None
    return selector


def _incomplete_summary() -> RoutingEpisodeSummary:
    return RoutingEpisodeSummary(
        initial_layout=[0, 1],
        final_layout=[0, 1],
        steps_executed=10,
        total_reward=-100.0,
        completed=False,
        truncated=True,
        truncation_reason="stagnation",
        total_swaps=0,
        gates_executed_count=0,
    )


def _valid_summary(*, swaps=0) -> RoutingEpisodeSummary:
    return RoutingEpisodeSummary(
        initial_layout=[0, 1],
        final_layout=[0, 1],
        steps_executed=1,
        total_reward=100.0,
        completed=True,
        truncated=False,
        total_swaps=swaps,
        gates_executed_count=0,
        swap_trace=[(0, 1)] * swaps,
    )


def test_incomplete_attempts_before_first_solution_do_not_consume_patience(monkeypatch, tmp_path) -> None:
    selector = _selector(tmp_path, min_evals=2, max_no_improvement_evals=1)
    monkeypatch.setattr(
        "src.integration.post_routing_selector.evaluate_routing_episode",
        lambda **kwargs: _incomplete_summary(),
    )
    monkeypatch.setattr(
        "src.integration.post_routing_selector.build_routed_circuit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not reconstruct incomplete routing")),
    )

    for timestep in (5_000, 10_000, 15_000):
        selector.n_calls += 1
        selector.num_timesteps = timestep
        assert selector._on_step() is True

    assert selector.phase == "searching_first_solution"
    assert selector.no_improvement_evals == 0
    assert selector.model.saved_paths == []
    assert [attempt["status"] for attempt in selector.attempts] == ["incomplete_routing"] * 3


def test_incomplete_regression_after_first_solution_consumes_patience(monkeypatch, tmp_path) -> None:
    selector = _selector(tmp_path, min_evals=1, max_no_improvement_evals=1)
    summaries = iter([_valid_summary(), _incomplete_summary()])
    monkeypatch.setattr(
        "src.integration.post_routing_selector.evaluate_routing_episode",
        lambda **kwargs: next(summaries),
    )
    monkeypatch.setattr(
        "src.integration.post_routing_selector.build_routed_circuit",
        lambda **kwargs: (QuantumCircuit(2), [0, 1]),
    )
    monkeypatch.setattr(
        "src.integration.post_routing_selector.qiskit_interface.transpile_post_routing",
        lambda *args, **kwargs: SimpleNamespace(
            transpiled_metrics=SimpleNamespace(cnot_equivalent=4, depth=8),
        ),
    )

    selector.n_calls = 1
    selector.num_timesteps = 5_000
    assert selector._on_step() is True
    selector.n_calls = 2
    selector.num_timesteps = 10_000
    assert selector._on_step() is False

    assert selector.first_solution_timestep == 5_000
    assert selector.no_improvement_evals == 1
    assert selector.stop_reason == "post_routing_no_improvement"


def test_selector_uses_lexicographic_cnot_depth_swaps_score(monkeypatch, tmp_path) -> None:
    selector = _selector(tmp_path)
    monkeypatch.setattr(
        "src.integration.post_routing_selector.build_routed_circuit",
        lambda **kwargs: (QuantumCircuit(2), [0, 1]),
    )
    scores = iter([(5, 7), (4, 99), (4, 8)])
    monkeypatch.setattr(
        "src.integration.post_routing_selector.qiskit_interface.transpile_post_routing",
        lambda *args, **kwargs: SimpleNamespace(
            transpiled_metrics=SimpleNamespace(
                **dict(zip(("cnot_equivalent", "depth"), next(scores)))
            ),
        ),
    )

    for swaps in (1, 9, 2):
        attempt, _ = selector._evaluate_completed_routing(_valid_summary(swaps=swaps))
        selector.attempts.append(attempt)

    assert selector.best_score == (4.0, 8, 2)
    assert len(selector.model.saved_paths) == 3


def test_selector_finalize_without_solution_preserves_budget_exhaustion_state(tmp_path) -> None:
    selector = _selector(tmp_path)

    selector.finalize(actual_timesteps=500_000)
    summary = selector.to_summary(actual_timesteps=500_000)

    assert summary["has_valid_solution"] is False
    assert summary["phase"] == "searching_first_solution"
    assert summary["stop_reason"] == "training_budget_exhausted"
    assert summary["actual_timesteps"] == 500_000
