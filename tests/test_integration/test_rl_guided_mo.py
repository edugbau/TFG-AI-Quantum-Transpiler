import json
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit

from src.integration.rl_guided_mo import (
    RL_GUIDED_MO_INVALID_SCORE,
    RLGuidedMoAttempt,
    RLGuidedFitnessEvaluator,
    _select_valid_compromise,
    optimize_rl_guided_layouts,
)


def _contract():
    return SimpleNamespace(
        masked=True,
        algorithm="MaskablePPO",
        frontier_mode="dag",
        max_steps=100,
        lookahead_window=10,
        mask_semantics="v2",
        routing_mask_config=None,
    )


def test_rl_guided_fitness_caches_post_routing_score(monkeypatch) -> None:
    calls = []
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)

    def fake_evaluate_routing_episode(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            completed=True,
            truncated=False,
            truncation_reason=None,
            termination_reason="completed",
            total_swaps=0,
            swap_trace=[],
            executed_gate_trace=[("cx", 0, 1)],
        )

    monkeypatch.setattr("src.integration.rl_guided_mo.evaluate_routing_episode", fake_evaluate_routing_episode)
    monkeypatch.setattr(
        "src.integration.rl_guided_mo.build_routed_circuit",
        lambda **kwargs: (circuit, [0, 1]),
    )
    monkeypatch.setattr(
        "src.integration.rl_guided_mo.qiskit_interface.transpile_post_routing",
        lambda *args, **kwargs: SimpleNamespace(
            transpiled_metrics=SimpleNamespace(depth=4, cnot_equivalent=6.5)
        ),
    )
    evaluator = RLGuidedFitnessEvaluator(
        circuit=circuit,
        backend_bundle=SimpleNamespace(backend=object(), backend_name="synthetic_ring_2q"),
        coupling_edges=[(0, 1), (1, 0)],
        model_path="checkpoint.zip",
        seed=7,
        agent=object(),
        contract=_contract(),
    )

    assert evaluator.evaluate([0, 1]).tolist() == [4.0, 6.5]
    assert evaluator.evaluate([0, 1]).tolist() == [4.0, 6.5]
    assert len(calls) == 1
    assert calls[0]["coupling_edges"] == [(0, 1), (1, 0)]
    assert evaluator.cache_stats == {"hits": 1, "misses": 1, "size": 1}


def test_rl_guided_fitness_penalizes_incomplete_episode(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.integration.rl_guided_mo.evaluate_routing_episode",
        lambda **kwargs: SimpleNamespace(
            completed=False,
            truncated=True,
            truncation_reason="max_steps",
            termination_reason=None,
            total_swaps=9,
        ),
    )
    evaluator = RLGuidedFitnessEvaluator(
        circuit=QuantumCircuit(2),
        backend_bundle=SimpleNamespace(backend=object(), backend_name="synthetic_ring_2q"),
        coupling_edges=[(0, 1), (1, 0)],
        model_path="checkpoint.zip",
        seed=7,
        agent=object(),
        contract=_contract(),
    )

    assert evaluator.evaluate([0, 1]).tolist() == list(RL_GUIDED_MO_INVALID_SCORE)
    assert evaluator.attempts[0].status == "incomplete_routing"


def test_rl_guided_fitness_reports_progress_for_unique_layouts(monkeypatch) -> None:
    progress_messages = []
    monkeypatch.setattr(
        "src.integration.rl_guided_mo.evaluate_routing_episode",
        lambda **kwargs: SimpleNamespace(
            completed=False,
            truncated=True,
            truncation_reason="max_steps",
            termination_reason=None,
            total_swaps=9,
        ),
    )
    evaluator = RLGuidedFitnessEvaluator(
        circuit=QuantumCircuit(2),
        backend_bundle=SimpleNamespace(backend=object(), backend_name="synthetic_ring_2q"),
        coupling_edges=[(0, 1), (1, 0)],
        model_path="checkpoint.zip",
        seed=7,
        agent=object(),
        contract=_contract(),
        progress_callback=progress_messages.append,
        progress_interval=2,
    )

    evaluator.evaluate([0, 1])
    evaluator.evaluate([0, 1])
    evaluator.evaluate([1, 0])

    assert progress_messages == [
        "RL-guided MO progress: evaluated=1, cache_hits=0, valid=0, incomplete=1, errors=0",
        "RL-guided MO progress: evaluated=2, cache_hits=1, valid=0, incomplete=2, errors=0",
    ]


def test_rl_guided_selection_fails_when_no_candidate_completed() -> None:
    mo_result = SimpleNamespace(
        pareto_layouts=[[0, 1], [1, 0]],
        pareto_fitness=np.asarray([[1e12, 1e12], [1e12, 1e12]]),
    )
    evaluator = SimpleNamespace(
        evaluate_attempt=lambda layout: SimpleNamespace(score=None),
    )

    try:
        _select_valid_compromise(mo_result, evaluator)
    except ValueError as exc:
        assert "did not produce any completed routing candidate" in str(exc)
    else:
        raise AssertionError("Expected controlled failure when every RL-guided candidate is invalid")


def test_optimize_rl_guided_layouts_persists_valid_front_cache_and_controls(monkeypatch, tmp_path) -> None:
    progress_messages = []
    valid = RLGuidedMoAttempt(
        layout=[0, 1],
        status="valid_candidate",
        completed=True,
        truncated=False,
        truncation_reason=None,
        termination_reason="completed",
        total_swaps=0,
        score=[4.0, 6.0],
    )
    invalid = RLGuidedMoAttempt(
        layout=[1, 0],
        status="incomplete_routing",
        completed=False,
        truncated=True,
        truncation_reason="max_steps",
        termination_reason=None,
        total_swaps=3,
        score=None,
    )

    class FakeEvaluator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.attempts = [valid, invalid]
            self.cache_stats = {"hits": 2, "misses": 2, "size": 2}

        def evaluate_attempt(self, layout):
            return valid if list(layout) == [0, 1] else invalid

    monkeypatch.setattr("src.integration.rl_guided_mo.RLGuidedFitnessEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        "src.integration.rl_guided_mo.mo_module.optimize_layout",
        lambda **kwargs: SimpleNamespace(
            pareto_layouts=[[0, 1], [1, 0]],
            pareto_fitness=np.asarray([[4.0, 6.0], [1e12, 1e12]]),
        ),
    )
    artifact_path = tmp_path / "rl_guided_mo.json"

    result = optimize_rl_guided_layouts(
        circuit=QuantumCircuit(2),
        backend_bundle=SimpleNamespace(backend=object(), backend_name="synthetic_ring_2q"),
        coupling_edges=[(0, 1), (1, 0)],
        model_path="pretrained.zip",
        qiskit_initial_layout=[0, 1],
        mo_only_layout=[1, 0],
        population_size=4,
        n_generations=2,
        seed=7,
        artifact_path=artifact_path,
        progress_callback=progress_messages.append,
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert result.selected_layout == [0, 1]
    assert payload["checkpoint_source"] == "pretrained.zip"
    assert payload["pareto_layouts"] == [[0, 1]]
    assert payload["valid_candidate_count"] == 1
    assert payload["invalid_candidate_count"] == 1
    assert payload["cache_stats"] == {"hits": 2, "misses": 2, "size": 2}
    assert set(payload["controls"]) == {
        "qiskit_initial_layout",
        "mo_only_layout",
        "rl_guided_layout",
    }
    assert progress_messages == [
        "RL-guided MO started: population_size=4, n_generations=2, max_candidate_evaluations=8",
        (
            "RL-guided MO completed: selected_layout=[0, 1], valid_candidates=1, "
            "invalid_candidates=1, cache={'hits': 2, 'misses': 2, 'size': 2}"
        ),
    ]
