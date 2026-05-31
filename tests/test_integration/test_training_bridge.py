from pathlib import Path

from qiskit import QuantumCircuit

from src.integration.campaign_contracts import CampaignCase, CampaignCircuitSpec, CampaignConfig
from src.integration.contracts import LayoutSelectionPolicy
from src.integration.training_bridge import train_case


def _build_campaign_config() -> CampaignConfig:
    return CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        rl_cycle_window=6,
        rl_stagnation_patience=11,
        rl_sabre_top_k=3,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mo_effort_mode="custom",
    )


def _build_campaign_case() -> CampaignCase:
    return CampaignCase(
        case_id="ghz_3__fake_torino",
        circuit_family="ghz",
        num_qubits=3,
        backend_name="fake_torino",
    )


def test_train_case_returns_best_model_when_available(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    expected_log_base_dir = case_output_dir / "training" / "logs"
    expected_model_base_dir = case_output_dir / "training" / "models"
    actual_run_log_dir = expected_log_base_dir / "rl_20260504_120000"
    actual_run_model_dir = expected_model_base_dir / "rl_20260504_120000"
    best_model_path = actual_run_model_dir / "best_model.zip"
    final_model_path = actual_run_model_dir / "final_routing_MaskablePPO.zip"
    captured_kwargs = {}

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = str(actual_run_model_dir)
            self.run_log_dir = str(actual_run_log_dir)
            self.best_model_path = str(best_model_path)
            self.last_model_path = str(final_model_path)

    def fake_setup_training_pipeline(**kwargs):
        captured_kwargs.update(kwargs)
        actual_run_log_dir.mkdir(parents=True)
        actual_run_model_dir.mkdir(parents=True)
        best_model_path.write_bytes(b"best")
        final_model_path.write_bytes(b"final")
        return DummyAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert captured_kwargs["algorithm"] == "MaskablePPO"
    assert captured_kwargs["total_timesteps"] == 5000
    assert captured_kwargs["frontier_mode"] == "dag"
    assert captured_kwargs["lookahead_window"] == 12
    assert captured_kwargs["max_steps"] == 256
    assert captured_kwargs["seed"] == 42
    assert captured_kwargs["hyperparams"] == {
        "learning_rate": 1e-4,
        "clip_range": 0.1,
        "target_kl": 0.03,
    }
    assert captured_kwargs["n_eval_episodes"] == 1
    assert captured_kwargs["routing_mask_config"].cycle_window == 6
    assert captured_kwargs["routing_mask_config"].stagnation_patience == 11
    assert captured_kwargs["routing_mask_config"].sabre_top_k == 3
    assert Path(captured_kwargs["log_dir"]) == expected_log_base_dir
    assert Path(captured_kwargs["model_save_dir"]) == expected_model_base_dir
    assert result.status == "completed"
    assert result.selected_artifact_path == best_model_path
    assert result.best_model_path == best_model_path
    assert result.final_model_path == final_model_path
    assert result.run_model_dir == actual_run_model_dir
    assert result.run_log_dir == actual_run_log_dir
    assert result.effective_training_config.algorithm == "MaskablePPO"
    assert result.effective_training_config.total_timesteps == 5000
    assert result.effective_training_config.frontier_mode == "dag"
    assert result.effective_training_config.lookahead_window == 12
    assert result.effective_training_config.max_steps == 256
    assert result.effective_training_config.seed == 42
    assert result.effective_training_config.learning_rate == 1e-4
    assert result.effective_training_config.clip_range == 0.1
    assert result.effective_training_config.target_kl == 0.03
    assert result.effective_training_config.n_eval_episodes == 1
    assert result.effective_training_config.cycle_window == 6
    assert result.effective_training_config.stagnation_patience == 11
    assert result.effective_training_config.sabre_top_k == 3


def test_train_case_forwards_initial_layout_to_setup_training_pipeline(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    captured_kwargs = {}

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = case_output_dir / "training" / "models" / "run-001"
            self.run_log_dir = case_output_dir / "training" / "logs" / "run-001"
            self.best_model_path = self.run_model_dir / "best_model.zip"
            self.last_model_path = self.run_model_dir / "final_model.zip"

    def fake_setup_training_pipeline(**kwargs):
        captured_kwargs.update(kwargs)
        agent = DummyAgent()
        agent.run_model_dir.mkdir(parents=True)
        agent.run_log_dir.mkdir(parents=True)
        agent.best_model_path.write_bytes(b"best")
        agent.last_model_path.write_bytes(b"final")
        return agent

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
        initial_layout=[2, 1, 0],
    )

    assert captured_kwargs["initial_layout"] == [2, 1, 0]
    assert result.status == "completed"


def test_train_case_falls_back_to_final_model_when_best_model_is_missing(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    actual_run_log_dir = case_output_dir / "training" / "logs" / "rl_20260504_120000"
    actual_run_model_dir = case_output_dir / "training" / "models" / "rl_20260504_120000"
    final_model_path = actual_run_model_dir / "final_routing_MaskablePPO.zip"

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = str(actual_run_model_dir)
            self.run_log_dir = str(actual_run_log_dir)
            self.best_model_path = None
            self.last_model_path = str(final_model_path)

    def fake_setup_training_pipeline(**kwargs):
        actual_run_log_dir.mkdir(parents=True)
        actual_run_model_dir.mkdir(parents=True)
        final_model_path.write_bytes(b"final")
        return DummyAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert result.status == "completed"
    assert result.selected_artifact_path == final_model_path
    assert result.best_model_path is None
    assert result.final_model_path == final_model_path
    assert result.run_model_dir == actual_run_model_dir
    assert result.run_log_dir == actual_run_log_dir


def test_train_case_forwards_none_initial_layout_when_omitted(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    captured_kwargs = {}

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = case_output_dir / "training" / "models" / "run-001"
            self.run_log_dir = case_output_dir / "training" / "logs" / "run-001"
            self.best_model_path = self.run_model_dir / "best_model.zip"
            self.last_model_path = self.run_model_dir / "final_model.zip"

    def fake_setup_training_pipeline(**kwargs):
        captured_kwargs.update(kwargs)
        agent = DummyAgent()
        agent.run_model_dir.mkdir(parents=True)
        agent.run_log_dir.mkdir(parents=True)
        agent.best_model_path.write_bytes(b"best")
        agent.last_model_path.write_bytes(b"final")
        return agent

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert "initial_layout" in captured_kwargs
    assert captured_kwargs["initial_layout"] is None
    assert result.status == "completed"


def test_train_case_surfaces_training_failure_with_paths_and_status(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id

    def fake_setup_training_pipeline(**kwargs):
        assert Path(kwargs["log_dir"]) == case_output_dir / "training" / "logs"
        assert Path(kwargs["model_save_dir"]) == case_output_dir / "training" / "models"
        raise RuntimeError("training exploded")

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert result.status == "failed"
    assert result.selected_artifact_path is None
    assert result.best_model_path is None
    assert result.final_model_path is None
    assert result.run_model_dir == case_output_dir / "training" / "models"
    assert result.run_log_dir == case_output_dir / "training" / "logs"
    assert result.effective_training_config.algorithm == "MaskablePPO"
    assert result.effective_training_config.total_timesteps == 5000
    assert result.effective_training_config.frontier_mode == "dag"
    assert result.effective_training_config.lookahead_window == 12
    assert result.effective_training_config.max_steps == 256
    assert result.effective_training_config.seed == 42


def test_train_case_fails_when_training_finishes_without_any_usable_artifact(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    actual_run_log_dir = case_output_dir / "training" / "logs" / "rl_20260504_120000"
    actual_run_model_dir = case_output_dir / "training" / "models" / "rl_20260504_120000"

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = str(actual_run_model_dir)
            self.run_log_dir = str(actual_run_log_dir)
            self.best_model_path = None
            self.last_model_path = None

    def fake_setup_training_pipeline(**kwargs):
        actual_run_log_dir.mkdir(parents=True)
        actual_run_model_dir.mkdir(parents=True)
        return DummyAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert result.status == "failed"
    assert result.selected_artifact_path is None
    assert result.best_model_path is None
    assert result.final_model_path is None
    assert result.run_model_dir == actual_run_model_dir
    assert result.run_log_dir == actual_run_log_dir


def test_train_case_clears_stale_internal_artifact_paths_when_files_do_not_exist(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    actual_run_log_dir = case_output_dir / "training" / "logs" / "rl_20260504_120000"
    actual_run_model_dir = case_output_dir / "training" / "models" / "rl_20260504_120000"
    stale_best_model_path = actual_run_model_dir / "best_model.zip"
    stale_final_model_path = actual_run_model_dir / "final_routing_MaskablePPO.zip"

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = str(actual_run_model_dir)
            self.run_log_dir = str(actual_run_log_dir)
            self.best_model_path = str(stale_best_model_path)
            self.last_model_path = str(stale_final_model_path)

    def fake_setup_training_pipeline(**kwargs):
        actual_run_log_dir.mkdir(parents=True)
        actual_run_model_dir.mkdir(parents=True)
        return DummyAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert result.status == "failed"
    assert result.selected_artifact_path is None
    assert result.best_model_path is None
    assert result.final_model_path is None
    assert result.run_model_dir == actual_run_model_dir
    assert result.run_log_dir == actual_run_log_dir


def test_train_case_ignores_blank_agent_paths_instead_of_treating_them_as_workspace(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    expected_log_base_dir = case_output_dir / "training" / "logs"
    expected_model_base_dir = case_output_dir / "training" / "models"

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = ""
            self.run_log_dir = ""
            self.best_model_path = ""
            self.last_model_path = ""

    def fake_setup_training_pipeline(**kwargs):
        return DummyAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert result.status == "failed"
    assert result.selected_artifact_path is None
    assert result.best_model_path is None
    assert result.final_model_path is None
    assert result.run_model_dir == expected_model_base_dir
    assert result.run_log_dir == expected_log_base_dir


def test_train_case_rejects_artifacts_outside_the_selected_run_model_dir(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    actual_run_log_dir = case_output_dir / "training" / "logs" / "rl_20260504_120000"
    actual_run_model_dir = case_output_dir / "training" / "models" / "rl_20260504_120000"
    foreign_best_model_path = tmp_path / "other_run" / "best_model.zip"

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = str(actual_run_model_dir)
            self.run_log_dir = str(actual_run_log_dir)
            self.best_model_path = str(foreign_best_model_path)
            self.last_model_path = None

    def fake_setup_training_pipeline(**kwargs):
        actual_run_log_dir.mkdir(parents=True)
        actual_run_model_dir.mkdir(parents=True)
        foreign_best_model_path.parent.mkdir(parents=True)
        foreign_best_model_path.write_bytes(b"foreign")
        return DummyAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert result.status == "failed"
    assert result.selected_artifact_path is None
    assert result.best_model_path is None
    assert result.final_model_path is None
    assert result.run_model_dir == actual_run_model_dir
    assert result.run_log_dir == actual_run_log_dir


def test_train_case_rejects_foreign_run_model_dir_even_if_artifact_lives_under_it(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    expected_log_base_dir = case_output_dir / "training" / "logs"
    expected_model_base_dir = case_output_dir / "training" / "models"
    foreign_run_log_dir = tmp_path / "other_case" / "training" / "logs" / "rl_20260504_120000"
    foreign_run_model_dir = tmp_path / "other_case" / "training" / "models" / "rl_20260504_120000"
    foreign_best_model_path = foreign_run_model_dir / "best_model.zip"

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = str(foreign_run_model_dir)
            self.run_log_dir = str(foreign_run_log_dir)
            self.best_model_path = str(foreign_best_model_path)
            self.last_model_path = None

    def fake_setup_training_pipeline(**kwargs):
        foreign_run_log_dir.mkdir(parents=True)
        foreign_run_model_dir.mkdir(parents=True)
        foreign_best_model_path.write_bytes(b"foreign")
        return DummyAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert result.status == "failed"
    assert result.selected_artifact_path is None
    assert result.best_model_path is None
    assert result.final_model_path is None
    assert result.run_model_dir == expected_model_base_dir
    assert result.run_log_dir == expected_log_base_dir


def test_train_case_rejects_foreign_run_model_dir_outside_case_model_base_dir(monkeypatch, tmp_path) -> None:
    config = _build_campaign_config()
    case = _build_campaign_case()
    case_output_dir = tmp_path / "cases" / case.case_id
    expected_model_base_dir = case_output_dir / "training" / "models"
    expected_log_base_dir = case_output_dir / "training" / "logs"
    foreign_run_model_dir = tmp_path / "other_case" / "training" / "models" / "rl_20260504_120000"
    foreign_run_log_dir = tmp_path / "other_case" / "training" / "logs" / "rl_20260504_120000"
    foreign_best_model_path = foreign_run_model_dir / "best_model.zip"

    class DummyAgent:
        def __init__(self):
            self.run_model_dir = str(foreign_run_model_dir)
            self.run_log_dir = str(foreign_run_log_dir)
            self.best_model_path = str(foreign_best_model_path)
            self.last_model_path = None

    def fake_setup_training_pipeline(**kwargs):
        foreign_run_log_dir.mkdir(parents=True)
        foreign_run_model_dir.mkdir(parents=True)
        foreign_best_model_path.write_bytes(b"foreign")
        return DummyAgent()

    monkeypatch.setattr("src.integration.training_bridge.setup_training_pipeline", fake_setup_training_pipeline)

    result = train_case(
        campaign_case=case,
        campaign_config=config,
        target_circuit=QuantumCircuit(3),
        coupling_map=[(0, 1), (1, 2)],
        case_output_dir=case_output_dir,
    )

    assert result.status == "failed"
    assert result.selected_artifact_path is None
    assert result.best_model_path is None
    assert result.final_model_path is None
    assert result.run_model_dir == expected_model_base_dir
    assert result.run_log_dir == expected_log_base_dir
