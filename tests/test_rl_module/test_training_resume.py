import json
from pathlib import Path

from qiskit import QuantumCircuit


def test_setup_training_pipeline_loads_checkpoint_and_keeps_local_timestep_budget(monkeypatch, tmp_path) -> None:
    from src.rl_module import training

    captured = {}

    class DummyEnv:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed=None, options=None):
            return {}, {}

    class DummyAgent:
        def __init__(self):
            self.model = type("Model", (), {"num_timesteps": 100, "tensorboard_log": None})()

        @classmethod
        def load(cls, path, *, env, algorithm):
            captured["load"] = (path, env, algorithm)
            return cls()

        def train(self, total_timesteps, callbacks=None, progress_bar=True, reset_num_timesteps=True):
            captured["train"] = (total_timesteps, reset_num_timesteps)
            self.model.num_timesteps += total_timesteps

        def save(self, path):
            Path(path).write_text("model", encoding="utf-8")

    monkeypatch.setattr(training, "QuantumTranspilationEnv", DummyEnv)
    monkeypatch.setattr(training, "Monitor", lambda env: env)
    monkeypatch.setattr(training, "QuantumRLAgent", DummyAgent)
    monkeypatch.setattr(training, "CheckpointCallback", lambda **kwargs: object())
    monkeypatch.setattr(training, "EvalCallback", lambda *args, **kwargs: object())
    monkeypatch.setattr(training, "MaskableEvalCallback", lambda *args, **kwargs: object())

    source_model = tmp_path / "pretraining" / "best_model.zip"
    agent = training.setup_training_pipeline(
        target_circuit=QuantumCircuit(2),
        coupling_map=[(0, 1), (1, 0)],
        algorithm="MaskablePPO",
        total_timesteps=7,
        initial_model_path=str(source_model),
        log_dir=str(tmp_path / "logs"),
        model_save_dir=str(tmp_path / "models"),
    )

    assert captured["load"][0] == str(source_model)
    assert captured["load"][2] == "MaskablePPO"
    assert captured["train"] == (7, False)
    assert agent.actual_timesteps == 7
    metadata = json.loads(Path(agent.run_model_dir, "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_resume"] == {
        "initial_model_path": str(source_model),
        "reset_num_timesteps": False,
    }
    assert metadata["training_runtime"]["actual_timesteps"] == 7
