from pathlib import Path
from types import SimpleNamespace

from src.integration.campaign_contracts import CampaignCircuitSpec
from src.integration.contracts import LayoutSelectionPolicy
from src.integration.mo_effort import MIN_CUSTOM_MO_POPULATION_SIZE


def _make_io(responses: list[str]) -> tuple[callable, list[str]]:
    remaining = list(responses)
    outputs: list[str] = []

    def input_fn(prompt: str = "") -> str:
        outputs.append(prompt)
        if not remaining:
            raise AssertionError("test input exhausted")
        return remaining.pop(0)

    def output_fn(message: str = "") -> None:
        outputs.append(str(message))

    return input_fn, outputs


def test_build_default_campaign_config_uses_canonical_defaults() -> None:
    from src.integration.campaign_cli import build_default_campaign_config

    config = build_default_campaign_config(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)]
    )

    assert config.circuit_specs == (CampaignCircuitSpec(family="ghz", num_qubits=3),)
    assert config.backend_names == ("fake_torino",)
    assert config.rl_algorithm == "MaskablePPO"
    assert config.rl_total_timesteps == 5000
    assert config.rl_frontier_mode == "dag"
    assert config.rl_lookahead_window == 10
    assert config.rl_max_steps == 200
    assert config.seed == 42
    assert config.mo_use_quick is True
    assert config.mo_population_size == 30
    assert config.mo_n_generations == 50
    assert config.mo_effort_mode == "auto"
    assert config.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert config.mo_objective_name is None
    assert config.mode == "default"


def test_run_interactive_campaign_cli_allows_multiple_backends_in_advanced_mode() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz,qft",
            "3,5",
            "advanced",
            "fake_torino,fake_brisbane",
            "MaskablePPO",
            "7000",
            "dag",
            "15",
            "300",
            "123",
            "custom",
            "false",
            "40",
            "60",
            "compromise",
            "y",
        ]
    )
    captured = {}

    def run_campaign_fn(campaign, *, output_root):
        captured["campaign"] = campaign
        captured["output_root"] = output_root
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-adv-001",
        output_root=Path("campaigns"),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert captured["campaign"].campaign_id == "campaign-adv-001"
    assert captured["campaign"].config.backend_names == (
        "fake_torino",
        "fake_brisbane",
    )
    assert captured["campaign"].config.mode == "advanced"
    assert captured["campaign"].config.rl_algorithm == "MaskablePPO"
    assert captured["campaign"].config.rl_total_timesteps == 7000
    assert captured["campaign"].config.rl_frontier_mode == "dag"
    assert captured["campaign"].config.rl_lookahead_window == 15
    assert captured["campaign"].config.rl_max_steps == 300
    assert captured["campaign"].config.seed == 123
    assert captured["campaign"].config.mo_effort_mode == "custom"
    assert captured["campaign"].config.mo_use_quick is False
    assert captured["campaign"].config.mo_population_size == 40
    assert captured["campaign"].config.mo_n_generations == 60
    assert captured["campaign"].config.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert "MO Effort Mode: custom" in rendered
    assert "MO Quick: False" in rendered
    assert "MO Population Size: 40" in rendered
    assert "MO Generations: 60" in rendered
    assert "MO Auto Preview" not in rendered


def test_run_interactive_campaign_cli_auto_effort_skips_manual_mo_knobs_and_prints_preview() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz,qft",
            "3,8",
            "advanced",
            "fake_torino",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "42",
            "auto",
            "compromise",
            "y",
        ]
    )
    captured = {}

    def run_campaign_fn(campaign, *, output_root):
        captured["campaign"] = campaign
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-auto-001",
        output_root=Path("campaigns"),
    )

    rendered = "\n".join(outputs)
    config = captured["campaign"].config

    assert exit_code == 0
    assert config.mo_effort_mode == "auto"
    assert config.mo_use_quick is True
    assert config.mo_population_size == 30
    assert config.mo_n_generations == 50
    assert "MO population size:" not in rendered
    assert "MO generations:" not in rendered
    assert "MO Effort Mode: auto" in rendered
    assert "MO Auto Preview (3q): quick=True, population_size=30, n_generations=50" in rendered
    assert "MO Auto Preview (8q): quick=False, population_size=60, n_generations=120" in rendered
    assert "MO Quick:" not in rendered
    assert "MO Population Size:" not in rendered
    assert "MO Generations:" not in rendered


def test_run_interactive_campaign_cli_collects_best_on_objective_objective_name(monkeypatch) -> None:
    from src.integration import campaign_cli

    monkeypatch.setattr(campaign_cli, "_available_objective_names", lambda: ("depth", "cnot_count"))
    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "advanced",
            "fake_torino",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "42",
            "custom",
            "true",
            "30",
            "50",
            "best_on_objective",
            "cnot_count",
            "y",
        ]
    )
    captured = {}

    def run_campaign_fn(campaign, *, output_root):
        captured["campaign"] = campaign
        return SimpleNamespace(campaign_status="completed")

    campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-adv-002",
        output_root=Path("campaigns"),
    )

    assert captured["campaign"].config.layout_policy is LayoutSelectionPolicy.BEST_ON_OBJECTIVE
    assert captured["campaign"].config.mo_objective_name == "cnot_count"


def test_run_interactive_campaign_cli_reprompts_on_invalid_input() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "default",
            "n",
        ]
    )
    run_calls: list[object] = []

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=lambda campaign, *, output_root: run_calls.append(campaign),
        campaign_id_factory=lambda: "campaign-default-001",
        output_root=Path("campaigns"),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert run_calls == []


def test_run_interactive_campaign_cli_reprompts_when_custom_mo_population_is_below_minimum() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "advanced",
            "fake_torino",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "42",
            "custom",
            "false",
            "3",
            str(MIN_CUSTOM_MO_POPULATION_SIZE),
            "60",
            "compromise",
            "y",
        ]
    )
    captured = {}

    def run_campaign_fn(campaign, *, output_root):
        captured["campaign"] = campaign
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-adv-003",
        output_root=Path("campaigns"),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert captured["campaign"].config.mo_population_size == MIN_CUSTOM_MO_POPULATION_SIZE
    assert f"Invalid selection. Enter an integer >= {MIN_CUSTOM_MO_POPULATION_SIZE}." in rendered


def test_run_interactive_campaign_cli_default_mode_uses_fake_torino_without_prompt() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "default",
            "y",
        ]
    )
    captured = {}

    def run_campaign_fn(campaign, *, output_root):
        captured["campaign"] = campaign
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-default-004",
        output_root=Path("campaigns"),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert captured["campaign"].config.backend_names == ("fake_torino",)
    assert "Choose backend" not in rendered


def test_run_interactive_campaign_cli_prints_confirmation_and_final_paths_on_execute() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz,clifford",
            "3,5",
            "default",
            "y",
        ]
    )

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=lambda campaign, *, output_root: SimpleNamespace(campaign_status="completed"),
        campaign_id_factory=lambda: "campaign-default-002",
        output_root=Path("tmp_campaigns"),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert "Confirmation Summary" in rendered
    assert "campaign-default-002" in rendered
    assert "completed" in rendered
    assert str(Path("tmp_campaigns") / "campaign-default-002" / "summary.md") in rendered
    assert str(Path("tmp_campaigns") / "campaign-default-002" / "campaign.json") in rendered


def test_run_interactive_campaign_cli_aborts_when_confirmation_rejected() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "default",
            "n",
        ]
    )
    run_calls: list[object] = []

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=lambda campaign, *, output_root: run_calls.append(campaign),
        campaign_id_factory=lambda: "campaign-default-003",
        output_root=Path("campaigns"),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert run_calls == []
    assert "aborted" in rendered.lower()
