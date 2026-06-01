import json
from pathlib import Path
from types import SimpleNamespace

from src.integration.campaign_contracts import CampaignCase, CampaignCircuitSpec
from src.integration.campaign_reporting import CampaignCaseReport, build_campaign_report
from src.integration.contracts import LayoutSelectionPolicy, ScenarioResult
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


def _write_batch_file(tmp_path: Path, payload: dict) -> Path:
    batch_path = tmp_path / "campaigns.json"
    batch_path.write_text(json.dumps(payload), encoding="utf-8")
    return batch_path


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
    assert config.rl_learning_rate == 1e-4
    assert config.rl_clip_range == 0.1
    assert config.rl_target_kl == 0.03
    assert config.rl_n_eval_episodes == 1
    assert config.seed == 42
    assert config.mo_use_quick is True
    assert config.mo_population_size == 30
    assert config.mo_n_generations == 50
    assert config.mo_effort_mode == "auto"
    assert config.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert config.mo_objective_name is None
    assert config.mode == "default"
    assert config.topology_source == "backend"
    assert config.synthetic_topology is None


def test_load_campaign_batch_builds_single_campaign_from_json(tmp_path) -> None:
    from src.integration.campaign_cli import load_campaign_batch

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "ghz-3-torino-fast",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                    "backend_names": ["fake_torino"],
                    "mode": "advanced",
                    "rl": {
                        "algorithm": "MaskablePPO",
                        "total_timesteps": 5000,
                        "frontier_mode": "dag",
                        "lookahead_window": 10,
                        "max_steps": 200,
                        "learning_rate": 0.0001,
                        "clip_range": 0.1,
                        "target_kl": 0.03,
                        "n_eval_episodes": 5,
                        "cycle_window": 6,
                        "stagnation_patience": 11,
                        "sabre_top_k": 3,
                    },
                    "mo": {
                        "effort_mode": "auto",
                        "layout_policy": "compromise",
                    },
                    "seed": 42,
                }
            ]
        },
    )

    campaigns = load_campaign_batch(batch_path)
    config = campaigns[0].config

    assert len(campaigns) == 1
    assert campaigns[0].campaign_id == "ghz-3-torino-fast"
    assert config.circuit_specs == (CampaignCircuitSpec(family="ghz", num_qubits=3),)
    assert config.backend_names == ("fake_torino",)
    assert config.mode == "advanced"
    assert config.rl_algorithm == "MaskablePPO"
    assert config.rl_total_timesteps == 5000
    assert config.rl_frontier_mode == "dag"
    assert config.rl_lookahead_window == 10
    assert config.rl_max_steps == 200
    assert config.rl_learning_rate == 0.0001
    assert config.rl_clip_range == 0.1
    assert config.rl_target_kl == 0.03
    assert config.rl_n_eval_episodes == 5
    assert config.rl_cycle_window == 6
    assert config.rl_stagnation_patience == 11
    assert config.rl_sabre_top_k == 3
    assert config.mo_effort_mode == "auto"
    assert config.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert config.seed == 42
    assert config.seeds == (42,)
    assert config.mo_selection_modes == ("compromise",)


def test_load_campaign_batch_builds_t_synthetic_topology_from_json(tmp_path) -> None:
    from src.integration.campaign_cli import load_campaign_batch

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "ghz-8-synthetic-t",
                    "circuit": {"family": "ghz", "num_qubits": 8},
                    "mode": "advanced",
                    "topology_source": "synthetic",
                    "synthetic_topology": {"shape": "t", "num_qubits": 8},
                    "mo": {
                        "effort_mode": "auto",
                        "layout_policy": "compromise",
                    },
                    "seed": 42,
                }
            ]
        },
    )

    campaign = load_campaign_batch(batch_path)[0]
    config = campaign.config

    assert config.topology_source == "synthetic"
    assert config.backend_names == ("synthetic_t_8q",)
    assert config.synthetic_topology.shape == "t"
    assert config.synthetic_topology.num_qubits == 8
    assert config.synthetic_topology.physical_qubits == 8


def test_load_campaign_batch_rejects_mismatched_synthetic_backend_name(tmp_path) -> None:
    from src.integration.campaign_cli import load_campaign_batch

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "ghz-8-synthetic-t",
                    "circuit": {"family": "ghz", "num_qubits": 8},
                    "backend_names": ["synthetic_t_11q"],
                    "mode": "advanced",
                    "topology_source": "synthetic",
                    "synthetic_topology": {"shape": "t", "num_qubits": 8},
                }
            ]
        },
    )

    try:
        load_campaign_batch(batch_path)
    except ValueError as exc:
        assert "synthetic topology backend_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched synthetic backend name")


def test_load_campaign_batch_accepts_multi_seed_all_mo_selection_modes(tmp_path) -> None:
    from src.integration.campaign_cli import load_campaign_batch

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "ghz-3-matrix",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                    "backend_names": ["fake_torino"],
                    "mode": "advanced",
                    "seeds": [11, 12],
                    "parallel_workers": 2,
                    "mo": {
                        "effort_mode": "auto",
                        "selection_modes": "all",
                    },
                }
            ]
        },
    )

    campaign = load_campaign_batch(batch_path)[0]
    config = campaign.config

    assert config.seed == 11
    assert config.seeds == (11, 12)
    assert config.mo_selection_modes == ("compromise", "best_depth", "best_cnot_count")
    assert config.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert config.mo_objective_name is None
    assert config.parallel_workers == 2


def test_load_campaign_batch_accepts_explicit_hybrid_probe_without_changing_all_alias(tmp_path) -> None:
    from src.integration.campaign_cli import load_campaign_batch

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "ghz-3-hybrid-probe",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                    "backend_names": ["fake_torino"],
                    "mode": "advanced",
                    "rl": {"algorithm": "MaskablePPO"},
                    "mo": {"selection_modes": ["hybrid_probe"]},
                }
            ]
        },
    )

    config = load_campaign_batch(batch_path)[0].config

    assert config.mo_selection_modes == ("hybrid_probe",)
    assert config.layout_policy is LayoutSelectionPolicy.COMPROMISE


def test_load_campaign_batch_accepts_explicit_rl_guided_synthetic_campaign(tmp_path) -> None:
    from src.integration.campaign_cli import load_campaign_batch

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "ghz-3-rl-guided",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                    "mode": "advanced",
                    "topology_source": "synthetic",
                    "synthetic_topology": {"shape": "ring", "num_qubits": 3},
                    "rl": {
                        "algorithm": "MaskablePPO",
                        "total_timesteps": 5000,
                        "finetune_timesteps": 1250,
                    },
                    "mo": {"selection_modes": ["rl_guided"]},
                }
            ]
        },
    )

    config = load_campaign_batch(batch_path)[0].config

    assert config.backend_names == ("synthetic_ring_3q",)
    assert config.mo_selection_modes == ("rl_guided",)
    assert config.rl_total_timesteps == 5000
    assert config.rl_finetune_timesteps == 1250


def test_load_campaign_batch_preserves_independent_campaign_configs(tmp_path) -> None:
    from src.integration.campaign_cli import load_campaign_batch

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "ghz-default",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                    "backend_names": ["fake_torino"],
                    "seed": 11,
                },
                {
                    "campaign_id": "qft-advanced",
                    "circuit": {"family": "qft", "num_qubits": 5},
                    "backend_names": ["fake_brisbane"],
                    "mode": "advanced",
                    "rl": {
                        "algorithm": "PPO",
                        "total_timesteps": 9000,
                        "frontier_mode": "sequential",
                        "lookahead_window": 15,
                        "max_steps": 300,
                    },
                    "mo": {
                        "effort_mode": "custom",
                        "use_quick": False,
                        "population_size": MIN_CUSTOM_MO_POPULATION_SIZE,
                        "n_generations": 70,
                        "layout_policy": "compromise",
                    },
                    "seed": 99,
                },
            ]
        },
    )

    first, second = load_campaign_batch(batch_path)

    assert first.config.circuit_specs == (CampaignCircuitSpec(family="ghz", num_qubits=3),)
    assert first.config.backend_names == ("fake_torino",)
    assert first.config.rl_algorithm == "MaskablePPO"
    assert first.config.seed == 11
    assert first.config.mo_effort_mode == "auto"
    assert second.config.circuit_specs == (CampaignCircuitSpec(family="qft", num_qubits=5),)
    assert second.config.backend_names == ("fake_brisbane",)
    assert second.config.rl_algorithm == "PPO"
    assert second.config.rl_total_timesteps == 9000
    assert second.config.rl_frontier_mode == "sequential"
    assert second.config.rl_lookahead_window == 15
    assert second.config.rl_max_steps == 300
    assert second.config.seed == 99
    assert second.config.mo_effort_mode == "custom"
    assert second.config.mo_use_quick is False
    assert second.config.mo_population_size == MIN_CUSTOM_MO_POPULATION_SIZE
    assert second.config.mo_n_generations == 70


def test_campaign_batch_dry_run_validates_without_executing(tmp_path) -> None:
    from src.integration import campaign_cli

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "dry-run-campaign",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                }
            ]
        },
    )
    outputs: list[str] = []
    run_calls: list[str] = []
    output_root = tmp_path / "campaigns"

    exit_code = campaign_cli.run_campaign_cli_from_args(
        ["--input", str(batch_path), "--output-root", str(output_root), "--dry-run"],
        output_fn=outputs.append,
        run_campaign_fn=lambda campaign, *, output_root: run_calls.append(campaign.campaign_id),
    )

    summary = json.loads((output_root / "batch_summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert run_calls == []
    assert "Validated campaign: dry-run-campaign" in outputs
    assert summary["status"] == "dry_run"
    assert summary["dry_run"] is True
    assert summary["results"][0]["status"] == "validated"


def test_campaign_batch_executes_queue_sequentially_in_file_order(tmp_path) -> None:
    from src.integration import campaign_cli

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "campaign-one",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                },
                {
                    "campaign_id": "campaign-two",
                    "circuit": {"family": "qft", "num_qubits": 4},
                    "backend_names": ["fake_brisbane"],
                },
            ]
        },
    )
    run_calls: list[str] = []
    output_root = tmp_path / "campaigns"

    def run_campaign_fn(campaign, *, output_root):
        run_calls.append(campaign.campaign_id)
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_campaign_cli_from_args(
        ["--input", str(batch_path), "--output-root", str(output_root)],
        output_fn=lambda message="": None,
        run_campaign_fn=run_campaign_fn,
    )

    summary = json.loads((output_root / "batch_summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert run_calls == ["campaign-one", "campaign-two"]
    assert summary["status"] == "completed"
    assert [result["campaign_id"] for result in summary["results"]] == [
        "campaign-one",
        "campaign-two",
    ]


def test_campaign_batch_non_verbose_prints_only_campaign_result_paths(tmp_path) -> None:
    from src.integration import campaign_cli

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "campaign-one",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                },
                {
                    "campaign_id": "campaign-two",
                    "circuit": {"family": "qft", "num_qubits": 4},
                },
            ]
        },
    )
    outputs: list[str] = []
    output_root = tmp_path / "campaigns"

    def run_campaign_fn(campaign, *, output_root, verbose=False):
        print("noisy child execution")
        assert verbose is False
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_campaign_cli_from_args(
        ["--input", str(batch_path), "--output-root", str(output_root)],
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
    )

    assert exit_code == 0
    assert outputs == [
        f"Campaign Result: {output_root / 'campaign-one'}",
        f"Campaign Result: {output_root / 'campaign-two'}",
    ]


def test_campaign_batch_verbose_preserves_detailed_final_output(tmp_path) -> None:
    from src.integration import campaign_cli

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "campaign-one",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                }
            ]
        },
    )
    outputs: list[str] = []
    output_root = tmp_path / "campaigns"

    def run_campaign_fn(campaign, *, output_root, verbose=False):
        assert verbose is True
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_campaign_cli_from_args(
        ["--input", str(batch_path), "--output-root", str(output_root), "--verbose"],
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert f"Executing 1 campaign(s) from {batch_path}..." in rendered
    assert "Campaign ID: campaign-one" in rendered
    assert "Final Status: completed" in rendered
    assert f"Summary Document: {output_root / 'campaign-one' / 'summary.md'}" in rendered
    assert f"Structured Campaign Output: {output_root / 'campaign-one' / 'campaign.json'}" in rendered
    assert f"Batch Summary: {output_root / 'batch_summary.json'}" in rendered


def test_campaign_batch_continues_after_campaign_execution_exception(tmp_path) -> None:
    from src.integration import campaign_cli

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "campaign-broken",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                },
                {
                    "campaign_id": "campaign-after",
                    "circuit": {"family": "qft", "num_qubits": 4},
                },
            ]
        },
    )
    run_calls: list[str] = []
    output_root = tmp_path / "campaigns"

    def run_campaign_fn(campaign, *, output_root):
        run_calls.append(campaign.campaign_id)
        if campaign.campaign_id == "campaign-broken":
            raise RuntimeError("training crashed")
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_campaign_cli_from_args(
        ["--input", str(batch_path), "--output-root", str(output_root)],
        output_fn=lambda message="": None,
        run_campaign_fn=run_campaign_fn,
    )

    summary = json.loads((output_root / "batch_summary.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert run_calls == ["campaign-broken", "campaign-after"]
    assert summary["status"] == "failed"
    assert [result["status"] for result in summary["results"]] == ["failed", "completed"]
    assert summary["results"][0]["error"] == "training crashed"


def test_campaign_batch_validation_errors_prevent_execution(tmp_path) -> None:
    from src.integration import campaign_cli

    batch_path = _write_batch_file(
        tmp_path,
        {
            "campaigns": [
                {
                    "campaign_id": "bad-family",
                    "circuit": {"family": "not_a_family", "num_qubits": 3},
                },
                {
                    "campaign_id": "bad-backend",
                    "circuit": {"family": "ghz", "num_qubits": 3},
                    "backend_names": ["not_a_backend"],
                },
                {
                    "campaign_id": "bad-policy",
                    "circuit": {"family": "qft", "num_qubits": 3},
                    "mo": {"layout_policy": "not_a_policy"},
                },
            ]
        },
    )
    outputs: list[str] = []
    run_calls: list[str] = []

    exit_code = campaign_cli.run_campaign_cli_from_args(
        ["--input", str(batch_path), "--output-root", str(tmp_path / "campaigns")],
        output_fn=outputs.append,
        run_campaign_fn=lambda campaign, *, output_root: run_calls.append(campaign.campaign_id),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 1
    assert run_calls == []
    assert "Invalid campaign batch" in rendered
    assert "not_a_family" in rendered
    assert "not_a_backend" in rendered
    assert "not_a_policy" in rendered


def test_run_interactive_campaign_cli_allows_multiple_backends_in_advanced_mode() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz,qft",
            "3,5",
            "advanced",
            "backend",
            "fake_torino,fake_brisbane",
            "MaskablePPO",
            "7000",
            "dag",
            "15",
            "300",
            "4",
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
        verbose=True,
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
    assert captured["campaign"].config.rl_learning_rate == 1e-4
    assert captured["campaign"].config.rl_clip_range == 0.1
    assert captured["campaign"].config.rl_target_kl == 0.03
    assert captured["campaign"].config.rl_n_eval_episodes == 1
    assert captured["campaign"].config.rl_sabre_top_k == 4
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
    assert "SABRE top-k (blank to disable): " in rendered


def test_run_interactive_campaign_cli_auto_effort_skips_manual_mo_knobs_and_prints_preview() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz,qft",
            "3,8",
            "advanced",
            "backend",
            "fake_torino",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "",
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
        verbose=True,
    )

    rendered = "\n".join(outputs)
    config = captured["campaign"].config

    assert exit_code == 0
    assert config.mo_effort_mode == "auto"
    assert config.mo_use_quick is True
    assert config.mo_population_size == 30
    assert config.mo_n_generations == 50
    assert config.rl_sabre_top_k is None
    assert "MO population size:" not in rendered
    assert "MO generations:" not in rendered
    assert "MO Effort Mode: auto" in rendered
    assert "MO Auto Preview (3q): quick=True, population_size=30, n_generations=50" in rendered
    assert "MO Auto Preview (8q): quick=False, population_size=60, n_generations=120" in rendered
    assert "MO Quick:" not in rendered
    assert "MO Population Size:" not in rendered
    assert "MO Generations:" not in rendered


def test_run_interactive_campaign_cli_collects_synthetic_topology_in_advanced_mode() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "advanced",
            "synthetic",
            "grid",
            "2",
            "2",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "",
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
        campaign_id_factory=lambda: "campaign-synth-001",
        output_root=Path("campaigns"),
        verbose=True,
    )

    rendered = "\n".join(outputs)
    config = captured["campaign"].config

    assert exit_code == 0
    assert config.topology_source == "synthetic"
    assert config.backend_names == ("synthetic_grid_2x2",)
    assert config.synthetic_topology.shape == "grid"
    assert config.synthetic_topology.physical_qubits == 4
    assert config.synthetic_topology.basis_gates == ("id", "rz", "sx", "x", "cx")
    assert "Choose backend" not in rendered
    assert "Topology Source: synthetic" in rendered
    assert "Synthetic Topology: synthetic_grid_2x2" in rendered
    assert "Synthetic Basis Gates: id, rz, sx, x, cx" in rendered


def test_run_interactive_campaign_cli_collects_rl_guided_finetune_budget() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "advanced",
            "synthetic",
            "ring",
            "3",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "",
            "42",
            "auto",
            "rl_guided",
            "750",
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
        campaign_id_factory=lambda: "campaign-rl-guided-001",
        output_root=Path("campaigns"),
        verbose=True,
    )

    rendered = "\n".join(outputs)
    config = captured["campaign"].config

    assert exit_code == 0
    assert config.mo_selection_modes == ("rl_guided",)
    assert config.rl_total_timesteps == 5000
    assert config.rl_finetune_timesteps == 750
    assert config.synthetic_topology.backend_name == "synthetic_ring_3q"
    assert "RL fine-tune timesteps: " in rendered
    assert "finetune_timesteps=750" in rendered


def test_run_interactive_campaign_cli_collects_t_synthetic_topology_in_advanced_mode() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "8",
            "advanced",
            "synthetic",
            "t",
            "8",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "",
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
        campaign_id_factory=lambda: "campaign-synth-t-001",
        output_root=Path("campaigns"),
        verbose=True,
    )

    rendered = "\n".join(outputs)
    config = captured["campaign"].config

    assert exit_code == 0
    assert config.topology_source == "synthetic"
    assert config.backend_names == ("synthetic_t_8q",)
    assert config.synthetic_topology.shape == "t"
    assert config.synthetic_topology.physical_qubits == 8
    assert "Synthetic Topology: synthetic_t_8q" in rendered
    assert "Synthetic Physical Qubits: 8" in rendered


def test_run_interactive_campaign_cli_reprompts_when_synthetic_topology_is_too_small() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "advanced",
            "synthetic",
            "line",
            "2",
            "line",
            "3",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "",
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
        campaign_id_factory=lambda: "campaign-synth-002",
        output_root=Path("campaigns"),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert captured["campaign"].config.backend_names == ("synthetic_line_3q",)
    assert "Requires at least 3 physical qubits, got 2" in rendered


def test_run_interactive_campaign_cli_collects_best_on_objective_objective_name(monkeypatch) -> None:
    from src.integration import campaign_cli

    monkeypatch.setattr(campaign_cli, "_available_objective_names", lambda: ("depth", "cnot_count"))
    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "advanced",
            "backend",
            "fake_torino",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "",
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
    assert captured["campaign"].config.mo_selection_modes == ("best_cnot_count",)


def test_run_interactive_campaign_cli_collects_multi_seed_all_mo_modes() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz",
            "3",
            "advanced",
            "backend",
            "fake_torino",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "",
            "41,42",
            "auto",
            "all",
            "2",
            "y",
        ]
    )
    captured = {}

    def _result(child, scenario_name: str, depth: int) -> ScenarioResult:
        return ScenarioResult(
            scenario_name=scenario_name,
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=child.config.seed,
            success=True,
            selected_layout=None if scenario_name == "Baseline" else [0, 1, 2],
            transpilation_metrics={
                "trans_depth": depth,
                "trans_two_qubit_gates": depth // 2,
                "trans_cnot_equivalent": float(depth),
                "elapsed_time_s": float(depth) / 100.0,
            },
        )

    def run_campaign_fn(campaign, *, output_root):
        captured.setdefault("children", []).append(campaign)
        case = CampaignCase(
            case_id="ghz_3__fake_torino",
            circuit_family="ghz",
            num_qubits=3,
            backend_name="fake_torino",
        )
        return build_campaign_report(
            campaign_id=campaign.campaign_id,
            campaign_status="completed",
            campaign_config=campaign.config,
            case_reports=[
                CampaignCaseReport(
                    case=case,
                    status="completed",
                    baseline_result=_result(campaign, "Baseline", 100),
                    rl_only_result=_result(campaign, "RL_Only", 90),
                    mo_only_result=_result(campaign, "MO_Only", 80),
                    mo_rl_result=_result(campaign, "MO+RL", 70),
                )
            ],
        )

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-matrix-cli",
        output_root=Path("campaigns"),
        verbose=True,
    )

    rendered = "\n".join(outputs)
    assert exit_code == 0
    assert len(captured["children"]) == 6
    assert {child.config.seed for child in captured["children"]} == {41, 42}
    assert {child.config.mo_selection_modes[0] for child in captured["children"]} == {
        "compromise",
        "best_depth",
        "best_cnot_count",
    }
    assert "Seeds: 41, 42" in rendered
    assert "MO Selection Modes: compromise, best_depth, best_cnot_count" in rendered
    assert "Parallel Workers: 2" in rendered
    assert "matrix_summary.md" in rendered


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
            "backend",
            "fake_torino",
            "MaskablePPO",
            "5000",
            "dag",
            "10",
            "200",
            "",
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


def test_run_interactive_campaign_cli_non_verbose_prints_only_campaign_result_after_execute() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz,clifford",
            "3,5",
            "default",
            "y",
        ]
    )

    def run_campaign_fn(campaign, *, output_root, verbose=False):
        print("internal training noise")
        assert verbose is False
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-default-002",
        output_root=Path("tmp_campaigns"),
    )

    rendered = "\n".join(outputs)

    assert exit_code == 0
    assert "Confirmation Summary" not in rendered
    assert "Executing campaign..." not in rendered
    assert "Final Status" not in rendered
    assert "Summary Document" not in rendered
    assert "Structured Campaign Output" not in rendered
    assert "internal training noise" not in rendered
    assert f"Campaign Result: {Path('tmp_campaigns') / 'campaign-default-002'}" in rendered


def test_run_interactive_campaign_cli_verbose_prints_confirmation_and_final_paths_on_execute() -> None:
    from src.integration import campaign_cli

    input_fn, outputs = _make_io(
        [
            "ghz,clifford",
            "3,5",
            "default",
            "y",
        ]
    )

    def run_campaign_fn(campaign, *, output_root, verbose=False):
        assert verbose is True
        print("internal training noise")
        return SimpleNamespace(campaign_status="completed")

    exit_code = campaign_cli.run_interactive_campaign_cli(
        input_fn=input_fn,
        output_fn=lambda message="": outputs.append(str(message)),
        run_campaign_fn=run_campaign_fn,
        campaign_id_factory=lambda: "campaign-default-002",
        output_root=Path("tmp_campaigns"),
        verbose=True,
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
