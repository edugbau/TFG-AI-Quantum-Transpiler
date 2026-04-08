from dataclasses import asdict

from src.integration.contracts import RoutingEpisodeSummary, ScenarioResult


def _make_result(scenario_name: str, *, with_routing_summary: bool = False) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=scenario_name,
        circuit_name="ghz_3",
        backend_name="fake_backend",
        seed=17,
        success=True,
        selected_layout=[0, 1, 2] if scenario_name == "MO+RL" else None,
        transpilation_metrics={"trans_depth": 7} if scenario_name == "Baseline" else None,
        routing_summary=(
            RoutingEpisodeSummary(
                initial_layout=[0, 1, 2],
                final_layout=[1, 0, 2],
                steps_executed=3,
                total_reward=1.5,
                completed=True,
                truncated=False,
                total_swaps=1,
                gates_executed_count=2,
            )
            if with_routing_summary
            else None
        ),
        notes=["note"],
    )


def test_build_parser_accepts_supported_scenarios_and_required_args() -> None:
    from src.integration.runner import build_parser

    parser = build_parser()

    args = parser.parse_args(
        [
            "--scenario",
            "MO_Only",
            "--circuit",
            "ghz_3",
            "--num-qubits",
            "3",
            "--backend",
            "fake_backend",
        ]
    )

    assert args.scenario == "MO_Only"
    assert args.circuit == "ghz_3"
    assert args.num_qubits == 3
    assert args.backend == "fake_backend"
    assert args.seed is None
    assert args.rl_model_path is None


def test_run_from_args_dispatches_baseline_and_returns_serializable_dict(monkeypatch) -> None:
    from src.integration import runner

    calls = []
    expected = _make_result("Baseline")

    monkeypatch.setattr(
        runner,
        "run_baseline_scenario",
        lambda request: calls.append(request) or expected,
    )

    payload = runner.run_from_args(
        [
            "--scenario",
            "Baseline",
            "--circuit",
            "ghz_3",
            "--num-qubits",
            "3",
            "--backend",
            "fake_backend",
            "--seed",
            "17",
        ]
    )

    assert payload == asdict(expected)
    assert len(calls) == 1
    assert calls[0].scenario_name == "Baseline"
    assert calls[0].circuit_name == "ghz_3"
    assert calls[0].num_qubits == 3
    assert calls[0].backend_name == "fake_backend"
    assert calls[0].seed == 17


def test_run_from_args_dispatches_mo_rl_with_rl_model_path(monkeypatch) -> None:
    from src.integration import runner

    calls = []
    expected = _make_result("MO+RL", with_routing_summary=True)

    monkeypatch.setattr(
        runner,
        "run_mo_rl_scenario",
        lambda request: calls.append(request) or expected,
    )

    payload = runner.run_from_args(
        [
            "--scenario",
            "MO+RL",
            "--circuit",
            "ghz_3",
            "--num-qubits",
            "3",
            "--backend",
            "fake_backend",
            "--seed",
            "17",
            "--rl-model-path",
            "models/policy.zip",
        ]
    )

    assert payload == asdict(expected)
    assert len(calls) == 1
    assert calls[0].scenario_name == "MO+RL"
    assert calls[0].rl_model_path == "models/policy.zip"


def test_run_from_args_rejects_rl_only_without_rl_model_path(monkeypatch) -> None:
    from src.integration import runner

    monkeypatch.setattr(
        runner,
        "_dispatch",
        lambda request: _make_result("RL_Only", with_routing_summary=True),
    )

    try:
        runner.run_from_args(
            [
                "--scenario",
                "RL_Only",
                "--circuit",
                "ghz_3",
                "--num-qubits",
                "3",
                "--backend",
                "fake_backend",
            ]
        )
    except ValueError as exc:
        assert "RL_Only" in str(exc)
        assert "rl_model_path" in str(exc)
    else:
        raise AssertionError("expected ValueError when RL_Only has no rl_model_path")


def test_dispatch_rejects_unknown_scenario_name_instead_of_falling_through(monkeypatch) -> None:
    from src.integration import runner

    invalid_request = type("Request", (), {"scenario_name": "Unexpected"})()
    mo_rl_calls = []

    monkeypatch.setattr(
        runner,
        "run_mo_rl_scenario",
        lambda request: mo_rl_calls.append(request) or _make_result("MO+RL", with_routing_summary=True),
    )

    try:
        runner._dispatch(invalid_request)
    except ValueError as exc:
        assert "Unexpected" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown scenario")
    assert mo_rl_calls == []


def test_run_from_args_requires_scenario_result_from_dispatch(monkeypatch) -> None:
    from src.integration import runner

    monkeypatch.setattr(runner, "run_baseline_scenario", lambda request: {"success": True})

    try:
        runner.run_from_args(
            [
                "--scenario",
                "Baseline",
                "--circuit",
                "ghz_3",
                "--num-qubits",
                "3",
                "--backend",
                "fake_backend",
            ]
        )
    except TypeError as exc:
        assert "ScenarioResult" in str(exc)
    else:
        raise AssertionError("expected TypeError for non-ScenarioResult result")


def test_main_returns_zero_and_prints_payload(monkeypatch, capsys) -> None:
    from src.integration import runner

    monkeypatch.setattr(
        runner,
        "run_from_args",
        lambda argv=None: {"scenario_name": "Baseline", "success": True},
    )

    exit_code = runner.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == '{"scenario_name": "Baseline", "success": true}\n'
