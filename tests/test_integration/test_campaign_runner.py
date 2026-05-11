import json
from pathlib import Path
from types import SimpleNamespace

from qiskit import QuantumCircuit

from src.integration.campaign_contracts import Campaign, CampaignCase, CampaignCircuitSpec, CampaignConfig
from src.integration.contracts import LayoutSelectionPolicy, ScenarioResult
from src.integration.training_bridge import TrainingBridgeResult, TrainingConfigSummary


def _build_campaign(
    *,
    circuit_specs: list[CampaignCircuitSpec] | None = None,
    backend_names: list[str] | None = None,
) -> Campaign:
    config = CampaignConfig(
        circuit_specs=circuit_specs or [CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=backend_names or ["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.COMPROMISE,
        mode="advanced",
    )
    return Campaign.from_config(campaign_id="campaign-001", config=config)


def _case_library_name(case: CampaignCase) -> str:
    return f"{case.circuit_family}_{case.num_qubits}"


def _case_id_from_request(request) -> str:
    return f"{request.circuit_name}__{request.backend_name}"


def _build_metrics(depth: int) -> dict[str, float | int]:
    return {
        "trans_depth": depth,
        "trans_two_qubit_gates": depth // 2,
        "trans_cnot_equivalent": float(depth) / 3.0,
        "elapsed_time_s": float(depth) / 100.0,
    }


def _build_result(
    scenario_name: str,
    case: CampaignCase,
    *,
    success: bool = True,
    metrics: dict[str, float | int] | None = None,
    errors: list[str] | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=scenario_name,
        circuit_name=_case_library_name(case),
        backend_name=case.backend_name,
        seed=42,
        success=success,
        selected_layout=[0, 1, 2] if scenario_name in {"MO_Only", "MO+RL"} else None,
        transpilation_metrics=metrics,
        errors=errors or [],
    )


def _build_training_result(case: CampaignCase, *, status: str = "completed") -> TrainingBridgeResult:
    artifact_path = (
        Path("campaigns") / "campaign-001" / "cases" / case.case_id / "training" / "models" / "run-001" / "best_model.zip"
        if status == "completed"
        else None
    )
    return TrainingBridgeResult(
        status=status,
        selected_artifact_path=artifact_path,
        best_model_path=artifact_path,
        final_model_path=artifact_path,
        run_model_dir=Path("campaigns") / "campaign-001" / "cases" / case.case_id / "training" / "models" / "run-001",
        run_log_dir=Path("campaigns") / "campaign-001" / "cases" / case.case_id / "training" / "logs" / "run-001",
        effective_training_config=TrainingConfigSummary(
            algorithm="MaskablePPO",
            total_timesteps=5000,
            frontier_mode="dag",
            lookahead_window=12,
            max_steps=256,
            seed=42,
        ),
    )


def _make_case_circuit(num_qubits: int = 3) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    if num_qubits >= 2:
        circuit.cx(0, 1)
    if num_qubits >= 3:
        circuit.cx(1, 2)
    return circuit


def test_run_campaign_executes_cases_in_stable_sequential_order(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="random_shallow", num_qubits=3),
            CampaignCircuitSpec(family="ghz", num_qubits=3),
        ],
        backend_names=["fake_torino", "fake_brisbane"],
    )
    cases = campaign.build_cases()
    call_log: list[tuple[str, str, int]] = []
    loaded_circuits: dict[str, object] = {}
    write_calls: list[int] = []

    def fake_load_circuit(case: CampaignCase):
        circuit = _make_case_circuit()
        loaded_circuits[case.case_id] = circuit
        call_log.append(("load", case.case_id, id(circuit)))
        return circuit

    def fake_run_baseline(request, *, circuit):
        case = next(case for case in cases if case.case_id == _case_id_from_request(request))
        call_log.append(("Baseline", case.case_id, id(circuit)))
        return _build_result("Baseline", case, metrics=_build_metrics(100))

    def fake_run_mo_only(request, *, circuit):
        case = next(case for case in cases if case.case_id == _case_id_from_request(request))
        call_log.append(("MO_Only", case.case_id, id(circuit)))
        return _build_result("MO_Only", case, metrics=_build_metrics(90))

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, case_output_dir, initial_layout
        assert coupling_map == [(0, 1), (1, 2)]
        call_log.append(("train", campaign_case.case_id, id(target_circuit)))
        return _build_training_result(campaign_case)

    def fake_run_mo_rl(request, *, circuit, injected_layout):
        case = next(case for case in cases if case.case_id == _case_id_from_request(request))
        assert injected_layout == [0, 1, 2]
        call_log.append(("MO+RL", case.case_id, id(circuit)))
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    def fake_write_outputs(*, output_dir, report):
        del output_dir
        write_calls.append(len(report.case_reports))
        return None

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=fake_load_circuit,
        run_baseline=fake_run_baseline,
        run_mo_only=fake_run_mo_only,
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=fake_write_outputs,
    )

    assert [entry[:2] for entry in call_log] == [
        ("load", cases[0].case_id),
        ("Baseline", cases[0].case_id),
        ("MO_Only", cases[0].case_id),
        ("train", cases[0].case_id),
        ("MO+RL", cases[0].case_id),
        ("load", cases[1].case_id),
        ("Baseline", cases[1].case_id),
        ("MO_Only", cases[1].case_id),
        ("train", cases[1].case_id),
        ("MO+RL", cases[1].case_id),
        ("load", cases[2].case_id),
        ("Baseline", cases[2].case_id),
        ("MO_Only", cases[2].case_id),
        ("train", cases[2].case_id),
        ("MO+RL", cases[2].case_id),
        ("load", cases[3].case_id),
        ("Baseline", cases[3].case_id),
        ("MO_Only", cases[3].case_id),
        ("train", cases[3].case_id),
        ("MO+RL", cases[3].case_id),
    ]
    assert [case_report.case.case_id for case_report in report.case_reports] == [case.case_id for case in cases]
    assert write_calls == [1, 2, 3, 4, 4]
    for case in cases:
        case_circuit_ids = {
            circuit_id
            for name, case_id, circuit_id in call_log
            if case_id == case.case_id and name in {"Baseline", "MO_Only", "train", "MO+RL"}
        }
        assert case_circuit_ids == {id(loaded_circuits[case.case_id])}


def test_run_campaign_trains_then_runs_baseline_mo_only_and_mo_rl(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    call_order: list[str] = []
    write_calls: list[int] = []
    circuit = _make_case_circuit()

    def fake_run_baseline(request, *, circuit):
        assert request.scenario_name == "Baseline"
        assert request.rl_model_path is None
        assert request.circuit_name == _case_library_name(case)
        call_order.append("Baseline")
        return _build_result("Baseline", case, metrics=_build_metrics(100))

    def fake_run_mo_only(request, *, circuit):
        assert request.scenario_name == "MO_Only"
        assert request.rl_model_path is None
        call_order.append("MO_Only")
        return _build_result("MO_Only", case, metrics=_build_metrics(90))

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, case_output_dir
        assert campaign_case == case
        assert target_circuit is circuit
        assert coupling_map == [(0, 1), (1, 2)]
        assert initial_layout == [0, 1, 2]
        call_order.append("train")
        return _build_training_result(case)

    def fake_run_mo_rl(request, *, circuit, injected_layout):
        assert request.scenario_name == "MO+RL"
        assert request.rl_model_path == str(_build_training_result(case).selected_artifact_path)
        assert injected_layout == [0, 1, 2]
        call_order.append("MO+RL")
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    def fake_write_outputs(*, output_dir, report):
        del output_dir
        write_calls.append(len(report.case_reports))
        return None

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: circuit,
        run_baseline=fake_run_baseline,
        run_mo_only=fake_run_mo_only,
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=fake_write_outputs,
    )

    assert call_order == ["Baseline", "MO_Only", "train", "MO+RL"]
    assert report.case_reports[0].status == "completed"
    assert report.case_reports[0].training_result.status == "completed"
    assert report.case_reports[0].baseline_result.scenario_name == "Baseline"
    assert report.case_reports[0].mo_only_result.scenario_name == "MO_Only"
    assert report.case_reports[0].mo_rl_result.scenario_name == "MO+RL"
    assert campaign.status == "completed"
    assert campaign.summary.status == "completed"
    assert campaign.summary.comparable_completed_cases == 1
    assert write_calls == [1, 1]


def test_run_campaign_trains_mo_rl_cases_from_mo_only_selected_layout(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    captured_training_layouts: list[list[int] | None] = []

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, target_circuit, case_output_dir
        assert campaign_case == case
        assert coupling_map == [(0, 1), (1, 2)]
        captured_training_layouts.append(list(initial_layout) if initial_layout is not None else None)
        return _build_training_result(campaign_case)

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: _build_result("Baseline", case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: ScenarioResult(
            scenario_name="MO_Only",
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=42,
            success=True,
            selected_layout=[2, 1, 0],
            transpilation_metrics=_build_metrics(90),
        ),
        train_case_fn=fake_train_case,
        run_mo_rl=lambda request, *, circuit, injected_layout: _build_result("MO+RL", case, metrics=_build_metrics(80)),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert captured_training_layouts == [[2, 1, 0]]
    assert report.case_reports[0].status == "completed"


def test_run_campaign_defensively_copies_selected_layout_across_training_and_mo_rl(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    mo_only_layout = [2, 1, 0]
    captured_training_layouts: list[list[int] | None] = []
    captured_mo_rl_layouts: list[list[int] | None] = []

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, target_circuit, case_output_dir
        assert campaign_case == case
        assert coupling_map == [(0, 1), (1, 2)]
        captured_training_layouts.append(list(initial_layout) if initial_layout is not None else None)
        initial_layout[0] = 99
        return _build_training_result(campaign_case)

    def fake_run_mo_rl(request, *, circuit, injected_layout):
        del request, circuit
        captured_mo_rl_layouts.append(list(injected_layout) if injected_layout is not None else None)
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: _build_result("Baseline", case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: ScenarioResult(
            scenario_name="MO_Only",
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=42,
            success=True,
            selected_layout=list(mo_only_layout),
            transpilation_metrics=_build_metrics(90),
        ),
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert captured_training_layouts == [[2, 1, 0]]
    assert captured_mo_rl_layouts == [[2, 1, 0]]
    assert report.case_reports[0].status == "completed"


def test_run_campaign_uses_derived_routing_subgraph_for_training_and_mo_rl(tmp_path, monkeypatch) -> None:
    from src.integration import campaign_runner

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    circuit = _make_case_circuit()
    derived_routing_graph = SimpleNamespace(
        mode="path_expanded_subgraph",
        coupling_edges=[(0, 2), (2, 3)],
        node_count=3,
        edge_count=2,
        added_intermediate_qubits=[2],
        interacting_pair_count=1,
        fallback_reason=None,
    )
    captured_training_coupling_maps: list[list[tuple[int, int]]] = []
    captured_mo_rl_kwargs: list[dict[str, object]] = []

    monkeypatch.setattr(
        campaign_runner,
        "build_path_expanded_subgraph",
        lambda *, circuit, selected_layout, coupling_edges: derived_routing_graph,
        raising=False,
    )

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, case_output_dir
        assert campaign_case == case
        assert target_circuit is circuit
        assert initial_layout == [2, 1, 0]
        captured_training_coupling_maps.append(list(coupling_map))
        return _build_training_result(campaign_case)

    def fake_run_mo_rl(request, *, circuit, injected_layout, injected_coupling_edges, injected_routing_graph):
        del request
        captured_mo_rl_kwargs.append(
            {
                "circuit": circuit,
                "injected_layout": list(injected_layout),
                "injected_coupling_edges": list(injected_coupling_edges),
                "injected_routing_graph": injected_routing_graph,
            }
        )
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    report = campaign_runner.run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: circuit,
        run_baseline=lambda request, *, circuit: _build_result("Baseline", case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: ScenarioResult(
            scenario_name="MO_Only",
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=42,
            success=True,
            selected_layout=[2, 1, 0],
            transpilation_metrics=_build_metrics(90),
        ),
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2), (2, 3)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert captured_training_coupling_maps == [[(0, 2), (2, 3)]]
    assert captured_mo_rl_kwargs == [
        {
            "circuit": circuit,
            "injected_layout": [2, 1, 0],
            "injected_coupling_edges": [(0, 2), (2, 3)],
            "injected_routing_graph": derived_routing_graph,
        }
    ]
    assert report.case_reports[0].status == "completed"


def test_run_campaign_fails_case_when_mo_only_result_has_no_selected_layout(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    train_calls: list[str] = []
    mo_rl_calls: list[str] = []

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout
        train_calls.append(campaign_case.case_id)
        return _build_training_result(campaign_case)

    def fake_run_mo_rl(request, *, circuit, injected_layout):
        del circuit
        mo_rl_calls.append(_case_id_from_request(request))
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: _build_result("Baseline", case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: ScenarioResult(
            scenario_name="MO_Only",
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=42,
            success=True,
            selected_layout=None,
            transpilation_metrics=_build_metrics(90),
        ),
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert report.case_reports[0].status == "failed"
    assert report.case_reports[0].training_result is None
    assert report.case_reports[0].mo_rl_result is None
    assert len(report.case_reports[0].incidents) == 1
    assert "selected layout" in report.case_reports[0].incidents[0]
    assert train_calls == []
    assert mo_rl_calls == []


def test_run_campaign_preserves_legacy_mo_rl_runner_seam_without_injected_layout(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    call_order: list[str] = []

    def fake_run_mo_rl(request, *, circuit):
        assert request.scenario_name == "MO+RL"
        call_order.append("MO+RL")
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: _build_result("Baseline", case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: ScenarioResult(
            scenario_name="MO_Only",
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=42,
            success=True,
            selected_layout=[2, 1, 0],
            transpilation_metrics=_build_metrics(90),
        ),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert call_order == ["MO+RL"]
    assert report.case_reports[0].status == "completed"


def test_run_campaign_passes_injected_layout_to_kwargs_aware_mo_rl_runner(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    captured_injected_layouts: list[list[int] | None] = []

    def fake_run_mo_rl(request, **kwargs):
        assert request.scenario_name == "MO+RL"
        injected_layout = kwargs.get("injected_layout")
        captured_injected_layouts.append(list(injected_layout) if injected_layout is not None else None)
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: _build_result("Baseline", case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: ScenarioResult(
            scenario_name="MO_Only",
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=42,
            success=True,
            selected_layout=[2, 1, 0],
            transpilation_metrics=_build_metrics(90),
        ),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert captured_injected_layouts == [[2, 1, 0]]
    assert report.case_reports[0].status == "completed"


def test_run_campaign_forces_baseline_safe_request_defaults(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_use_quick=False,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.BEST_ON_OBJECTIVE,
        mo_objective_name="cnot_count",
        mode="advanced",
    )
    campaign = Campaign.from_config(campaign_id="campaign-001", config=config)
    case = campaign.build_cases()[0]
    captured_baseline_request = None

    def fake_run_baseline(request, *, circuit):
        nonlocal captured_baseline_request
        captured_baseline_request = request
        return _build_result("Baseline", case, metrics=_build_metrics(100))

    run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=fake_run_baseline,
        run_mo_only=lambda request, *, circuit: _build_result("MO_Only", case, metrics=_build_metrics(90)),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=lambda request, *, circuit, injected_layout: _build_result("MO+RL", case, metrics=_build_metrics(80)),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert captured_baseline_request is not None
    assert captured_baseline_request.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert captured_baseline_request.mo_use_quick is True
    assert captured_baseline_request.mo_objective_index == 0


def test_run_campaign_resolves_effective_mo_settings_per_campaign_case(monkeypatch, tmp_path) -> None:
    from src.integration import campaign_runner

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="ghz", num_qubits=8),
        ]
    )
    cases = campaign.build_cases()
    baseline_requests = {}
    mo_only_requests = {}
    mo_rl_requests = {}

    monkeypatch.setattr(
        campaign_runner,
        "build_path_expanded_subgraph",
        lambda *, circuit, selected_layout, coupling_edges: SimpleNamespace(coupling_edges=list(coupling_edges)),
        raising=False,
    )

    def fake_load_case_circuit(campaign_case: CampaignCase):
        return _make_case_circuit(campaign_case.num_qubits)

    def fake_run_baseline(request, *, circuit):
        del circuit
        baseline_requests[_case_id_from_request(request)] = request
        case = next(case for case in cases if case.case_id == _case_id_from_request(request))
        return _build_result("Baseline", case, metrics=_build_metrics(100))

    def fake_run_mo_only(request, *, circuit):
        del circuit
        mo_only_requests[_case_id_from_request(request)] = request
        case = next(case for case in cases if case.case_id == _case_id_from_request(request))
        return ScenarioResult(
            scenario_name="MO_Only",
            circuit_name=_case_library_name(case),
            backend_name=case.backend_name,
            seed=42,
            success=True,
            selected_layout=list(range(case.num_qubits)),
            transpilation_metrics=_build_metrics(90),
        )

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, target_circuit, case_output_dir
        assert initial_layout == list(range(campaign_case.num_qubits))
        assert coupling_map
        return _build_training_result(campaign_case)

    def fake_run_mo_rl(request, *, circuit, injected_layout, injected_coupling_edges, injected_routing_graph):
        del circuit, injected_coupling_edges, injected_routing_graph
        mo_rl_requests[_case_id_from_request(request)] = request
        case = next(case for case in cases if case.case_id == _case_id_from_request(request))
        assert injected_layout == list(range(case.num_qubits))
        return ScenarioResult(
            scenario_name="MO+RL",
            circuit_name=_case_library_name(case),
            backend_name=case.backend_name,
            seed=42,
            success=True,
            selected_layout=list(range(case.num_qubits)),
            transpilation_metrics=_build_metrics(80),
        )

    report = campaign_runner.run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=fake_load_case_circuit,
        run_baseline=fake_run_baseline,
        run_mo_only=fake_run_mo_only,
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(index, index + 1) for index in range(7)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    case_3q, case_8q = cases

    assert baseline_requests[case_3q.case_id].mo_use_quick is True
    assert baseline_requests[case_3q.case_id].mo_population_size == 30
    assert baseline_requests[case_3q.case_id].mo_n_generations == 50
    assert baseline_requests[case_3q.case_id].layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert baseline_requests[case_8q.case_id].mo_use_quick is True
    assert baseline_requests[case_8q.case_id].mo_population_size == 30
    assert baseline_requests[case_8q.case_id].mo_n_generations == 50
    assert baseline_requests[case_8q.case_id].layout_policy is LayoutSelectionPolicy.COMPROMISE

    assert mo_only_requests[case_3q.case_id].mo_use_quick is True
    assert mo_only_requests[case_3q.case_id].mo_population_size == 30
    assert mo_only_requests[case_3q.case_id].mo_n_generations == 50
    assert mo_only_requests[case_8q.case_id].mo_use_quick is False
    assert mo_only_requests[case_8q.case_id].mo_population_size == 60
    assert mo_only_requests[case_8q.case_id].mo_n_generations == 120

    assert mo_rl_requests[case_3q.case_id].mo_use_quick is True
    assert mo_rl_requests[case_3q.case_id].mo_population_size == 30
    assert mo_rl_requests[case_3q.case_id].mo_n_generations == 50
    assert mo_rl_requests[case_8q.case_id].mo_use_quick is False
    assert mo_rl_requests[case_8q.case_id].mo_population_size == 60
    assert mo_rl_requests[case_8q.case_id].mo_n_generations == 120
    assert report.case_reports[0].status == "completed"
    assert report.case_reports[1].status == "completed"


def test_run_campaign_rejects_unknown_mo_objective_name(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    config = CampaignConfig(
        circuit_specs=[CampaignCircuitSpec(family="ghz", num_qubits=3)],
        backend_names=["fake_torino"],
        rl_algorithm="MaskablePPO",
        rl_total_timesteps=5000,
        rl_frontier_mode="dag",
        rl_lookahead_window=12,
        rl_max_steps=256,
        seed=42,
        mo_use_quick=True,
        mo_population_size=30,
        mo_n_generations=50,
        layout_policy=LayoutSelectionPolicy.BEST_ON_OBJECTIVE,
        mo_objective_name="unknown_objective",
        mode="advanced",
    )
    campaign = Campaign.from_config(campaign_id="campaign-001", config=config)

    try:
        run_campaign(
            campaign,
            output_root=tmp_path / "campaigns",
            load_case_circuit=lambda campaign_case: _make_case_circuit(),
            run_baseline=lambda request, *, circuit: _build_result(
                "Baseline",
                campaign.build_cases()[0],
                metrics=_build_metrics(100),
            ),
            run_mo_only=lambda request, *, circuit: _build_result(
                "MO_Only",
                campaign.build_cases()[0],
                metrics=_build_metrics(90),
            ),
            train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
            run_mo_rl=lambda request, *, circuit, injected_layout: _build_result(
                "MO+RL",
                campaign.build_cases()[0],
                metrics=_build_metrics(80),
            ),
            resolve_backend_bundle=lambda backend_name: SimpleNamespace(
                backend_name=backend_name,
                coupling_edges=[(0, 1), (1, 2)],
            ),
            write_outputs=lambda *, output_dir, report: None,
        )
    except ValueError as exc:
        assert "unknown_objective" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown mo_objective_name")


def test_run_campaign_intermediate_persistence_does_not_mark_campaign_terminal(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign
    from src.integration.campaign_reporting import write_campaign_outputs

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=3),
        ]
    )
    cases = campaign.build_cases()
    snapshots: list[tuple[str, str, str | None, int]] = []

    def fake_write_outputs(*, output_dir, report):
        output_paths = write_campaign_outputs(output_dir=output_dir, report=report)
        persisted_payload = json.loads(output_paths.campaign_json_path.read_text(encoding="utf-8"))
        snapshots.append(
            (
                persisted_payload["campaign_status"],
                persisted_payload["summary"]["status"],
                campaign.status,
                len(report.case_reports),
            )
        )
        return output_paths

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: _build_result(
            "Baseline",
            next(case for case in cases if case.case_id == _case_id_from_request(request)),
            metrics=_build_metrics(100),
        ),
        run_mo_only=lambda request, *, circuit: _build_result(
            "MO_Only",
            next(case for case in cases if case.case_id == _case_id_from_request(request)),
            metrics=_build_metrics(90),
        ),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=lambda request, *, circuit, injected_layout: _build_result(
            "MO+RL",
            next(case for case in cases if case.case_id == _case_id_from_request(request)),
            metrics=_build_metrics(80),
        ),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=fake_write_outputs,
    )

    assert snapshots == [("running", "running", "running", 1), ("running", "running", "running", 2), ("completed", "completed", "completed", 2)]
    assert report.campaign_status == "completed"
    assert campaign.status == "completed"


def test_run_campaign_intermediate_persistence_passes_coherent_running_report_to_write_outputs(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=3),
        ]
    )
    cases = campaign.build_cases()
    snapshots: list[tuple[str, str, int, int]] = []

    def fake_write_outputs(*, output_dir, report):
        del output_dir
        snapshots.append(
            (
                report.campaign_status,
                report.summary.status,
                report.summary.total_cases,
                len(report.case_reports),
            )
        )
        return None

    run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: _build_result(
            "Baseline",
            next(case for case in cases if case.case_id == _case_id_from_request(request)),
            metrics=_build_metrics(100),
        ),
        run_mo_only=lambda request, *, circuit: _build_result(
            "MO_Only",
            next(case for case in cases if case.case_id == _case_id_from_request(request)),
            metrics=_build_metrics(90),
        ),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=lambda request, *, circuit, injected_layout: _build_result(
            "MO+RL",
            next(case for case in cases if case.case_id == _case_id_from_request(request)),
            metrics=_build_metrics(80),
        ),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=fake_write_outputs,
    )

    assert snapshots == [
        ("running", "running", 2, 1),
        ("running", "running", 2, 2),
        ("completed", "completed", 2, 2),
    ]


def test_run_campaign_default_scenario_runners_accept_real_public_signatures(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    call_order: list[str] = []

    def fake_result(scenario_name: str, depth: int) -> ScenarioResult:
        return _build_result(scenario_name, case, metrics=_build_metrics(depth))

    def fake_load_case_circuit(campaign_case: CampaignCase):
        assert campaign_case == case
        return _make_case_circuit()

    def fake_public_run_baseline(request):
        assert request.scenario_name == "Baseline"
        call_order.append("Baseline")
        return fake_result("Baseline", 100)

    def fake_public_run_mo_only(request):
        assert request.scenario_name == "MO_Only"
        call_order.append("MO_Only")
        return fake_result("MO_Only", 90)

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, target_circuit, case_output_dir
        assert campaign_case == case
        assert coupling_map == [(0, 1), (1, 2)]
        assert initial_layout == [0, 1, 2]
        call_order.append("train")
        return _build_training_result(case)

    def fake_public_run_mo_rl(request, *, injected_layout):
        assert request.scenario_name == "MO+RL"
        assert request.rl_model_path == str(_build_training_result(case).selected_artifact_path)
        assert injected_layout == [0, 1, 2]
        call_order.append("MO+RL")
        return fake_result("MO+RL", 80)

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=fake_load_case_circuit,
        run_baseline=fake_public_run_baseline,
        run_mo_only=fake_public_run_mo_only,
        train_case_fn=fake_train_case,
        run_mo_rl=fake_public_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert call_order == ["Baseline", "MO_Only", "train", "MO+RL"]
    assert report.case_reports[0].status == "completed"


def test_run_campaign_default_scenario_runners_use_real_wiring_and_frozen_case_circuit(monkeypatch, tmp_path) -> None:
    from src.integration import scenarios
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    selected_layout = [2, 1, 0]
    call_log: list[tuple[str, object]] = []

    def _baseline_payload(baseline_name: str, layout: list[int] | None):
        row = {
            "backend_name": case.backend_name,
            "baseline_name": baseline_name,
            "optimization_level": 1,
            "seed": 42,
            "elapsed_time_s": 0.25,
            "depth_reduction": 0.1,
            "two_qubit_gate_overhead": 1.0,
            "orig_num_qubits": case.num_qubits,
            "orig_depth": 5,
            "orig_total_gates": 6,
            "orig_two_qubit_gates": 2,
            "trans_depth": 12,
            "trans_num_qubits": case.num_qubits,
            "trans_total_gates": 9,
            "trans_two_qubit_gates": 4,
            "initial_layout": layout,
        }
        artifact = {
            "artifact_version": "transpilation_result.v1",
            "baseline_name": baseline_name,
            "transpilation": {
                "baseline_name": baseline_name,
                "initial_layout": layout,
            },
        }
        return row, artifact

    class _FakePostRoutingResult:
        def to_dict(self):
            return {
                "backend_name": case.backend_name,
                "trans_depth": 10,
                "trans_two_qubit_gates": 3,
                "trans_cnot_equivalent": 3.0,
                "elapsed_time_s": 0.1,
            }

        def to_artifact_dict(self):
            return {"artifact_version": "transpilation_result.v1"}

    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "load_circuit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scenario should reuse frozen case circuit")),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "run_named_baseline",
        lambda baseline_name, circuit, backend_names, seed, layout=None, include_artifact=False: call_log.append(
            (baseline_name, circuit)
        )
        or [_baseline_payload(baseline_name, layout)],
    )
    monkeypatch.setattr(
        scenarios,
        "resolve_backend_bundle",
        lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            backend=SimpleNamespace(num_qubits=5),
            coupling_edges=[(0, 1), (1, 2)],
        ),
    )
    monkeypatch.setattr(
        scenarios.mo_module,
        "optimize_layout_quick",
        lambda circuit, backend, population_size, n_generations, seed: call_log.append(("mo", circuit)) or "mo-result",
    )
    monkeypatch.setattr(
        scenarios,
        "select_layout_from_mo_result",
        lambda result, *, policy, objective_index=0: selected_layout,
    )
    monkeypatch.setattr(
        scenarios,
        "resolve_routing_model_contract",
        lambda path: SimpleNamespace(
            algorithm="PPO",
            frontier_mode="dag",
            max_steps=256,
            lookahead_window=4,
            masked=False,
            metadata_source="defaults",
        ),
    )
    monkeypatch.setattr(scenarios, "_load_agent", lambda request, *, algorithm="PPO": "agent-object")
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: call_log.append(("eval", kwargs["circuit"]))
        or SimpleNamespace(
            completed=True,
            truncated=False,
            initial_layout=list(kwargs["initial_layout"]),
            final_layout=list(kwargs["initial_layout"]),
            swap_trace=[],
            executed_gate_trace=[],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: call_log.append(("rebuild", kwargs["circuit"])) or ("routed-circuit", list(kwargs["initial_layout"])),
    )
    monkeypatch.setattr(
        scenarios.qiskit_interface,
        "transpile_post_routing",
        lambda routed_circuit, **kwargs: call_log.append(("post", kwargs["reference_circuit"])) or _FakePostRoutingResult(),
    )

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: circuit,
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert [entry[0] for entry in call_log] == [
        "qiskit_level_1",
        "mo",
        "custom_layout_level_1",
        "eval",
        "rebuild",
        "post",
    ]
    assert all(entry[1] is circuit for entry in call_log)
    assert report.case_reports[0].status == "completed"


def test_run_campaign_preserves_baseline_and_mo_only_when_training_fails(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=3),
        ]
    )
    first_case, second_case = campaign.build_cases()
    mo_rl_calls: list[str] = []

    def fake_result_for_request(request, scenario_name: str, depth: int) -> ScenarioResult:
        case_id = _case_id_from_request(request)
        case = first_case if case_id == first_case.case_id else second_case
        return _build_result(scenario_name, case, metrics=_build_metrics(depth))

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout
        if campaign_case.case_id == first_case.case_id:
            return _build_training_result(campaign_case, status="failed")
        return _build_training_result(campaign_case)

    def fake_run_mo_rl(request, *, circuit, injected_layout):
        del circuit
        assert injected_layout == [0, 1, 2]
        mo_rl_calls.append(_case_id_from_request(request))
        return fake_result_for_request(request, "MO+RL", 80)

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: fake_result_for_request(request, "Baseline", 100),
        run_mo_only=lambda request, *, circuit: fake_result_for_request(request, "MO_Only", 90),
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert [case_report.status for case_report in report.case_reports] == ["failed", "completed"]
    assert report.case_reports[0].baseline_result.scenario_name == "Baseline"
    assert report.case_reports[0].mo_only_result.scenario_name == "MO_Only"
    assert report.case_reports[0].mo_rl_result is None
    assert report.case_reports[0].training_result.status == "failed"
    assert mo_rl_calls == [second_case.case_id]
    assert campaign.status == "completed"
    assert campaign.summary.failed_cases == 1
    assert campaign.summary.comparable_completed_cases == 1


def test_run_campaign_marks_case_failed_when_baseline_or_mo_only_fails(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=3),
        ]
    )
    first_case, second_case = campaign.build_cases()
    train_calls: list[str] = []
    mo_rl_calls: list[str] = []

    def fake_baseline(request, *, circuit):
        del circuit
        case_id = _case_id_from_request(request)
        if case_id == first_case.case_id:
            return _build_result(
                "Baseline",
                first_case,
                success=False,
                metrics=None,
                errors=["Baseline transpilation failed."],
            )
        return _build_result("Baseline", second_case, metrics=_build_metrics(100))

    def fake_mo_only(request, *, circuit):
        del circuit
        case_id = _case_id_from_request(request)
        if case_id == second_case.case_id:
            return _build_result(
                "MO_Only",
                second_case,
                success=False,
                metrics=None,
                errors=["MO selection failed."],
            )
        return _build_result("MO_Only", first_case, metrics=_build_metrics(90))

    def fake_train_case(**kwargs):
        campaign_case = kwargs["campaign_case"]
        train_calls.append(campaign_case.case_id)
        return _build_training_result(campaign_case)

    def fake_run_mo_rl(request, *, circuit, injected_layout):
        del circuit
        assert injected_layout == [0, 1, 2]
        mo_rl_calls.append(_case_id_from_request(request))
        return _build_result("MO+RL", second_case, metrics=_build_metrics(80))

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=fake_baseline,
        run_mo_only=fake_mo_only,
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert [case_report.status for case_report in report.case_reports] == ["failed", "failed"]
    assert report.case_reports[0].baseline_result.success is False
    assert report.case_reports[0].mo_only_result is None
    assert report.case_reports[0].training_result is None
    assert report.case_reports[0].mo_rl_result is None
    assert report.case_reports[1].baseline_result.success is True
    assert report.case_reports[1].mo_only_result.success is False
    assert report.case_reports[1].training_result is None
    assert report.case_reports[1].mo_rl_result is None
    assert train_calls == []
    assert mo_rl_calls == []
    assert campaign.status == "completed"
    assert campaign.summary.failed_cases == 2
    assert campaign.summary.comparable_completed_cases == 0


def test_run_campaign_records_incomplete_mo_rl_without_aborting_remaining_cases(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=3),
        ]
    )
    first_case, second_case = campaign.build_cases()
    mo_rl_calls: list[str] = []

    def fake_result_for_request(request, scenario_name: str, depth: int) -> ScenarioResult:
        case_id = _case_id_from_request(request)
        case = first_case if case_id == first_case.case_id else second_case
        return _build_result(scenario_name, case, metrics=_build_metrics(depth))

    def fake_run_mo_rl(request, *, circuit, injected_layout):
        del circuit
        assert injected_layout == [0, 1, 2]
        case_id = _case_id_from_request(request)
        mo_rl_calls.append(case_id)
        if case_id == first_case.case_id:
            return _build_result(
                "MO+RL",
                first_case,
                success=False,
                metrics=None,
                errors=["MO+RL routing episode did not complete."],
            )
        return _build_result("MO+RL", second_case, metrics=_build_metrics(80))

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: fake_result_for_request(request, "Baseline", 100),
        run_mo_only=lambda request, *, circuit: fake_result_for_request(request, "MO_Only", 90),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert [case_report.status for case_report in report.case_reports] == ["incomplete", "completed"]
    assert report.case_reports[0].baseline_result.scenario_name == "Baseline"
    assert report.case_reports[0].mo_only_result.scenario_name == "MO_Only"
    assert report.case_reports[0].mo_rl_result.success is False
    assert report.case_reports[1].mo_rl_result.success is True
    assert mo_rl_calls == [first_case.case_id, second_case.case_id]
    assert campaign.status == "completed"
    assert campaign.summary.incomplete_cases == 1
    assert campaign.summary.comparable_completed_cases == 1


def test_run_campaign_marks_remaining_cases_cancelled_after_explicit_cancellation_signal(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=3),
            CampaignCircuitSpec(family="random_shallow", num_qubits=3),
        ]
    )
    first_case, second_case, third_case = campaign.build_cases()
    started_cases: list[str] = []
    cancellation_checks = iter([False, True, True])

    def fake_baseline(request, *, circuit):
        del circuit
        started_cases.append(_case_id_from_request(request))
        case = next(case for case in (first_case, second_case, third_case) if case.case_id == started_cases[-1])
        return _build_result("Baseline", case, metrics=_build_metrics(100))

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=fake_baseline,
        run_mo_only=lambda request, *, circuit: _build_result(
            "MO_Only",
            first_case,
            metrics=_build_metrics(90),
        ),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=lambda request, *, circuit, injected_layout: _build_result(
            "MO+RL",
            first_case,
            metrics=_build_metrics(80),
        ),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
        cancellation_requested=lambda: next(cancellation_checks),
    )

    assert started_cases == [first_case.case_id]
    assert [case_report.status for case_report in report.case_reports] == ["completed", "cancelled", "cancelled"]
    assert campaign.status == "cancelled"
    assert campaign.summary.cancelled_cases == 2


def test_run_campaign_marks_interrupted_when_circuit_load_is_interrupted(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=3),
        ]
    )
    first_case, second_case = campaign.build_cases()

    def fake_load_case_circuit(campaign_case: CampaignCase):
        if campaign_case.case_id == first_case.case_id:
            raise KeyboardInterrupt()
        return _make_case_circuit()

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=fake_load_case_circuit,
        run_baseline=lambda request, *, circuit: _build_result("Baseline", second_case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: _build_result("MO_Only", second_case, metrics=_build_metrics(90)),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=lambda request, *, circuit, injected_layout: _build_result("MO+RL", second_case, metrics=_build_metrics(80)),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert [case_report.status for case_report in report.case_reports] == ["cancelled", "cancelled"]
    assert report.case_reports[0].case.case_id == first_case.case_id
    assert report.case_reports[1].case.case_id == second_case.case_id
    assert campaign.status == "interrupted"
    assert campaign.summary.status == "interrupted"
    assert campaign.summary.cancelled_cases == 2


def test_run_campaign_preserves_completed_results_when_persistence_is_interrupted(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign(
        circuit_specs=[
            CampaignCircuitSpec(family="ghz", num_qubits=3),
            CampaignCircuitSpec(family="qft", num_qubits=3),
        ]
    )
    first_case, second_case = campaign.build_cases()
    write_attempts: list[tuple[str, list[str]]] = []
    interrupt_seen = False

    def fake_result_for_case(case: CampaignCase, scenario_name: str, depth: int) -> ScenarioResult:
        return _build_result(scenario_name, case, metrics=_build_metrics(depth))

    def fake_write_outputs(*, output_dir, report):
        nonlocal interrupt_seen
        del output_dir
        statuses = [case_report.status for case_report in report.case_reports]
        write_attempts.append((report.campaign_status, statuses))
        if report.campaign_status == "running" and not interrupt_seen:
            interrupt_seen = True
            raise KeyboardInterrupt()
        return None

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: fake_result_for_case(first_case, "Baseline", 100),
        run_mo_only=lambda request, *, circuit: fake_result_for_case(first_case, "MO_Only", 90),
        train_case_fn=lambda **kwargs: _build_training_result(kwargs["campaign_case"]),
        run_mo_rl=lambda request, *, circuit, injected_layout: fake_result_for_case(first_case, "MO+RL", 80),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=fake_write_outputs,
    )

    assert write_attempts == [
        ("running", ["completed"]),
        ("interrupted", ["completed", "cancelled"]),
    ]
    assert [case_report.status for case_report in report.case_reports] == ["completed", "cancelled"]
    assert report.case_reports[0].case.case_id == first_case.case_id
    assert report.case_reports[0].baseline_result.scenario_name == "Baseline"
    assert report.case_reports[0].mo_only_result.scenario_name == "MO_Only"
    assert report.case_reports[0].mo_rl_result.scenario_name == "MO+RL"
    assert report.case_reports[1].case.case_id == second_case.case_id
    assert campaign.status == "interrupted"
    assert campaign.summary.status == "interrupted"
    assert campaign.summary.cancelled_cases == 1


def test_run_campaign_aligns_training_case_anchor_with_persisted_case_directory(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign
    from src.integration.campaign_reporting import write_campaign_outputs

    campaign = _build_campaign(
        circuit_specs=[CampaignCircuitSpec(family="ghz..bad", num_qubits=3)],
        backend_names=["fake_torino"],
    )
    case = campaign.build_cases()[0]
    captured_case_output_dirs: list[Path] = []

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, target_circuit, coupling_map, initial_layout
        assert campaign_case == case
        case_output_path = Path(case_output_dir)
        captured_case_output_dirs.append(case_output_path)
        artifact_path = case_output_path / "training" / "models" / "run-001" / "best_model.zip"
        return TrainingBridgeResult(
            status="completed",
            selected_artifact_path=artifact_path,
            best_model_path=artifact_path,
            final_model_path=artifact_path.parent / "final_model.zip",
            run_model_dir=artifact_path.parent,
            run_log_dir=case_output_path / "training" / "logs" / "run-001",
            effective_training_config=TrainingConfigSummary(
                algorithm="MaskablePPO",
                total_timesteps=5000,
                frontier_mode="dag",
                lookahead_window=12,
                max_steps=256,
                seed=42,
            ),
        )

    run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: _make_case_circuit(),
        run_baseline=lambda request, *, circuit: _build_result("Baseline", case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: _build_result("MO_Only", case, metrics=_build_metrics(90)),
        train_case_fn=fake_train_case,
        run_mo_rl=lambda request, *, circuit, injected_layout: _build_result("MO+RL", case, metrics=_build_metrics(80)),
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2)],
        ),
        write_outputs=write_campaign_outputs,
    )

    assert captured_case_output_dirs == [tmp_path / "campaigns" / campaign.campaign_id / "cases" / "ghz_bad_3__fake_torino"]
    assert (captured_case_output_dirs[0] / "result.json").exists()
