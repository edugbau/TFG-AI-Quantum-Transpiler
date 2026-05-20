import ast
from pathlib import Path

import pytest

import src.integration as integration
from src.integration import LayoutSelectionPolicy, ScenarioRequest
from src.integration.synthetic_topology import SyntheticTopologySpec


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def assert_contains_all(text: str, expected_tokens: tuple[str, ...]) -> None:
    for token in expected_tokens:
        assert token in text


def assert_excludes_all(text: str, forbidden_tokens: tuple[str, ...]) -> None:
    for token in forbidden_tokens:
        assert token not in text


def iter_python_files(relative_dir: str):
    return (ROOT / relative_dir).rglob("*.py")


def assert_module_tree_has_no_imports(relative_dir: str, forbidden_modules: tuple[str, ...]) -> None:
    forbidden_prefixes = forbidden_modules + tuple(f"src.{module}" for module in forbidden_modules)

    for python_file in iter_python_files(relative_dir):
        tree = ast.parse(python_file.read_text(encoding="utf-8"), filename=str(python_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(
                        alias.name == module or alias.name.startswith(f"{module}.")
                        for module in forbidden_prefixes
                    ), f"{python_file} imports forbidden module {alias.name!r}"
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module
                if module_name is not None:
                    assert not any(
                        module_name == module or module_name.startswith(f"{module}.")
                        for module in forbidden_prefixes
                    ), f"{python_file} imports forbidden module {module_name!r}"
                elif node.level > 0:
                    for alias in node.names:
                        assert alias.name not in forbidden_modules, (
                            f"{python_file} imports forbidden relative module {alias.name}"
                        )


def test_scenario_request_defaults_and_normalization() -> None:
    request = ScenarioRequest(
        scenario_name="MO_Only",
        circuit_name="ghz_5",
        num_qubits=5,
        backend_name="fake_backend",
        layout_policy="compromise",
    )

    assert request.seed == 42
    assert request.layout_policy is LayoutSelectionPolicy.COMPROMISE
    assert request.circuit_source.value == "library"
    assert request.circuit_format.value == "auto"


def test_scenario_request_accepts_matching_synthetic_topology() -> None:
    synthetic_topology = SyntheticTopologySpec(shape="line", num_qubits=5)

    request = ScenarioRequest(
        scenario_name="MO_Only",
        circuit_name="ghz_5",
        num_qubits=5,
        backend_name=synthetic_topology.backend_name,
        synthetic_topology=synthetic_topology,
    )

    assert request.synthetic_topology is synthetic_topology


def test_scenario_request_rejects_invalid_baseline_inputs() -> None:
    with pytest.raises(ValueError):
        ScenarioRequest(
            scenario_name="Baseline",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
            initial_layout=[0, 1, 2],
        )


def test_scenario_request_requires_rl_model_for_rl_only() -> None:
    with pytest.raises(ValueError):
        ScenarioRequest(
            scenario_name="RL_Only",
            circuit_name="ghz",
            num_qubits=3,
            backend_name="fake_torino",
        )


def test_public_integration_contracts_do_not_export_scenario_name() -> None:
    assert integration.__all__ == [
        "LayoutSelectionPolicy",
        "RoutingEpisodeSummary",
        "ScenarioRequest",
        "ScenarioResult",
    ]
    assert not hasattr(integration, "ScenarioName")


def test_docs_and_workspace_metadata_define_four_module_ownership() -> None:
    workspace_agents_text = read_text(".github/AGENTS.md")
    readme_text = read_text("README.md")

    assert_contains_all(
        workspace_agents_text,
        (
            "The project has 4 interconnected modules:",
            "src/integration/",
            "Module boundaries are respected",
            "integration owns",
        ),
    )
    assert_contains_all(
        readme_text,
        (
            "Como leer el proyecto",
            "src/qiskit_interface/README.md",
            "src/mo_module/README.md",
            "src/rl_module/README.md",
            "src/integration/README.md",
            "Scenario",
            "Campaign",
            "layout[i] = physical_qubit_for_logical_qubit_i",
        ),
    )


def test_qiskit_interface_docs_define_central_contracts() -> None:
    transpiler_text = read_text("src/qiskit_interface/transpiler.py")
    qiskit_readme_text = read_text("src/qiskit_interface/README.md")

    assert_contains_all(
        qiskit_readme_text,
        (
            "CircuitMetrics",
            "BackendInfo",
            "TranspilationResult",
            "two_qubit_gates",
            "cnot_equivalent",
            "load_circuit",
            "fake_torino",
            "fake_brisbane",
            "transpile_post_routing",
            "run_baseline",
            "run_named_baseline",
        ),
    )
    assert_contains_all(qiskit_readme_text, ("initial_layout", "mo_module", "integration"))
    assert "initial_layout" in transpiler_text
    assert "src/integration/" in transpiler_text


def test_mo_docs_define_layout_flow_and_local_benchmarking() -> None:
    mo_readme_text = read_text("src/mo_module/README.md")

    assert_contains_all(
        mo_readme_text,
        (
            "encoding",
            "fitness",
            "optimizer",
            "pareto",
            "tuning",
            "benchmark",
            "layout_campaigns",
            "local",
            "handoff MO -> RL",
        ),
    )


def test_mo_appendices_keep_tuning_and_campaign_details() -> None:
    mo_readme_text = read_text("src/mo_module/README.md")

    assert_contains_all(
        mo_readme_text,
        (
            "docs/tuning.md",
            "docs/benchmark_documentation.md",
            "docs/analisis_resultados.md",
            "docs/generacion_soluciones.md",
        ),
    )


def test_rl_docs_define_routing_synthesis_and_metadata_contracts() -> None:
    rl_readme_text = read_text("src/rl_module/README.md")
    rl_internal_text = read_text("src/rl_module/docs/internal_documentation.md")

    assert_contains_all(
        rl_readme_text,
        (
            "routing",
            "masked routing",
            "synthesis",
            "training",
            "model_metadata",
            "RLBenchmarkGUI",
            "integration",
            "MaskablePPO",
        ),
    )
    assert_contains_all(
        rl_internal_text,
        (
            "run_metadata.json",
            "synthesis_primitives",
            "synthesis_clifford",
            "MaskablePPO",
            "integration",
        ),
    )


def test_integration_docs_define_scenario_and_campaign_layers() -> None:
    integration_text = read_text("src/integration/__init__.py")
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert_contains_all(
        integration_text,
        ("routing-evaluation v1", "RL-based scenarios rebuild routed circuits"),
    )
    assert_contains_all(
        integration_readme_text,
        (
            "Scenario",
            "Campaign",
            "SyntheticTopologySpec",
            "mo_effort_mode",
            "load_campaign_batch",
            "run_campaign_batch",
            "run_metadata.json",
            "path-expanded routing subgraph",
            "fake_torino",
            "fake_brisbane",
            "Baseline",
            "MO_Only",
            "RL_Only",
            "MO+RL",
        ),
    )
    assert_contains_all(
        internal_doc_text,
        (
            "ScenarioRequest",
            "CampaignConfig",
            "SyntheticTopologySpec",
            "run_campaign_batch",
            "run_metadata.json",
            "path-expanded routing subgraph",
        ),
    )


def test_masked_routing_docs_describe_public_contracts() -> None:
    repo_readme_text = read_text("README.md")
    rl_readme_text = read_text("src/rl_module/README.md")
    rl_internal_text = read_text("src/rl_module/docs/internal_documentation.md")
    integration_readme_text = read_text("src/integration/README.md")
    integration_internal_text = read_text("src/integration/docs/internal_documentation.md")

    assert_contains_all(repo_readme_text, ("masked routing", "MaskablePPO", "run_metadata.json"))

    for text in (rl_readme_text, rl_internal_text):
        assert_contains_all(
            text,
            (
                "masked routing",
                "action_masks()",
                "frontier-aware",
                "MaskablePPO",
                "legacy",
            ),
        )

    assert_contains_all(
        integration_readme_text,
        (
            "masked routing",
            "legacy",
            "run_metadata.json",
            "Campaign",
        ),
    )
    assert_contains_all(
        integration_internal_text,
        (
            "masked routing",
            "legacy",
            "run_metadata.json",
            "Campaign",
        ),
    )


def test_mo_module_has_no_direct_rl_imports() -> None:
    assert_module_tree_has_no_imports("src/mo_module", ("rl_module",))


def test_rl_module_has_no_direct_mo_imports() -> None:
    assert_module_tree_has_no_imports("src/rl_module", ("mo_module",))


def test_qiskit_interface_has_no_direct_mo_or_rl_imports() -> None:
    assert_module_tree_has_no_imports("src/qiskit_interface", ("mo_module", "rl_module"))
