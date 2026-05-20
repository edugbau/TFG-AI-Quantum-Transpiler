from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def assert_contains_all(text: str, expected_tokens: tuple[str, ...]) -> None:
    for token in expected_tokens:
        assert token in text


def test_root_readme_summarizes_integration_boundary() -> None:
    repo_readme_text = read_text("README.md")

    assert_contains_all(
        repo_readme_text,
        (
            "qiskit_interface",
            "mo_module",
            "rl_module",
            "integration",
            "Scenario",
            "Campaign",
            "masked routing",
            "run_metadata.json",
            "trans_active_qubits",
        ),
    )


def test_integration_docs_describe_scenarios_and_campaigns() -> None:
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert_contains_all(
        integration_readme_text,
        (
            "Scenario",
            "Campaign",
            "Baseline",
            "MO_Only",
            "RL_Only",
            "MO+RL",
            "SyntheticTopologySpec",
            "mo_effort_mode",
            "load_campaign_batch",
            "run_campaign_batch",
            "run_metadata.json",
            "path-expanded routing subgraph",
            "fake_torino",
            "fake_brisbane",
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


def test_integration_docs_describe_rl_metadata_sidecar_contract() -> None:
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert_contains_all(
        integration_readme_text,
        (
            "masked routing",
            "legacy",
            "run_metadata.json",
            "Training Artifact",
        ),
    )
    assert_contains_all(
        internal_doc_text,
        (
            "masked routing",
            "legacy",
            "run_metadata.json",
            "ScenarioResult",
        ),
    )


def test_integration_package_docstring_keeps_routing_v1_scope() -> None:
    integration_text = read_text("src/integration/__init__.py")

    assert_contains_all(
        integration_text,
        ("routing-evaluation v1", "RL-based scenarios rebuild routed circuits"),
    )
