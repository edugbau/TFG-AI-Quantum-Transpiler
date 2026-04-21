from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_integration_docs_describe_routing_evaluation_v1_scope_and_known_limits() -> None:
    repo_readme_text = read_text("README.md")
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert "episode summaries" in repo_readme_text
    assert "not final circuits" in repo_readme_text
    assert "QASM input is available for `Baseline` and `MO_Only`" in repo_readme_text
    assert "QASM input is also deferred" not in repo_readme_text
    assert "stub" not in repo_readme_text.lower()

    assert "QASM input is available" in integration_readme_text
    assert "Qiskit-facing scenarios" in integration_readme_text
    assert "episode summaries, not final circuits" in integration_readme_text

    assert "QASM input is available for the Qiskit-facing scenarios" in internal_doc_text
    assert "qiskit_interface.load_circuit(...)" in internal_doc_text
    assert "--circuit-source" in internal_doc_text
    assert "--circuit-path" in internal_doc_text
    assert "--circuit-format" in internal_doc_text
    assert "Baseline" in internal_doc_text and "MO_Only" in internal_doc_text
    assert "backend catalog is intentionally limited" in internal_doc_text


def test_integration_docs_describe_rl_metadata_sidecar_contract() -> None:
    repo_readme_text = read_text("README.md")
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert "run_metadata.json" in repo_readme_text
    assert "saved routing contract from that sidecar when available" in integration_readme_text
    assert "reports that condition through an extra note" in integration_readme_text
    assert "ScenarioResult.notes" in internal_doc_text
    assert "Legacy RL evaluation defaults were used because no run metadata sidecar was found." in internal_doc_text
    assert "metadata_source" not in internal_doc_text
