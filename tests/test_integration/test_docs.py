from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_integration_docs_describe_routing_evaluation_v1_scope_and_known_limits() -> None:
    repo_readme_text = read_text("README.md")
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert "RL_Only" in repo_readme_text
    assert "episode summaries" in repo_readme_text
    assert "MO+RL" in repo_readme_text
    assert "swap_trace" in repo_readme_text
    assert "executed_gate_trace" in repo_readme_text
    assert "QASM input is available for `Baseline` and `MO_Only`" in repo_readme_text
    assert "trans_active_qubits" in repo_readme_text
    assert "total_swaps == len(swap_trace)" in repo_readme_text
    assert "QASM input is also deferred" not in repo_readme_text
    assert "stub" not in repo_readme_text.lower()

    assert "QASM input is available" in integration_readme_text
    assert "Qiskit-facing scenarios" in integration_readme_text
    assert "`RL_Only` returns episode summaries, not final circuits" in integration_readme_text
    assert "`MO+RL` now attempts to reconstruct the routed circuit from the RL trace" in integration_readme_text
    assert "trans_active_qubits" in integration_readme_text

    assert "QASM input is available for the Qiskit-facing scenarios" in internal_doc_text
    assert "qiskit_interface.load_circuit(...)" in internal_doc_text
    assert "--circuit-source" in internal_doc_text
    assert "--circuit-path" in internal_doc_text
    assert "--circuit-format" in internal_doc_text
    assert "Baseline" in internal_doc_text and "MO_Only" in internal_doc_text
    assert "backend catalog is intentionally limited" in internal_doc_text
    assert "executed_gate_trace" in internal_doc_text
    assert "total_swaps == len(swap_trace)" in internal_doc_text
    assert "trans_active_qubits" in internal_doc_text


def test_integration_docs_describe_rl_metadata_sidecar_contract() -> None:
    repo_readme_text = read_text("README.md")
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert "run_metadata.json" in repo_readme_text
    assert "saved routing contract from that sidecar when available" in integration_readme_text
    assert "reports that condition through an extra note" in integration_readme_text
    assert "ScenarioResult.notes" in internal_doc_text
    assert "resolve_routing_model_contract()" in internal_doc_text
    assert "Legacy RL evaluation defaults were used because no run metadata sidecar was found." in internal_doc_text
    assert "metadata_source" not in internal_doc_text
    assert "_load_routing_contract()" not in internal_doc_text


def test_integration_docs_lock_campaign_mo_conditioned_layout_reuse_semantics() -> None:
    repo_readme_text = read_text("README.md")
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert "Cada Campaign Case corresponde a una combinación `circuit x backend` y ejecuta la comparación canónica `Baseline`, `MO_Only` y `MO+RL`." in repo_readme_text
    assert "`MO_Only` es el Scenario que selecciona el layout. El training de Campaign para `MO+RL` arranca desde ese layout exacto y la evaluación posterior de `MO+RL` reutiliza ese mismo layout junto con el Training Artifact producido para el mismo Campaign Case." in repo_readme_text
    assert "En el camino híbrido de Campaign, `MO_Only` selecciona el layout, `integration` lo reenvía como `initial_layout` al training RL y la evaluación `MO+RL` reutiliza ese mismo layout junto con el Training Artifact del caso." in repo_readme_text

    assert "The canonical Campaign comparison set is `Baseline`, `MO_Only`, and `MO+RL`; `RL_Only` remains available as a standalone Scenario outside the primary guided Campaign flow." in integration_readme_text
    assert "Within that guided Campaign comparison, `MO_Only` selects the layout for the Campaign Case. Campaign training for `MO+RL` starts from that exact layout, and `MO+RL` evaluation reuses the same layout together with the resulting Training Artifact for the same Campaign Case." in integration_readme_text
    assert "For the Campaign hybrid path, the sequence is explicit: `MO_Only` selects the layout, Campaign training produces the Training Artifact starting from that exact layout, and `MO+RL` evaluation uses the same layout and that artifact when it runs the routed comparison." in integration_readme_text

    assert "El conjunto canónico de comparación dentro de esa Campaign es `Baseline`, `MO_Only` y `MO+RL`. `RL_Only` sigue existiendo como Scenario, pero queda fuera del flujo guiado principal de Campaign." in internal_doc_text
    assert "Dentro de ese flujo guiado, `MO_Only` es el Scenario que selecciona el layout del Campaign Case. El training de Campaign para `MO+RL` arranca desde ese layout exacto, produce el Training Artifact del caso y la evaluación posterior de `MO+RL` reutiliza tanto ese mismo layout como ese artifacto resultante." in internal_doc_text
    assert "usa el layout exacto seleccionado por `MO_Only` para lanzar el training RL del camino `MO+RL` y reutiliza ese mismo layout en la evaluación híbrida del caso;" in internal_doc_text
