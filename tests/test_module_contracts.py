from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_agents_exists_and_describes_four_modules():
    agents_doc = ROOT / "docs" / "agents.md"
    assert agents_doc.exists(), "README y .github/AGENTS.md apuntan a docs/agents.md"

    text = agents_doc.read_text(encoding="utf-8")
    for token in (
        "src/qiskit_interface/",
        "src/rl_module/",
        "src/mo_module/",
        "src/integration/",
    ):
        assert token in text

    transpiler_text = (ROOT / "src" / "qiskit_interface" / "transpiler.py").read_text(
        encoding="utf-8"
    )
    assert "evaluación local de layouts suministrados por el llamador" in transpiler_text
    assert "No implementa la integración MO -> RL" in transpiler_text
    assert "puente principal" not in transpiler_text
    assert "híbrido MO+RL" not in transpiler_text


def test_readme_architecture_reference_points_to_real_doc():
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[agents.md](docs/agents.md)" in readme_text
    assert (ROOT / "docs" / "agents.md").exists()

    qiskit_readme_text = (ROOT / "src" / "qiskit_interface" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "helper de evaluación local" in qiskit_readme_text
    assert "Función puente para el Módulo MO" not in qiskit_readme_text
