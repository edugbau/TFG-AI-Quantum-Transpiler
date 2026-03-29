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

    mo_doc_text = (ROOT / "src" / "mo_module" / "docs" / "internal_documentation.md").read_text(
        encoding="utf-8"
    )
    assert "Salida hacia el módulo `rl_module`" not in mo_doc_text
    assert "consumibles por el módulo `integration`" in mo_doc_text

    rl_lookahead_text = (
        ROOT / "src" / "rl_module" / "docs" / "lookahead_frontier.md"
    ).read_text(encoding="utf-8")
    assert "productor del `initial_layout` es externo al módulo" in rl_lookahead_text
    assert "MO -> RL pertenecerá a `src/integration/`" in rl_lookahead_text

    for mo_python_file in (ROOT / "src" / "mo_module").rglob("*.py"):
        mo_python_text = mo_python_file.read_text(encoding="utf-8")
        assert "from src.rl_module" not in mo_python_text
        assert "import src.rl_module" not in mo_python_text
        assert "from ..rl_module" not in mo_python_text


def test_readme_architecture_reference_points_to_real_doc():
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[agents.md](docs/agents.md)" in readme_text
    assert (ROOT / "docs" / "agents.md").exists()

    qiskit_readme_text = (ROOT / "src" / "qiskit_interface" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "helper de evaluación local" in qiskit_readme_text
    assert "Función puente para el Módulo MO" not in qiskit_readme_text

    rl_environment_text = (ROOT / "src" / "rl_module" / "environment.py").read_text(
        encoding="utf-8"
    )
    assert "layout inicial externo" in rl_environment_text
    assert "desde el Módulo MO" not in rl_environment_text

    integration_text = (ROOT / "src" / "integration" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "único dueño del handoff MO -> RL" in integration_text
    assert "stub" in integration_text.lower()
