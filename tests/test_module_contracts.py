import ast
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
    assert "layout inicial personalizado (para recibir layouts\n    suministrados por el llamador)" in transpiler_text
    assert "No implementa la integración MO -> RL" in transpiler_text
    assert "layouts del módulo MO" not in transpiler_text
    assert "puente principal" not in transpiler_text
    assert "híbrido MO+RL" not in transpiler_text

    workspace_agents_text = (ROOT / ".github" / "AGENTS.md").read_text(encoding="utf-8")
    assert "The project has 4 interconnected modules:" in workspace_agents_text
    assert "Orchestration of handoff and benchmark scenarios across modules" in workspace_agents_text
    assert "Pipeline orchestration: MO layouts feed RL agent" not in workspace_agents_text
    assert (
        "Module boundaries are respected: integration owns orchestration and handoff scenarios across the other modules."
        in workspace_agents_text
    )
    assert (
        "Qiskit interface → MO optimization → RL synthesis → integration orchestration"
        not in workspace_agents_text
    )


def test_readme_architecture_reference_points_to_real_doc():
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[agents.md](docs/agents.md)" in readme_text
    assert (ROOT / "docs" / "agents.md").exists()
    assert "Proyecto de transpilación cuántica organizado en cuatro módulos" in readme_text
    assert "MO y RL evolucionan como módulos separados" in readme_text
    assert "integration` posee el futuro handoff y la orquestación" in readme_text
    assert "Pipeline híbrido de transpilación de circuitos cuánticos que combina" not in readme_text
    assert "recibe los layouts optimizados como entrada" not in readme_text

    qiskit_readme_text = (ROOT / "src" / "qiskit_interface" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "helper de evaluación local" in qiskit_readme_text
    assert "Función puente para el Módulo MO" not in qiskit_readme_text


def test_mo_docs_route_future_rl_consumption_through_integration():
    mo_doc_text = (ROOT / "src" / "mo_module" / "docs" / "internal_documentation.md").read_text(
        encoding="utf-8"
    )
    assert "Salida hacia el módulo `rl_module`" not in mo_doc_text
    assert "consumibles por el módulo `integration`" in mo_doc_text
    assert (
        "`get_compromise_layout()` y `get_best_layout()` proporcionan un layout único para escenarios `MO_Only` y futuros flujos `MO+RL`."
        in mo_doc_text
    )


def test_rl_environment_reset_docstring_is_source_agnostic():
    rl_environment_text = (ROOT / "src" / "rl_module" / "environment.py").read_text(
        encoding="utf-8"
    )
    assert "Permite inyectar un `initial_layout` externo a través de `options`." in rl_environment_text
    assert "# Ingesta genérica de layout inicial desde el llamador" in rl_environment_text
    assert "desde el Módulo MO" not in rl_environment_text


def test_rl_frontier_docs_keep_initial_layout_generic():
    rl_lookahead_text = (
        ROOT / "src" / "rl_module" / "docs" / "lookahead_frontier.md"
    ).read_text(encoding="utf-8")
    assert (
        "- El productor del `initial_layout` es externo al módulo; el handoff MO -> RL pertenecerá a `src/integration/`."
        in rl_lookahead_text
    )


def test_integration_stub_declares_handoff_ownership():
    integration_text = (ROOT / "src" / "integration" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert '"""Módulo 4: Integración y experimentación.' in integration_text
    assert "único dueño del handoff MO -> RL" in integration_text
    assert "Estado actual: stub." in integration_text
    assert "__all__: list[str] = []" in integration_text


def test_mo_module_has_no_direct_rl_imports():
    for mo_python_file in (ROOT / "src" / "mo_module").rglob("*.py"):
        mo_python_tree = ast.parse(mo_python_file.read_text(encoding="utf-8"))
        for node in ast.walk(mo_python_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "rl_module"
                    assert not alias.name.startswith("rl_module.")
                    assert alias.name != "src.rl_module"
                    assert not alias.name.startswith("src.rl_module.")
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    assert node.module != "rl_module"
                    assert not node.module.startswith("rl_module.")
                    if node.level > 0:
                        assert node.module != "rl_module"
                        assert not node.module.startswith("rl_module.")
                    assert node.module != "src.rl_module"
                    assert not node.module.startswith("src.rl_module.")
                    if node.module == "src":
                        for alias in node.names:
                            assert alias.name != "rl_module"
                elif node.level > 0:
                    for alias in node.names:
                        assert alias.name != "rl_module"


def test_rl_module_has_no_direct_mo_imports():
    for rl_python_file in (ROOT / "src" / "rl_module").rglob("*.py"):
        rl_python_tree = ast.parse(rl_python_file.read_text(encoding="utf-8"))
        for node in ast.walk(rl_python_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "mo_module"
                    assert not alias.name.startswith("mo_module.")
                    assert alias.name != "src.mo_module"
                    assert not alias.name.startswith("src.mo_module.")
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    assert node.module != "mo_module"
                    assert not node.module.startswith("mo_module.")
                    if node.level > 0:
                        assert node.module != "mo_module"
                        assert not node.module.startswith("mo_module.")
                    assert node.module != "src.mo_module"
                    assert not node.module.startswith("src.mo_module.")
                    if node.module == "src":
                        for alias in node.names:
                            assert alias.name != "mo_module"
                elif node.level > 0:
                    for alias in node.names:
                        assert alias.name != "mo_module"
