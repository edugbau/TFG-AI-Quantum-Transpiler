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
    assert "`src/integration/` owns MO->RL orchestration" in text


def test_readme_architecture_reference_points_to_real_doc():
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    github_agents_text = (ROOT / ".github" / "AGENTS.md").read_text(encoding="utf-8")
    assert "[agents.md](docs/agents.md)" in readme_text
    assert "../docs/agents.md" in github_agents_text
    assert (ROOT / "docs" / "agents.md").exists()
