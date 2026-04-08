from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_integration_docs_describe_routing_evaluation_v1_scope_and_known_limits() -> None:
    readme_text = read_text("README.md")

    assert "episode summaries" in readme_text
    assert "not final circuits" in readme_text
    assert "future iteration" in readme_text
    assert "QASM" in readme_text
    assert "stub" not in readme_text.lower()
