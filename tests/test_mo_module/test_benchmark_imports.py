import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _last_nonempty_stdout_line(result: subprocess.CompletedProcess[str]) -> str:
    return next(
        (line.strip() for line in reversed(result.stdout.splitlines()) if line.strip()),
        "",
    )


class TestBenchmarkImportIsolation:
    def test_importing_mo_module_does_not_eagerly_load_benchmark_analysis(self):
        result = _run_python(
            """
            import sys
            import src.mo_module

            print(f"analysis_loaded={('src.mo_module.benchmark.analysis' in sys.modules)}")
            """
        )

        assert result.returncode == 0, result.stderr
        assert _last_nonempty_stdout_line(result) == "analysis_loaded=False"

    def test_importing_benchmark_gui_does_not_eagerly_load_benchmark_analysis(self):
        result = _run_python(
            """
            import sys
            import src.mo_module.benchmark.benchmark_gui

            print(f"analysis_loaded={('src.mo_module.benchmark.analysis' in sys.modules)}")
            """
        )

        assert result.returncode == 0, result.stderr
        assert _last_nonempty_stdout_line(result) == "analysis_loaded=False"
