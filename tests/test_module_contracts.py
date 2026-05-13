import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def assert_contains_all(text: str, expected_tokens: tuple[str, ...]) -> None:
    for token in expected_tokens:
        assert token in text


def assert_excludes_all(text: str, forbidden_tokens: tuple[str, ...]) -> None:
    for token in forbidden_tokens:
        assert token not in text


def assert_any_contains(text: str, alternatives: tuple[str, ...]) -> None:
    assert any(option in text for option in alternatives)


def assert_mentions_initial_layout_from_caller(text: str) -> None:
    assert "initial_layout" in text
    assert_any_contains(
        text,
        (
            "suministrados por el llamador",
            "suministrado por el llamador",
            "desde el llamador",
        ),
    )


def assert_mentions_fixed_session_ref_point(text: str) -> None:
    lowered = text.lower()

    assert_contains_all(text, ("session_ref_point", "calibrated", "manual", "HV=0.0"))
    assert "trial" in lowered
    assert "seed" in lowered
    assert "recalcul" in lowered
    assert "warning" in lowered or "aviso" in lowered
    assert "promedi" in lowered or "media" in lowered
    assert "1.3x" in text or "30 %" in text or "30%" in text
    assert "ValueError" not in text
    assert "fijo" in lowered or "fija" in lowered
    assert "resultado/frente" in lowered


def assert_mentions_event_specific_payloads(text: str) -> None:
    lowered = text.lower()

    assert_contains_all(
        text,
        (
            "calibration_started",
            "calibration_progress",
            "calibration_completed",
            "trial_completed",
            "tuning_completed",
        ),
    )
    assert "payload" in lowered or "evento" in lowered
    assert "campos relevantes" in lowered or "espec" in lowered
    assert "mismo payload" in lowered or "payloads son específicos" in lowered
    assert "no todos" in lowered or "cada evento" in lowered


def assert_mentions_calibration_progress_fields(text: str) -> None:
    assert "calibration_progress" in text
    assert_contains_all(text, ("current_step", "total_steps", "config", "ref_point_candidate"))


def assert_mentions_trial_completed_fields(text: str) -> None:
    assert "trial_completed" in text
    assert_contains_all(text, ("score", "best_score", "params", "ref_point"))


def assert_mentions_supported_crossover_options(text: str) -> None:
    lowered = text.lower()

    assert_contains_all(text, ("crossover_operator", "DPXCrossover", "LayoutCrossover"))
    assert "`dpx`" in lowered
    assert "`ox`" in lowered
    assert "por defecto" in lowered
    assert_any_contains(lowered, ("alternativa", "opcional"))
    assert "prob_crossover" not in text


def assert_integration_v1_doc_scope(text: str) -> None:
    assert_contains_all(text, ("Baseline", "MO_Only", "RL_Only", "MO+RL"))

    lowered = text.lower()
    assert "routing" in lowered
    assert "qiskit-facing scenarios" in lowered
    assert "qasm input is available" in lowered
    assert "rebuilds the routed circuit" in lowered
    assert "post-routing qiskit stages" in lowered
    assert "deferred" in lowered
    assert "qasm" in lowered
    assert "backend catalog is intentionally limited" in lowered
    assert "fake backends" in lowered


def assert_mentions_masked_routing_regime(text: str) -> None:
    lowered = text.lower()

    assert "masked routing" in lowered
    assert "sabre" in lowered
    assert "fixed" in lowered or "fijo" in lowered or "fija" in lowered
    assert "coupling" in lowered
    assert "action_masks()" in text
    assert "deterministic" in lowered or "determinista" in lowered
    assert "frontier-aware" in lowered
    assert "hard mask" in lowered
    assert "maskableppo" in lowered or "maskableppo" in text
    assert "legacy ppo/dqn" in lowered or ("legacy" in lowered and "ppo" in lowered and "dqn" in lowered)
    assert "checkpoint" in lowered


def iter_python_files(relative_dir: str):
    return (ROOT / relative_dir).rglob("*.py")


def assert_module_tree_has_no_imports(relative_dir: str, forbidden_modules: tuple[str, ...]) -> None:
    forbidden_prefixes = forbidden_modules + tuple(
        f"src.{module}" for module in forbidden_modules
    )

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

                    if module_name == "src":
                        for alias in node.names:
                            assert alias.name not in forbidden_modules, (
                                f"{python_file} imports forbidden module src.{alias.name}"
                            )
                elif node.level > 0:
                    for alias in node.names:
                        assert alias.name not in forbidden_modules, (
                            f"{python_file} imports forbidden relative module {alias.name}"
                        )


def test_docs_and_workspace_metadata_define_four_module_ownership() -> None:
    workspace_agents_text = read_text(".github/AGENTS.md")
    readme_text = read_text("README.md")

    assert_contains_all(
        workspace_agents_text,
        (
            "The project has 4 interconnected modules:",
            "src/integration/",
            "benchmark scenarios",
        ),
    )
    assert_contains_all(workspace_agents_text, ("Module boundaries are respected", "integration owns"))
    assert_contains_all(workspace_agents_text, ("orchestration", "handoff scenarios"))
    assert_contains_all(
        readme_text,
        (
            "Proyecto de transpilación cuántica organizado en cuatro módulos",
            "MO y RL evolucionan como módulos separados",
            "MO+RL",
        ),
    )
    assert_contains_all(readme_text, ("integration`", "handoff", "orquestación"))
    assert_excludes_all(
        workspace_agents_text,
        (
            "Pipeline orchestration: MO layouts feed RL agent",
            "Qiskit interface → MO optimization → RL synthesis → integration orchestration",
        ),
    )
    assert_excludes_all(
        readme_text,
        (
            "Pipeline híbrido de transpilación de circuitos cuánticos que combina",
            "recibe los layouts optimizados como entrada",
        ),
    )


def test_qiskit_interface_docs_keep_initial_layout_generic() -> None:
    transpiler_text = read_text("src/qiskit_interface/transpiler.py")
    qiskit_readme_text = read_text("src/qiskit_interface/README.md")

    assert_mentions_initial_layout_from_caller(transpiler_text)
    assert "initial_layout" in qiskit_readme_text
    assert_contains_all(
        qiskit_readme_text,
        (
            "load_circuit(",
            "`qasm3`",
            "`auto`",
            "fake_torino",
            "fake_sherbrooke",
            "fake_brisbane",
            "Baseline",
            "MO_Only",
        ),
    )
    assert_any_contains(
        qiskit_readme_text.lower(),
        ("resúmenes de episodio", "resumenes de episodio"),
    )
    assert "helper de evaluación local" in qiskit_readme_text
    assert "src/integration/" in transpiler_text
    assert "No implementa la integración MO -> RL" in transpiler_text

    assert_excludes_all(
        transpiler_text,
        (
            "layout inicial personalizado (del módulo MO)",
            "layouts del módulo MO",
            "puente principal",
            "híbrido MO+RL",
        ),
    )
    assert_excludes_all(
        qiskit_readme_text,
        ("Función puente para el Módulo MO",),
    )


def test_mo_docs_route_layout_consumption_through_integration() -> None:
    mo_doc_text = read_text("src/mo_module/docs/internal_documentation.md")

    assert "consumibles por el módulo `integration`" in mo_doc_text
    assert "escenarios `MO_Only` y futuros flujos `MO+RL`" in mo_doc_text
    assert "Salida hacia el módulo `rl_module`" not in mo_doc_text


def test_mo_docs_keep_mo_to_rl_handoff_exclusive_to_integration() -> None:
    internal_text = read_text("src/mo_module/docs/internal_documentation.md")
    benchmark_text = read_text("src/mo_module/docs/benchmark_documentation.md")

    assert_contains_all(
        internal_text,
        (
            "consumibles por el módulo `integration`",
            "no debe comunicarse directamente con `rl_module`",
            "handoff MO -> RL pertenece exclusivamente a `src/integration/`",
        ),
    )
    assert_contains_all(
        benchmark_text,
        (
            "tooling experimental local al módulo MO",
            "no actúan como puente de orquestación hacia `rl_module`",
            "src/integration/",
        ),
    )


def test_mo_benchmark_docs_cover_phase3_layout_campaign_presets() -> None:
    benchmark_text = read_text("src/mo_module/docs/benchmark_documentation.md")
    campaigns_text = read_text("src/mo_module/benchmark/layout_campaigns.py")

    assert_contains_all(
        benchmark_text,
        (
            "layout_campaigns.py",
            "run_layout_selection_campaign",
            "build_reference_layouts",
            "quick",
            "balanced",
            "thorough",
            "reverse_trivial",
            "high_index_block",
            "heaviest_hex",
            "tooling experimental local al módulo MO",
            "no actúan como puente de orquestación hacia `rl_module`",
        ),
    )
    assert_contains_all(
        campaigns_text,
        (
            "preset",
            "quick",
            "balanced",
            "thorough",
            "heaviest_hex",
        ),
    )


def test_mo_tuning_docs_keep_fixed_session_ref_point_contract() -> None:
    tuning_doc_text = read_text("src/mo_module/docs/tuning.md")
    internal_doc_text = read_text("src/mo_module/docs/internal_documentation.md")

    for text in (tuning_doc_text, internal_doc_text):
        assert_mentions_fixed_session_ref_point(text)


def test_mo_docs_describe_event_specific_progress_payloads() -> None:
    tuning_doc_text = read_text("src/mo_module/docs/tuning.md")
    mo_doc_text = read_text("src/mo_module/docs/internal_documentation.md")

    assert_mentions_event_specific_payloads(mo_doc_text)
    assert_mentions_calibration_progress_fields(tuning_doc_text)
    assert_mentions_trial_completed_fields(tuning_doc_text)


def test_mo_generation_docs_match_supported_crossover_contract() -> None:
    generation_doc_text = read_text("src/mo_module/docs/generacion_soluciones.md")

    assert_mentions_supported_crossover_options(generation_doc_text)


def test_rl_docs_and_reset_contract_keep_initial_layout_generic() -> None:
    rl_environment_text = read_text("src/rl_module/environment.py")
    rl_lookahead_text = read_text("src/rl_module/docs/lookahead_frontier.md")
    rl_internal_text = read_text("src/rl_module/docs/internal_documentation.md")

    assert_contains_all(
        rl_environment_text,
        (
            "Permite inyectar un `initial_layout` externo a través de `options`.",
            "# Ingesta genérica de layout inicial desde el llamador",
        ),
    )
    assert_contains_all(
        rl_lookahead_text,
        (
            "Si se inyecta `initial_layout`, el entorno lo respeta exactamente.",
            "El productor del `initial_layout` es externo al módulo; el handoff MO -> RL pertenecerá a `src/integration/`.",
        ),
    )
    assert_contains_all(
        rl_internal_text,
        (
            "input externo",
            "handoff MO -> RL",
            "src/integration/",
        ),
    )
    assert_excludes_all(
        rl_environment_text,
        ("desde el Módulo MO",),
    )
    assert_excludes_all(
        rl_internal_text,
        (
            "generado típicamente por el módulo de Optimización Multiobjetivo (MO)",
            "traído desde el Algoritmo Genético Multiobjetivo",
        ),
    )


def test_integration_docs_declare_routing_v1_scope_and_rl_reconstruction_scope() -> None:
    integration_text = read_text("src/integration/__init__.py")
    integration_readme_text = read_text("src/integration/README.md")

    assert_contains_all(
        integration_text,
        ("routing-evaluation v1", "RL-based scenarios rebuild routed circuits"),
    )
    assert_integration_v1_doc_scope(integration_readme_text)


def test_masked_routing_docs_describe_public_contracts() -> None:
    repo_readme_text = read_text("README.md")
    rl_internal_text = read_text("src/rl_module/docs/internal_documentation.md")
    rl_lookahead_text = read_text("src/rl_module/docs/lookahead_frontier.md")
    roadmap_text = read_text("src/rl_module/docs/routing_stability_roadmap.md")
    integration_readme_text = read_text("src/integration/README.md")
    integration_internal_text = read_text("src/integration/docs/internal_documentation.md")
    future_iteration_text = read_text(
        "docs/future-iterations/chapter3-branch2-frontier-observation-and-feasible-actions.md"
    )

    for text in (
        repo_readme_text,
        rl_internal_text,
        rl_lookahead_text,
        roadmap_text,
        future_iteration_text,
    ):
        assert_mentions_masked_routing_regime(text)

    assert_contains_all(
        integration_readme_text,
        (
            "versioned masked routing metadata",
            "legacy fallback remains",
            "RL_Only` rebuilds the routed circuit",
            "post-routing Qiskit stages",
        ),
    )
    assert_contains_all(
        integration_internal_text,
        (
            "versioned masked routing metadata",
            "legacy defaults",
            "MaskablePPO",
        ),
    )


def test_mo_module_has_no_direct_rl_imports() -> None:
    assert_module_tree_has_no_imports("src/mo_module", ("rl_module",))


def test_rl_module_has_no_direct_mo_imports() -> None:
    assert_module_tree_has_no_imports("src/rl_module", ("mo_module",))


def test_qiskit_interface_has_no_direct_mo_or_rl_imports() -> None:
    assert_module_tree_has_no_imports("src/qiskit_interface", ("mo_module", "rl_module"))
