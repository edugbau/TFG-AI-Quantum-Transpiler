# Campaign MO+RL Path-Expanded Subgraph Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Campaign `MO+RL` derive a path-expanded routing subgraph from the `MO_Only` layout and use it as the default routing graph for both RL training and evaluation, while preserving non-Campaign scenario behavior.

**Architecture:** Add a small `integration` helper that derives deterministic shortest-path closures from interacting logical pairs, then thread that derived coupling graph through `campaign_runner` and the internal Campaign-only `run_mo_rl_scenario(...)` seam. Surface the chosen routing-graph mode and counts through `ScenarioResult.notes` and Campaign reporting so the case outputs remain auditable.

**Tech Stack:** Python 3.10, pytest, Qiskit `QuantumCircuit`, existing `src/integration/` Campaign orchestration, existing RL environment coupling-map contract.

---

## File Structure

- Create: `src/integration/routing_subgraph.py`
  - Responsibility: derive the Campaign `MO+RL` routing graph from `circuit`, `selected_layout`, and backend `coupling_edges`; own deterministic shortest-path tie-breaking and fallback metadata.
- Modify: `src/integration/campaign_runner.py`
  - Responsibility: derive the Campaign hybrid routing graph after `MO_Only`, pass derived `coupling_map` to RL training, and pass the same derived graph into Campaign `MO+RL` evaluation.
- Modify: `src/integration/scenarios.py`
  - Responsibility: accept Campaign-only injected coupling edges and routing-graph metadata for `MO+RL`, while keeping public single-scenario behavior unchanged.
- Modify: `src/integration/campaign_reporting.py`
  - Responsibility: render the new `MO+RL` routing-graph notes in `summary.md` so the behavior is visible outside `result.json`.
- Modify: `README.md`
  - Responsibility: document the new default Campaign `MO+RL` routing-graph semantics in the root overview.
- Modify: `src/integration/README.md`
  - Responsibility: document the new Campaign-only default in the integration module README.
- Modify: `src/integration/docs/internal_documentation.md`
  - Responsibility: explain ownership and fallback behavior for path-expanded subgraphs.
- Create: `tests/test_integration/test_routing_subgraph.py`
  - Responsibility: lock deterministic subgraph derivation behavior, ignored non-interacting pairs, and fallback behavior.
- Modify: `tests/test_integration/test_campaign_runner.py`
  - Responsibility: prove Campaign training/evaluation both receive the same derived coupling edges and preserve legacy injected seams.
- Modify: `tests/test_integration/test_scenarios.py`
  - Responsibility: prove internal Campaign `MO+RL` injection reuses both `selected_layout` and derived `coupling_edges`, while external behavior still falls back to MO + full backend.
- Modify: `tests/test_integration/test_campaign_reporting.py`
  - Responsibility: prove Campaign summary markdown renders routing-graph notes.
- Modify: `tests/test_integration/test_docs.py`
  - Responsibility: lock the new wording in docs.

### Task 1: Derive the Campaign Routing Subgraph

**Files:**
- Create: `src/integration/routing_subgraph.py`
- Test: `tests/test_integration/test_routing_subgraph.py`

- [ ] **Step 1: Write the failing derivation tests**

```python
from qiskit import QuantumCircuit


def test_build_path_expanded_subgraph_closes_shortest_paths_for_interacting_pairs() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    circuit.cx(1, 2)
    selected_layout = [4, 0, 2]
    backend_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.mode == "path_expanded_subgraph"
    assert result.coupling_edges == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert result.interacting_pair_count == 2
    assert result.added_intermediate_qubits == [1, 3]


def test_build_path_expanded_subgraph_ignores_non_interacting_layout_pairs() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    selected_layout = [5, 2, 4, 0]
    backend_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.mode == "path_expanded_subgraph"
    assert result.coupling_edges == [(2, 3), (3, 4), (4, 5)]
    assert result.interacting_pair_count == 1
    assert result.added_intermediate_qubits == [3, 4]


def test_build_path_expanded_subgraph_resolves_shortest_path_ties_deterministically() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    selected_layout = [0, 3]
    backend_edges = [(0, 1), (1, 3), (0, 2), (2, 3)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.coupling_edges == [(0, 1), (1, 3)]
    assert result.added_intermediate_qubits == [1]


def test_build_path_expanded_subgraph_falls_back_to_full_backend_when_no_two_qubit_pairs_exist() -> None:
    from src.integration.routing_subgraph import build_path_expanded_subgraph

    circuit = QuantumCircuit(3)
    circuit.h(0)
    selected_layout = [7, 3, 5]
    backend_edges = [(3, 4), (4, 5), (5, 6), (6, 7)]

    result = build_path_expanded_subgraph(
        circuit=circuit,
        selected_layout=selected_layout,
        coupling_edges=backend_edges,
    )

    assert result.mode == "full_backend_fallback"
    assert result.coupling_edges == [(3, 4), (4, 5), (5, 6), (6, 7)]
    assert result.fallback_reason == "no_interacting_pairs"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_routing_subgraph.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.integration.routing_subgraph'`

- [ ] **Step 3: Write the minimal derivation module**

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from qiskit import QuantumCircuit


@dataclass(frozen=True, slots=True)
class RoutingSubgraph:
    mode: str
    coupling_edges: list[tuple[int, int]]
    node_count: int
    edge_count: int
    added_intermediate_qubits: list[int]
    interacting_pair_count: int
    fallback_reason: str | None = None


def _normalize_edges(coupling_edges) -> list[tuple[int, int]]:
    normalized = {tuple(sorted((int(left), int(right)))) for left, right in coupling_edges}
    return sorted(normalized)


def _extract_interacting_logical_pairs(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for instruction in circuit.data:
        logical_qargs = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if len(logical_qargs) != 2:
            continue
        pair = tuple(sorted((int(logical_qargs[0]), int(logical_qargs[1]))))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _build_adjacency(coupling_edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    adjacency: dict[int, set[int]] = {}
    for left, right in coupling_edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return {node: sorted(neighbors) for node, neighbors in adjacency.items()}


def _shortest_path(adjacency: dict[int, list[int]], start: int, goal: int) -> list[int] | None:
    if start == goal:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor in visited:
                continue
            next_path = [*path, neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return None


def build_path_expanded_subgraph(*, circuit: QuantumCircuit, selected_layout: list[int], coupling_edges) -> RoutingSubgraph:
    backend_edges = _normalize_edges(coupling_edges)
    interacting_pairs = _extract_interacting_logical_pairs(circuit)
    if not interacting_pairs:
        return RoutingSubgraph(
            mode="full_backend_fallback",
            coupling_edges=backend_edges,
            node_count=len({node for edge in backend_edges for node in edge}),
            edge_count=len(backend_edges),
            added_intermediate_qubits=[],
            interacting_pair_count=0,
            fallback_reason="no_interacting_pairs",
        )

    adjacency = _build_adjacency(backend_edges)
    selected_qubits = {int(entry) for entry in selected_layout}
    chosen_edges: set[tuple[int, int]] = set()
    visited_nodes: set[int] = set(selected_qubits)

    for logical_q1, logical_q2 in interacting_pairs:
        physical_q1 = int(selected_layout[logical_q1])
        physical_q2 = int(selected_layout[logical_q2])
        if physical_q1 == physical_q2:
            continue
        edge = tuple(sorted((physical_q1, physical_q2)))
        if edge in backend_edges:
            chosen_edges.add(edge)
            visited_nodes.update(edge)
            continue
        path = _shortest_path(adjacency, physical_q1, physical_q2)
        if path is None:
            return RoutingSubgraph(
                mode="full_backend_fallback",
                coupling_edges=backend_edges,
                node_count=len({node for item in backend_edges for node in item}),
                edge_count=len(backend_edges),
                added_intermediate_qubits=[],
                interacting_pair_count=len(interacting_pairs),
                fallback_reason=f"missing_path:{physical_q1}-{physical_q2}",
            )
        for left, right in zip(path, path[1:]):
            chosen_edges.add(tuple(sorted((left, right))))
        visited_nodes.update(path)

    derived_edges = sorted(chosen_edges)
    if not derived_edges and backend_edges:
        return RoutingSubgraph(
            mode="full_backend_fallback",
            coupling_edges=backend_edges,
            node_count=len({node for item in backend_edges for node in item}),
            edge_count=len(backend_edges),
            added_intermediate_qubits=[],
            interacting_pair_count=len(interacting_pairs),
            fallback_reason="empty_derived_graph",
        )

    added_nodes = sorted(node for node in visited_nodes if node not in selected_qubits)
    return RoutingSubgraph(
        mode="path_expanded_subgraph",
        coupling_edges=derived_edges,
        node_count=len(visited_nodes),
        edge_count=len(derived_edges),
        added_intermediate_qubits=added_nodes,
        interacting_pair_count=len(interacting_pairs),
    )
```

- [ ] **Step 4: Run the derivation tests to verify they pass**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_routing_subgraph.py -q`

Expected: `4 passed`

- [ ] **Step 5: Commit the derivation helper**

```bash
git add src/integration/routing_subgraph.py tests/test_integration/test_routing_subgraph.py
git commit -m "feat: derive campaign routing subgraphs"
```

### Task 2: Thread the Derived Graph Through Campaign Training and Evaluation

**Files:**
- Modify: `src/integration/campaign_runner.py`
- Modify: `src/integration/scenarios.py`
- Modify: `tests/test_integration/test_campaign_runner.py`
- Modify: `tests/test_integration/test_scenarios.py`

- [ ] **Step 1: Write failing Campaign and Scenario tests**

```python
def test_run_campaign_uses_path_expanded_coupling_edges_for_training_and_mo_rl(tmp_path) -> None:
    from src.integration.campaign_runner import run_campaign

    campaign = _build_campaign()
    case = campaign.build_cases()[0]
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    captured_training_coupling_maps = []
    captured_mo_rl_coupling_maps = []

    def fake_train_case(*, campaign_case, campaign_config, target_circuit, coupling_map, case_output_dir, initial_layout=None):
        del campaign_config, target_circuit, case_output_dir, initial_layout
        captured_training_coupling_maps.append(list(coupling_map))
        return _build_training_result(campaign_case)

    def fake_run_mo_rl(request, *, circuit, injected_layout, injected_coupling_edges, injected_routing_graph):
        del request, circuit, injected_layout, injected_routing_graph
        captured_mo_rl_coupling_maps.append(list(injected_coupling_edges))
        return _build_result("MO+RL", case, metrics=_build_metrics(80))

    report = run_campaign(
        campaign,
        output_root=tmp_path / "campaigns",
        load_case_circuit=lambda campaign_case: circuit,
        run_baseline=lambda request, *, circuit: _build_result("Baseline", case, metrics=_build_metrics(100)),
        run_mo_only=lambda request, *, circuit: ScenarioResult(
            scenario_name="MO_Only",
            circuit_name="ghz_3",
            backend_name="fake_torino",
            seed=42,
            success=True,
            selected_layout=[2, 0, 1],
            transpilation_metrics=_build_metrics(90),
        ),
        train_case_fn=fake_train_case,
        run_mo_rl=fake_run_mo_rl,
        resolve_backend_bundle=lambda backend_name: SimpleNamespace(
            backend_name=backend_name,
            coupling_edges=[(0, 1), (1, 2), (2, 3)],
        ),
        write_outputs=lambda *, output_dir, report: None,
    )

    assert captured_training_coupling_maps == [[(1, 2)]]
    assert captured_mo_rl_coupling_maps == [[(1, 2)]]
    assert report.case_reports[0].status == "completed"


def test_run_mo_rl_scenario_uses_injected_coupling_edges_for_campaign_eval(monkeypatch) -> None:
    from src.integration import scenarios

    circuit = QuantumCircuit(3)
    request = _make_request("MO+RL", rl_model_path="models/policy.zip")
    bundle = SimpleNamespace(
        backend_name="fake_backend",
        backend=SimpleNamespace(num_qubits=3),
        coupling_edges=[(0, 1), (1, 2), (2, 3)],
    )
    eval_calls = []
    rebuild_calls = []

    monkeypatch.setattr(scenarios, "_load_circuit", lambda request: circuit)
    monkeypatch.setattr(scenarios, "resolve_backend_bundle", lambda backend_name: bundle)
    monkeypatch.setattr(scenarios, "_load_agent", lambda request, *, algorithm="PPO": "agent-object")
    monkeypatch.setattr(
        scenarios,
        "evaluate_routing_episode",
        lambda **kwargs: eval_calls.append(kwargs)
        or RoutingEpisodeSummary(
            initial_layout=[1, 2, 0],
            final_layout=[1, 2, 0],
            steps_executed=0,
            total_reward=0.0,
            completed=True,
            truncated=False,
            total_swaps=0,
            gates_executed_count=2,
            swap_trace=[],
        ),
    )
    monkeypatch.setattr(
        scenarios,
        "build_routed_circuit",
        lambda **kwargs: rebuild_calls.append(kwargs) or ("routed-circuit", [1, 2, 0]),
    )
    monkeypatch.setattr(scenarios.qiskit_interface, "transpile_post_routing", lambda *args, **kwargs: _make_transpilation_result())

    result = scenarios.run_mo_rl_scenario(
        request,
        injected_layout=[1, 2, 0],
        injected_coupling_edges=[(1, 2), (0, 2)],
    )

    assert result.success is True
    assert eval_calls[0]["coupling_edges"] == [(1, 2), (0, 2)]
    assert rebuild_calls[0]["coupling_edges"] == [(1, 2), (0, 2)]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_runner.py -k "path_expanded" tests/test_integration/test_scenarios.py -k "injected_coupling_edges" -q`

Expected: FAIL because `run_campaign(...)` does not derive a subgraph yet and `run_mo_rl_scenario(...)` does not accept `injected_coupling_edges`.

- [ ] **Step 3: Implement Campaign and Scenario wiring with the new helper**

```python
# src/integration/campaign_runner.py
from src.integration.routing_subgraph import build_path_expanded_subgraph


selected_layout_for_training = list(selected_layout)
selected_layout_for_mo_rl = list(selected_layout)
derived_routing_graph = build_path_expanded_subgraph(
    circuit=circuit,
    selected_layout=selected_layout_for_mo_rl,
    coupling_edges=list(getattr(backend_bundle, "coupling_edges")),
)
training_coupling_map = list(derived_routing_graph.coupling_edges)

case_report.training_result = train_case_fn(
    campaign_case=campaign_case,
    campaign_config=campaign.config,
    target_circuit=circuit,
    coupling_map=training_coupling_map,
    case_output_dir=case_output_dir,
    initial_layout=selected_layout_for_training,
)

case_report.mo_rl_result = _invoke_scenario_runner(
    run_mo_rl,
    mo_rl_request,
    circuit=circuit,
    injected_layout=selected_layout_for_mo_rl,
    injected_coupling_edges=list(derived_routing_graph.coupling_edges),
    injected_routing_graph=derived_routing_graph,
)
```

```python
# src/integration/scenarios.py
def _build_routing_graph_notes(routing_graph) -> list[str]:
    if routing_graph is None:
        return []
    summary = (
        f"Campaign routing graph: mode={routing_graph.mode}, "
        f"nodes={routing_graph.node_count}, edges={routing_graph.edge_count}, "
        f"interacting_pairs={routing_graph.interacting_pair_count}, "
        f"added_intermediate_qubits={routing_graph.added_intermediate_qubits}"
    )
    if routing_graph.fallback_reason is not None:
        summary = f"{summary}, fallback_reason={routing_graph.fallback_reason}"
    return [summary]


def run_mo_rl_scenario(
    request: ScenarioRequest,
    *,
    circuit=None,
    injected_layout: list[int] | None = None,
    injected_coupling_edges: list[tuple[int, int]] | None = None,
    injected_routing_graph=None,
) -> ScenarioResult:
    ...
    routing_coupling_edges = (
        list(injected_coupling_edges)
        if injected_coupling_edges is not None
        else list(backend_bundle.coupling_edges)
    )
    routing_summary = evaluate_routing_episode(
        circuit=circuit,
        coupling_edges=routing_coupling_edges,
        agent=agent,
        seed=request.seed,
        initial_layout=selected_layout,
        frontier_mode=contract.frontier_mode,
        max_steps=contract.max_steps,
        lookahead_window=contract.lookahead_window,
        masked=contract.masked,
    )
    ...
    routed_circuit, final_layout = build_routed_circuit(
        circuit=circuit,
        coupling_edges=routing_coupling_edges,
        initial_layout=selected_layout,
        swap_trace=routing_summary.swap_trace,
        frontier_mode=contract.frontier_mode,
        executed_gate_trace=routing_summary.executed_gate_trace,
    )
    ...
    return ScenarioResult(
        ...,
        notes=_build_mo_rl_notes(contract.metadata_source) + _build_routing_graph_notes(injected_routing_graph),
    )
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_runner.py tests/test_integration/test_scenarios.py tests/test_integration/test_routing_subgraph.py -q`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the Campaign/scenario wiring**

```bash
git add src/integration/routing_subgraph.py src/integration/campaign_runner.py src/integration/scenarios.py tests/test_integration/test_routing_subgraph.py tests/test_integration/test_campaign_runner.py tests/test_integration/test_scenarios.py
git commit -m "feat: run campaign MO+RL on path-expanded subgraphs"
```

### Task 3: Report and Document the New Default Behavior

**Files:**
- Modify: `src/integration/campaign_reporting.py`
- Modify: `README.md`
- Modify: `src/integration/README.md`
- Modify: `src/integration/docs/internal_documentation.md`
- Modify: `tests/test_integration/test_campaign_reporting.py`
- Modify: `tests/test_integration/test_docs.py`

- [ ] **Step 1: Write failing reporting and docs tests**

```python
def test_render_campaign_summary_markdown_includes_mo_rl_routing_graph_notes() -> None:
    report = build_campaign_report(
        campaign_id="campaign-001",
        campaign_status="completed",
        campaign_config=_build_campaign_config(),
        case_reports=[
            CampaignCaseReport(
                case=_build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino"),
                status="completed",
                baseline_result=_build_scenario_result("Baseline", _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino"), metrics=_build_metrics(100, 30, 45.0, 1.0)),
                mo_only_result=_build_scenario_result("MO_Only", _build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino"), metrics=_build_metrics(90, 28, 40.0, 1.5)),
                mo_rl_result=ScenarioResult(
                    scenario_name="MO+RL",
                    circuit_name="ghz_3",
                    backend_name="fake_torino",
                    seed=42,
                    success=True,
                    selected_layout=[2, 0, 1],
                    transpilation_metrics=_build_metrics(80, 24, 36.0, 2.0),
                    notes=[
                        "Campaign routing graph: mode=path_expanded_subgraph, nodes=3, edges=2, interacting_pairs=1, added_intermediate_qubits=[]"
                    ],
                ),
                training_result=_build_training_result(_build_case("ghz_3__fake_torino", "ghz", 3, "fake_torino")),
            )
        ],
    )

    markdown = render_campaign_summary_markdown(report)

    assert "MO+RL Notes" in markdown
    assert "Campaign routing graph: mode=path_expanded_subgraph" in markdown


def test_integration_docs_lock_campaign_path_expanded_subgraph_default() -> None:
    repo_readme_text = read_text("README.md")
    integration_readme_text = read_text("src/integration/README.md")
    internal_doc_text = read_text("src/integration/docs/internal_documentation.md")

    assert "Campaign `MO+RL` derives a path-expanded routing subgraph from the interacting logical pairs in the circuit" in repo_readme_text
    assert "Campaign `MO+RL` trains and evaluates RL on that derived routing graph" in integration_readme_text
    assert "If subgraph derivation fails, Campaign falls back to the full backend coupling map and records that fallback" in internal_doc_text
```

- [ ] **Step 2: Run the reporting/docs tests to verify they fail**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_reporting.py -k "routing_graph_notes" tests/test_integration/test_docs.py -k "path_expanded_subgraph" -q`

Expected: FAIL because markdown does not render `ScenarioResult.notes` yet and the docs do not mention the new default.

- [ ] **Step 3: Implement reporting and docs updates**

```python
# src/integration/campaign_reporting.py
def _render_scenario_notes(label: str, result: ScenarioResult | None) -> list[str]:
    if result is None or not result.notes:
        return []
    return [f"- {label} Notes: " + " | ".join(result.notes)]


def _render_case_detail(report: CampaignReport) -> list[str]:
    lines = ["", "## Per-Case Detail"]
    for case_report in report.case_reports:
        ...
        lines.extend(_render_scenario_notes("MO+RL", case_report.mo_rl_result))
        lines.extend(_render_training_summary(case_report.training_result))
```

```markdown
<!-- README.md -->
Cada Campaign Case sigue ejecutando `Baseline`, `MO_Only` y `MO+RL`, pero el camino híbrido ahora usa por defecto un subgrafo de routing expandido por caminos mínimos. `MO_Only` selecciona el layout, `integration` toma sólo los pares lógicos que realmente interactúan en el circuito, construye el cierre por shortest paths sobre el backend real y usa ese grafo derivado para entrenar y evaluar RL. Si esa derivación falla, Campaign cae al coupling map completo y deja constancia de ese fallback en la salida del caso.
```

```markdown
<!-- src/integration/README.md -->
Within Campaign `MO+RL`, `MO_Only` still selects the layout for the Campaign Case, but RL no longer defaults to the full backend routing graph. The Campaign runner derives a path-expanded routing subgraph from the interacting logical pairs in the circuit, trains RL on that derived graph, and evaluates `MO+RL` on the same graph and layout. If derivation fails, Campaign falls back to the full backend coupling map and records that fallback in case output notes.
```

```markdown
<!-- src/integration/docs/internal_documentation.md -->
Para Campaign `MO+RL`, `integration` sigue siendo dueño del handoff MO -> RL. En la nueva semántica por defecto, toma el layout exacto de `MO_Only`, extrae sólo los pares lógicos que interactúan en puertas de 2 qubits, construye un subgrafo expandido por shortest paths sobre el coupling map real del backend y entrega ese coupling map derivado tanto al training RL como a la evaluación híbrida. Si la derivación no produce un grafo válido, el runner hace fallback al coupling map completo y registra ese modo en las notas del resultado `MO+RL`.
```

- [ ] **Step 4: Run the reporting/docs tests to verify they pass**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_campaign_reporting.py tests/test_integration/test_docs.py -q`

Expected: all selected tests PASS.

- [ ] **Step 5: Run the integration verification sweep**

Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration -q`

Expected: full integration suite PASS.

- [ ] **Step 6: Commit the reporting and docs updates**

```bash
git add src/integration/campaign_reporting.py README.md src/integration/README.md src/integration/docs/internal_documentation.md tests/test_integration/test_campaign_reporting.py tests/test_integration/test_docs.py
git commit -m "docs: describe campaign path-expanded routing graph"
```

## Final Verification

- [ ] Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration/test_routing_subgraph.py tests/test_integration/test_campaign_runner.py tests/test_integration/test_scenarios.py tests/test_integration/test_campaign_reporting.py tests/test_integration/test_docs.py -q`
- [ ] Expected: all targeted tests PASS.
- [ ] Run: `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_integration -q`
- [ ] Expected: full integration suite PASS.
