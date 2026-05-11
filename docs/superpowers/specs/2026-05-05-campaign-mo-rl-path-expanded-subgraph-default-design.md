# Campaign MO+RL Path-Expanded Subgraph Default Design

## Goal

Make Campaign `MO+RL` use a path-expanded backend subgraph as the default routing graph, instead of always training and evaluating RL on the full backend coupling map.

The subgraph must be derived from the exact `MO_Only` layout selected for the Campaign Case and from the logical qubit pairs that actually interact in the circuit.

## Context

The current Campaign hybrid flow already does two important things:

1. `MO_Only` selects the layout for the Campaign Case.
2. Campaign RL training and Campaign `MO+RL` evaluation both reuse that exact layout.

That change fixes the train-vs-eval mismatch discovered during diagnosis, but Campaign `MO+RL` still gives RL the full backend coupling map.

The new desired behavior is stricter:

- keep the exact `MO_Only` layout as the anchor for the Campaign Case;
- derive a smaller routing graph from that layout;
- use that derived graph for both Campaign RL training and Campaign `MO+RL` evaluation.

The design discussed and approved during brainstorming is not the induced subgraph over only the layout qubits. Instead, it is a path-expanded subgraph:

- start from the physical qubits used by the selected layout;
- inspect only the logical qubit pairs that actually interact in two-qubit gates in the circuit;
- map those logical pairs to physical pairs through the selected layout;
- close the graph by adding shortest backend paths between those physical endpoints when needed.

## Scope

This work covers:

- Campaign-only `MO+RL` behavior;
- deriving a path-expanded subgraph from `selected_layout`, `circuit`, and backend coupling edges;
- using that derived subgraph consistently in Campaign RL training and Campaign `MO+RL` evaluation;
- reporting enough metadata to understand which graph Campaign `MO+RL` used.

## Out of scope

This work does not cover:

- changing the public single-scenario `MO+RL` behavior outside Campaign;
- changing `RL_Only` public behavior;
- changing how `MO_Only` selects layouts;
- dynamic graph growth during routing;
- weighted or heuristic shortest-path selection beyond a deterministic tie-break rule.

## Ownership and boundaries

`src/integration/` remains the owner of MO -> RL orchestration.

The path-expanded subgraph is a Campaign orchestration concern, so it belongs in `integration`, not in `mo_module` and not in `rl_module`.

That keeps module responsibilities clear:

- `mo_module` selects a layout;
- `integration` derives the Campaign routing graph from that layout and the circuit;
- `rl_module` consumes the provided `coupling_map` and `initial_layout` without owning the derivation policy.

## Functional design

### 1. Relevant interacting pairs

The derived graph uses only logical qubit pairs that actually appear in two-qubit gates in the Campaign Case circuit.

Rules:

- ignore one-qubit gates;
- ignore repeated occurrences after deduplication of the same unordered logical pair;
- preserve deterministic processing order by recording pairs in first-seen order.

Each logical pair is mapped to a physical pair through the selected layout using the existing `logical_qubit -> physical_qubit` convention.

### 2. Path-expanded subgraph construction

Treat the backend coupling map as an undirected graph for path-finding purposes.

For each relevant physical pair `(pq1, pq2)`:

- if `pq1 == pq2`, ignore it;
- if the backend already contains an edge between them, include that edge in the derived graph;
- otherwise, find one shortest path between `pq1` and `pq2` in the full backend graph and add all nodes and edges from that path.

The final Campaign routing graph is the union of all such selected path edges.

Important design choice:

- include only the edges that belong to the chosen shortest paths;
- do not expand again to the induced subgraph over all visited nodes.

This keeps the action space smaller than the full backend and avoids reintroducing many irrelevant edges.

### 3. Determinism

If multiple shortest paths exist, Campaign must choose one deterministically.

Approved rule:

- use BFS on the undirected backend graph;
- visit neighbor qubits in ascending numeric order;
- use the first shortest path discovered under that stable traversal.

With that rule, the same circuit, layout, and backend always produce the same derived graph.

### 4. Fallback behavior

If Campaign cannot derive a valid path-expanded graph, it must fall back to the full backend coupling map instead of failing the whole Campaign Case.

Fallback triggers include:

- some relevant physical pair has no path in the backend graph;
- the derivation result is empty while the backend graph is not empty;
- any internal validation of the derived graph fails.

This is intentionally conservative. The new default should not make Campaign more fragile than today.

### 5. Where the derived graph is used

For Campaign `MO+RL`, the derived graph becomes the routing graph for both:

- RL training for the Campaign Case;
- Campaign `MO+RL` evaluation for the same Campaign Case.

That means training and evaluation stay aligned on both:

- the same `initial_layout`;
- the same routing graph.

The full backend object still remains available for MO and for Qiskit post-routing evaluation.

### 6. Qiskit post-routing semantics

Qiskit post-routing comparison must still evaluate against the real backend, not against a synthetic reduced backend object.

So Campaign `MO+RL` should:

- run RL on the derived subgraph coupling edges;
- reconstruct the routed circuit from the RL traces using that same derived graph;
- pass the resulting routed circuit to the existing post-routing Qiskit flow against the real backend.

This preserves scenario comparability at the final metrics stage.

## Data and reporting

Campaign should record enough graph metadata to make the new behavior auditable.

Recommended minimum metadata for the `MO+RL` result or case-level reporting:

- whether Campaign used `path_expanded_subgraph` or `full_backend_fallback`;
- number of nodes in the derived graph;
- number of edges in the derived graph;
- number of added intermediate qubits;
- number of relevant interacting logical pairs.

The exact storage location can be decided in implementation as long as it ends up visible in case outputs.

## Error handling

- If `MO_Only` does not provide a valid `selected_layout`, Campaign hybrid execution still fails early exactly as in the current behavior.
- If subgraph derivation fails, Campaign logs the incident and falls back to the full backend coupling map.
- If RL still fails to complete on the derived graph, the case remains `incomplete` exactly as today.

## Testing

Required coverage:

1. Derivation builds the expected shortest-path closure for interacting logical pairs.
2. Derivation ignores non-interacting layout pairs.
3. Multiple shortest paths resolve deterministically.
4. Campaign training uses derived coupling edges for `MO+RL` cases.
5. Campaign `MO+RL` evaluation uses the same derived coupling edges and same selected layout.
6. Fallback to full backend occurs when derivation cannot produce a valid graph.
7. Existing non-Campaign scenario behavior remains unchanged.

## Documentation impact

Campaign documentation must be updated so `MO+RL` no longer means only “train and evaluate from the MO-selected layout.”

It must now say that Campaign `MO+RL`:

1. uses the exact layout selected by `MO_Only`;
2. derives a path-expanded routing subgraph from the interacting logical pairs in the circuit;
3. trains and evaluates RL on that derived routing graph;
4. still compares final post-routing metrics against the real backend.

## Rollout decision

This design makes the path-expanded subgraph the new default for Campaign `MO+RL`.

There is no user-facing toggle in this iteration. If derivation fails, the system silently falls back to the current full-backend behavior and records that fallback in Campaign output metadata.
