# Rendimiento de `count_cnot_equivalent` en el proceso de optimización

## Contexto

`CnotCountFitness` necesita contar CNOTs equivalentes, lo que requiere una segunda llamada a
`qiskit.transpile` (con `optimization_level=0`, base `['cx', 'u']`) sobre el circuito ya
transpilado al backend. Esta segunda llamada se realiza dentro de `count_cnot_equivalent`,
que es invocada por `extract_metrics` al construir `CircuitMetrics`.

## Por qué no ralentiza el optimizador

### 1. Cálculo único por layout (caché)

El flujo de evaluación es el siguiente:

```
FitnessEvaluator.evaluate(layout)
  └─ TranspilationCache.get(layout)
       ├─ HIT  → devuelve TranspilationResult almacenado (O(1), dict lookup)
       └─ MISS → transpile_with_custom_layout(...)
                   └─ extract_metrics(transpiled_circuit)
                        └─ count_cnot_equivalent(transpiled_circuit)  ← única llamada
                   └─ guarda en _cache[tuple(layout)]
```

Durante una ejecución del optimizador multiobjetivo (NSGA-II, MOEA/D), el mismo layout puede
evaluarse múltiples veces a lo largo de varias generaciones. Todas las evaluaciones posteriores
al primer cálculo son simples accesos a diccionario: `count_cnot_equivalent` **no se vuelve a
llamar** para layouts ya vistos.

### 2. La segunda transpilación es mucho más barata

La llamada de `count_cnot_equivalent` recibe como entrada el circuito **ya transpilado** al
backend (puertas nativas, routing ya aplicado). Con `optimization_level=0` y sin información
de backend, Qiskit solo realiza substituciones de puertas individuales —equivalentes a una
tabla de conversión— sin ninguno de los algoritmos costosos de la transpilación principal:

| Fase                | Transpilación principal | `count_cnot_equivalent` |
|---------------------|------------------------|-------------------------|
| `optimization_level`| 2 (por defecto)        | 0                       |
| Routing (SABRE)     | Sí                     | No                      |
| Layout search       | Sí                     | No                      |
| Basis translation   | Sí (hacia nativas)     | Sí (hacia CX+U, trivial)|
| Coste relativo      | ~100 %                 | ~5–15 %                 |

El coste extra es real pero menor, y se paga **una sola vez** por layout único durante toda la
ejecución del optimizador.

## Equivalencias de referencia

La descomposición a base CX convierte cada puerta nativa en su coste real en CNOTs:

| Puerta nativa  | Backend              | CNOTs equivalentes |
|----------------|----------------------|--------------------|
| CX / CNOT      | Simuladores genéricos| 1                  |
| CZ             | FakeTorino           | 1                  |
| ECR            | FakeSherbrooke/Brisbane | 1               |
| SWAP           | Cualquiera           | 3                  |
| iSWAP          | Cualquiera           | 2                  |
| CRZ / CRX      | Cualquiera           | 2                  |

Estas equivalencias justifican usar la descomposición a CX como métrica más fiel del coste
real de un layout que la simple cuenta de puertas 2Q nativas.
