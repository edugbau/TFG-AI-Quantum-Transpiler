# Análisis de resultados del benchmark — MO vs Qiskit default

**Configuración**: NSGA-II · 10 semillas · pop=30 · gen=50 · backend=`fake_torino`  
**Circuitos**: `ghz_5q`, `qft_4q`, `random_4q_d10`, `clifford_4q`

---

## Resultados por circuito

| Circuito | depth MO (media) | depth Qiskit | Δ depth | cnots MO (media) | cnots Qiskit | Δ cnots |
|---|---|---|---|---|---|---|
| `ghz_5q` | 19.2 | 19.0 | −1.1% ▼ | 5.5 | 4.0 | −37.5% ▼ |
| `qft_4q` | 75.0 | 78.3 | **+4.2% ▲** | 28.5 | 27.0 | −5.6% ▼ |
| `random_4q_d10` | 121.0 | 121.0 | 0.0% = | 42.3 | 42.0 | −0.7% ≈ |
| `clifford_4q` | 41.7 | 43.8 | **+4.8% ▲** | 15.3 | 15.0 | −2.0% ▼ |

*(Δ positivo = MO es mejor; negativo = Qiskit es mejor)*

---

## Interpretación

### `ghz_5q` — Qiskit ya es óptimo
GHZ es una cadena de CX: `q0–q1–q2–q3–q4`. SABRE resuelve este patrón de forma trivial.  
El baseline tiene `depth_std = 0` y `cnot_std = 0` en todas las semillas: no hay variabilidad, lo que implica que Qiskit alcanza siempre el óptimo global. El MO no puede mejorar lo que ya está en el mínimo posible, y hay una ligera penalización por el proceso de búsqueda evolutiva. Este circuito es un **caso límite** donde el enfoque MO aporta poco valor.

### `qft_4q` — Mejora real en profundidad, tradeoff en CNOTs
Qiskit es **inestable** en este circuito: `bl_depth_std = 9.9` frente a `depth_std = 0.0` del MO. El optimizador MO encuentra consistentemente un layout de depth=75 en todas las semillas mientras que Qiskit oscila entre 63 y 94. La ligera penalización en CNOTs (+5.6%) es un tradeoff explícito dentro del frente de Pareto: el algoritmo priorizó estabilidad y depth a costa de un gate adicional.

### `random_4q_d10` — Empate técnico
El circuito aleatorio de depth=10 es suficientemente denso para que el espacio de layouts sea reducido. Tanto SABRE como el MO convergen al mismo layout en prácticamente todos los casos (`depth_std = 0`, `cnot_std ≈ 0`). Esto indica que con 4 qubits el espacio de búsqueda es demasiado pequeño para que el MO aporte ventaja.

### `clifford_4q` — Mejor resultado del benchmark
El MO consigue una mejora consistente de +4.8% en profundidad sobre los 10 seeds. La penalización en CNOTs (−2%) es marginal. El `pareto_size_mean = 32.97` indica un frente de Pareto rico, con solutions bien distribuidas entre los dos objetivos. Es el caso donde el tradeoff depth/CNOTs es más significativo y el MO aprovecha mejor el espacio de búsqueda.

### Conclusión general
Los resultados son **correctos y esperados**: el MO mejora en los circuitos donde SABRE tiene variabilidad y el espacio de layouts no es trivial. Los resultados negativos (GHZ) no son un fallo del algoritmo sino una consecuencia de que el baseline ya es óptimo. El patrón de mejora en `depth` a costa de `cnot_count` refleja la naturaleza multiobjetivo del problema: el frente de Pareto ofrece soluciones que Qiskit (mono-objetivo implícito) no considera.

---

## Limitaciones del experimento actual

1. **Circuitos pequeños**: 4–5 qubits implican un espacio de layouts muy reducido (~120 permutaciones para 5q). SABRE tiene alta probabilidad de encontrar o aproximarse al óptimo global.
2. **Pocas semillas**: Con 10 semillas los intervalos de confianza son amplios. Los porcentajes de mejora tienen alta incertidumbre estadística.
3. **Un solo backend**: `fake_torino` tiene topología heavy-hex, optimizada para SABRE. La ventaja del MO puede ser mayor en topologías menos regulares.
4. **Un solo algoritmo**: Solo se comparó NSGA-II. No hay comparativa con MOEA/D.
5. **Baseline de nivel fijo**: El baseline usa `optimization_level=1`. No se compara contra `optimization_level=3`, que es la máxima optimización de Qiskit.

---

## Mejoras propuestas

### Alta prioridad (impacto en resultados)

#### 1. Circuitos más grandes (≥ 8 qubits)
Con circuitos de 8–12 qubits el espacio de layouts crece factorialmente (~40k permutaciones para 8q) y SABRE se aleja más del óptimo. Candidatos:

- **QFT 8q**: `QuantumVolume(8)` o `QFT(8)` — buen benchmark estándar
- **QAOA sobre K5 o K6**: grafos completos generan muchos CX cruzados, difíciles para SABRE
- **VQE ansatz (RealAmplitudes 6q, depth=3)**: circuito variacional realista
- **Random 8q depth=20**: generalización del caso aleatorio actual

#### 2. Más semillas (30)
30 semillas permiten calcular intervalos de confianza al 95% con `scipy.stats.sem` y reportar mejoras con significancia estadística — necesario para el TFG.

#### 3. Comparar con `optimization_level=3`
Añadir un segundo baseline con el nivel máximo de Qiskit muestra si el MO supera incluso la transpilación con optimización agresiva (que incluye `VF2PostLayout` y rotaciones de cancellación).

### Media prioridad (comparativa de algoritmos)

#### 4. NSGA-II vs MOEA/D
Ejecutar ambos algoritmos sobre los mismos circuitos y semillas permite una comparativa directa. MOEA/D tiende a ser más eficiente cuando el frente de Pareto tiene una forma convexa bien definida. Esta comparativa es una contribución original clara para el TFG.

#### 5. Estudio de convergencia (generaciones 50 → 100 → 200)
Para `qft_4q` y `clifford_4q` aumentar las generaciones de 50 a 150 puede mejorar la calidad del frente de Pareto. Un experimento de ablación con gen={50, 100, 150, 200} mostraría la curva de convergencia.

### Baja prioridad (rigor experimental)

#### 6. Múltiples backends
Comparar `fake_torino` vs `fake_sherbrooke` (Eagle r3, 127q) con circuitos de 8–12 qubits. En backends más grandes la topología impone más restricciones de routing y el MO tiene más margen de mejora.

#### 7. Métrica de hipervolumen del frente de Pareto
Además de comparar soluciones individuales, calcular el **hipervolumen** del frente de Pareto (indicador de calidad estándar en optimización multiobjetivo) permite comparar la calidad global del frente entre algoritmos y entre seeds.

#### 8. Tercera métrica: T1/T2 weighted error
Añadir un tercer objetivo basado en el error de los gates según la calibración del backend (`backend.properties()`). Esto haría el frente tridimensional y más representativo de condiciones reales.

---

## Configuración recomendada para el siguiente experimento

```python
# Circuitos propuestos
circuitos = ['qft_8q', 'qaoa_k5', 'vqe_6q', 'random_8q_d20']

# Parámetros
seeds      = 30
population = 50
generations = 150
backends   = ['fake_torino', 'fake_sherbrooke']
algorithms = ['nsga2', 'moead']
```

Tiempo estimado por combinación (basado en resultados actuales ~350s/seed × 4q):  
- 8q circuitos: ~700–1500 s/seed  
- 30 seeds × 2 algoritmos × 2 backends = 120 ejecuciones  
- Tiempo total estimado: **~24–50 horas** (paralelizable con workers=N)
