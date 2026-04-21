# Roadmap de Futuras Iteraciones del Proyecto

## Propósito del documento

Este documento recoge una visión estructurada de las funcionalidades y líneas de evolución que convendría abordar en futuras iteraciones del proyecto. No pretende ser un plan de implementación detallado ni una lista cerrada de tareas técnicas, sino un mapa de maduración del sistema a nivel funcional, experimental y de producto/TFG.

El objetivo es identificar qué grandes capacidades faltan por consolidar para que el proyecto evolucione desde un conjunto de módulos ya operativos hacia una plataforma híbrida de transpilación cuántica más completa, comparable, usable y defendible académicamente.

## Estado actual resumido

En el estado actual del repositorio, el proyecto ya dispone de una base sólida dividida en cuatro módulos:

- `qiskit_interface` como capa de abstracción sobre backends, transpilación, métricas y baselines.
- `mo_module` como motor de optimización multiobjetivo de layouts, con algoritmos evolutivos, fitness extensible y capacidades de análisis del frente de Pareto.
- `rl_module` como entorno de aprendizaje por refuerzo para routing y synthesis, con una base más madura en routing y un primer alcance funcional en synthesis.
- `integration` como primera capa de orquestación para escenarios de evaluación en `routing` (`Baseline`, `MO_Only`, `RL_Only`, `MO+RL`).

El proyecto ya ha superado la fase de prototipo puramente conceptual, pero todavía no puede considerarse funcionalmente cerrado. La madurez es desigual entre módulos: la base de transpilación y la optimización multiobjetivo están más consolidadas; la parte de RL ha avanzado mucho en observabilidad, inspección y estructura interna, pero aún tiene retos de robustez y generalización; y la integración existe ya para `routing`, aunque todavía no resuelve el pipeline híbrido completo de forma homogénea ni comparable extremo a extremo.

Sobre esta base, las futuras iteraciones deberían orientarse a cerrar capacidades incompletas, unificar resultados entre módulos, ampliar el alcance experimental y mejorar la usabilidad general del sistema.

---

## Capítulo 1. Evolución de la Interfaz Base y de la Capacidad de Transpilación

Este capítulo agrupa las funcionalidades necesarias para consolidar la capa de base sobre la que operan todos los demás módulos.

### Funcionalidades a desarrollar

- Ampliar el soporte de entrada de circuitos más allá de la librería interna actual.
- Consolidar una gestión más rica de formatos de intercambio de circuitos y artefactos experimentales.
- Extender el catálogo de backends y configuraciones de referencia para que la evaluación no dependa de un conjunto reducido de dispositivos simulados.
- Mejorar la uniformidad de métricas de transpilación para que puedan reutilizarse de forma consistente en MO, integración y análisis experimental.
- Incorporar capacidades de baseline más variadas que permitan comparar el sistema híbrido frente a más estrategias de referencia, no solo frente al baseline principal actual.
- Evolucionar el tratamiento de propiedades de hardware para soportar evaluaciones más ricas y más próximas a escenarios realistas.
- Hacer más expresiva la capa de abstracción sobre Qiskit para absorber futuras diferencias de versión, formatos y modos de evaluación sin propagar complejidad al resto de módulos.

### Valor de este capítulo

Sin una base de transpilación suficientemente flexible y estable, el resto del sistema queda limitado a un conjunto pequeño de experimentos. Este capítulo es clave para que el proyecto pueda escalar en variedad de entradas, riqueza de métricas y comparabilidad frente a baselines externos.

---

## Capítulo 2. Maduración de la Optimización Multiobjetivo de Layouts

Este capítulo agrupa las futuras capacidades necesarias para que el bloque de optimización multiobjetivo pase de ser un optimizador ya funcional a una herramienta más completa de exploración, selección y análisis de layouts.

### Funcionalidades a desarrollar

- Hacer más expresivas las estrategias de selección de una solución única a partir del frente de Pareto.
- Profundizar en la interpretación del frente de Pareto y en la explicación de trade-offs entre soluciones.
- Reforzar la capacidad de tuning y calibración para convertir el módulo en una herramienta experimental más estable y menos dependiente de ajustes manuales.
- Mejorar la capacidad de comparación entre layouts producidos por MO, heurísticas externas y layouts triviales o adversariales.
- Avanzar hacia presets experimentales más claros para distintos tipos de circuito, objetivos o condiciones de evaluación.
- Dar un salto desde la optimización “por ejecución aislada” hacia una visión más sistemática de campañas experimentales sobre layouts.

### Valor de este capítulo

El módulo MO ya es una pieza fuerte del proyecto, pero todavía puede madurar mucho en capacidad explicativa y en variedad de criterios de optimización. Este bloque es esencial para sostener una narrativa fuerte en el TFG sobre exploración multiobjetivo, calidad de layouts y selección informada de soluciones.

---

## Capítulo 3. Consolidación del Aprendizaje por Refuerzo para Routing

Este capítulo reúne las futuras iteraciones necesarias para que la parte de RL en `routing` pase de ser funcional y observable a ser robusta, estable y generalizable.

### Funcionalidades a desarrollar

- Mejorar la robustez de las políticas entrenadas frente a oscilaciones, ciclos y comportamientos inestables en evaluación.
- Aumentar la capacidad de generalización del agente frente a más circuitos, topologías y layouts iniciales.
- Refinar la representación del estado y de la frontera observable para capturar mejor el contexto de routing.
- Ampliar la experimentación con distintas estrategias de aprendizaje, políticas y configuraciones de entrenamiento.
- Reforzar el diseño de recompensas para que el agente no solo aprenda a completar episodios, sino a producir decisiones de routing más eficientes y consistentes.
- Introducir una visión más curricular o escalonada del entrenamiento, de manera que el agente pueda madurar desde casos simples hasta escenarios complejos.
- Consolidar métricas específicas de calidad de routing más allá de la recompensa agregada.
- Convertir el bloque de evaluación RL en una base más sólida para comparación experimental y no solo para inspección puntual de episodios.

### Valor de este capítulo

La parte de routing en RL es una de las áreas más visibles y prometedoras del proyecto, pero también una de las más sensibles a inestabilidad. Este capítulo es crítico para transformar una demostración funcional en un componente científicamente defendible y técnicamente fiable.

---

## Capítulo 4. Expansión del Aprendizaje por Refuerzo para Synthesis

Este capítulo aborda la evolución del bloque de synthesis, que actualmente está en un estadio mucho más inicial que routing.

### Funcionalidades a desarrollar

- Ampliar el alcance de synthesis más allá del caso inicial y restringido disponible en la actualidad.
- Extender el repertorio de circuitos, operaciones y familias lógicas que el sistema puede sintetizar.
- Profundizar en el catálogo de primitivas hardware-aware para que synthesis deje de ser solo una prueba de concepto y se convierta en una línea fuerte del proyecto.
- Mejorar la semántica de observación y acción en synthesis para reflejar mejor el progreso real del episodio.
- Hacer más rica la evaluación de calidad de synthesis, tanto en coste como en equivalencia funcional del resultado.
- Explorar la relación entre synthesis y layout, incluyendo futuras variantes donde ambas dimensiones no estén completamente separadas.
- Reducir las limitaciones del primer alcance funcional actual para convertir synthesis en un segundo gran pilar del proyecto junto a routing.

### Valor de este capítulo

El bloque de synthesis es una gran oportunidad de diferenciación del proyecto, pero todavía requiere una fase importante de maduración. Su desarrollo futuro puede aportar mucho valor al TFG, tanto por originalidad como por amplitud de alcance frente a una solución centrada solo en routing.

---

## Capítulo 5. Integración y Pipeline Híbrido MO+RL

Este capítulo agrupa las capacidades necesarias para que el proyecto deje de ser una colección de módulos cooperantes y pase a comportarse como un pipeline híbrido más cerrado y coherente.

### Funcionalidades a desarrollar

- Consolidar la integración más allá del primer alcance de `routing` ya implementado.
- Extender la orquestación para cubrir más escenarios y variantes de evaluación de manera homogénea.
- Incorporar un pipeline híbrido completo que no solo conecte MO y RL mediante un `initial_layout`, sino que produzca resultados comparables de extremo a extremo.
- Resolver la principal limitación actual de la integración con RL: la ausencia de un circuito final materializado como salida pública.
- Evolucionar la integración para cubrir también futuros escenarios de synthesis.
- Convertir la capa de integración en un punto único para ejecutar experimentos híbridos, no solo en una colección de runners individuales.
- Hacer más rica la selección y configuración de escenarios, para permitir campañas comparativas más completas.
- Reforzar el papel de `integration` como dueño real de los flujos `Baseline`, `MO_Only`, `RL_Only` y `MO+RL`, evitando que esa lógica se disperse entre módulos.

### Valor de este capítulo

Este es el capítulo más estratégico del proyecto. Mientras la integración permanezca parcial, el valor del sistema híbrido se percibe como suma de piezas. Cuando este bloque madure, el proyecto pasará a tener una narrativa clara como pipeline completo de transpilación híbrida.

---

## Capítulo 6. Experimentación, Benchmarking y Validación Empírica

Este capítulo reúne todo lo necesario para que el proyecto gane peso como trabajo experimental y no solo como desarrollo de software avanzado.

### Funcionalidades a desarrollar

- Definir un conjunto de benchmarks más amplio, sistemático y representativo.
- Formalizar protocolos de comparación entre escenarios, módulos y configuraciones.
- Mejorar la trazabilidad de campañas experimentales para poder repetir y auditar resultados.
- Unificar la recolección de métricas entre los distintos bloques del sistema.
- Incorporar análisis estadístico más sólido sobre semillas, variabilidad y estabilidad de resultados.
- Mejorar la capacidad de generar tablas, resúmenes y reportes directamente utilizables en memoria, presentaciones o anexos.
- Estructurar mejor la narrativa de resultados para responder preguntas como: cuándo ayuda MO, cuándo ayuda RL, cuándo compensa el pipeline híbrido y bajo qué condiciones.
- Preparar el proyecto para comparativas más ricas frente a heurísticas, configuraciones y variantes internas.

### Valor de este capítulo

Este bloque es indispensable para la parte académica del TFG. Sin una validación empírica fuerte, incluso una arquitectura muy buena queda incompleta como contribución. Este capítulo convierte el sistema en un objeto de estudio y no solo en una implementación técnica.

---

## Capítulo 7. Interacción, Visualización y Herramientas de Uso

Este capítulo agrupa las funcionalidades orientadas a hacer el proyecto más accesible, interpretable y usable, tanto durante el desarrollo como en demos, evaluación y defensa.

### Funcionalidades a desarrollar

- Seguir ampliando la capacidad de inspección visual de episodios, decisiones y resultados intermedios.
- Mejorar las herramientas de exploración de resultados para entender por qué un layout, una política o una ejecución han funcionado mejor que otra.
- Consolidar una experiencia de uso más unificada entre GUI, CLI y runners de experimentación.
- Convertir la visualización en una herramienta de análisis, no solo en una herramienta de demostración.
- Ampliar la exportación de resultados, configuraciones y trazas a formatos cómodos para análisis posterior.
- Mejorar la experiencia de configuración de escenarios y campañas para reducir fricción al evaluar distintas variantes del sistema.
- Reforzar las capacidades de explicación visual del pipeline híbrido para memoria, presentación y debugging.

### Valor de este capítulo

La usabilidad y la visualización son especialmente importantes en un TFG de este tipo porque ayudan tanto al desarrollo como a la defensa del trabajo. Este capítulo hace que el sistema sea más comprensible, más demostrable y más fácil de analizar por terceros.

---

## Capítulo 8. Robustez, Calidad Técnica y Documentación

Este capítulo agrupa las capacidades necesarias para que el proyecto gane solidez a largo plazo y no dependa en exceso de conocimiento tácito o de validaciones ad hoc.

### Funcionalidades a desarrollar

- Seguir consolidando contratos intermodulares claros y estables.
- Reducir inconsistencias entre documentación de alto nivel, documentación interna y estado real del repositorio.
- Ampliar la cobertura de pruebas en rutas críticas y escenarios cruzados entre módulos.
- Reforzar la reproducibilidad técnica del proyecto en distintos entornos y configuraciones.
- Mejorar la observabilidad de errores, estados intermedios y fallos experimentales.
- Preparar el proyecto para futuras extensiones sin aumentar el acoplamiento entre módulos.
- Estructurar mejor la documentación orientada a mantenimiento, extensión y uso por terceros.
- Reducir la dependencia de decisiones implícitas o conocimiento local no documentado.

### Valor de este capítulo

Este bloque no suele ser el más vistoso, pero es el que más influye en la sostenibilidad real del proyecto. Además, en un trabajo académico largo, la calidad documental y la coherencia técnica marcan una gran diferencia en la percepción global del resultado.

---

## Prioridades Transversales

Además de los capítulos anteriores, hay varias prioridades que atraviesan a todo el proyecto y que deberían guiar las futuras iteraciones.

### 1. Comparabilidad real entre módulos y escenarios

El proyecto necesita una noción cada vez más homogénea de “resultado comparable”. Hoy existen buenas piezas de comparación parcial, pero aún falta cerrar una visión unificada entre baselines de Qiskit, resultados de MO, episodios RL y futuras salidas de synthesis.

### 2. Reproducibilidad experimental

La reproducibilidad debe considerarse una capacidad central, no un añadido. Toda futura iteración importante debería reforzar la capacidad de repetir resultados con seeds, configuraciones y artefactos bien registrados.

### 3. Generalización frente a demos puntuales

El proyecto debe seguir alejándose de ejemplos aislados y avanzar hacia comportamientos más generales, sobre más circuitos, más backends y más condiciones de evaluación.

### 4. Mantenimiento de los límites modulares

La evolución futura no debería resolver necesidades prácticas a costa de romper el desacoplamiento entre módulos. El valor del proyecto también está en que sus bloques sean comprensibles y testables de forma independiente.

### 5. Coherencia entre producto técnico y relato de TFG

Las futuras iteraciones deberían ayudar no solo a “tener más funcionalidades”, sino también a construir una narrativa académica clara: qué problema resuelve cada bloque, qué evidencia lo respalda y cómo se conecta con la propuesta híbrida global.

---

## Visión de Madurez del Proyecto

El proyecto podrá considerarse sustancialmente más maduro cuando cumpla, al menos, estas condiciones de alto nivel:

- la capa base permita evaluar circuitos y backends con más flexibilidad;
- MO tenga una narrativa más rica de objetivos, trade-offs y selección;
- RL en routing sea más robusto y menos dependiente de casos simples;
- synthesis deje de ser un primer alcance restringido y pase a formar parte del valor central del sistema;
- la integración produzca resultados híbridos realmente comparables y utilizables de extremo a extremo;
- la validación empírica permita sostener conclusiones defendibles;
- el proyecto sea más fácil de usar, entender, mantener y presentar.

En conjunto, las futuras iteraciones no deberían verse como una mera suma de mejoras aisladas, sino como un proceso de convergencia hacia un sistema híbrido de transpilación cuántica más completo, más sólido y más convincente tanto técnica como académicamente.
