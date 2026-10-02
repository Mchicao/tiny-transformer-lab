# Papers e ideas para el laboratorio

Última revisión: **2026-10-02**. Documento vivo mantenido según [AGENTS.md](AGENTS.md). Planes derivados: [EXPERIMENTOS_FUTUROS.md](EXPERIMENTOS_FUTUROS.md).

Las fichas resumen fuentes primarias, no garantizan reproducibilidad ni transferencia al laboratorio. `Resumen` significa que se consultó el abstract; `Secciones` indica texto y tablas pertinentes, sin afirmar revisión exhaustiva. Los resultados del paper y los nuestros se presentan por separado. Mantener IDs y versiones; al cambiar una ficha, registrar el motivo al final.

## Terminología técnica

Usar los nombres técnicos en inglés en ambos documentos, conservando la explicación en español. Mantener la capitalización habitual: `loss`, `accuracy`, `checkpoint`, `SFT`, `GRPO`, `GPU`. Los títulos originales de las fuentes se conservan literalmente.

| Término | Significado en este laboratorio |
| --- | --- |
| loss / NLL | Objetivo de optimización / negative log-likelihood. Menor NLL no implica automáticamente mayor task accuracy. |
| accuracy / pass@k | Proporción de respuestas correctas / probabilidad estimada de obtener al menos una correcta entre k muestras, con sampling protocol fijado. Greedy accuracy usa decoding determinista. |
| starting checkpoint / frozen baseline | Pesos desde los que empieza un contraste / esos mismos pesos evaluados sin optimizer updates. |
| SFT / additional SFT | Supervised fine-tuning con targets correctos / continuación de SFT desde el starting checkpoint compartido. |
| RLVR / correctness reward | Reinforcement learning with verifiable rewards / score que asigna el verifier a la corrección de la respuesta. No es una loss supervisada. |
| GRPO / advantage | Group Relative Policy Optimization / señal relativa usada para ponderar el aprendizaje de cada completion. |
| rollout / completion | Ejecución muestreada de la policy; aquí, una respuesta aritmética / tokens generados para un prompt. En tareas agentic, un rollout puede incluir varias llamadas y tool results. |
| behavior policy / logprob / loss mask | Policy que generó el rollout / logaritmo de una probabilidad / máscara que indica qué tokens contribuyen a la loss. |
| informative group / reward contrast | Grupo de completions con rewards diferentes / diferencia de rewards que permite obtener una advantage relativa no nula. |
| arm / compute budget / held-out test | Condición experimental / límite de trabajo computacional / datos reservados para evaluation final. |
| language retention / reward shaping | Conservación de capacidades de lenguaje previas / modificación del reward con señales adicionales a correctness. |
| GPU / LR / seed | Graphics processing unit / learning rate / semilla del generador pseudoaleatorio. |

## Datos, escala y entrenamiento

### P01 — TinyStories: How Small Can Language Models Be and Still Speak Coherent English?

- **Fuente:** [arXiv:2305.07759v2](https://arxiv.org/abs/2305.07759v2). Lectura: resumen.
- **Resultado de los autores:** un corpus de historias con vocabulario y conceptos sencillos permite producir texto coherente con modelos inferiores a 10M parámetros.
- **Límite:** evidencia en un dominio restringido; no demuestra razonamiento matemático general ni fija nuestro presupuesto óptimo.
- **Idea local:** separar insuficiencia de datos, presupuesto y capacidad mediante F01/F03. El 3M sigue siendo útil para experimentos controlados.

### P02 — SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model

- **Fuente:** [arXiv:2502.02737v1](https://arxiv.org/abs/2502.02737v1). Lectura: resumen.
- **Resultado de los autores:** entrenamiento por etapas y curación de texto, matemáticas, código e instrucciones; el modelo central de 1,7B consume aproximadamente 11T tokens.
- **Límite:** “pequeño” aquí no equivale a 3M ni a pocos millones de tokens. No copiar presupuestos ni mezclar etapas sin controles.
- **Idea local:** priorizar cobertura y datos adecuados a cada tarea; F01 y F02. E11 respalda localmente ampliar cobertura, no toda la receta SmolLM2.

### P03 — Practical Efficiency of Muon for Pretraining

- **Fuente:** [arXiv:2505.02222v4](https://arxiv.org/abs/2505.02222v4). Lectura: resumen y contraste con implementación local.
- **Resultado de los autores:** estudia eficiencia de Muon frente a AdamW, tamaños de batch y transferencia de hiperparámetros mediante muP, hasta 4B parámetros.
- **Límite:** ventajas dependen de receta, batch y escala; igualar tokens no basta para igualar tiempo o FLOPs.
- **Resultado local:** Muon terminó peor que AdamW en la comparación 3M histórica; QK-Norm fijo tampoco ganó. [Reporte](output/campaign-3m-20261001/report.md). No equivale a refutar el paper.
- **Idea local:** reconsiderar sólo tras cambiar y estabilizar el régimen de datos/presupuesto; F08.

### P04 — Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful

- **Fuente:** [arXiv:2507.07101v4](https://arxiv.org/abs/2507.07101v4). Lectura: resumen.
- **Resultado de los autores:** batches pequeños pueden entrenar establemente; proponen ajustar hiperparámetros de Adam preservando la semivida del segundo momento en tokens.
- **Límite:** quitar acumulación cambia batch efectivo, cantidad de actualizaciones y dinámica del optimizador. No es una optimización transparente.
- **Idea local:** F08, ablación independiente tras F01; medir calidad por tokens y tiempo, con schedule y calibración definidos antes.

### P05 — Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws

- **Fuente:** [arXiv:2606.06888v2](https://arxiv.org/abs/2606.06888v2). Lectura: resumen.
- **Resultado de los autores:** MIR añade next-token prediction loss sobre entradas enmascaradas; en 72M–1,4B mejora sobre controles con weight decay fuerte. SoftQ modela interacción entre tamaño y datos repetidos.
- **Límite:** auxiliary regularization añade trabajo de entrenamiento; sus ajustes no se trasladan automáticamente a 3M ni sustituyen datos disponibles.
- **Idea local:** comparar regularización sólo si volvemos a un régimen realmente limitado por datos; F08. E11 favorece primero ampliar historias únicas.

## Recurrencia, arquitectura e inferencia

### P06 — Squeezing More from Limited Data with Recursive Transformers

- **Fuente:** [arXiv:2608.26973v1](https://arxiv.org/abs/2608.26973v1). Lectura: resumen.
- **Resultado de los autores:** estudia presupuestos de 10M–100M **palabras**, recurrencia y embeddings factorizados; el tamaño óptimo depende del corpus y la evaluación.
- **Límite:** palabras no son tokens BPE. Recurrencia y factorización son intervenciones distintas; no atribuir una ganancia conjunta sólo al loop.
- **Idea local:** F03/F06; variar tamaño y presupuesto con el mismo tokenizer antes de añadir factorización.

### P07 — SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers

- **Fuente:** [arXiv:2609.01343v3](https://arxiv.org/abs/2609.01343v3). Lectura: resumen.
- **Resultado de los autores:** repite dos veces la mitad central de modelos MoE; compara FLOPs por token, parámetros no embedding y KV cache, hasta 54B parámetros no embedding.
- **Límite:** arquitectura y escala difieren del denso 3M. Igual número de layer-passes es una aproximación al coste, no una prueba de FLOPs o latencia iguales.
- **Idea local:** conservar controles densos y medir frontera calidad/coste en F06; MoE queda pospuesto.

### P08 — Decoding Looped Transformers Better for (Almost) Free

- **Fuente:** [arXiv:2610.02185v1](https://arxiv.org/html/2610.02185v1). Lectura: método, experimentos y apéndices pertinentes.
- **Resultado de los autores:** LoopCD contrasta estados o logits de una pasada temprana y la final sin actualizar pesos; evalúa cuatro familias recurrentes. La variante Hidden evita otra pasada de salida; Logits añade una.
- **Límite:** el ahorro de 22,5–48,2% son FLOPs teóricos en el estudio de profundidad reducida sobre elección múltiple, no latencia demostrada. Referencia y fuerza requieren calibración; una fuerza excesiva puede empeorar generación.
- **Resultado local:** K2 de 3M mejora de primera a segunda pasada, pero pierde frente al denso; E7 temprano empeora en la segunda pasada. [Registro](docs/architecture/experimentos.md).
- **Idea local:** F05, diagnóstico con pesos fijos sobre 3M. Con K2 reducido a K1 desaparece el contraste entre primera y última pasada.

### P09 — Scaling Laws for Looped Mixture of Experts

- **Fuente:** [arXiv:2609.40316v1](https://arxiv.org/html/2609.40316v1). Lectura: método, resultados y apéndices pertinentes.
- **Resultado de los autores:** ajusta una ganancia efectiva de parámetros acotada por recurrencia y condicionada por sparsity. Barrido de 0,3–1,0B parámetros activos, 100–500B tokens; distingue capacidad efectiva y cómputo del desenrollado.
- **Límite:** coeficientes empíricos no extrapolables directamente a 3M. En la tabla 2, el modelo recurrente pequeño a R5 aproxima resultados de razonamiento del mayor, pero cuesta 1,8× inferencia y obtiene menor puntuación global. Optimiza memoria de pesos, no toda la VRAM.
- **Idea local:** F03/F06; reportar parámetros almacenados, activos y trabajo ejecutado. No iniciar MoE por sus titulares.

### P10 — Looped Diffusion Language Models

- **Fuente:** [arXiv:2605.26106v1](https://arxiv.org/abs/2605.26106v1). Lectura: resumen.
- **Resultado de los autores:** en modelos de difusión enmascarada, repetir capas tempranas/intermedias mejora eficiencia y permite variar cómputo al inferir.
- **Límite:** interacciones entre posiciones enmascaradas difieren de atención causal. No establece que nuestra receta autorregresiva deba ganar.
- **Idea local:** inspiración para ubicación del loop en F06; revisar texto y separar factores antes de un protocolo.

### P11 — Looped Diffusion Transformer

- **Fuente:** [arXiv:2609.40305v1](https://arxiv.org/abs/2609.40305v1). Lectura: resumen.
- **Resultado de los autores:** en generación texto-a-imagen, estabiliza loops con supervisión intermedia y atención automodulada; reporta ganancias con parámetros y cómputo comparados.
- **Límite:** es difusión de imágenes, no un LM autorregresivo. Combina dos cambios; no demuestra que supervisión profunda aislada resuelva nuestro K2.
- **Idea local:** F06/F08, como hipótesis de estabilidad y ablaciones; no réplica directa.

### P12 — Explore Broadly, Reason Sharply: Push Small Models toward the Frontier via Sampling

- **Fuente:** [arXiv:2609.38104v1](https://arxiv.org/abs/2609.38104v1). Lectura: resumen; adaptación local documentada en E10.
- **Resultado de los autores:** Parallel Power Tempering usa cadenas con distintas potencias e intercambios para explorar y favorecer secuencias probables, sin actualizar pesos ni external rewards.
- **Límite:** probabilidad alta no garantiza corrección; generación, propuestas e intercambios consumen cómputo. No equivale a RL.
- **Resultado local:** E10 sólo tiene una prueba greedy persistida y una limitación del extractor. [Resultado](output/e10-greedy-s42-i00-20261002-v1/results.json), [protocolo](docs/architecture/protocolo-e10-piloto-20261002.md).
- **Follow-up local F00:** parser v1 y baseline 96M repetidos en proceso nuevo; score estricto 0% por format/boundary failures. Primera línea correcta en 25/32 greedy sólo como diagnóstico post hoc; no se ejecutó búsqueda ni se evaluó Parallel Power Tempering. [Reporte](output/f00-96m-20261002-142456-cee1e8/report.md).
- **F00 v2:** newline/EOS fijado y validado en dev, control pareado0/12→9/12 greedy sin updates; test nuevo31/32 greedy ypass@8 100%,full repeat exacto. Cambio de dataset/régimen declarado, sin búsqueda/PPT ni aprendizaje. [Reporte v2](output/f00-boundary-test-20261002-154935-1b4bc2/report.md).
- **Idea local:** F00/F07. “PPT” aquí significa Parallel Power Tempering; en P13/E9 significa pre-pretraining: no confundirlos.

## Preparación sintética, post-training y RL

### P13 — Synthetic Pre-pretraining Survives Scale, but Not as a Grammatical Prior

- **Fuente:** [arXiv:2609.39827v1](https://arxiv.org/abs/2609.39827v1). Lectura: resumen y contraste con protocolo local.
- **Resultado de los autores:** estudia preparación sintética antes del pretraining, modelos de 500M–7B y hasta 100B tokens naturales; vincula ganancias a retrieval distante más que a un prior gramatical consistente.
- **Límite:** nuestros datos programáticos y presupuesto no replican esas condiciones; dificultad y entropía de los controles importan.
- **Resultado local:** E9 retrieval gana 1/3 seeds y empeora en media +0,008024 nats; grammatical mejora −0,012714 en 3/3, con 16,33% de tokens adicionales. No confirma un mecanismo gramatical ni ahorro total. [Reporte](output/e9-campaign-20261001-v1/report.md).
- **Idea local:** F08; no repetir sin demostrar primero aprendizaje de retrieval y controlar el coste frente a más texto natural.

### P14 — L20-Edu-135M: An Auditable Single-GPU Study of Data-Efficient Small Language Modeling

- **Fuente:** [arXiv:2606.22189v1](https://arxiv.org/html/2606.22189v1). Lectura: secciones de entrenamiento y post-training.
- **Resultado de los autores:** caso de 134,5M parámetros con aproximadamente 13B tokens previos; RLVR tipo GRPO sobre GSM8K reduce accuracy de 1,82% a 1,59%/1,21% según horizonte.
- **Límite:** corridas únicas; no identifica un tamaño mínimo de RL ni demuestra que la escasez de aciertos sea la única causa. SFT también requiere controles de retención.
- **Idea local:** F00/F02/F04; usar tareas fáciles, medir aciertos previos y reward signal antes de RL.

### P15 — Effective Learning for Small Reasoning Models: An Empirical Study on 0.5B Reasoning LLMs

- **Fuente:** [arXiv:2506.13404v3](https://arxiv.org/html/2506.13404v3). Lectura: tablas y protocolo pertinentes.
- **Resultado reportado:** tabla 2, Qwen2.5-0.5B-Instruct + RL: GSM8K 45,5%→54,0%; también compara SFT, distillation y combinaciones.
- **Límite:** el starting checkpoint ya está preentrenado e instruction-tuned; hay inconsistencias entre texto y tablas. No presentar como garantía de que SFT→RL siempre gana ni trasladar el salto a 96M.
- **Idea local:** F04; comparar RL con additional SFT desde el mismo checkpoint y conservar resultados negativos.

### P16 — Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

- **Fuente:** [arXiv:2504.13837v5](https://arxiv.org/abs/2504.13837v5). Lectura: resumen.
- **Resultado de los autores:** en modelos/tareas estudiados, RL mejora pass@1 pero no necesariamente pass@k con k grande; interpreta parte de la ganancia como selección más eficiente de soluciones ya muestreables.
- **Límite:** no demuestra que todo RL sea incapaz de aprender nuevas habilidades; depende de tarea, exploración y objetivo. El abstract no basta para reproducir metodología.
- **Idea local:** F04; medir accuracy de una respuesta y probabilidad de hallar una correcta entre varias, sin confundir selección con adquisición de capacidad.

### P17 — Frontier Learning: Training LLM Reasoners at the Edge of Capability

- **Fuente:** [arXiv:2609.35426v1](https://arxiv.org/html/2609.35426v1). Lectura: método, tablas y apéndices D/F/G/H; código no auditado.
- **Resumen:** GRPO con generación procedural, prioridad por aciertos parciales, exploración y mutación de niveles.
- **Escala/evidencia:** modelos 3B–7B, cinco tareas, cuatro seeds; en Dice reporta 71,8% frente a SEC 33,3%, y 68,0% a tiempo comparable con PLR.
- **Límites:** mejoras variables; Dice consume más tiempo total. No prueba transferencia a 3M/96M ni razonamiento general. Los controles de niveles fijos también generan ejemplos nuevos.
- **Idea local:** F09, condicionado a F00/F04; aislar selección adaptativa de frescura de datos y cambios del optimizador.

### P18 — The ultimate guide to multi-harness RL

- **Fuente primaria/versionada:** [HF Space de FineEnvs](https://huggingface.co/spaces/FineEnvs/multi-harness-rl), artículo actualizado el 2026-10-01; [source snapshot f1d9a931](https://huggingface.co/spaces/FineEnvs/multi-harness-rl/tree/f1d9a9311956925696736dde3d2721bec733d5b9). Es una guía técnica con experimentos, no un paper arXiv. Lectura: secciones de token capture, training setup, rewards, resultados y limitations; código no auditado ni ejecutado.
- **Resumen:** OpenEnv captura token IDs, behavior logprobs y loss masks de agentes; Harbor aporta tasks/sandboxes y TRL ejecuta Async GRPO conservando los harnesses reales.
- **Resultados de los autores:** en SmolDataEnvs, LFM2.5-2.6B pasa de 42,2% a 54,2% pass@1 con multi-harness RL. Reportan 31% menos tool calls en tasks resueltas por ambos modelos. Usan correctness reward más un efficiency bonus condicionado a success; Qwen3.5-2B muestra riesgos de largas trayectorias y grupos sin reward contrast.
- **Escala/dominio:** modelos 2–2,6B, tareas con herramientas, cuatro harnesses; receta LFM de 1.000 steps con dos H100 por run, una para training y otra para vLLM.
- **Limitaciones:** una seed por configuración y exposición desigual a datos/compute; no establece superioridad general de multi-harness ni RL sobre SFT. El efecto aislado del efficiency bonus no está identificado. No demuestra transferencia a 3M/96M ni compatibilidad del stack con nuestra GPU AMD.
- **Interpretación/idea local:** F04 toma token fidelity, logprob checks, informative-group monitoring y controles de coste/evaluation; synchronous GRPO y binary correctness reward para aislar RLVR. Multi-harness y efficiency reward se posponen. F00 v1 detectó response-boundary failures y rewards uniformes: [resultado negativo conservado](output/f00-96m-20261002-142456-cee1e8/report.md). F00 v2 validó un nuevo boundary y readiness: [reporte](output/f00-boundary-test-20261002-154935-1b4bc2/report.md). F09 queda separado.
- **Resultado local F04 Stage1:** original96M FP32 sobreAMD,512 rollouts/16 updates,52/64 informative groups; logprob agreement,recovery ycheckpoint audit PASS. En32 tasks pareadas,sampled accuracy200/256→229/256,greedy31/32 ypass@8 100% sin cambio. Canary1024 targetsΔNLL+0,000200748 nats. Señal exploratoria de sampling con una seed,sin control additional SFT; no demuestra reasoning nuevo,superioridad de RL ni réplica del stack multi-harness. [Protocolo](docs/architecture/protocolo-f04-grpo-feasibility-20261002.md), [reporte y costes](output/f04-grpo-20261002-160826-741c93/report.md).
- **Resultado local F04 Stage2,readiness gate previo al training:** en nuevo dev128 tasks,original96.2M obtiene29/128 greedy y186/1024 samples;word problems de dos operaciones0/256 samples correctos bajo contrato estricto. Readiness por categoría FAIL;contraste SFT/GRPO no iniciado,0 updates,test reservado. El fallo combina formato/cálculo yno prueba imposibilidad de RL ni ausencia de capacidad en otros regímenes. [Protocolo](docs/architecture/protocolo-f04-stage2-20261002.md), [reporte negativo](output/f04-stage2-dev-20261002-165126-9603f0/report.md).
- **Follow-up F02/F04 autorizado:** nueva campaña local hasta21:15 Santiago. Foundation integer SFT completado:512 updates,1K ejemplos únicos/8 epochs,293K tokens procesados y34K completion targets;ΔNLL de lenguaje+0,00519 nats sobre4K targets. Recovery8 vs3+5 exacto. Dev nuevo en curso;sin conclusión de accuracy ni contraste SFT/GRPO todavía. No reinterpretar el gate negativo anterior. [Protocolo](docs/architecture/protocolo-posttraining-evening-20261002.md), [reporte inicial](output/f02-foundation-integer-20261002-172136-edabd0/report.md).

### P19 — Finetuning with Sampling: SFT Learns Better Than You Think

- **Fuente:** [arXiv:2610.02140v1](https://arxiv.org/abs/2610.02140), [Proyecto](https://aakaran.github.io/finetuning_with_sampling/), [Código](https://github.com/aakaran/finetuning-with-sampling). Lectura: web del proyecto, abstract y formulación matemática de projection sampling; código no ejecutado localmente.
- **Resultado de los autores:** proponen Projection Sampling, un algoritmo MCMC (Metropolis-Hastings) que transforma trazas off-policy con información privilegiada $\mathcal{C}$ en trayectorias con mayor verosimilitud bajo la policy base $p$. SFT estándar sobre estos datos proyectados iguala o supera a baselines de RL on-policy (GRPO, UFT) y destilación (OPSD) en generalización y retención de capacidades previas (mitigando catastrophic forgetting), permitiendo además resolver problemas donde la tasa base pass@k era 0% (aprendizaje más allá de simple distribution sharpening).
- **Límite:** requiere un mecanismo de propuesta fiel a las restricciones y evaluación iterativa de verosimilitud/candidatos MCMC, lo que añade coste de cómputo previo al entrenamiento. En modelos extremadamente reducidos (como nuestro 96M o 3M), la capacidad de seguir instrucciones complejas en-contexto para generar candidatos restringidos sin un modelo maestro externo es limitada.
- **Idea local:** F02/F04; ofrece una base teórica para entender por qué SFT con datos sintéticos rígidos u off-policy puede inducir olvido o baja generalización, y sugiere filtrar o proyectar los ejemplos aritméticos hacia regiones de alta verosimilitud del base model antes de SFT, en vez de depender exclusivamente de exploración estocástica no guiada en RLVR.

## Mantenimiento

Añadir fichas verificables y breves; una idea derivada debe tener entrada o referencia en el plan. No eliminar un paper porque una adaptación local falle. Para una nueva versión, indicar qué conclusión cambia y preservar la referencia anterior cuando explica una decisión.

| Fecha | Cambio material |
| --- | --- |
| 2026-10-02 | Creación con 16 fichas, alcance de lectura explícito y vínculos a resultados E7/E9/E11 y propuestas F00–F08. |
| 2026-10-02 | Añadida P17 tras revisar método, resultados y cómputo; vinculada a F09, sin extrapolar a nuestros tamaños ni autorizar entrenamiento. |
| 2026-10-02 | Añadida P18 con source snapshot de la guía HF, alcance de lectura y límites de transferencia; vinculada a F00/F04/F09. Normalizada terminología técnica en inglés. |
| 2026-10-02 | Vinculados P12/P18 al F00 v1 ejecutado: verifier y repeat checks PASS; response-boundary readiness pendiente. Sin extrapolar el score estricto a capacidad aritmética ni a eficacia de RL. |
| 2026-10-02 | Vinculados P12/P18 a F00 boundary v2 y F04 Stage1 ejecutados/auditados; distinguir corrección del evaluator,señal de sampled accuracy y efficacy frente a SFT aún no probada. |
| 2026-10-02 | P18 vinculado al readiness negativo de F04 Stage2 sobreword problems;sin training ni contraste de eficacia,no extrapolar el gate local a una limitación universal de RL. |
| 2026-10-02 | Vinculado follow-up foundation SFT/F04 autorizado hasta21:15;dev nuevo yretention/recovery antes deprincipales. |
| 2026-10-02 | Añadida P19 (Finetuning with Sampling) con alcance de lectura declarado y vinculación teórica a F02 (SFT de preparación) y F04 (RLVR vs SFT). |
