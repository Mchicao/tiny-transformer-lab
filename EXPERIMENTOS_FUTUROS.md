# Ideas y planes de experimentos futuros

Última revisión: **2026-10-02**. Fuentes: [PAPERS.md](PAPERS.md). Mantenimiento obligatorio: [AGENTS.md](AGENTS.md).

**Documento de planificación: ninguna propuesta nueva está autorizada para entrenar por aparecer aquí.** La creación y mantenimiento de estos archivos no amplía presupuestos ni permisos de campañas anteriores. La autorización nocturna tenía límite 2026-10-02 09:00 de Santiago; no inferir de ella permiso para estos planes.

Este archivo es el índice actual de prioridades. Los protocolos y resultados históricos permanecen en [docs/architecture/experimentos.md](docs/architecture/experimentos.md) y [plan looped histórico](docs/architecture/plan-looped-20261001.md); sus órdenes antiguos no sustituyen el estado resumido aquí. IDs Fxx son propuestas, no nuevos IDs Ex ejecutados.

**Terminología:** nombres técnicos en inglés y explicación en español; ver [glosario compartido](PAPERS.md#terminología-técnica). Usar, por ejemplo, `correctness reward`, `accuracy` y `starting checkpoint` para distinguir score del verifier, métrica de evaluación y pesos iniciales.

## Evidencia de partida

| Experimento | Estado verificado al revisar | Consecuencia para las próximas pruebas |
| --- | --- | --- |
| E1/E2/E4, looped 3M | K2 pierde contra denso; segunda pasada mejora sobre primera. Receta histórica sin cooldown y datos repetidos. [Registro](docs/architecture/experimentos.md). | No descartar recurrencia universalmente; repetir sólo con un contraste nuevo justificado. |
| E8, cooldown | Mejora media final aproximadamente −0,0835 nats, 3/3 seeds; adoptado localmente. [Registro](docs/architecture/experimentos.md). | Control de receta: GELU + AdamW + cooldown final 15%. |
| E9, preparación sintética | Retrieval no supera gate; grammatical tiene señal exploratoria, con coste extra y dificultad no igualada. [Reporte](output/e9-campaign-20261001-v1/report.md). | No adoptar retrieval ni atribuir causalmente la mejora a gramática. |
| E7, adaptación 96M | Pareja temprana v2 completa: ambos empeoran; revisión media abortada por FP16. FP32 y pilotos LR comprobados, pero no adaptación larga estable. [Pareja](output/e7x-early-pair-20261001-v1/report.md), [pilotos](output/e7-lr-pilot-20261002-v1/report.md). | Usar original final para nueva línea de tareas; no reutilizar pesos deteriorados como baseline control. |
| E11, datos únicos | Media final 2,878252 frente a E8 3,096211: −0,217959 nats, 3/3 seeds, iguales 6,144M tokens. [Reporte](output/e11-coverage-campaign-20261002-v1/report.md). | Nuevo control para este corpus; ampliar cobertura antes de sumar técnicas. |
| E10, búsqueda al inferir | Sólo una prueba greedy persistida; genera `18 + 76 = 94`, pero el extractor devuelve 18. [Resultado](output/e10-greedy-s42-i00-20261002-v1/results.json). | Corregir protocolo/evaluador antes de interpretar accuracy o construir rewards. |
| F00 v1, arithmetic baseline | Dos runs FP32 del original 96M reproducen exactamente 32 greedy + 256 samples; accuracy/pass@8 estrictos 0%, format validity 0%. Las completions suelen continuar con otra pregunta. [Reporte](output/f00-96m-20261002-142456-cee1e8/report.md). | Separar answer boundary de capacidad: el diagnóstico post hoc de primera línea no reemplaza el score. Versionar ese contrato en dev y usar un test nuevo antes de SFT/RL. |
| F00 v2, answer boundary | Newline/EOS validado en dev; test nuevo31/32 greedy,200/256 samples,pass@8 100%; full fresh-process repeat exacto,0 updates. [Reporte](output/f00-boundary-test-20261002-154935-1b4bc2/report.md). | Readiness PASS. El control dev pareado0/12→9/12 identifica boundary failures, no aprendizaje. Dataset/régimen distintos de v1. |
| F04 Stage1, synchronous GRPO | 512 rollouts/16 updates,52/64 informative groups; sampled accuracy200/256→229/256,greedy31/32 sin cambio; restore/evaluation audit exactos. [Reporte](output/f04-grpo-20261002-160826-741c93/report.md). | Feasibility PASS; señal exploratoria de sampling, sin control SFT ni réplicas. No adoptar RL ni extrapolar. |
| F04 Stage2, readiness dev | Original96.2M:29/128 greedy,186/1024 samples,44/128 informative groups;word problems0/256 samples correctos. [Reporte](output/f04-stage2-dev-20261002-165126-9603f0/report.md). | Gate por categoría FAIL;0 updates,contraste SFT/GRPO no iniciado,test reservado. Proponer SFT de preparación yreadiness nuevo antes de comparar. |

E7 fue continuación/adaptación mediante next-token prediction. Posteriormente se completaron F04 Stage1 RLVR y F02 foundation SFT por tareas; sus resultados se registran por separado. Las NLL de TinyStories y FineWeb-Edu/tinctura no son comparables directamente.

## Orden de trabajo propuesto

| ID | Prioridad | Idea | Estado / autorización | Dependencia |
| --- | --- | --- | --- | --- |
| F00 | Previa a tareas | Arithmetic verifier y baseline fiable | V1/v2 completados y auditados; boundary readiness PASS | Conservar contrato v2 para comparaciones |
| F01 | Alta | Más tokens únicos para el denso 3M | Propuesto; entrenamiento no autorizado | Corpus ampliado y protocolo de presupuesto |
| F02 | Alta | SFT aritmético del original 96M | Autorizado hasta21:15 Santiago;foundation integer SFT completado,dev en curso | Selección por readiness yretention;test reservado |
| F03 | Media | Separar tamaño de modelo y presupuesto | Propuesto; entrenamiento no autorizado | F01 y tokenizer común |
| F04 | Condicionada | RLVR frente a additional SFT | Stage1 PASS; piloto Stage2 detenido por readiness FAIL antes de principals | Starting checkpoint con señal en todas las categorías;contraste nuevo |
| F05 | Piloto barato | LoopCD sobre checkpoints 3M existentes | Propuesto; inferencia no ejecutada | Restauración fiable y dev/test separados |
| F06 | Condicionada | Reevaluar K2 con datos nuevos/más entrenamiento | Propuesto; entrenamiento no autorizado | Control denso del mismo régimen |
| F07 | Condicionada | Completar comparación E10 de búsqueda | Pendiente; no asumir vigencia del permiso anterior | F00, baseline capabilities y nuevo cap |
| F08 | Reserva | Ablaciones de receta y arquitecturas más complejas | Pospuesto; sin autorización nueva | Baseline y pregunta causal específicos |
| F09 | Condicionada | Curriculum adaptativo para RL procedural | Propuesto; entrenamiento no autorizado | F00 y baseline F04 estable con grupos informativos |

Siguientes pilotos recomendados: **F01 a 24,576M tokens** y **F02**, después de F00. Son líneas distintas: escala del pretraining y aprendizaje por tareas. No ejecutar toda la tabla como una campaña automática.

## Estimaciones de tiempo — 2026-10-02

Rangos de planificación para la RX 6750 GRE local, sin carga GPU concurrente y reutilizando entorno/checkpoints. Preparación incluye implementación, datos y comprobaciones previas; ejecución incluye el piloto, evaluación acotada y auditoría enfocada. No son benchmarks nuevos ni compromisos de finalización. No incluyen rehash completo de todo el inventario histórico en cada run; si se exige, sumar su coste real por separado. Fallos, descargas limitadas y cambios de alcance pueden extenderlos.

**Anclas medidas:** E11 procesa 6.144.000 tokens en 153,74–161,65 s de bucle, 212,98–221,47 s wall sin auditoría posterior: aproximadamente 38–40k tokens/s. [Métricas seed42](runs/e11-fresh-3m-20261002-037-seed42/metrics.json), [seed44](runs/e11-fresh-3m-20261002-039-seed44/metrics.json). El piloto denso 96M FP32 procesó 409.600 tokens en 147,98 s de bucle y 202,37 s wall, con contexto2048; esto no mide SFT corto. [Métricas](runs/e7f-early-dense-20261002-035-pilot100/metrics.json). E10 generó 24 tokens en 3,38 s para un único prompt, sin KV cache; no es un benchmark de rollouts batched. [Resultado](output/e10-greedy-s42-i00-20261002-v1/results.json).

| Piloto y alcance supuesto | Preparación estimada | Ejecución y verificaciones estimadas | Confianza |
| --- | --- | --- | --- |
| F00: parser + baseline de 32–64 preguntas, 8 muestras, horizonte24–32 | 30–60 min | 20–45 min | Media-baja; inferencia extrapolada de un prompt |
| F01: 3M, 24,576M tokens, una seed | 1–3 h para ampliar datos y adaptar contrato | 30–60 min por seed, de los cuales 10–11 min son bucle train | Media para train, menor para descarga/auditoría |
| F02: SFT 96M, 5k–20k ejemplos cortos, 1–3 epochs, evaluación acotada | 2–5 h | 30–120 min | Baja hasta medir batching, longitud y SFT loss |
| F04: primera feasibility probe de RLVR, 512–1.024 completions totales de24–32 tokens | 4–8 h tras F00/starting checkpoint estable | 1–3 h | Baja; rollouts y optimizer updates todavía no medidos; no incluye Stage 2 |
| F05: LoopCD, pocos valores de fuerza en dev y evaluación retenida acotada | 30–90 min | 15–45 min | Baja; adaptador y cap todavía no definidos |

Para F01, extrapolar sólo el bucle de E11 da **41–43 min por seed a 98,304M tokens**. A 24,576M, tres seeds suman aproximadamente31–33 min de bucle, pero ejecución con comprobaciones puede ocupar1,5–3 h, además de preparación compartida. El corpus E11 actual no alcanza estos presupuestos sin repetición: ampliar datos forma parte del tiempo, no está preparado aún.

F02 no incluye repetir tres seeds ni una búsqueda de hiperparámetros. F04 describe una sonda de viabilidad, no un entrenamiento RL hasta convergencia; un presupuesto largo o respuestas de razonamiento extensas aumentan mucho el tiempo. F03/10M requiere medir throughput de su configuración nueva antes de una estimación fiable; F06 depende del presupuesto seleccionado y F07 del cap de búsqueda. F08 no tiene duración asignada porque es una reserva de ideas.

Orden de magnitud para los siguientes pilotos F00 + F01 + F02, ejecutados secuencialmente: **5–13 h de trabajo total**, de las cuales sólo una parte es GPU entrenando. Implementar y verificar RL después añade aproximadamente **5–11 h** para la sonda F04 descrita. Recalcular tras preparar datos y medir el primer batch/rollout; no interpretar estos rangos como autorización ni garantía.

**Medición posterior, no extrapolación a Stage2:** F00 v2 dev/test/repeat50,78/38,70/43,24s. F04 Stage1 principal512 rollouts/16 updates y evaluation119,74s; peak allocated3022,2MiB. Preparación, continuity checks y auditoría aparte. Los rangos anteriores incluían implementación; no describen estos tiempos de ejecución medidos.

## F00 — Evaluación y verificador aritmético

- **Hipótesis:** parte de la accuracy observada puede depender del formato/extractor, no del cálculo.
- **Diseño:** preservar E10 histórico; definir un formato inequívoco y parser para un protocolo nuevo. Validar respuestas correctas, incorrectas, vacías, ambiguas, con ecuaciones y números negativos. No seleccionar el parser buscando aumentar el score del test.
- **Datos/métricas:** train/dev/test disjuntos por problemas; accuracy, formato válido y aciertos entre 8 respuestas. Reservar acarreo/mayor longitud como pruebas de generalización separadas.
- **Presupuesto/control ejecutado:** original 96M FP32, 32 tasks (24 test + 8 generalization), una greedy + ocho samples por task, horizon24, T=0,8/top-p=0,9; 1.800 s/7.152 generated tokens como caps por run incluyendo checks. Piloto y repeat completo en proceso nuevo; 298 completions por run, 596 contando la réplica, con 0 optimizer updates. No son 596 samples independientes ni una réplica con otra seed.
- **Autorización/estado:** usuario «ejecuta algun experimento», 2026-10-02; se seleccionó y comunicó F00 evaluation. Piloto v1 completado, sin training ni permiso inferido para F02/F04. [Protocolo fijado](docs/architecture/protocolo-f00-20261002.md), [run principal](runs/f00-96m-20261002-142456-cee1e8/results.json), [repeat](runs/f00-96m-repeat-20261002-142717-2972fb/results.json), [reporte](output/f00-96m-20261002-142456-cee1e8/report.md).
- **Gate/resultados:** PASS en 15 parser cases, split disjointness por identidad ordenada de task, full fresh-process repeat exacto y preservation de pesos/RNG/mode/source files. FAIL de readiness: greedy accuracy y pass@8 estrictos 0%, format validity 0%, informative-group fraction 0%; la generación suele continuar más allá de una ecuación correcta. Esto no demuestra ausencia de capacidad aritmética. La primera línea correcta en 25/32 greedy es un diagnóstico post hoc separado, no un score autorizado.
- **Decisión/follow-up:** conservar v1 y E10. Definir answer boundary/stop delimiter en dev y verificar con test nuevo, sin elegir reglas contra estas 32 tasks para subir accuracy. No iniciar RL con rewards uniformes; decidir si F02 es necesario después de un baseline con response boundary válido. La separación por `(a, operation, b)` ordenado no excluye equivalencias algebraicas ni overlap con pretraining externo.
- **Fuentes:** P12/P14/P16/P18 y artefacto E10. No se autorizó ni ejecutó el follow-up al cerrar v1; la autorización siguiente se registra por separado.
- **Follow-up v2 autorizado/completado:** usuario «CONTINUA HACIENDO EXPERIMENTOS», 2026-10-02. [Protocolo nuevo](docs/architecture/protocolo-f00-boundary-v2-20261002.md), [dataset44 tasks](data/arithmetic/f00-boundary-data-20261002-v2/tasks.json), [dev gate](output/f00-boundary-dev-20261002-154638-db1124/dev-gate.json), [test y reporte](output/f00-boundary-test-20261002-154935-1b4bc2/report.md), [repeat](runs/f00-boundary-repeat-20261002-155654-eda5bf/results.json). First nonempty line terminada en newline/EOS, parser v1 sin cambios; rule fijada antes del test. Canonical addition exclusions eliminan operandos invertidos;0 overlap con129 identidades históricas. Two-digit pool10–99 frente10–49 en v1: cambio de régimen explícito.
- **V2 resultados/decisión:** dev pareado boundary v1/v2 greedy0/12→9/12 con mismos pesos/prompts/seeds. Test nuevo31/32 greedy,200/256 samples,pass@8 32/32; format validity greedy32/32 y sampled229/256,1 truncada,27/32 informative groups. Dev/test/repeat50,78/38,70/43,24s;0 optimizer updates. Token IDs/logprobs/scoring exactos en fresh-process repeat y hashes preservados. Readiness PASS; esto permite verificar una F04 Stage1 propia, no demuestra mejora aprendida ni habilita Stage2 automáticamente.

## F01 — Curva de aprendizaje con más datos únicos

- **Hipótesis:** 6,144M tokens no permiten aprovechar completamente el denso 3M; ampliar datos/entrenamiento cambia la interpretación de negativos previos.
- **Control y variable:** GELU/AdamW/BPE4096/contexto256/validación de E11; variar presupuesto y el prefijo consumido de un corpus común versionado, manteniendo idéntico orden de datos en los prefijos compartidos. Reconocer que aquí aumenta exposición total y cobertura; E11 ya aisló cobertura a iguales tokens.
- **Presupuesto sugerido:** 6.144.000 / 24.576.000 / 98.304.000 tokens por seed. El primero ya tiene controles E11; verificar identidad antes de reutilizarlos. Piloto seed42, confirmación seeds42/43/44 sólo para contrastes seleccionados y explícitamente autorizados.
- **Protocolo:** nueva versión de datos suficiente para no wrap, misma fuente/revisión, tokenizer y validación; duplicados/overlap controlados. Schedule definido por presupuesto antes de ejecutar. No prolongar directamente un final con LR cero ni cambiar fuentes congeladas.
- **Métricas/gate:** NLL final fijo y curva, calidad narrativa sobre prompts retenidos, coste de preparación y entrenamiento. Fijar mejora mínima y techo de coste antes del piloto; seguir a 98,304M sólo si 24,576M aporta ganancia reproducible sin problemas de integridad. No llamar óptimo a un presupuesto probado.
- **Fuentes:** P01/P02/P05 y E11. Resultado/decisión: pendientes.

## F02 — SFT aritmético sobre el 96M original final

- **Hipótesis:** ejemplos supervisados pueden enseñar formato y operaciones simples sin necesitar entrenar otro modelo desde cero.
- **Control:** original final checkpoint weights tinctura-v1, revisión fijada en E7/E10; nunca pesos adaptados/degradados de E7. Mantener tokenizer y flujo numérico validado en FP32.
- **Variable/datos:** SFT con respuesta corta y correcta; propuesta de 5.000–20.000 ejemplos programáticos de sumas/restas fáciles. Fijar un tamaño para el primer piloto, particiones y política de loss sobre completion tokens antes de entrenar; ampliar tamaño es otro contraste.
- **Presupuesto:** número de ejemplos no equivale a tokens de entrenamiento. Tokens, epochs, LR, actualizaciones y límite de tiempo pendientes de protocolo y benchmark; no lanzar un sweep implícito.
- **Métricas/gate:** accuracy y formato sobre test independiente, acarreo/longitud fuera del régimen train, NLL de lenguaje retenida, tiempo y memoria. Umbrales de mejora/retención prefijados; detener por no finitos u omisiones. Si no aprende tareas fáciles, diagnosticar datos/loss antes de escalar o hacer RL.
- **Dependencias/fuentes:** F00; continuidad propia de la receta; P14/P15/P19. Resultado/decisión: pendientes.
- **Nueva prioridad tras dev F04 Stage2:** preparar formato yoperaciones con negativos/word problems mediante SFT antes de otro contraste RL. Es una propuesta de protocolo propio,distinta del control SFT principal detenido por el gate;no se ejecutó ni se hereda automáticamente su presupuesto. Evaluar en dev nuevo si produce respuestas válidas/correctas,con retention checks;después ambos arms del contraste compartirían ese starting checkpoint.
- **Autorización posterior,2026-10-02:** usuario «realiza experimentos hasta esa hora aprox,te autorizo a hacer todo lo que creas necesario» hasta21:15 Santiago,seguido de«continue». [Protocolo campaña](docs/architecture/protocolo-posttraining-evening-20261002.md). Foundation integer SFT:original96.2M,train1024 reutilizado,dev nuevo128 canónico disjunto,AdamW3e-6,batch16,≤512 updates/8 epochs. Receta [fijada](output/posttraining-evening-20261002-v1/recipes/foundation-integer.json),deadline UTC00:15Z,machine guard60s. Recovery8 vs3+5 antes del principal;no usar verification weights. Si readiness PASS,comparar additional SFT/GRPO desde foundation común;variantes ydecisiones quedan separadas del negativo Stage2 v1. Estado:foundation integer completado,512 updates/8 epochs,293K tokens procesados y34K completion targets;ΔNLL+0,00519 nats en4K targets. Recovery8 vs3+5 exacto,16 updates de verificación aparte. [Reporte inicial](output/f02-foundation-integer-20261002-172136-edabd0/report.md). Dev yselección en curso;test reservado.

## F03 — Capacidad frente a entrenamiento

- **Hipótesis:** un modelo mayor puede aprovechar los datos mejor, pero necesitar más entrenamiento; no basta aumentar parámetros a presupuesto corto.
- **Diseño:** denso 3M frente a aproximadamente 10M, ambos desde cero, mismos datos/tokenizer/validación y receta de referencia. `configs/10m.json` usa vocab8192: no utilizarlo tal cual frente al BPE4096 si se pretende aislar arquitectura; fijar y contar los parámetros reales de la configuración nueva.
- **Presupuesto sugerido:** 24.576.000 y, condicionado, 98.304.000 tokens por modelo/seed. Control de tamaño a iguales tokens; adicionalmente reportar calidad por tiempo/FLOPs. Calibración específica de LR, si necesaria, separada y contabilizada.
- **Métricas/gate:** NLL final, curvas y tareas narrativas comunes; memoria y coste. Continuar si capacidad extra mueve la frontera dentro del techo previamente fijado. No mezclar el 96M ya preentrenado en esta comparación causal de tamaño.
- **Dependencias/fuentes:** F01; P01/P02/P06/P09. Resultado/decisión: pendientes.

## F04 — RLVR frente a additional SFT

- **Hipótesis:** desde el mismo starting checkpoint, RLVR con binary correctness reward mejora held-out accuracy más que additional SFT dentro de un compute budget comparable, sin degradar language retention. El starting checkpoint debe producir algunas respuestas correctas y grupos con reward contrast; esto es una condición local de viabilidad, no un tamaño mínimo universal para RL.
- **Starting checkpoint y arms:** elegir y congelar antes del contraste el original 96M o un checkpoint estable de F02, con revisión/hash y tokenizer fijados. A: frozen baseline, sin optimizer updates; B: additional SFT; C: RLVR. B y C parten de los mismos pesos y del mismo espacio de problemas de training; SFT usa targets correctos y RLVR sampled completions evaluadas por el verifier. Si se usa F02, registrar su coste como preparación común. Cambiar el starting checkpoint constituye otro contraste; SFT no es un requisito universal de RL.
- **Task y reward:** aritmética sencilla; binary correctness reward = 1 sólo para una respuesta final inequívoca y correcta según F00, y 0 para incorrecta, vacía o ambigua. Registrar format validity por separado. Sin format reward, length reward, efficiency bonus, reward model ni LLM judge en el primer contraste; cualquier reward shaping posterior requiere una ablación independiente.
- **Training loop:** synchronous GRPO: mantener la behavior policy fija durante la generación de cada grupo, calcular rewards y advantages, y después ejecutar los optimizer updates previstos. Fijar group size, advantage normalization, LR, clipping, KL/reference policy si aplica, precision y updates por batch en el protocolo. Async GRPO, OpenEnv/Harbor y multi-harness quedan pospuestos; F09 mantiene separado el adaptive curriculum.
- **Generation contract:** conservar prompt/completion token IDs originales, attention/loss masks, per-token behavior logprobs y policy checkpoint/version; no reconstruir targets mediante decode→re-tokenize. Fijar temperature, top-p/top-k, EOS, max completion tokens y política de truncation. Distinguir logprobs de la sampling distribution de las de la policy usada en la loss y explicitar cualquier correction. Antes de updates, verificar alignment y agreement de logprobs al recalcular con los mismos pesos, contexto, distribución y precision, con tolerancia numérica prefijada. Aplicar la policy loss sólo a completion tokens válidos; excluir prompt/padding y prefijar el tratamiento de completions truncadas. La loss y sus gradient checks deben tener una comprobación runnable de comportamiento.
- **Stage 1 — Feasibility probe:** después de F00 y del rollout benchmark en la GPU real, fijar un cap pequeño y un mínimo de informative groups (grupos con rewards diferentes), así como el máximo de batches consecutivos sin reward contrast. Comprobar baseline accuracy, logprob agreement, finite loss/gradients y estabilidad durante updates acotados. La sonda no prueba eficacia; contabilizar su coste y usar datos de dev para cualquier decisión. Al pasar a Stage 2, restaurar el starting checkpoint común en B/C, sin dar a C updates extra ocultos de la sonda.
- **Stage 2 — Efficacy comparison:** comparar A/B/C en el mismo held-out test, con greedy accuracy y pass@8 como métricas distintas, y un protocolo de decoding común. Mantener train/dev/test disjuntos por problemas e incluir un generalization test separado con carry y longitudes fuera del régimen de training. Usar dev para elegir checkpoints y reservar test para el contraste final. Prefijar minimum accuracy gain, uncertainty reporting, language-retention tolerance y cost ceiling antes del training; confirmar el contraste seleccionado con réplicas explícitamente autorizadas.
- **Compute budget:** fijar caps por arm después del benchmark; registrar prompts, completions, generated tokens, processed tokens, tokens incluidos en la loss, optimizer updates, wall time y peak VRAM. Para B/C comparar curvas por coste medido en el mismo hardware y reportar también tokens; igual número de updates no equivale a igual compute. Separar preparación, rollout generation, verifier/evaluation y training; el frozen baseline sólo incurre en evaluation. Los rangos de tiempo previos son estimaciones, no un benchmark de esta receta ni autorización.
- **Metrics y stop criteria:** además de accuracy/pass@8, registrar format validity, informative-group fraction, completion length, truncation rate, completion diversity y held-out language NLL. Reward train no sustituye evaluation. Detener por non-finite loss/gradients, collapse, pérdida de language retention fuera de tolerancia o ausencia persistente de informative groups. Si falta señal, revisar task difficulty o starting checkpoint antes de ampliar; cambiar esos factores exige un contraste nuevo. Adoptar sólo si C supera B en held-out accuracy dentro de los umbrales de coste/retention y la mejora se confirma; un negativo local no refuta todo RL.
- **Dependencias/fuentes:** F00, numerical stability y F02 si aporta el starting checkpoint; [P14/P15/P16/P18/P19](PAPERS.md), [TRL GRPO reference v0.28.0](https://huggingface.co/docs/trl/v0.28.0/en/grpo_trainer). Stage1 ejecutado bajo autorización nueva; Stage2 efficacy comparison sigue pendiente de protocolo, caps y thresholds propios.
- **Readiness tras F00 v1:** response-boundary gate pendiente e informative-group fraction 0% bajo ese contrato. El diagnóstico de primera línea no se convierte en correctness reward. Resolver y versionar evaluation antes de la feasibility probe; no interpretar este resultado como un fallo de RL.
- **Stage1 autorización/protocolo:** usuario «CONTINUA HACIENDO EXPERIMENTOS», 2026-10-02; se anunció scope condicionado a F00 dev, máximo512 rollouts/16 optimizer updates. [Protocolo](docs/architecture/protocolo-f04-grpo-feasibility-20261002.md), [training64 tasks](data/arithmetic/f04-data-20261002-v1/tasks.json), [run](runs/f04-grpo-20261002-160826-741c93/results.json), [reporte/auditoría](output/f04-grpo-20261002-160826-741c93/report.md). Original final96M FP32,64 prompts únicos disjuntos,group8,T1/top-p1,AdamW3e-6,clip ratio0,2,KL0. Una seed; sin SFT previo ni reutilización de verification weights.
- **Stage1 resultado:** 16/16 updates,52/64 informative groups (gate≥8),finite nonzero gradients y logprob max absolute error0,000075341 nats (cap0,0005). Canary1024 targets:2,129828453→2,130029202 nats,Δ+0,000200748(cap+0,05). Post-test mismos32 problems/seeds v2:greedy31/32→31/32,pass@8 100%→100%,sampled accuracy200/256→229/256(78,125%→89,453%),sampled format validity229/256→241/256,truncation1/256→0. Cero greedy tasks ganadas/perdidas. Señal descriptiva sobre sampling, no mejora de greedy ni prueba de reasoning nuevo.
- **Integridad/coste:** [Recovery PASS](output/f04-recovery-v2-20261002-160613-1e1602/recovery.json),3 vs1+2 fresh processes exactos y identity/corruption rechazadas. Primer check falló por Python/NumPy RNG no inicializados; [fallo y fuentes preservados](output/f04-recovery-20261002-160313-dcabc8/failure.json). Preparación total384 candidates/12 updates separados. Audit final restaura weights/optimizer/RNG/canary y repite288 held-out completions exactas,0 updates. Principal119,744s,5810 rollout generated tokens/244272 processed forward tokens;18304 padded training tokens/4682 loss-mask targets,canary/evaluation/audit aparte.
- **Decisión:** feasibility PASS. No adoptar RL ni aumentar el cap para perseguir el score. Stage2 necesita additional SFT con compute comparable, réplicas y test nuevo; baseline casi al techo y test ya observado en F00. La canary pequeña no certifica retención general. Esta ejecución no autoriza automáticamente F02/F09 ni replica multi-harness.
- **Stage2 autorización yprotocolo fijado:** usuario «empieza ese experimento»,2026-10-02,tras propuesta de un piloto one-seed/~1K train/original96.2M/baseline-SFT-GRPO ybenchmark para compute matching. [Protocolo](docs/architecture/protocolo-f04-stage2-20261002.md). Dataset [manifest](data/arithmetic/f04-stage2-data-20261002-165106/manifest.json):1024 train,128 dev,192 test,64 generalization;1408 identidades únicas,0 overlap canónico histórico. Nuevo régimen:negative subtraction,carry,signed addition yword problems de dos operaciones;no comparar directamente con Stage1. Parser/boundary congelados antes de dev;11 parser cases,target roundtrip ySFT gradient direction PASS.
- **Stage2 resultado/stop:** [Dev run](runs/f04-stage2-dev-20261002-165126-9603f0/results.json), [reporte yverification](output/f04-stage2-dev-20261002-165126-9603f0/report.md):29/128 greedy(22,66%),186/1024 samples(18,16%),44/128 informative groups. Carry26/32 greedy y162/256 samples;negative subtraction2/32 y17/256;signed addition1/32 y7/256;word problems0/32 y0/256. Global gates pasan,pero falla el criterio prefijado de algún sample correcto por categoría. Sólo2/256 word samples tienen formato válido;también hay errores de cálculo. No inferir ausencia universal de capacidad ni fallo de GRPO.
- **Stage2 estado/decisión:** detener antes de calibration/recovery/training principals,0 optimizer updates. Test/generalization preparados pero no evaluados. No se ejecutó SFT vsGRPO ni se identificó ganador. Dev186,780s;retention baseline4096 targets en results;fuentes/dataset/assets preservados por hash. Proponer SFT preparatorio ydev nuevo antes de fijar otro contraste desde checkpoint común;no cambiar retrospectivamente gate/parser para salvar el piloto. Presupuesto de Stage2 no consumido,no hay autorización automática para otra campaña.

## F05 — LoopCD sin actualizar pesos

- **Hipótesis:** el refinamiento primera→segunda pasada del K2 3M permite mejorar la salida mediante contraste.
- **Diseño/control:** restaurar el mismo checkpoint para decodificación habitual y contrastiva, con prompts/precisión/protocolo idénticos; fuerza elegida en dev separado, sin afinar contra test. Comparar Hidden y Logits sólo dentro de un cap de inferencia definido.
- **Presupuesto/métricas:** cero entrenamiento; cap de inferencia pendiente. NLL, tareas/generaciones retenidas y tiempo real, incluyendo head y almacenamiento de estados. Mejora visual de un ejemplo no prueba calidad.
- **Gate:** continuar si hay ganancia en evaluación independiente sin coste o degradación inaceptable. K2→K1 elimina el contraste; no prometer el ahorro de profundidad del paper. E7 temprano no tiene una trayectoria útil para extrapolar sin diagnóstico adicional.
- **Fuentes:** P08 y resultados K2 históricos. Resultado/decisión: pendientes.

## F06 — Recurrencia en el nuevo régimen de datos

- **Hipótesis:** la pérdida de K2 a presupuesto corto puede cambiar con datos únicos/cooldown y suficiente entrenamiento.
- **Diseño/control:** comparar denso3M y K2 con el mismo corpus, presupuesto, inicialización base y receta; un control denso de profundidad comparable si la pregunta es eficiencia de cómputo. No usar E1 histórico contra E11 como contraste de arquitectura.
- **Presupuesto:** elegir primero un presupuesto de F01 y fijar techo por brazo; no escalar a K4 automáticamente. Separar entrenamiento desde cero de cirugía/adaptación del 96M.
- **Métricas/gate:** NLL final, NLL por pasada, tareas independientes, tiempo y pico VRAM. Superar un control deteriorado no basta. Abrir más profundidad sólo ante ganancia confirmada y coste aceptable; un negativo no refuta todos los modelos recurrentes.
- **Fuentes:** P06/P07/P09/P10/P11; protocolos históricos E1–E7. Resultado/decisión: pendientes.

## F07 — Retomar E10 con una evaluación fiable

- **Hipótesis:** búsqueda con potencia/tempering encuentra respuestas correctas más eficazmente que alternativas con trabajo comparable.
- **Diseño:** conservar protocolo y prueba históricos; versionar cualquier cambio de extractor. Original 96M fijo, greedy/sampling como referencias baratas; best-of-N, power-MH y tempering como búsqueda. No usar respuesta correcta del verificador para elegir el candidato final.
- **Presupuesto:** cap histórico de búsqueda 16.000 tokens forward por ítem/seed, contando prefijos; es antecedente, no autorización renovada ni igualdad exacta de FLOPs. Definir nuevo cap y tareas antes de ejecutar.
- **Métricas/gate:** accuracy, propuestas/aceptaciones/swaps y coste real. Sólo ampliar si mejora a trabajo comparable y pasan las invariantes del sampler; no atribuir una sola respuesta a razonamiento general.
- **Dependencias/fuentes:** F00 y baseline medible; P12, [protocolo E10](docs/architecture/protocolo-e10-piloto-20261002.md). Resultado/decisión: pendientes.

## F08 — Reserva de ideas, sin barridos abiertos

| Idea | Qué falta para justificarla | Fuente |
| --- | --- | --- |
| Menor batch/acumulación | Baseline nuevo; contrato de hiperparámetros en tokens, coste y continuidad. | P04 |
| Repetir Muon o SwiGLU | Cambio de régimen y pregunta concreta; controles separados, sin promesa de transferencia. | P03 y registro FFN |
| MIR | Corpus realmente limitado; control de regularización y coste auxiliar. | P05 |
| Preparación sintética retrieval | Demostrar aprendizaje de la tarea; igualar o contabilizar dificultad/coste y comparar contra más texto natural. | P13 |
| Loop selectivo/escalado residual | Recuperar protocolo E6, calibrar por separado y aislar cada cambio. | P07/P10/P11 |
| MoE / difusión / otro modelo externo | Ganancia esperada frente a complejidad, memoria, hardware y nueva pila; autorización específica. | P07/P09/P10/P11 |

Estas ideas necesitan su propia hipótesis, presupuesto, métricas y gates antes de salir de reserva. No convertir una lista bibliográfica en una lista de entrenamientos obligatorios.

## F09 — Curriculum adaptativo en la frontera de capacidad

- **Hipótesis:** seleccionar niveles donde el modelo aún obtiene aciertos parciales mejora la accuracy por coste frente a selección fija o aleatoria. Es una hipótesis local, no una transferencia demostrada por P17.
- **Modelo/datos:** mismo checkpoint original 96M o starting checkpoint de F02 elegido antes del contraste; tareas aritméticas verificables. Caracterizar niveles por dígitos, operaciones y acarreo; medir dificultad empírica sin asumir orden monotónico. Generar ejemplos nuevos en todos los brazos.
- **Control/variable:** A, niveles fijos con muestreo uniforme; B, niveles nuevos aleatorios dentro del mismo espacio permitido; C, selección adaptativa de niveles. Mantener GRPO, advantage normalization, LR, precisión, batch y candidatos por prompt idénticos. Comenzar C sin mutación del espacio; si aporta valor, comparar C frente a C+mutación como una segunda prueba. Esta simplificación no reproduce el algoritmo completo del paper.
- **Presupuesto:** pendiente de protocolo y benchmark de F04. Prefijar caps por brazo de candidatos, tokens y tiempo; contabilizar preparación, generación, evaluación y actualización. Reportar curvas a tiempo y tokens comparables, además de steps; no usar las horas de H200/GH200 del paper como estimación local.
- **Métricas:** accuracy greedy y pass@8, fracción de grupos con rewards diferentes, aciertos por nivel, longitud, coste y language retention. Evaluación fija con ejemplos y configuraciones de generador excluidos de entrenamiento; incluir niveles fáciles y difíciles para detectar especialización.
- **Gate:** avanzar sólo si C supera a B en evaluación retenida con coste comparable y retención dentro de umbrales prefijados; confirmar con tres seeds después de una sonda. Detener por no finitos, colapso o ausencia persistente de grupos informativos; facilitar tarea o mejorar el starting checkpoint sin ampliar automáticamente el presupuesto.
- **Dependencias/fuente:** F00, F04 estable y P17. Estado: propuesto; sin autorización de entrenamiento. Resultado/decisión: pendientes. Pospuesto hasta disponer de baseline RL; no reemplaza F01/F02.

## Reglas de actualización y registro

Estados: `propuesto`, `preparado`, `autorizado`, `en curso`, `completado`, `bloqueado`, `descartado` o `sustituido`; mantener autorización como dato separado. Preparación o una mejora observada no concede permiso de ejecución. Marcar qué dependencia bloquea cuando corresponda.

Al cambiar estado, añadir fecha, motivo y enlace a protocolo/artefacto. Al obtener resultados, actualizar síntesis aquí, vínculo del paper y registro histórico; preservar informes originales. Si una propuesta se sustituye, conservar su ID y señalar sucesora.

| Fecha | Cambio material |
| --- | --- |
| 2026-10-02 | Creación del índice F00–F08 con evidencia local, prioridades de datos y SFT/RL, límites de autorización y gates pendientes de protocolo. |
| 2026-10-02 | Añadidos rangos de tiempo para pilotos, anclas medidas E11/E7/E10 y límites de extrapolación; no se ejecutaron experimentos nuevos. |
| 2026-10-02 | Añadida F09 a partir de P17: selección adaptativa tras baseline RL, controles de niveles/frescura y coste real; sin ejecución ni presupuesto autorizado. |
| 2026-10-02 | Revisada F04 con P18: arms desde un starting checkpoint común, synchronous GRPO, generation contract, feasibility/efficacy gates y binary correctness reward; terminología técnica en inglés. Sin ejecución ni training autorizado. |
| 2026-10-02 | Ejecutado F00 v1 y full fresh-process repeat bajo «ejecuta algun experimento»: evaluation reproducible, score estricto 0% por response-format/boundary failures; diagnostic separado. Follow-up de answer boundary pendiente; F02/F04 sin ejecutar. |
| 2026-10-02 | Continuación autorizada: F00 boundary v2 preregistrado/dev/test/repeat PASS; F04 Stage1 completado con512 rollouts/16 updates y audit exacto. Sampled accuracy mejora exploratoria,greedy sin cambio; fallo RNG de preparación preservado y corregido. Stage2 pendiente de control SFT,réplicas y protocolo propio. |
| 2026-10-02 | Piloto Stage2 autorizado:dataset nuevo ydev ejecutados,readiness FAIL por0/256 word samples correctos;principals no iniciados,0 updates,test reservado. Preservado negativo ypriorizado SFT preparatorio como propuesta distinta. |
| 2026-10-02 | Ventana autorizada hasta21:15 Santiago:foundation integer SFT completado yrecovery exacto;dev nuevo en curso,contraste RL condicionado a readiness. |
| 2026-10-02 | Vinculado P19 (Projection Sampling / SFT vs RL) como fuente teórica para preparación de datos en F02 y contraste en F04. |
