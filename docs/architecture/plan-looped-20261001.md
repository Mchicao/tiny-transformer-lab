# Plan de experimentos — SML looped (propuesto, pendiente de autorización)

Fecha: 2026-10-01. Este documento es una propuesta; ningún entrenamiento aquí descrito está autorizado por existir. La preparación autorizada vigente no incluye entrenamientos. La ejecución de cada fase requiere instrucción explícita del usuario.

## Preguntas

- **P1 · Eficiencia.** ¿Un transformer con bloques compartidos ejecutados K veces ("looped") aprende más por unidad de cómputo que un denso equivalente? Métricas: NLL alcanzado al mismo coste total (layer-pass·tokens, wall-clock), y looped-3M frente a denso-6M con idéntico coste por token.
- **P2 · VRAM igual, mejor aunque más lento.** Con los mismos 3.000.384 parámetros (misma VRAM de pesos/optimizador) y K× cómputo por token, ¿mejora el NLL final? ¿Dónde satura K?

Fundamento: *Looped Diffusion Language Models* (arXiv:2605.26106) y *Looped Diffusion Transformer* (arXiv:2609.40305) — el looping ingenuo falla; la **supervisión profunda entre bucles** es el estabilizador clave; los bucles profundos corrigen errores de bucles previos. Para LM autorregresivos: *Scaling Test-Time Compute with Looped Transformers* (Geiping 2025) y *Reasoning with Looped Transformers* (Saunshi 2025) — receta con inyección de entrada y embeddings de bucle.

## Diseño del modelo looped

Sobre el modelo vigente (d=192, 5 capas, 6 cabezas, contexto 256, vocab 4.096, manual attention, FP16):

- Las **mismas 5 capas se ejecutan K veces** por token; K∈{2,4}.
- **Inyección de entrada**: en cada iteración se reinyecta el embedding del token (h₀ = emb(x); h_{k+1} = Block(h_k + emb(x))).
- **Embedding de bucle**: vector de d=192 por iteración, sumado a h_k. Únicos parámetros nuevos (K·192 ≈ 384–768; despreciables). Inicialización PCG64 con stream separado, patrón del artefacto SwiGLU seed43.
- **Pérdida**:
  - Receta A (*naive*): NLL sólo en la última iteración.
  - Receta B (*supervisión profunda*): media uniforme de la NLL de todas las iteraciones. Cada iteración predice el siguiente token desde su estado.
- Logits de generación: última iteración en ambas recetas.
- AdamW idéntico al vigente (LR 3e-4, betas 0,9/0,95, wd 0,1, warmup 10, clip 1,0, acumulación 4, microbatch 4). Sin QK-Norm (descartada en campaña previa). FFN: la ganadora de seed44 (criterio abajo).

## Matriz de experimentos

Presupuesto común: 6.144.000 tokens / 1.500 actualizaciones / eval cada 50 pasos con milestones históricos / NLL ponderada sobre 39.275 targets / seeds {42, 43, 44} pareadas con los runs existentes.

| ID | Configuración | K | Coste/token | Seeds | Runs | Est. bucle train |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | Denso 3M (existente) | 1 | 1× | 42/43/44 | 0 nuevos (seed44 en curso) | 130 s ×3 (ya ejecutados) |
| E1 | Looped, pérdida final (A) | 2 | 2× | 3 | 3 | ~270 s ×3 |
| E2 | Looped + supervisión profunda (B) | 2 | 2× | 3 | 3 | ~270 s ×3 |
| E3 | Looped receta ganadora | 4 | 4× | 3 | 3 | ~540 s ×3 |
| E4 | Denso 6M, 10 capas (matched-compute vs E1/E2) | 1 | 2× | 3 | 3 | ~260 s ×3 |
| E5 | *Reformulado tras E1–E4:* continuación pareada ×2 (12.288.000 tokens) de E1 + denso-3M + denso-10L, 3 seeds; ×4 condicionado a que las brechas se muevan | — | — | 3 | 9 | ~40 min |

- **E4 es la comparación de eficiencia limpia**: denso-6M tiene 10 layer-passes por token, igual que looped-K2. Mismo coste de entrenamiento por token; el looped tiene la mitad de parámetros. Si looped-K2 ≥ denso-6M, el looping "pega por encima de su peso" de parámetros (la afirmación central de los papers).
- **Análisis gratis desde las curvas existentes**: NLL-por-FLOP comparando denso-3M@3.072M tokens vs looped-K2@3.072M tokens (igual wall, igual cómputo total, distinto reparto datos/bucles).
- **Elección de FFN para E1–E3**: si seed44 rompe el empate a favor de una variante (mejor NLL medio de mejores checkpoints en 2 de 3 seeds), se usa esa; si mantiene el empate (|Δ| < 0,01 en media), se usa GELU por simplicidad y se documenta.

## Métricas nuevas por run

El registro actual (`metrics.json`, `results.jsonl`) se extiende sin romper formato:

- `loop_k` y `loss_recipe` en identidad de run y checkpoint (los checkpoints de recetas distintas son incompatibles entre sí; el loader debe rechazarlos).
- `loops_nll`: NLL de **cada iteración de bucle** en cada evaluación (37 por run). Evidencia de refinamiento iterativo medido: si NLL(bucle k+1) < NLL(bucle k) de forma sostenida, el modelo aprende a refinar; si no, K extra es cómputo desperdiciado. **No interpretar como "razonamiento latente"** — la mejora media entre bucles no muestra qué tokens mejoran ni predice exactitud en tareas (SMELT/TaH2 miden NLL y tareas por separado y difieren).
- `loops_delta_distribution`: distribución por token de loss(bucle 1) − loss(bucle 2) en validación (cuantiles + fracción de tokens que empeoran), no sólo medias — TaH2 encuentra que muchos tokens no se benefician de iteraciones extra.
- **Sonda de tareas pequeña, retenida y prefijada** (no GSM8K/AIME, fuera de escala para TinyStories): copia literal, patrones cortos, referencia larga y composición simple sobre el split de validación, con exactitud exact-match y pérdida de respuesta. Complementa la NLL global, que no captura la ventaja tipo razonamiento que citan los papers.
- `layer_pass_tokens` = tokens × layer-passes reales (para eficiencia-por-FLOP; distinto por variante: 5 denso, 7 selectivo, 10 full-K2).
- Norma global de gradiente por actualización (barato, detecta erosión de gradiente a través de K bucles).

Métricas existentes que se mantienen: NLL final/mejor, cruces de umbral (añadir 3,15 bajo el actual 3,20/3,3/3,5), train_seconds, wall_seconds, tokens/s, pico MiB, generaciones fijas (greedy + T=0,8/top-p 0,9).

## Criterios de decisión (gates)

- **G-estabilidad** (duro, todas las fases): 0 omisiones, pérdidas/pesos finitos, restauración bit a bit en proceso nuevo (5 vs 2+3), auditoría post-run sin entrenar. Igual que el contrato vigente.
- **G-supervisión**: E2 vs E1 pareado por seed. Adoptar receta B si Δ best-NLL medio ≤ −0,01 nats; si |Δ| < 0,01, quedarse con A (más simple) y documentar.
- **G-P2**: receta ganadora K=2 supera a denso-3M por ≥ 0,02 nats de best-NLL medio y en ≥ 2 de 3 seeds pareadas → "misma VRAM, mejor rendimiento aunque más lento" confirmado. Reportar Δwall (esperado ~2×).
- **G-K**: K=4 mejora ≥ 0,02 sobre K=2 → proponer K=8 como nueva autorización; mejora < 0,01 → techo de K documentado, no escalar más.
- **G-P1**: looped-K2 vs denso-6M al mismo coste: si best-NLL(looped) ≤ best-NSL(denso-6M) − 0,01 → el looping es superior a comprar profundidad con parámetros. Si denso-6M gana claramente, el looping en esta escala no paga y se documenta como negativo — resultado válido.
- **E5** sólo se propone si E1–E4 muestran mejora positiva pero con señal de que emerge con presupuesto (curvas looped aún descendiendo cuando el denso meseta). La literatura sugiere que el beneficio del loop crece con el presupuesto de entrenamiento.

Con n=3 seeds sólo comparación descriptiva pareada; sin tests de significancia. Umbral de acción 0,02 nats ≈ borde superior de la banda de ruido entre seeds observada en la campaña FFN (±0,02).

## Coste y memoria estimados

| Brazo | Pico VRAM esperado | Bucle train total (3 seeds) |
| --- | --- | --- |
| E1/E2 looped K=2 | ~420–480 MiB (activaciones ×2) | ~27 min |
| E3 looped K=4 | ~700–900 MiB | ~27 min |
| E4 denso 6M | ~290–310 MiB | ~13 min |
| E5 extensión ×2 | según ganador | ~36 min |

Total E1–E4 ≈ 67 min de bucle train (~90–110 min wall con evaluaciones). E5 opcional ~1 h más. Todo muy por debajo de los 10.224 MiB: la VRAM no es el límite en ningún brazo; el coste real es wall-clock, que es precisamente la variable de P2.

## Implementación (fase 0, sin entrenamiento)

1. `src/looped.py`: módulo nuevo; **no tocar** `src/model.py`, `src/data.py`, `src/training.py`, `scripts/train.py`, `scripts/experiment.py` (sus hashes forman la identidad de checkpoints existentes). Patrón ya usado por `src/swiglu.py`.
2. `configs/3m-looped-k2.json`, `configs/3m-looped-k4.json`, `configs/6m-dense.json` (d=192, 10 capas, misma familia).
3. `scripts/looped_experiment.py`: reutiliza el bucle y AdamW existentes vía fábrica limitada al proceso (patrón `scripts/replicate.py`).
4. Artefactos de inicialización: para cada seed, reutilizar el artefacto aleatorio existente (mismos 5 bloques) + embedding de bucle con stream PCG64 separado; denso-6M con artefacto propio por seed.
5. `scripts/utils/check_loop.py`: continuidad 5 vs 2+3 en procesos nuevos, 10 actualizaciones / 40.960 tokens separados del presupuesto, `--audit-run` sin entrenar, rechazo de identidad K/receta incompatible.
6. `scripts/benchmark.py`: extensión opcional con flag looped; el contrato actual queda intacto.
7. Validación de fase 0 en HIP local: forward/backward looped, VRAM medida vs estimada, `loops_nll` con pesos aleatorios (debe empezar ≈ ln(4096) ≈ 8,32 en todas las iteraciones).

## Riesgos y mitigaciones

- **Looping ingenuo inestable** (hallazgo central de ambos papers): E1 existe precisamente para medirlo; E2 es la mitigación; `loops_nll` lo diagnostica por evaluación.
- **Erosión de atención/gradiente a través de K bucles**: clip 1,0 + GradScaler ya activos; norma de gradiente registrada; QK-Norm no se reintroduce (fracasó en la campaña previa y añadiría una segunda variable).
- **Confundir receta con K**: E3 sólo se ejecuta con la receta ganadora de E1 vs E2.
- **Señal dependiente del presupuesto**: E5 con gate explícito; no extender "por si acaso".
- **Interferencia con seed44 en curso**: la fase 0 puede implementarse en paralelo sin tocar runs; ningún entrenamiento empieza hasta que seed44 termine y se registre.
- **Resultado negativo válido**: si el denso gana en ambos ejes, el experimento cierra la pregunta para esta escala con evidencia pareada — se documenta igual.

## Fase E6 — ablaciones de receta bajo presupuesto ya comparado (propuesta, corregida tras revisión externa)

Añadida el 2026-10-01 y corregida tras revisión adversarial (Codex gpt-6-sol, xhigh, solo lectura; 13 hallazgos, 12 incorporados). Motivación verificada: "On the Residual Scaling of Looped Transformers" (arXiv:2606.18524) — con peso compartido las actualizaciones residuales quedan correlacionadas entre iteraciones y el escalado por rama debe ser **ε = λ/(N·√L)** (parametrización factorial; el LR óptimo depende de L, no de N, *dentro de esa familia*); SMELT (arXiv:2609.01343) — loop de capas medias gana con FLOPs/params/KV **igualados** en arquitectura MoE ajustada a tal efecto, y TaH2 (arXiv:2609.35748) documenta el patrón E1 (pendiente empinada, derrota a cómputo igualado). Nota de alcance: **estas fases no replican SMELT** (su triple control exige rediseño MoE); son ablaciones de receta dentro del lab, comparadas contra E1 con los layer-passes reportados aparte. DeepLoop (arXiv:2607.13491) queda retirado como respaldo directo: es Post-LN/DeepNorm y el lab es Pre-LN.

Se mantiene lo ya decidido: GELU, AdamW, receta de pérdida *final*, seeds {42,43,44}, 6.144.000 tokens / 1.500 actualizaciones, mismos artefactos base por seed.

| ID | Diseño | Pases/token | Pregunta aislada |
| --- | --- | --- | --- |
| E6a · loop medio (ablación) | Secuencia fija elegida a priori: **L0, L1, L2, L3, L2, L3, L4** (entrada y salida fuera del bucle; tramo medio índices 2–3 decidido antes de mirar validación). Mismo escalado por rama que E1 — **sin** factor 1/N aquí, para no cambiar dos factores | 7 | ¿Dónde conviene repetir capas? E6a vs E1 a iguales tokens mide concentración del cómputo (7 vs 10 pases); reportar además frontera NLL–FLOPs. No es comparación "presupuesto igualado" contra el denso-3M (5 pases) |
| E6b · escalado por rama 1/N | Loop completo K=2 con **factor ε por rama dentro de cada capa del stack looped** (atención y FFN: x ← x + ε·rama(norm(x)), ε = λ/(N·√L), λ=1 por defecto), implementado como `ScaledBlock` que reutiliza los mismos tensores de `model.py` con forward propio; inyección de entrada conservada explícitamente. La forma h ← h + c·f(h) del stack completo queda como hipótesis separada, no como E6b | 10 | ¿Estabiliza el peso compartido lo suficiente para cerrar la brecha de +0,036 contra el denso? |
| E6c · condicional | Selectivo (E6a) **+** escalado 1/N, sólo si E6a y E6b mejoran por separado; comparado contra full-K2 escalado (aísla el tramo bajo escalado) | 7 | Interacción tramo×escalado |

**E6b-0 · calibración previa** (micro-presupuesto etiquetado, no comparación): la parametrización nueva no garantiza que LR 3e-4 siga óptimo — el paper transfiere LR *entre valores de N dentro de la familia*, no desde el modelo sin reparametrizar. Ejecutar λ∈{0,5, 1} × LR∈{1,5e-4, 3e-4} a 100 actualizaciones (655.360 tokens, 1 seed) y elegir por NLL; registrar normas de gradiente y de pesos como diagnóstico.

**Métrica de decisión (E5/E6, corregida)**: la selección por mejor-checkpoint de 37 puntos es frágil — el run looped s42 saltó de 3,2103 (paso 1450) a 3,2525 (1500). Decisión principal: **NLL en el paso final fijo (1500) + media de las últimas 3 evaluaciones**; mejor-checkpoint pasa a métrica secundaria. Publicar las 3 diferencias pareadas por seed y las curvas. n=3 sigue siendo descriptivo.

**Denso-7L opcional** (sólo si se quiere la comparación por token del E6a contra un denso de sus mismos 7 pases; 3.885.888 parámetros): no forma parte del gate.

Implementación en archivos **nuevos** — lección de la campaña: editar `src/looped.py` o `scripts/looped_experiment.py` tras ejecutar runs invalida la restauración de sus checkpoints (el self-hash forma la identidad; el contrato actual además sólo admite 1.500 pasos y variantes E1–E4). Por eso: `src/loop_selective.py` (secuencia fija, ScaledBlock con ε por rama), `configs/` por variante, `scripts/loop_e6_experiment.py` (hereda el patrón de fábricas y congela), artefactos por (variante, seed) en el índice existente, y **verificador nuevo `scripts/utils/check_e6.py`** que invoque la entrada E6 — `check_loop.py` apunta al script E1–E4 congelado y no sirve para E6.

**Gates E6**: pareado por seed contra E1 con la métrica de decisión corregida. Adoptar una variante si mejora ≥ 0,01 nats en media y en ≥ 2/3 seeds. Objetivo declarado: si alguna supera al denso-3M (NLL@1500 medio 3,187... refrescar con la métrica nueva) o se acerca a < 0,01, el loop paga por primera vez en el lab. Resultado negativo válido y barato: cerrar la receta y dejar E5 como única vía.

**Coste**: E6b-0 (6×100 updates ≈ 4 min) + E6a/E6b ×3 seeds ≈ 6×190–270 s de bucle + evaluaciones ≈ **~40–50 min GPU** en total; VRAM pico esperada 300–350 MiB. Auditoría con `check_e6.py --audit-run` sin tocar scripts congelados.

## Fase E8 — cooldown minimal de LR (COMPLETADA — 2026-10-01)

Diseño de variable única: denso-3M GELU, seeds 42/43/44, 6.144.000 tokens, idéntico en todo a los basales constantes salvo el schedule de LR — constante 3e-4 hasta el paso 1.275 y coseno a cero en el último 15%. Implementado en `scripts/e8_decay_experiment.py` + `scripts/utils/check_e8.py` (continuidad 5 vs 2+3 bitwise; schedule verificado por función pura); nada congelado se tocó.

Resultado (pareado por seed, final vs final): **−0,081 / −0,095 / −0,074 nats** (media −0,0835, ~8,0% de reducción de ppl, gate 0,005 superado 16×). Nuevo mejor NLL del lab: **3,079776** (s44 @1450, ppl ≈ 21,7). Brecha final−mejor cae de 0,009–0,081 a 0,001–0,006: el síntoma "mejor checkpoint antes del final" quedó eliminado. **Decisión: cooldown adoptado como receta del lab**; toda fase nueva lo declara. Referencias actualizadas: la frontera del denso-3M a 6,144M tokens ya no es 3,177 sino **3,096210** (media de finales decay, corregida contra los artefactos). Las comparaciones E1–E4 siguen válidas como pareadas (ambos brazos con receta constante). Detalle en `experimentos.md`.

## Fase E7 — loopify de modelo preentrenado con control denso (EN CURSO — autorizada)

Contrato efectivo y recuperación numérica en [protocolo E7](protocolo-e7-20261001.md); registro vivo en `experimentos.md`. E7-v1 abortó por overflow FP16 XSA; fuentes/checkpoints conservados. E7-v2 corrige las reducciones XSA en archivos nuevos y ambos brazos reinician desde preentrenados. Revisión temprana completada: looped pierde **+0,248256 nats** frente al denso, con coste **1,87×** y−4,69 puntos en la sonda propia. Media/final pendientes; resultado no generalizable a las otras revisiones.

Motivación: E1–E4 operaron a 6,1M tokens, ~6 órdenes de magnitud por debajo del régimen donde los papers demuestran ventaja del loop. E7 traslada la pregunta a un modelo con preentrenamiento real: **bench-labs/tinctura-v1** (96,2M parámetros, from-scratch sobre 75B tokens, Apache-2.0, arquitectura documentada `CagliostroForCausalLM` — 18 capas, d=640, GQA 10Q/5KV, SwiGLU, RoPE θ=100k, embeddings atados, logit cap 15, pesos FP32) y con **historial de checkpoints por revisión** (push cada ~30 min de entrenamiento) que habilita el eje experimental adicional de madurez.

Diseño (incorpora la corrección de control denso de la segunda revisión externa):

1. **Cirugía**: mismo stack de 18 capas ejecutado K=2 con inyección de entrada + loop embeddings (K·640 parámetros nuevos); pesos originales intactos.
2. **Pre-check de equivalencia de un pase**: antes de post-entrenar, demostrar que la variante loopificada en configuración equivalente-a-un-pase reproduce los logits del original con tolerancia declarada. Sin esto, una caída posterior se confunde con bugs de cirugía.
3. **Control denso por revisión**: para cada una de 3 revisiones de madurez (temprana / media / final del historial), dos brazos con **idénticos 2M tokens, datos y evaluación**: (a) continuación densa del original, (b) loopified + post-train. Post-train con la receta E8 (decay al final).
4. **Mediciones**: delta de calidad pareado por revisión (NLL retenida sobre corpus común + sonda de tareas tipo ArithMark — a 96M/75B la aritmética deja de estar fuera de escala); `loops_nll` (refinamiento bucle 1→2); tiempo/FLOPs por token (P1); **pico VRAM real medido** (P2: los pesos no crecen, las activaciones sí); la "media de últimas 3 evaluaciones" no aplica en el tramo decay — NLL@final.

Coste estimado: VRAM full-FT ≈ 2–2,5 GB ✓; throughput estimado 0,6–1,2k tok/s a ctx 2048 (eager) → 6 runs × 2M tokens ≈ **3–5 h GPU** + 4–8 h de implementación (descarga 385 MB, port de arquitectura `trust_remote_code` o adaptación a `src/`, corpus de post-train común — p.ej. muestra de FineWeb-Edu —, sonda retenida). Riesgos declarados: 2M tokens es una intervención breve sobre 75B (la señal puede vivir en la sonda, no en la NLL); `lm-eval` en Windows/ROCm puede friccionar — preferir sonda propia; Unsloth no aporta ruta optimizada para arquitectura custom y queda fuera.

Gates: (a) el brazo loopified recupera y supera a su control denso en ≥1 revisión → el loop paga con preentrenamiento real, primera evidencia positiva del lab; (b) la brecha loopified−denso depende de la madurez → hallazgo publicable sobre loop-ability; (c) negativo limpio en todas las revisiones → el loop no se instala con post-train corto a esta escala, se documenta.

## Fase E5-piloto (propuesta, pendiente de autorización)

Como estaba definida: continuación pareada ×2 (12.288M tokens) de E1 + denso-3M + denso-10L, 3 seeds c/u, desde `latest` con estado completo, run-ids nuevos, script nuevo con contrato a 3.000 pasos. **Nota post-E8**: continúa con receta constante por diseño (cambiar el schedule a mitad rompería el aislamiento de la variable presupuesto); las brechas pareadas siguen siendo la lectura válida, los absolutos ya no son frontera del lab. Decide por evolución de brechas; habilita o cierra E5-×4.

## Fase E9 — pre-pretraining sintético (COMPLETADA — 2026-10-01)

Preparación y primer principal autorizados el 2026-10-01. Contrato efectivo en [protocolo E9](protocolo-e9-20261001.md), resultados y autorizaciones pendientes en `experimentos.md`. Nada de lo pendiente queda autorizado por esta sección.

La autorización posterior de continuar hasta indicación de parada cubrió los otros cinco principales. Resultado: retrieval **+0,008024 nats** medio frente a E8 (gana 1/3); grammatical **−0,012714** (gana 3/3); retrieval pierde contra grammatical 3/3 (**+0,020738** medio). **Gate de adopción retrieval cerrado**, sin confirmación del mecanismo. Seis runs / 252 checkpoints auditados, cero omisiones. Receta vigente sin PPT. Señal del control local conservada como exploratoria; [reporte final](../../output/e9-campaign-20261001-v1/report.md).

Inspiración: *Synthetic Pre-pretraining Survives Scale, but Not as a Grammatical Prior* (arXiv:2609.39827): el PPT sobre datos sintéticos no naturales ahorra tokens de pretraining (≥21B a escala 3B en su régimen 500M–7B / hasta 100B tokens), las ganancias **no** vienen de un prior gramatical sino de tareas que entrenan **recuperación de largo alcance**, y son robustas a la mezcla de datos pero se debilitan sin texto web.

Diseño pareado sobre la receta vigente (GELU, AdamW, cooldown E8):

| Brazo | Fase previa (1M tokens) | Entrenamiento | Coste |
| --- | --- | --- | --- |
| A · control | ninguna | TinyStories 6,144M + cooldown | **0** (son los runs E8 existentes, seeds 42/43/44) |
| B · PPT-retrieval | tareas sintéticas programáticas de recuperación de largo alcance (repetir token visto hace N posiciones, copiar pares, asociaciones distribuidas sobre el vocab 4.096) | idéntico a A | 3 runs |
| C · PPT-grammatical (opcional, aisla mecanismo) | tareas sintéticas gramatical-like | idéntico a A | 3 runs |

El contraste B vs C es una **comparación exploratoria inspirada en el paper** a escala 3M. El control programático local no iguala dificultad/entropía ni replica su generador: retrieval > gramatical sería compatible con el mecanismo, no su confirmación. Contabilidad de tokens declarada: el PPT añade 1.003.520 tokens (16,33% de sobrecoste); la ganancia debe leerse como NLL a presupuesto PT igualado, con el coste PPT anotado aparte.

Implementación: generador sintético determinista (PCG64, shards uint16 en `data/synthetic-ppt/`, misma interfaz que PackedTokens), `scripts/ppt_experiment.py` nuevo (fábrica que añade la fase PPT antes del bucle estándar con cooldown); nada congelado se toca. Métrica: NLL@1500 + sonda; pareado por seed contra los runs E8.

Gates: (a) B supera a A por ≥0,01 nats medio y ≥2/3 seeds → PPT adoptado para esta escala; transferencia a 10M requiere otra validación; (b) B > C → evidencia exploratoria compatible con retrieval, contrastada también con la sonda; (c) sin señal → documentado como límite del régimen/generador (TinyStories no es web text — la propia salvedad del paper). Si la sonda no muestra aprendizaje de retrieval, no atribuir una mejora TinyStories a ese mecanismo.

Coste: 6 runs × ~300 s ≈ **~35 min GPU** + 2–3 h implementación. Nota de alcance: nuestro régimen (3M, 6M tokens, texto no web) está lejos del suyo (≥500M, ≤100B); el valor está en el contraste de mecanismo barato y en la receta si funciona.

## Fase E10 — Parallel Power Tempering en inferencia (propuesta, sin ejecutar)

Referencia: [*Explore Broadly, Reason Sharply: Push Small Models toward the Frontier via Sampling* — arXiv:2609.38104v1](https://arxiv.org/html/2609.38104v1), publicado el 2026-09-29. Propone **Parallel Power Tempering (PPT)**: cadenas de generación con distintas potencias de la probabilidad de secuencias completas, refinamiento Metropolis–Hastings e intercambio de estados entre cadenas. **No modifica pesos ni usa recompensas externas**. Este PPT no es el pre-pretraining sintético de E9 ni el looping arquitectónico de E7; temperatura/top-p por token tampoco equivalen al objetivo sobre secuencias completas.

**Prioridad:** después de diagnosticar la degradación de adaptación E7 y resolver baseline/datos. Añadir esta fase al roadmap no inicia su implementación ni ejecución; el piloto requiere fijar y autorizar su presupuesto de inferencia. Entrenamiento: **0 tokens / 0 actualizaciones**.

Diseño mínimo propuesto:

| Factor | Contrato a cerrar antes del piloto |
| --- | --- |
| Modelo | Un checkpoint fijo con calidad y estabilidad verificadas; candidato inicial: tinctura-v1 original de 96M sin adaptación. No usar un checkpoint E7 deteriorado como único baseline |
| Controles | Sampling convencional, power sampling de una cadena y Parallel Power Tempering; greedy como referencia diagnóstica |
| Variable | Estrategia de inferencia; mismos pesos, tokenizer, precisión, prompts, horizonte máximo y protocolo de extracción de respuesta |
| Réplicas | Seeds de muestreo 42/43/44 pareadas entre métodos; no son réplicas de entrenamiento |
| Tareas | Sonda generativa pequeña y retenida de aritmética/composición, con respuesta verificable y exact-match; fijar prompts y extracción antes de observar resultados |
| Presupuesto | Igualar el cómputo de los métodos de búsqueda; contar propuestas, rescoring, tokens realmente procesados en forward y longitudes de contexto, además de tiempo real. No igualar sólo tokens devueltos |
| Salida | Una respuesta por ítem y seed (pass@1); no elegir retrospectivamente la mejor de varias respuestas mediante el verificador |

**Implementación:** compartir los pesos del mismo modelo entre estados de cadenas; fijar la escalera de potencias, número de cadenas y pasos de refinamiento antes de evaluar. Implementar la corrección de horizonte fijo/EOS y las probabilidades de propuesta directa/inversa del paper: la truncación unilateral sesga el objetivo. Registrar aceptación de refinamientos/intercambios y contabilizar todo el trabajo de inferencia. Archivos nuevos, IDs únicos, fuentes congeladas intactas; verificar hash de pesos antes/después, RNG aislado y repetición determinista sin optimizador.

**Métricas y gate:** exactitud por tarea y diferencias pareadas por seed frente a los controles, calidad por segundo/cómputo, pico VRAM y tasas de aceptación. Ampliar sólo si el piloto muestra una ganancia consistente a presupuesto comparable; sin señal, documentar el límite del checkpoint/régimen. NLL del modelo base no cambia y no es la métrica principal de este experimento. No interpretar mayor likelihood de una secuencia como garantía de corrección.

**Alcance y coste:** el paper evalúa Qwen3-4B, Qwen3-8B y Qwen3.5-9B; probar 96M (o posteriormente 3M) sería una adaptación exploratoria, no una réplica de sus resultados de frontera. No presupone que un modelo sin capacidades de razonamiento previas las adquiera por sampling. Coste de inferencia/VRAM pendiente de benchmark local antes de autorizar el cap; los intercambios pueden reutilizar likelihoods, pero generar y refinar cadenas sí consume cómputo. No se incorporan modelos 4B–9B ni nuevas pilas por añadir esta propuesta.

## Fase E11 — datos únicos frente a repetidos (COMPLETADA — 2026-10-02)

Diseño del [handoff de datos](../technical/handoff-e7-datos-20261001.md), autorizado en la [campaña nocturna](campana-nocturna-20261002.md). Misma receta denso3M GELU/AdamW/cooldown E8, seeds42/43/44,6.144M tokens por run, mismos pesos iniciales reales, BPE4096 y39.275 targets val. Variable única: corpus train ampliado a26.864 historias de la misma revisión/fuente, sin duplicados/overlap exacto val y suficiente para no repetir targets; prefijo histórico reproducido por hash. E8 reutilizado como control de calidad, no como prueba temporal confirmatoria.

Resultados medios E11/E8 = **2,878252 /3,096211**, Δ **−0,217959 nats**,3/3 seeds favorables y19,58% menos perplexity. Tres principales/111 checkpoints auditados, cero omisiones, cursor0→6.144.000 sin wrap; primeros104 batches iguales antes de la repetición del control. [Reporte completo](../../output/e11-coverage-campaign-20261002-v1/report.md). Coste principal18.432M tokens,~7,91min de bucle, preparación/continuidad aparte.

**Decisión:** priorizar cobertura de datos únicos antes de nuevas técnicas; comparaciones posteriores sobre el corpus ampliado usan E11 como baseline, conservando los linajes históricos de datos repetidos. La receta numérica no cambia. Evidencia local descriptiva, sin significancia ni transferencia automática a10M.

## Orden propuesto (revisado tras E8 y dos revisiones externas)

1. ~~Fase 0~~, ~~E1+E2~~, ~~E4~~, ~~E8~~ — completados; E3 descartado por gate cerrado (ver `experimentos.md`).
2. **E7** (~3–5 h GPU): la apuesta principal — único experimento con preentrenamiento real, donde la literatura dice que el loop paga. Hereda la receta decay recién validada.
3. **E5-piloto** (~55 min GPU): pendiente local de presupuesto en régimen multi-época (~29 épocas del shard).
4. **E6** condicionado: sólo si E5 muestra brecha persistente o E7 señala problema de receta.
5. **E9** (~35 min GPU): encaja en cualquier hueco — barato, sin dependencias; atractivo mientras se implementa E7 (sus ~4–8 h de implementación dejan la GPU libre).
6. **E10 · Parallel Power Tempering** (inferencia, sin entrenar): después de resolver baseline/receta/datos; comparar estrategias de sampling sobre un checkpoint fijo a cómputo comparable. Presupuesto pendiente de benchmark y autorización.
7. Fuera de alcance, anotado: escalado con datos únicos (shard mayor + baselines reentrenados); comparación looped-vs-híbrido (LFM2.5) a presupuesto igualado; WSD completo con LR 3e-3 (sólo si se quiere explorar más allá del cooldown).
