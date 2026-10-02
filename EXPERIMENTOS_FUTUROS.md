# Ideas y planes de experimentos futuros

Última revisión: **2026-10-02**. Fuentes: [PAPERS.md](PAPERS.md). Mantenimiento obligatorio: [AGENTS.md](AGENTS.md).

**Documento de planificación: ninguna propuesta nueva está autorizada para entrenar por aparecer aquí.** La creación y mantenimiento de estos archivos no amplía presupuestos ni permisos de campañas anteriores. La autorización nocturna tenía límite 2026-10-02 09:00 de Santiago; no inferir de ella permiso para estos planes.

Este archivo es el índice actual de prioridades. Los protocolos y resultados históricos permanecen en [docs/architecture/experimentos.md](docs/architecture/experimentos.md) y [plan looped histórico](docs/architecture/plan-looped-20261001.md); sus órdenes antiguos no sustituyen el estado resumido aquí. IDs Fxx son propuestas, no nuevos IDs Ex ejecutados.

## Evidencia de partida

| Experimento | Estado verificado al revisar | Consecuencia para las próximas pruebas |
| --- | --- | --- |
| E1/E2/E4, looped 3M | K2 pierde contra denso; segunda pasada mejora sobre primera. Receta histórica sin cooldown y datos repetidos. [Registro](docs/architecture/experimentos.md). | No descartar recurrencia universalmente; repetir sólo con un contraste nuevo justificado. |
| E8, cooldown | Mejora media final aproximadamente −0,0835 nats, 3/3 seeds; adoptado localmente. [Registro](docs/architecture/experimentos.md). | Control de receta: GELU + AdamW + cooldown final 15%. |
| E9, preparación sintética | Retrieval no supera gate; grammatical tiene señal exploratoria, con coste extra y dificultad no igualada. [Reporte](output/e9-campaign-20261001-v1/report.md). | No adoptar retrieval ni atribuir causalmente la mejora a gramática. |
| E7, adaptación 96M | Pareja temprana v2 completa: ambos empeoran; revisión media abortada por FP16. FP32 y pilotos LR comprobados, pero no adaptación larga estable. [Pareja](output/e7x-early-pair-20261001-v1/report.md), [pilotos](output/e7-lr-pilot-20261002-v1/report.md). | Usar original final para nueva línea de tareas; no reutilizar pesos deteriorados como control basal. |
| E11, datos únicos | Media final 2,878252 frente a E8 3,096211: −0,217959 nats, 3/3 seeds, iguales 6,144M tokens. [Reporte](output/e11-coverage-campaign-20261002-v1/report.md). | Nuevo control para este corpus; ampliar cobertura antes de sumar técnicas. |
| E10, búsqueda al inferir | Sólo una prueba greedy persistida; genera `18 + 76 = 94`, pero el extractor devuelve 18. [Resultado](output/e10-greedy-s42-i00-20261002-v1/results.json). | Corregir protocolo/evaluador antes de interpretar exactitud o construir recompensas. |

E7 fue continuación/adaptación mediante predicción de siguiente token. No se encontró un experimento local completado de SFT por tareas ni RL con recompensas. Las NLL de TinyStories y FineWeb-Edu/tinctura no son comparables directamente.

## Orden de trabajo propuesto

| ID | Prioridad | Idea | Estado / autorización | Dependencia |
| --- | --- | --- | --- | --- |
| F00 | Previa a tareas | Evaluador aritmético y medida basal fiable | Propuesto; sin ejecutar cambios | Ninguna |
| F01 | Alta | Más tokens únicos para el denso 3M | Propuesto; entrenamiento no autorizado | Corpus ampliado y protocolo de presupuesto |
| F02 | Alta | SFT aritmético del original 96M | Propuesto; entrenamiento no autorizado | F00 y verificación numérica de nueva receta |
| F03 | Media | Separar tamaño de modelo y presupuesto | Propuesto; entrenamiento no autorizado | F01 y tokenizer común |
| F04 | Condicionada | RL verificable frente a SFT adicional | Propuesto; entrenamiento no autorizado | F00, arranque estable y señal de recompensa |
| F05 | Piloto barato | LoopCD sobre checkpoints 3M existentes | Propuesto; inferencia no ejecutada | Restauración fiable y dev/test separados |
| F06 | Condicionada | Reevaluar K2 con datos nuevos/más entrenamiento | Propuesto; entrenamiento no autorizado | Control denso del mismo régimen |
| F07 | Condicionada | Completar comparación E10 de búsqueda | Pendiente; no asumir vigencia del permiso anterior | F00, capacidades basales y nuevo cap |
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
| F02: SFT 96M, 5k–20k ejemplos cortos, 1–3 epochs, evaluación acotada | 2–5 h | 30–120 min | Baja hasta medir batching, longitud y pérdida de SFT |
| F04: primer RLVR pequeño, 512–1.024 respuestas totales de24–32 tokens | 4–8 h tras F00/arranque estable | 1–3 h | Baja; rollouts y actualización todavía no medidos |
| F05: LoopCD, pocos valores de fuerza en dev y evaluación retenida acotada | 30–90 min | 15–45 min | Baja; adaptador y cap todavía no definidos |

Para F01, extrapolar sólo el bucle de E11 da **41–43 min por seed a 98,304M tokens**. A 24,576M, tres seeds suman aproximadamente31–33 min de bucle, pero ejecución con comprobaciones puede ocupar1,5–3 h, además de preparación compartida. El corpus E11 actual no alcanza estos presupuestos sin repetición: ampliar datos forma parte del tiempo, no está preparado aún.

F02 no incluye repetir tres seeds ni una búsqueda de hiperparámetros. F04 describe una sonda de viabilidad, no un entrenamiento RL hasta convergencia; un presupuesto largo o respuestas de razonamiento extensas aumentan mucho el tiempo. F03/10M requiere medir throughput de su configuración nueva antes de una estimación fiable; F06 depende del presupuesto seleccionado y F07 del cap de búsqueda. F08 no tiene duración asignada porque es una reserva de ideas.

Orden de magnitud para los siguientes pilotos F00 + F01 + F02, ejecutados secuencialmente: **5–13 h de trabajo total**, de las cuales sólo una parte es GPU entrenando. Implementar y verificar RL después añade aproximadamente **5–11 h** para la sonda F04 descrita. Recalcular tras preparar datos y medir el primer batch/rollout; no interpretar estos rangos como autorización ni garantía.

## F00 — Evaluación y verificador aritmético

- **Hipótesis:** parte de la exactitud observada puede depender del formato/extractor, no del cálculo.
- **Diseño:** preservar E10 histórico; definir un formato inequívoco y parser para un protocolo nuevo. Validar respuestas correctas, incorrectas, vacías, ambiguas, con ecuaciones y números negativos. No seleccionar el parser buscando aumentar el score del test.
- **Datos/métricas:** train/dev/test disjuntos por problemas; exactitud, formato válido y aciertos entre 8 respuestas. Reservar acarreo/mayor longitud como pruebas de generalización separadas.
- **Presupuesto/control:** cero actualizaciones; inferencia sobre original 96M con prompts y horizonte fijados. Cap de ejemplos/tokens/tiempo pendiente de benchmark y autorización del piloto.
- **Gate:** continuar cuando el evaluador tenga un chequeo runnable de comportamiento y el baseline se repita. Si la extracción es ambigua, reportarla; no dar recompensa por encontrar cualquier número correcto dentro de una respuesta.
- **Fuentes:** P12/P14/P16 y artefacto E10. Resultado/decisión: pendientes.

## F01 — Curva de aprendizaje con más datos únicos

- **Hipótesis:** 6,144M tokens no permiten aprovechar completamente el denso 3M; ampliar datos/entrenamiento cambia la interpretación de negativos previos.
- **Control y variable:** GELU/AdamW/BPE4096/contexto256/validación de E11; variar presupuesto y el prefijo consumido de un corpus común versionado, manteniendo idéntico orden de datos en los prefijos compartidos. Reconocer que aquí aumenta exposición total y cobertura; E11 ya aisló cobertura a iguales tokens.
- **Presupuesto sugerido:** 6.144.000 / 24.576.000 / 98.304.000 tokens por seed. El primero ya tiene controles E11; verificar identidad antes de reutilizarlos. Piloto seed42, confirmación seeds42/43/44 sólo para contrastes seleccionados y explícitamente autorizados.
- **Protocolo:** nueva versión de datos suficiente para no wrap, misma fuente/revisión, tokenizer y validación; duplicados/overlap controlados. Schedule definido por presupuesto antes de ejecutar. No prolongar directamente un final con LR cero ni cambiar fuentes congeladas.
- **Métricas/gate:** NLL final fijo y curva, calidad narrativa sobre prompts retenidos, coste de preparación y entrenamiento. Fijar mejora mínima y techo de coste antes del piloto; seguir a 98,304M sólo si 24,576M aporta ganancia reproducible sin problemas de integridad. No llamar óptimo a un presupuesto probado.
- **Fuentes:** P01/P02/P05 y E11. Resultado/decisión: pendientes.

## F02 — SFT aritmético sobre el 96M original final

- **Hipótesis:** ejemplos supervisados pueden enseñar formato y operaciones simples sin necesitar entrenar otro modelo desde cero.
- **Control:** pesos originales finales tinctura-v1, revisión fijada en E7/E10; nunca pesos adaptados/degradados de E7. Mantener tokenizer y flujo numérico validado en FP32.
- **Variable/datos:** ajuste supervisado con respuesta corta y correcta; propuesta de 5.000–20.000 ejemplos programáticos de sumas/restas fáciles. Fijar un tamaño para el primer piloto, particiones y política de pérdida sobre respuestas antes de entrenar; ampliar tamaño es otro contraste.
- **Presupuesto:** número de ejemplos no equivale a tokens de entrenamiento. Tokens, epochs, LR, actualizaciones y límite de tiempo pendientes de protocolo y benchmark; no lanzar un sweep implícito.
- **Métricas/gate:** exactitud y formato sobre test independiente, acarreo/longitud fuera del régimen train, NLL de lenguaje retenida, tiempo y memoria. Umbrales de mejora/retención prefijados; detener por no finitos u omisiones. Si no aprende tareas fáciles, diagnosticar datos/loss antes de escalar o hacer RL.
- **Dependencias/fuentes:** F00; continuidad propia de la receta; P14/P15. Resultado/decisión: pendientes.

## F03 — Capacidad frente a entrenamiento

- **Hipótesis:** un modelo mayor puede aprovechar los datos mejor, pero necesitar más entrenamiento; no basta aumentar parámetros a presupuesto corto.
- **Diseño:** denso 3M frente a aproximadamente 10M, ambos desde cero, mismos datos/tokenizer/validación y receta de referencia. `configs/10m.json` usa vocab8192: no utilizarlo tal cual frente al BPE4096 si se pretende aislar arquitectura; fijar y contar los parámetros reales de la configuración nueva.
- **Presupuesto sugerido:** 24.576.000 y, condicionado, 98.304.000 tokens por modelo/seed. Control de tamaño a iguales tokens; adicionalmente reportar calidad por tiempo/FLOPs. Calibración específica de LR, si necesaria, separada y contabilizada.
- **Métricas/gate:** NLL final, curvas y tareas narrativas comunes; memoria y coste. Continuar si capacidad extra mueve la frontera dentro del techo previamente fijado. No mezclar el 96M ya preentrenado en esta comparación causal de tamaño.
- **Dependencias/fuentes:** F01; P01/P02/P06/P09. Resultado/decisión: pendientes.

## F04 — RL con recompensas verificables

- **Hipótesis:** la recompensa de corrección mejora exactitud sobre un arranque que ya produce algunos aciertos, más allá de entrenamiento supervisado adicional.
- **Diseño:** mismo checkpoint estable como punto de partida; referencia sin updates, SFT continuado y SFT+RLVR. RL directo desde original queda como brazo posterior si su muestreo basal aporta señal; SFT no es un requisito universal de RL.
- **Tarea:** aritmética sencilla y recompensa por corrección de respuesta, separando validez de formato. Nada de modelos de recompensa/LLM jueces para esta primera prueba.
- **Presupuesto:** piloto con cap fijado después de medir rollouts en la GPU real; registrar prompts, candidatos, tokens generados/procesados/actualizados, actualizaciones y tiempo. Igual número de updates no es igual cómputo frente a SFT.
- **Métricas:** exactitud de una respuesta, aciertos entre 8/16 muestras, generalización, retención de lenguaje, longitud y coste; no decidir sólo por reward train. En GRPO estándar registrar grupos con recompensas distintas: grupos uniformes tienen ventaja relativa cero.
- **Gate:** si faltan aciertos/señal, facilitar tarea o mejorar arranque antes de ampliar. Adoptar sólo si supera al control SFT adicional y cumple retención a coste aceptable, con umbrales prefijados y réplicas; detener por colapso/no finitos. No concluir que todo RL falla por un negativo local.
- **Dependencias/fuentes:** F00, estabilidad numérica, F02 si aporta el arranque; P14/P15/P16. Resultado/decisión: pendientes.

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
- **Métricas/gate:** exactitud, propuestas/aceptaciones/swaps y coste real. Sólo ampliar si mejora a trabajo comparable y pasan las invariantes del sampler; no atribuir una sola respuesta a razonamiento general.
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

- **Hipótesis:** seleccionar niveles donde el modelo aún obtiene aciertos parciales mejora la exactitud por coste frente a selección fija o aleatoria. Es una hipótesis local, no una transferencia demostrada por P17.
- **Modelo/datos:** mismo checkpoint original 96M o arranque de F02 elegido antes del contraste; tareas aritméticas verificables. Caracterizar niveles por dígitos, operaciones y acarreo; medir dificultad empírica sin asumir orden monotónico. Generar ejemplos nuevos en todos los brazos.
- **Control/variable:** A, niveles fijos con muestreo uniforme; B, niveles nuevos aleatorios dentro del mismo espacio permitido; C, selección adaptativa de niveles. Mantener GRPO, normalización de ventajas, LR, precisión, batch y candidatos por prompt idénticos. Comenzar C sin mutación del espacio; si aporta valor, comparar C frente a C+mutación como una segunda prueba. Esta simplificación no reproduce el algoritmo completo del paper.
- **Presupuesto:** pendiente de protocolo y benchmark de F04. Prefijar caps por brazo de candidatos, tokens y tiempo; contabilizar preparación, generación, evaluación y actualización. Reportar curvas a tiempo y tokens comparables, además de pasos; no usar las horas de H200/GH200 del paper como estimación local.
- **Métricas:** exactitud greedy y pass@8, fracción de grupos con recompensas distintas, aciertos por nivel, longitud, coste y retención de lenguaje. Evaluación fija con ejemplos y configuraciones de generador excluidos de entrenamiento; incluir niveles fáciles y difíciles para detectar especialización.
- **Gate:** avanzar sólo si C supera a B en evaluación retenida con coste comparable y retención dentro de umbrales prefijados; confirmar con tres seeds después de una sonda. Detener por no finitos, colapso o ausencia persistente de grupos informativos; facilitar tarea o mejorar arranque sin ampliar automáticamente el presupuesto.
- **Dependencias/fuente:** F00, F04 estable y P17. Estado: propuesto; sin autorización de entrenamiento. Resultado/decisión: pendientes. Pospuesto hasta disponer de baseline RL; no reemplaza F01/F02.

## Reglas de actualización y registro

Estados: `propuesto`, `preparado`, `autorizado`, `en curso`, `completado`, `bloqueado`, `descartado` o `sustituido`; mantener autorización como dato separado. Preparación o una mejora observada no concede permiso de ejecución. Marcar qué dependencia bloquea cuando corresponda.

Al cambiar estado, añadir fecha, motivo y enlace a protocolo/artefacto. Al obtener resultados, actualizar síntesis aquí, vínculo del paper y registro histórico; preservar informes originales. Si una propuesta se sustituye, conservar su ID y señalar sucesora.

| Fecha | Cambio material |
| --- | --- |
| 2026-10-02 | Creación del índice F00–F08 con evidencia local, prioridades de datos y SFT/RL, límites de autorización y gates pendientes de protocolo. |
| 2026-10-02 | Añadidos rangos de tiempo para pilotos, anclas medidas E11/E7/E10 y límites de extrapolación; no se ejecutaron experimentos nuevos. |
| 2026-10-02 | Añadida F09 a partir de P17: selección adaptativa tras baseline RL, controles de niveles/frescura y coste real; sin ejecución ni presupuesto autorizado. |
