# E7: loopify de tinctura con controles densos por madurez

Autorizado por la instrucción de continuar los experimentos hasta indicación de parada, después de completar E9. Se conserva la matriz del plan: **tres revisiones × dos brazos × 2.000.000 tokens**, sin barridos ni ampliaciones.

**Versión numérica vigente: E7-v2, XSA en FP32.** El primer intento v1 fue abortado por overflow FP16 antes de obtener una comparación completa. V2 conserva la matriz y presupuestos, reinicia desde los preentrenados y no admite checkpoints v1. Detalles y coste operativo al final del protocolo.

## Contrato fijado antes de los principales

| Factor | Decisión |
| --- | --- |
| Modelo | `bench-labs/tinctura-v1`, 96.200.064 parámetros, Apache-2.0 |
| Arquitectura | 18 capas, d640, GQA10Q/5KV, SwiGLU1536, QK-RMSNorm, XSA, RoPEθ100k, logit cap15, embeddings atados |
| Brazo denso | Un pase por el stack original |
| Brazo looped | K2, inyección del embedding en cada pase + embeddings de bucle PCG64 seed por revisión; 1.280 parámetros nuevos |
| Post-train | AdamW3e-4, betas0,9/0,95, decay0,1 matrices/0 normas, clip1, warmup10; coseno a cero en los últimos15% de actualizaciones |
| Tokens | Exactamente2.000.000 por principal: 488 actualizaciones de4.096 + última de1.152; total489 |
| Batch | Contexto2048, microbatch1, acumulación2; la última actualización usa una microsecuencia de1.152 |
| Precisión | FP16 autocast, pesos/AdamW FP32, GradScaler inicial1, cero omisiones exigidas |
| Activaciones | Checkpoint por capa en ambos brazos, nativo PyTorch no reentrante, sin Triton ni torch.compile |
| Datos | Mismo orden FineWeb-Edu `sample-10BT` para cada brazo/revisión, shards en `data/posttrain/fineweb-e7-v2/` |
| Métrica principal | NLL retenida@2M sobre65.536 targets comunes; final fijo, no selección por mejor checkpoint |
| Métricas complementarias | NLL por bucle, sonda aritmética retenida, generaciones greedy, tiempo y VRAM reales |

### Revisiones inmutables y seeds

| Madurez | Revisión HF | Presupuesto previo verificado | Seed pareada |
| --- | --- | ---: | ---: |
| Temprana | `b57a4f781c583bf0015cf6123fedab45d9c63436` | 2.162.688.000 tokens; título step22.000, contador interno21.999 | 42 |
| Media | `f5a0a6822f6727d7a51459514f0e14fb6c809121` | 37.355.520.000 tokens; título step380.000, contador interno379.999 | 43 |
| Final | `f475b8a2698b0f2de14fbae584014908fec4f719` | ~75B tokens, release final | 44 |

Cada seed está pareada entre denso y looped de su revisión. **Madurez y seed no están cruzadas**: seis runs no permiten separar una interacción con madurez del efecto de seed. Se reportan diferencias descriptivas por revisión; no se declara un hallazgo causal/publicable de madurez. La matriz cruzada3×3×2 queda fuera del presupuesto autorizado.

## Preparación y equivalencia

Artefactos descargados por revisión y checksum LFS en `data/pretrained/tinctura-v1/`. Temprana/media contienen sólo `latest.pt` de1,15GB; carga segura `torch.load(weights_only=True)`, pesos y metadatos inspeccionados, AdamW heredado descartado. Final incluye export de385MB y código/tokenizer originales. No se instala transformers ni otra pila: `src/tinctura_loop.py` reutiliza definiciones upstream verificadas por hash y el forward original como referencia numérica; los archivos descargados permanecen intactos.

El port sólo adapta la promoción de dtype de RMSNorm/RoPE para autocast FP16 con pesos maestros FP32, aplicada a ambos brazos. Mantiene QK-RMSNorm y XSA del preentrenado. La caché RoPE se crea fuera de inference_mode para poder reutilizarla en backward después de evaluar.

`output/e7-model-check-20261001-v2/model-check.json`: equivalencia de un pase frente al forward original en FP32 con **error máximo0,0** en contextos32/256/2048 para las tres revisiones. Deriva de NLL FP16 vs FP32 máxima observada **0,001708 nats**, inferior al límite prefijado0,02. Forward/backward sin optimizador en K1/K2: gradientes finitos, pesos sin cambios, pico~1,7GB sin estados AdamW. No extrapolar ese pico al full-FT; se mide en el principal. El benchmark temprano K1 incluye calentamiento y no sirve para comparar throughput contra K2.

Los primeros preparativos fallaron por un límite HTTP429 al descargar validación y por una caché RoPE creada en inference_mode que el backward rechazaba. Datos parciales/logs conservados; validación completada en corpusv2 con backoff y trainv1 reutilizado y revalidado. Caché corregida antes de crear checkpoints E7. Ningún entrenamiento principal se usó para ajustar estos detalles.

## Datos y tareas retenidas

FineWeb-Edu:1.767 documentos train,83 validation, IDs y hashes de texto disjuntos; train desde fila0, validación desde10.000. Tokenizer original32768, EOS0 entre documentos; binarios uint16. Train2.000.001 tokens permite exactamente2M targets sin repetir; validation65.537 tokens permite65.536 targets. Se rechazan celdas truncadas de la API. Licencia/procedencia ODC-By registradas en el manifest; es adaptación a texto web, no réplica de la mezcla original de75B.

Sonda propia tipo ArithMark, **no ArithMark oficial**:64 ejercicios prefijados de suma/resta/producto/división exacta, cuatro respuestas; seed202610017. Se tokeniza cada pregunta+respuesta conjuntamente y se valida la frontera. Métrica: exactitud de opción elegida por log-likelihood de respuesta normalizada por longitud. No se entrena ni selecciona LR con la sonda. Las tareas son cortas; su exactitud no prueba razonamiento general.

## Continuidad, ejecución y gates

`scripts/utils/check_tinctura_training.py`:5 pasos frente a2+3 en procesos nuevos con contexto real2048,10 pasos/40.960 tokens por revisión/brazo, fuera del presupuesto principal. Compara pesos, AdamW/scaler/RNG bitwise, batches/cursor/contadores y eval aislada; rechaza corrupción y contrato de verificación. Comprueba schedule y contabilidad exacta del último batch parcial. Cada principal exige su evidencia de continuidad y el pre-check de equivalencia.

`scripts/tinctura_experiment.py` ejecuta segmentos de≤100 actualizaciones en llamadas bash separadas de≤600s. Cada segmento tiene runID nuevo, reanuda el estado completo y conserva checkpoints anteriores. El presupuesto de2M es **por linaje**, no por segmento; sus métricas contienen el historial acumulado. Auditoría sin entrenar tras cada segmento y del artefacto final; la final reproduce también generaciones/sonda. Fuentes de modelo y entrypoint quedan congeladas desde el primer checkpoint.

Gate: una ganancia looped frente a su control en≥1 revisión es una señal exploratoria que requiere réplica, no adopción automática. Negativo limpio en las tres cierra la intervención corta de2M. Comparación a iguales tokens con distinto cómputo: se publica el sobrecoste y **no se afirma eficiencia P1 a cómputo igualado**. P2 usa el pico real: compartir pesos no iguala activaciones. No se aumenta K ni presupuesto sin un nuevo gate y alcance explícito.

## Incidente v1 y corrección numérica v2

El denso temprano v1 produjo pérdida no finita al intentar paso273. Último checkpoint durable250;22 actualizaciones251–272 reconstruidas exactamente desde éste y materializadas en `e7-early-dense-20261001-031-recovery272`. Todos los campos de los22 rows, excepto tiempo, coinciden con el intento original. Checkpoint272 auditado, sin entrenamiento adicional durante la auditoría.

Repro mínima retenida en `scripts/utils/diagnose_tinctura_precision.py`: batch del offset1.114.112 con checkpoint272. FP16 produce NaN y primer módulo no finito `layers.13.attn.o_proj`; FP32 produce NLL3,376299. XSA multiplica `v*v` en FP16 antes de reducir; en capa13 una multiplicación desborda, mientras la norma cuadrada FP32 máxima es87.890,90625. No son pesos corruptos ni labels inválidos. El batch con checkpoint250 y pesos preentrenados es finito: la amplitud cambia durante post-train.

`src/tinctura_xsa_fp32.py` implementa las multiplicaciones/reducciones de XSA en FP32; el resto sigue el forward original. `scripts/tinctura_xsa_experiment.py` reutiliza el runner congelado vía fábricas de proceso, añade contrato numérico2 e identidad de ambas fuentes nuevas. No se edita ningún archivo con checkpoints. Se reinician **ambos brazos y todas las revisiones desde sus preentrenados**, sin mezclar el cambio numérico con continuidad v1.

Regresión real: el comando con `--expect-finite` falló en la repro v1; con `--fp32-xsa --verify-backward` pasa con logits/gradientes finitos y pesos inmutables. `output/e7-precision-regression-20261001-xsa-fp32/diagnosis.json`, cero updates. Equivalencia v2 repetida en `output/e7-xsa-model-check-20261001-v2/model-check.json`: error FP32 máximo0,0 en los9 casos; deriva NLL FP16 máxima0,000679, backward finito y pesos sin cambios. La versión rechaza efectivamente el checkpoint272 v1 por identidad/contrato incompatible.

Snapshot v2 de6.519 archivos previos en `output/e7-xsa-preparation-20261001-v2/`. Continuidad temprano-denso v2 aprobada con `scripts/utils/check_tinctura_xsa.py`,5 frente a2+3 bitwise. Principales nuevos usan ese verificador y `scripts/tinctura_xsa_experiment.py`. Presupuesto científico sigue2M por linaje,12M para la matriz completa. Coste operativo extra del intento descartado:1.114.112 tokens de actualizaciones +90.112 tokens recomputados, aparte de pruebas y forwards de diagnóstico. No se interpreta el intento abortado como un resultado de looping ni se reutiliza para medir calidad.
