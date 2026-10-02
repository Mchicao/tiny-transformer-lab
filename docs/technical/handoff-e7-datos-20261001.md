# Handoff: degradación E7-v2 y siguientes experimentos

Estado observado: 2026-10-02 01:58:57 UTC (2026-10-01 en Chile).
Solicitud: revisar avances y preparar un handoff para el agente que ejecuta la campaña.

## Decisión recomendada

Mantener datos nuevos frente a repetidos como siguiente experimento del modelo TinyStories 3M. Para E7, priorizar el diagnóstico de la receta de adaptación: ya terminaron ambos brazos tempranos y ambos empeoran. Recomiendo diferir nuevos linajes de madurez media/final hasta revisar esa degradación, preservando la pareja terminada como resultado negativo del protocolo vigente.

Este documento es una recomendación técnica, no una autorización nueva de entrenamiento ni una orden ejecutada de pausa. La autorización anterior de la campaña E7 consta en su protocolo; cualquier experimento adicional debe tener alcance autorizado explícitamente. En esta revisión sólo se leyeron archivos y se escribió este handoff: cero actualizaciones, procesos de entrenamiento iniciados o detenidos, cambios de fuentes, commits y publicaciones.

## Evidencia actual, posterior a la captura del usuario

Las métricas finales de ambos linajes indican `completed`, 489 actualizaciones efectivas, 2.000.000 tokens y cero omisiones por brazo. Los diez archivos de auditoría de sus segmentos indican `passed` y cero actualizaciones durante auditoría. Se inspeccionaron los artefactos; no se repitieron las auditorías en GPU.

| Brazo temprano, seed42 | NLL inicial propia | NLL final | Cambio | Bucle train (s) |
| --- | ---: | ---: | ---: | ---: |
| Denso K1 | 3,227632 | 3,820782 | +0,593151 | 714,59 |
| Looped K2 | 3,628482 | 4,069039 | +0,440556 | 1.336,76 |

Looped final menos denso final: **+0,248256 nats**. El checkpoint preentrenado sin adaptación tiene NLL 3,227632 en esta validación. El looped empieza peor por la cirugía; su menor deterioro frente a su propio inicio no lo convierte en ganador. Su mejor NLL registrada, 3,368064, tampoco supera al preentrenado denso sin adaptación. Los resultados iniciales de K1 y K2 no representan la misma función.

El denso empeora progresivamente: NLL a 0 / 409.600 / 819.200 / 1.228.800 / 1.638.400 / 2.000.000 tokens = 3,227632 / 3,270820 / 3,337387 / 3,467029 / 3,700700 / 3,820782. Esto requiere explicación antes de atribuirlo exclusivamente al looping. Una pérdida finita y una recuperación exacta no garantizan una receta que conserve calidad.

Fuentes de verdad:

- [Métricas finales denso](../../runs/e7x-early-dense-20261001-032-seg05/metrics.json).
- [Métricas finales looped](../../runs/e7x-early-looped-20261001-033-seg05/metrics.json).
- `post-run-verification.json` en cada segmento `032-seg01..05` y `033-seg01..05`.
- [Reporte denso](../../output/e7x-early-dense-20261001-report-v1/report.md): todavía es parcial y contiene sólo el denso.
- [Protocolo E7](../architecture/protocolo-e7-20261001.md), [contrato general](../architecture/experimentos.md).

El contrato general aún describe resultados v2 como pendientes. El agente ejecutor debe actualizar el estado con ambos brazos y producir un reporte pareado nuevo, sin sobrescribir el reporte parcial. Reinspeccionar los archivos antes de actuar: otra sesión puede seguir avanzando después de este snapshot.

## Lectura científica y diagnóstico E7

La corrección XSA FP32 en ambos brazos es adecuada y debe mantenerse. No mezclar checkpoints v1 y v2 ni reutilizar el intento abortado como resultado de calidad. La equivalencia exacta del port se verificó para un pase; no prueba que K2 preserve la función inicial. Madurez y seed no están cruzadas. E7 iguala tokens, no cómputo, y sus NLL no son comparables con TinyStories por usar otro corpus/tokenizer.

Primero, trabajo sin optimizador:

1. Cerrar el reporte pareado temprano incluyendo el preentrenado sin adaptar, los dos puntos iniciales, finales y costes. Mantener el endpoint prefijado; mostrar mejores checkpoints sólo como diagnóstico.
2. Revisar las series completas de pérdidas, LR, gradientes y batches. Los primeros y últimos batches son distintos: no concluir sobreajuste a partir de dos pérdidas de entrenamiento puntuales.
3. Evaluar checkpoints inicial/intermedio/final sobre una misma muestra fija del corpus train y la validación retenida, sin actualizar pesos. Si ambos empeoran, priorizar optimización o errores del flujo; si sólo validación empeora, investigar generalización y distribución. Estas observaciones orientan, no confirman una causa.
4. Verificar alineación x/y, identidad de tokenizer/shards, ponderación de acumulación y último batch, gradientes tras unscale, schedule reanudado y evaluación independiente del cursor/RNG. La continuidad bitwise verifica reproducción, no sustituye estas comprobaciones.

Hipótesis abierta: LR de adaptación excesivo para este modelo/batch y reinicio de AdamW. El contrato usa LR 3e-4, betas 0,9/0,95, 4.096 tokens/update, warmup10 y cooldown final15%. No hay evidencia suficiente para confirmar LR como causa; tampoco para llamar al deterioro «catastrophic forgetting» sin evaluar capacidades retenidas adicionales.

Si las comprobaciones no encuentran un error, propuesta de calibración **separada y pendiente de autorización**: denso temprano desde el mismo preentrenado/seed42, LR 3e-5 frente al prefijo existente de LR 3e-4, 100 updates / 409.600 tokens. Cambiar sólo LR; conservar batches, AdamW, precisión y el schedule original definido para 2M, deteniendo el piloto a100 sin introducir un cooldown abreviado. Elegir otro LR requeriría declarar otra comparación. Cargar pesos iniciales, no continuar desde el denso degradado ni heredar estados de prueba. Es diagnóstico, no réplica confirmatoria ni validación de un LR óptimo. Selección repetida sobre la validación actual exige un split de desarrollo separado para futuras afirmaciones confirmatorias.

No cambiar LR, batch, betas y arquitectura simultáneamente. Una receta nueva requiere versión/linajes nuevos; no alterar retrospectivamente E7-v2, sus gates o los resultados.

## Siguiente experimento 3M: datos nuevos frente a repetidos

La recomendación original se mantiene para TinyStories. Este problema es distinto de E7: E7 ya usa2M targets sin repetir; TinyStories3M consume6,144M sobre un shard de427.856 tokens, unas14,4 vueltas, según [manifest](../../data/processed/manifest.json).

Diseño propuesto, no ejecutado:

| Factor | Contrato propuesto |
| --- | --- |
| Control | E8 existente, seeds42/43/44, shard pequeño repetido |
| Nuevo brazo | Corpus ampliado de historias distintas de la misma fuente/revisión, suficiente para6.144.000 targets sin repetir ventanas |
| Modelo y receta | Denso3M GELU, AdamW y cooldown E8; sin otra técnica nueva |
| Presupuesto |6.144.000 tokens /1.500 updates por seed;18.432.000 tokens nuevos principales en total |
| Inicialización | Mismos pesos reales por seed que E8; comprobar hash del checkpoint inicial, no confiar en su campo histórico `initial_weights_sha256`, incorrecto en seeds43/44 |
| Tokenizer / validación | BPE4096 existente y mismos39.275 targets retenidos; tokenizer congelado |
| Variable | Cobertura/diversidad del corpus de entrenamiento, conservando fuente y política de selección |
| Lectura | NLL final y diferencias pareadas; calidad alcanzada por tokens y segundos; costes de preparación separados |

Construir corpus nuevo en un directorio versionado de `data/`; conservar shards existentes. Verificar historias/IDs/hashes disjuntos de validación y duplicados internos. «Tokens nuevos» significa nuevas ocurrencias de texto/documentos, no IDs de vocabulario diferentes. Fijar selección antes de ver resultados. No retokenizar con un BPE nuevo ni cambiar simultáneamente la calidad/fuente del corpus.

El control E8 puede reutilizarse para calidad si identidad y receta coinciden. Sus tiempos históricos no prueban una aceleración bajo las condiciones actuales; una comparación temporal confirmatoria requeriría medición comparable y presupuesto adicional declarado. Tres seeds aportan evidencia local, no una conclusión universal. Este diseño responde a cobertura de datos; no es una comparación directa con E7.

## Prioridades posteriores, condicionadas

- **E9 grammatical:** comparar su PPT1.003.520 + TinyStories6.144M con una fase previa de texto natural del mismo tamaño seguida del mismo entrenamiento. Igualar traspaso sólo de pesos y reinicio de AdamW/scaler/cursor/RNG; así separar curriculum sintético de presupuesto/traspaso extra. No basta extender E8 con otro schedule.
- **E9 retrieval:** antes de un principal nuevo, exigir aprendizaje observable en la sonda. La receta actual aplica CE a toda la secuencia y sólo16/256 targets son respuestas; seed42 obtuvo0/2.048 después de PPT. Una tarea más sencilla o pérdida sólo de respuesta sería otra intervención, con contrato nuevo.
- **Batch/acumulación:** estudiar menor acumulación sólo después de disponer de un baseline estable. Expresar schedule en tokens y calibrar hiperparámetros de Adam por protocolo; cambiar batch altera número de updates y memoria temporal. Medir throughput real y NLL por segundo.
- **Muon, MIR o looping selectivo:** posponer hasta resolver receta/datos. Los negativos actuales delimitan configuraciones concretas, no descartan familias completas.

## Literatura y límites de transferencia

Fuentes primarias consultadas en la revisión anterior; SmolLM2 y Small Batch se revisaron nuevamente para este handoff.

| Paper | Uso en la decisión |
| --- | --- |
| [SmolLM2,2025](https://arxiv.org/html/2502.02737v1), sección6 |135M/360M se beneficiaron de una sola etapa con datos consistentemente buenos. Apoya priorizar datos; no prescribe una receta para3M. |
| [Small Batch Size Training,2025](https://arxiv.org/html/2507.07101v1) | Ajustar memoria de Adam en tokens y medir el menor batch que maximice throughput. No garantiza que eliminar acumulación mejore esta GPU. |
| [Data-Constrained Pretraining/MIR,junio2026](https://arxiv.org/html/2606.06888v1) | Distingue datos únicos/repetidos; masking auxiliar mejora modelos72M–1,4B. Su equivalencia estimada≈1,3x datos no es aceleración local comprobada. |
| [Recursive Transformers,agosto2026](https://arxiv.org/html/2608.26973v1) | Resultados con10M/100M palabras; recurrencia y embeddings factorizados deben aislarse antes de transferir. |
| [SMELT,septiembre2026](https://arxiv.org/abs/2609.01343) | Loops intermedios con presupuestos controlados en MoE. No extrapolar su ahorro6,8–18% de FLOPs a3M denso. |
| [Synthetic PPT,septiembre2026](https://arxiv.org/html/2609.39827v1) | Régimen500M–7B y datos distintos. E9 local no reproduce su escala/generador; fallo de adquisición limita la lectura del mecanismo. |
| [Practical Efficiency of Muon,2025](https://arxiv.org/html/2505.02222v2) | Comparar eficiencia y calibración, no sólo nombres de optimizadores con una receta fija. |

## Restricciones operativas

Usar `.venv` local. Preservar fuentes congeladas, runs y resultados. IDs únicos; nuevas pruebas/resultados en `output/`, `runs/`, `logs/` o `.cache/`; datos sólo en `data/`. No cambiar drivers, WSL, MCP ni dependencias globales. Proyecto privado: no publicar, commit/push o enviar mensajes a terceros por este handoff. Registrar costes científicos y operativos, incluyendo calibración, intentos descartados y replay. Este archivo es el único cambio de esta revisión.
