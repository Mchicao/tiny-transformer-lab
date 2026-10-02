# E9: pre-pretraining sintético con traspaso sólo de pesos

Preparación y primer entrenamiento retrieval/seed42 autorizados por el usuario el 2026-10-01. Cada principal posterior requiere su autorización individual. E7 no se ejecuta en esta fase.

**Ampliación de autorización, 2026-10-01:** después del primer resultado, el usuario indicó «sí, continua haciendo experimentos hasta que yo te diga que pares». Autoriza continuar los principales restantes del plan manteniendo sus presupuestos, gates y verificaciones; ya no se solicita permiso individual para cada run de esa campaña. E7 se prepara después de completar E9. No autoriza barridos ni presupuestos nuevos fuera del plan.

## Comparación fijada antes de entrenar

| Factor | Contrato |
| --- | --- |
| Modelo | Denso 3.000.384 parámetros, GELU, contexto 256, manual attention, HIP eager FP16 |
| Réplicas previstas | Seeds 42/43/44; mismos pesos aleatorios que los controles E8 |
| Control A | Runs E8 existentes, 6.144.000 tokens TinyStories |
| Brazo B | PPT retrieval: 245 actualizaciones / 1.003.520 tokens; luego 1.500 / 6.144.000 TinyStories |
| Brazo C | PPT grammatical: mismos presupuestos, arquitectura y traspaso que B |
| PPT | AdamW 3e-4, betas 0,9/0,95, warmup 10, LR constante después del warmup |
| Traspaso | Sólo pesos; AdamW, scaler, cursor TinyStories y RNG se reinician a la seed |
| Pretraining natural | Receta E8: warmup 10, LR 3e-4 hasta paso PT 1.275, coseno a cero al PT 1.500 |
| Métrica principal | NLL TinyStories en PT 1.500, diferencias pareadas contra E8 |
| Métrica secundaria | Sonda sintética retenida: NLL de respuesta, precisión por respuesta y exact-match de las 16 respuestas de cada secuencia |

El PPT agrega **16,33%** de tokens respecto al control. No es una comparación a presupuesto total igualado. No demuestra ahorro de tokens ni superioridad por cómputo frente a invertir ese coste adicional en texto natural.

## Datos sintéticos

PCG64 con `SeedSequence(seed, spawn_key=(9,))`, shards uint16 inmutables en `data/synthetic-ppt/e9-v1/`. Cada ventana de 256 tokens contiene 16 asociaciones `MEM clave valor` al comienzo, relleno aleatorio, y las 16 consultas `QUERY clave respuesta` en orden permutado, posiciones 160–207. Cada clave se consulta una sola vez: ninguna respuesta anterior de esa ventana resuelve la consulta siguiente. Las respuestas están a 115–205 posiciones de sus valores originales. Los tokens 0/1 son marcadores; los terminales cubren 16–4095.

- **Retrieval:** valores aleatorios nuevos en cada secuencia; requieren acceder a la tabla distante.
- **Grammatical:** gramática local de estados finitos con producción fija `QUERY k -> (k-16+2039) mod 4080 + 16`; la misma regla genera la tabla. No requiere memoria distante. Es un control programático gramatical-like, no una réplica del generador del paper.

Ambos brazos usan igual longitud, marcadores, claves, orden de consultas, relleno y presupuesto, con marginales de terminales uniformes por construcción. **No se iguala la entropía condicional ni la dificultad**: una diferencia B–C es exploratoria y no prueba por sí sola el mecanismo causal del paper. La CE se aplica a todos los tokens; las respuestas constituyen 16/256 targets, el resto incluye relleno impredecible.

La sonda usa seed 202610019, 128 secuencias por brazo y 2.048 respuestas. Sus datos nunca se entrenan ni se usan para seleccionar hiperparámetros. También se evalúa el control E8 con exactamente la misma sonda, sin actualizar pesos.

## Recuperación y gates

`scripts/utils/check_ppt.py`: contrato de prueba separado (2 pasos PPT + 3 PT), 5 continuos frente a 2+3 en procesos nuevos; además 3+2 para verificar restauración dentro de PT. Son **15 actualizaciones / 61.440 tokens por comprobación**, separados de los principales. Comprueba pesos, AdamW/scaler/RNG bitwise, batches/cursor, reinicio de AdamW y LR al traspasar, schedule completo por función pura, rechazo de corrupción y seed/brazo/contrato de prueba incompatibles. Un principal exige evidencia de continuidad de su propia seed/brazo y fuentes.

Una omisión AMP, pérdida/pesos no finitos o identidad incompatible detienen el principal. El checkpoint al final de PPT conserva todavía el estado PPT; el reinicio ocurre al comenzar el siguiente paso y se reproduce al reanudar. Los contadores son globales (245+1.500); `phase-accounting.json` separa tokens y tiempos PPT/PT.

Auditoría posterior: restauración de todos los checkpoints, NLL y generaciones exactas, sonda determinista y ausencia de actualizaciones. Fuente del modelo y entrypoints históricos congelados; E9 usa exclusivamente archivos nuevos. Snapshot de fuentes y hashes de artefactos previos en el directorio de preparación.

Gate de adopción: mejora media final frente a E8 ≥0,01 nats y en ≥2/3 seeds. Una sola seed sólo aporta una observación. B mejor que C es evidencia exploratoria compatible con retrieval; no se declara mecanismo confirmado. Con n=3 no se hacen afirmaciones de significancia ni se extrapola a 10M.
