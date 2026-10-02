# Campaña autónoma hasta las09:00 de Santiago

Autorización del usuario: «Continua con los experimentos hasta mañana a las9am aprox. trabaja de forma autónoma». La máquina marca2026-10-02 01:23−03:00; se interpreta como la próxima09:00 de Santiago, **2026-10-02 09:00−03:00**, explicitado al usuario. Correcciones posteriores del usuario prevalecen. No se inicia un segmento cuyo límite de600s exceda esta hora; resultados/checkpoints existentes se conservan.

## Prioridades y presupuestos fijados

1. **E7, diagnóstico sin optimizador.** Comparar pesos iniciales y checkpoints100/200/final de la pareja temprana v2 sobre una muestra train fija y la misma validación. Incluir referencia FP32 de los mismos pesos y checks de alineación, acumulación, contabilidad del último batch y aislamiento RNG/cursor. La continuidad no prueba calidad de adaptación.
2. **E7, calibración LR en FP32.** Dos pilotos del denso temprano/seed42 desde los mismos pesos originales: LR3e-4 y3e-5, **100 updates/409.600 tokens cada uno**, cambiando sólo LR. Mantener schedule de2M/warmup10/cooldown415–489, sin abreviarlo al piloto. Datos dev nuevos disjuntos, fijados antes de entrenar, para seleccionar; validación histórica sólo descriptiva. Ambos necesitan continuidad propia y auditoría. Total principal piloto819.200 tokens; pruebas separadas.
3. **E11, TinyStories nuevos frente a repetidos.** Denso3M GELU/AdamW/cooldown E8, seeds42/43/44,6.144.000 tokens/1.500 updates cada uno; **18.432.000 tokens principales nuevos**. Control: E8 existente. Ampliar historias de la misma fuente/revisión observada, conservar el tokenizerBPE4096 y los39.275 targets val. Prefijo histórico intacto, documentos adicionales en orden de filas prefijado, duplicados/overlap rechazados, shard suficiente para no volver al comienzo. Nueva cobertura de datos es la única variable.
4. **E7, continuación condicionada.** Sólo continuar a2M o a las otras revisiones si una receta conserva calidad en dev y supera al piloto LR3e-4; no completar una matriz que sólo confirme degradación no diagnosticada. Criterio piloto: dev final no empeora más de0,01 nats respecto al inicial y mejora≥0,005 sobre el control alto. Esto es selección exploratoria, no significancia ni LR óptimo. No mezclar pesos/optimizer de pruebas con principales ni versiones FP16/FP32.
5. **E10, inferencia condicionada.** Sólo después de baseline estable y con un cap de inferencia escrito antes de ejecutar. Cero entrenamiento. No descargar modelos4B–9B ni instalar una pila nueva para llenar la ventana.

## Reglas de ejecución

- Una variable por comparación, seeds pareadas y presupuesto por tokens; runs únicos y archivos nuevos para fuentes numéricas/de datos/LR.
- E7 actual: FP16 falló en XSA y luego en MLP de la revisión media. BF16/FP32 resolvieron el caso real; FP32 reprodujo logits originales exactamente y mostró throughput adecuado en la RX6750. Las fuentes y artefactos v1/v2 se conservan; no se usan como controles de una receta FP32 distinta.
- Cada principal/segmento en una llamada PowerShell independiente,≤600s. Auditoría antes de continuar. Cualquier pérdida/gradiente/peso no finito, omisión o fallo de identidad detiene el brazo; no se omiten batches.
- Registrar costes de preparación, diagnóstico y recuperación aparte. El intento v1/replay consumió1.204.224 tokens de updates. V2 consumió4M en la pareja temprana más1.277.952 en el medio abortado; dos replays de12 updates sumaron98.304 tokens adicionales. Estos consumos no desaparecen por preparar FP32.
- No modificar drivers, WSL, MCP, dependencias globales, fuentes congeladas ni servicios externos. No publicar ni hacer push. Datos en `data/`, evidencias en `output/`, `runs/`, `logs/`, `.cache/`.

## Entrega

Reporte final con tabla de calidad/costes, comprobaciones ejecutadas, gates y pendientes. Registrar en `experimentos.md`; detener nuevos entrenamientos a la hora límite. Si un gate cierra una rama, conservar el negativo y pasar a la siguiente prioridad autorizada.
