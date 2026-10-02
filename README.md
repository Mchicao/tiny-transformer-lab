# Tiny Transformer Lab

Laboratorio local en `D:\Proyectos_B\tiny-transformer-lab`; primer baseline AdamW 3M ejecutado y recuperado en HIP.
Un único modelo PyTorch para HIP/AMD y CUDA/Colab. No depende de Triton ni FlashAttention.

## Tercera réplica FFN seed44 — 2026-10-01

- Ambos AdamW 3M completados desde pesos aleatorios: GELU NLL **8,346746 → 3,160184** (también su mejor); SwiGLU **8,342689 → 3,214367**, mejor **3,165126** a 1.200 pasos. **6.144.000 tokens / 1.500 efectivas / 0 omitidas por modelo**.
- GELU gana ambas medidas en seed44. En seeds42/43/44, diferencia media SwiGLU−GELU: final **+0,018757**, mejor checkpoint **−0,005979 nats/target**. GELU gana al final en dos seeds; SwiGLU gana por mejor checkpoint en dos. No hay ventaja consistente en las tres ni significancia demostrada. SwiGLU seed44 empeora después de su mínimo.
- Nuevos principales: 12.288.000 tokens / 3.000 pasos; pruebas aparte: 81.920 tokens / 20 pasos. Continuidad bit a bit igual en ambas arquitecturas; 74 checkpoints recuperados, evaluación y generaciones exactas. 1.714 archivos históricos y prefijo de `results.jsonl` preservados.
- GELU: 162,65 s train / 222,67 s wall, 37.775 tokens/s, pico PyTorch 257,5/282,0 MiB. SwiGLU: 179,00 / 243,94 s, 34.323 tokens/s, 262,5/288,0 MiB. Mediciones secuenciales, sin atribuir causalmente la diferencia de tiempo a la FFN.
- [Reporte de tres seeds, curva, generaciones y checkpoints](output/ffn-seed44-20261001-3f197002/report.md). Entry `scripts/ffn_seed44.py --ffn gelu|swiglu`; auditoría sin entrenamiento `scripts/utils/check_ffn_seed44.py --ffn gelu|swiglu --audit-run RUN`. Siguiente propuesta: misma pareja seed44 con decay de LR, sin aumentar modelo ni tokens; no se ejecutó.

## Réplica FFN seed43 — 2026-10-01

- SwiGLU AdamW 3M: NLL **8,318994 → 3,195542**, mejor **3,169331** a 1.200 pasos. 6.144.000 tokens / 1.500 efectivas / 0 omitidas; pruebas separadas 40.960 tokens / 10 pasos.
- GELU seed43 conservado: final **3,186252**, mejor **3,177860**. SwiGLU pierde por **+0,009290** al final, aunque su mejor checkpoint mejora **−0,008530**. La ventaja final de seed42 no se replicó: media pareada final **+0,001044**; no hay ganador consistente al presupuesto final ni significancia demostrada con dos seeds.
- 132,32 s train / 183,54 s wall, 46.433 tokens/s; pico PyTorch 262,5/288,0 MiB asignado/reservado. Recuperación previa bit a bit igual; 37 checkpoints recuperados y evaluación/generaciones exactas sin updates. 1.450 archivos anteriores preservados por SHA-256.
- [Reporte, curva, ambas seeds, generaciones y checkpoints](output/swiglu-seed43-20261001-b6cfe902/report.md). Entry `scripts/ffn_seed43.py`; auditoría sin entrenamiento: `scripts/utils/check_ffn_seed43.py --audit-run RUN`.

## FFN GELU / SwiGLU seed42 — 2026-10-01

- SwiGLU con AdamW, seed42, 3.000.384 parámetros: NLL **8,321924 → 3,185541**, mejor **3,178393** a 4.915.200 tokens. Principal completo: 6.144.000 tokens / 1.500 efectivas / 0 omitidas.
- GELU conservado terminó en 3,192743: ventaja final SwiGLU **−0,007201 nats/target**, perplexity 0,72 % menor. Una seed y FFN con inicialización propia: señal pequeña y descriptiva, sin significancia demostrada.
- 122,30 s train / 168,34 s wall, 50.239 tokens/s; pico PyTorch asignado/reservado 262,5/288,0 MiB. Continuidad 5 frente a 2 + 3 bit a bit igual, 37 checkpoints recuperados; 40.960 tokens de prueba separados y 1.215 archivos previos preservados.
- [Reporte, curva, generaciones, identidades y límites](output/swiglu-3m-20261001-01cfc7e0/report.md). `scripts/ffn_experiment.py` reutiliza el loop sin alterar fuentes anteriores; `scripts/utils/check_ffn.py --audit-run RUN` verifica sin entrenar. [Explicación de NLL, perplexity y t pareado](docs/technical/metricas-nll-perplexity-t-pareado.md).

## Réplica AdamW / Muon seed42 — 2026-10-01

- Completada a 6.144.000 tokens / 1.500 actualizaciones efectivas / 0 omitidas por variante: NLL final/mejor AdamW **3,192743 / 3,192743**, Muon **3,302172 / 3,244917**. NLL inicial compartida 8,322308.
- AdamW obtuvo mejor NLL en seeds42 y43; diferencia Muon−AdamW del mejor checkpoint **+0,052174 / +0,051871**. Dos seeds apoyan el baseline bajo estos hiperparámetros; no prueban superioridad universal.
- Nuevos principales: 10.280.960 tokens; prueba Muon separada: 40.960 tokens / 10 actualizaciones. Estado AdamW heredado exacto, 59 checkpoints nuevos recuperados y 840 archivos históricos/data/fuentes preservados.
- [Reporte, curva de ambas seeds, tiempos, memoria, checkpoints y generaciones](output/replication-seed42-20261001-d523c048/report.md). `scripts/replicate.py` reutiliza el loop anterior conservando sus fuentes; `scripts/utils/check_replication.py --audit-run RUN` verifica sin entrenar.

## Campaña AdamW / Muon / QK-Norm — 2026-10-01

- Tres variantes 3M seed43 comparadas con 6.144.000 tokens / 1.500 actualizaciones efectivas / 0 omitidas cada una. AdamW alcanzó la meseta operativa antes del máximo autorizado de 8.028.160 tokens.
- NLL final / mejor: AdamW **3,186252 / 3,177860**; Muon **3,271451 / 3,229731**; Muon + QK-Norm fijo **3,638630 / 3,354514**. QK-Norm mostró deterioro de validation después de su mejor checkpoint.
- Recuperación continua frente a interrumpida/reanudada en procesos nuevos, omisión AMP de ambas ramas y 96 checkpoints nuevos verificados. Pruebas separadas: 81.920 tokens / 20 actualizaciones. 294 archivos históricos/data/fuentes conservados.
- [Reporte comparativo, curva, tiempos, memoria, checkpoints y generaciones](output/campaign-3m-20261001/report.md). Contrato y límites en [experimentos](docs/architecture/experimentos.md#campaña-local-autorizada--2026-10-01).
- `scripts/experiment.py` gestiona la campaña sin modificar el núcleo ni la CLI anteriores. `scripts/utils/check_experiments.py --audit-run RUN` verifica checkpoints y reproduce evaluación/generaciones sin entrenar.

## Primer entrenamiento local — 2026-10-01

- AdamW 3M: validation NLL 8,322308 → 4,151172; 1.003.520 tokens, 245 actualizaciones efectivas y ninguna omitida.
- Continuidad verificada en procesos nuevos: 5 pasos continuos frente a 2 + 3 reanudados; 40.960 tokens / 10 actualizaciones aparte.
- Checkpoints inicial/final y milestones recuperados; evaluación y generaciones finales reproducidas sin entrenamiento adicional.
- [Reporte, curva, generaciones y límites](runs/adamw-3m-20261001-001/report.md). Resultados locales ignorados por Git.
- Un único sanity demuestra aprendizaje y estabilidad durante ese presupuesto, no convergencia completa. Nuevos entrenamientos requieren autorización.

## Continuación local AdamW 3M — 2026-10-01

- Desde el checkpoint final de 1M: 1.003.520 tokens adicionales, 245 actualizaciones efectivas / 0 omitidas; acumulado 2.007.040 tokens y 490 actualizaciones.
- Validation NLL 4,151172 → 3,709261; 25,59 s de bucle train / 41,89 s de run. Warmup, optimizer, scaler, cursor y RNG conservados.
- Seis checkpoints restaurados, evaluación y generaciones reproducidas; núcleo y run original intactos. [Reporte y curva acumulada](runs/adamw-3m-20261001-002-continuation/report.md).
- `scripts/train.py --additional-iterations 245 --resume CHECKPOINT --run-id ID_NUEVO` amplía sólo el baseline completo hasta 490 iteraciones. No implica autorización para otro entrenamiento.
- `scripts/utils/check_continuation.py --checkpoint CHECKPOINT` y `scripts/utils/sample_checkpoint.py --run-dir RUN` verifican y generan sin actualizar pesos; reconocen los presupuestos explícitos 245/490.

## Réplica local AdamW 3M, seed 43 — 2026-10-01

- Desde una inicialización aleatoria nueva: 2.007.040 tokens, 490 actualizaciones efectivas / 0 omitidas; validation NLL 8,355683 → 3,696174.
- Seed 42 terminó en 3,709261 con igual presupuesto; diferencia −0,013087 nats/target. Los 490 batches/cursor/LR/scale coinciden y sólo cambió la seed experimental.
- 53,80 s de bucle train / 78,16 s de run; 11 checkpoints recuperados, fuentes y resultados seed42 conservados. [Reporte, generaciones y curva comparativa](runs/adamw-3m-20261001-003-seed43/report.md).
- `scripts/utils/prepare_initialization.py` crea un artefacto aleatorio seed43 único en `output/initializations/`, sin entrenamiento ni cambios al tokenizer. `scripts/train.py --seed 43 --initialization DIRECTORIO --run-id ID_NUEVO` usa 490 iteraciones desde cero.
- La CLI y `scripts/utils/check_continuation.py`/`sample_checkpoint.py` restauran ambas seeds con sus identidades; nuevos entrenamientos requieren autorización. Dos seeds no prueban significancia estadística ni una meseta de convergencia.

## Primer entrenamiento Colab — 2026-10-01

- AdamW 3M en Tesla T4: validation NLL `8,322308 → 4,889388` en 409.600 tokens, 100 actualizaciones y ninguna omitida.
- Tres checkpoints completos recuperados y verificados en D:. El sanity de 1.003.520 tokens quedó pendiente al perderse la sesión; la siguiente T4 fue rechazada con HTTP 503.
- [Reporte Colab y recuperación pendiente](docs/technical/primer-entrenamiento-colab.md). Este segmento no demuestra convergencia ni calidad narrativa.

## Estado verificado

- Windows 11 Pro 26200, Python 3.12.11 local.
- RX 6750 GRE 10GB, target real `gfx1031`, 10.224 MiB de VRAM.
- TheRock estable: torch `2.13.0+rocm10.0.0`, runtime HIP reportado `7.15.26333`.
- Colab: Tesla T4, Python 3.13.15, torch `2.11.0+cu128`, CUDA 12.8.
- FP16, forward/backward, atención manual y SDPA local funcionan. Inductor/torch.compile falla por Triton ausente.
- MCP oficial probado: conectar, listar herramientas, leer, crear, modificar y ejecutar celdas; stdout/stderr, instalación remota y ejecución del repo.
- Ver [reporte de preparación](docs/technical/preparacion.md) y evidencia en `output/`, `runs/`.
- Continuación: recuperación de checkpoint aleatorio verificada en HIP y Colab CPU; persistencia Colab → MCP → SSD probada. Drive sigue bloqueado. Ver [pruebas y límites](docs/technical/persistencia-checkpoints.md).

## Uso local

```powershell
Set-Location D:\Proyectos_B\tiny-transformer-lab
.\scripts\utils\setup.ps1
.\.venv\Scripts\python.exe scripts\utils\check_model.py
.\.venv\Scripts\python.exe scripts\utils\probe_gpu.py
```

El setup instala versiones bloqueadas, dentro de entornos dedicados y con caché en D:. No inicia cómputo ni cambia drivers.
El baseline tiene RoPE, RMSNorm, GELU, proyecciones sin bias y embedding/output compartidos.
Configs: 3.000.384, 4.983.552 y 9.998.400 parámetros. Sólo 3M fue benchmarkeado.
Para 10M hay que preparar un tokenizer de 8.192; el shard actual corresponde exclusivamente al tokenizer 4.096.

Preparar de nuevo datos desde la API pública pequeña:

```powershell
.\.venv\Scripts\python.exe scripts\utils\prepare_data.py
```

Se archivan 2.000 historias train y 200 validation; tokenizer byte-level BPE 4.096, `<eos>` entre historias, shards little-endian uint16 y mmap.
Los hashes del subconjunto archivado son la identidad del experimento. La API de filas no garantiza reproducir una revisión histórica; para repetir exactamente, usar el bundle archivado.
Cambiar vocabulario exige un directorio/artefacto nuevo y actualizar ambos entornos antes de comparar.
Ejemplo futuro: `prepare_data.py --vocab-size 8192 --output-dir data/processed-8k`; reutiliza el subconjunto train/validation. Para cambiar también su tamaño, proporcionar otro `--raw-dir`.

Benchmark opcional sin entrenamiento, sólo cuando se solicite otra medición:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --run-id mi-medicion-unica --steps 3
```

No contiene `optimizer.step`. Registra GPU, backend, hashes, loss, tiempo, tokens/s y memoria; verifica pesos inalterados. No prueba convergencia.

## Colab MCP sólo en este proyecto

La configuración está en `.codex/config.toml`; no se modificó la configuración global.
Abrir este directorio como proyecto de confianza en Codex/T3 y comenzar una nueva sesión para cargarla.
La política de confianza del cliente puede requerir aceptación del usuario; no se alteró globalmente.
Usar `open_colab_browser_connection`; el servidor oficial abre Helium. Aceptar **Connect** en el notebook dentro de 60 segundos.
El cliente debe soportar `notifications/tools/list_changed`.

También hay un cliente local pequeño, probado contra el servidor STDIO oficial:

```powershell
.\scripts\agents\start-colab.ps1 -SessionId mi-sesion-unica
```

Desde otro proceso:

```powershell
.\.venv\Scripts\python.exe scripts\agents\colab_call.py list --session-id mi-sesion-unica
.\.venv\Scripts\python.exe scripts\agents\colab_call.py get_cells --session-id mi-sesion-unica
```

Para parámetros largos usar `--arguments-file` con un JSON dentro de `.cache/`.
El cliente usa archivos de solicitud/respuesta locales; no reemplaza el MCP oficial y no usa credenciales extraídas.
Una sola sesión/notebook puede conectarse al proxy. No iniciar a la vez este cliente y el MCP del agente para el mismo notebook.
El wrapper genera un ID único si se omite `-SessionId` y lo muestra al arrancar. Las nuevas colas y evidencias quedan separadas por sesión.
Si vence el primer plazo de conexión, detener ese cliente con `colab_call.py stop --session-id ID` y arrancar otro con un ID nuevo; aceptar **Connect** dentro de 60 segundos.

## CLI Colab desde WSL

CLI oficial `google-colab-cli` 0.7.4 instalada en un entorno dedicado de Ubuntu/WSL, separado del `.venv` AMD y del MCP. Cuenta autenticada, T4 utilizada y transferencia de checkpoints entrenados verificada. No asumir que hay una sesión activa.

```powershell
.\scripts\agents\colab-wsl.ps1 version
.\scripts\agents\colab-wsl.ps1 --help
```

Ver [guía de instalación y uso](docs/guides/colab-cli-wsl.md). No se iniciaron runtimes ni entrenamientos durante la instalación.

## Portabilidad y persistencia

`scripts/agents/build_colab_bundle.py` crea `output/colab-bundle.zip` y una celda de bootstrap con código + datos pequeños; no publica repositorios.
La celda instala dependencias comunes con hashes. Mantiene PyTorch CUDA del runtime y reporta su versión.
**La celda generada incluye un benchmark sin actualización de pesos al ejecutarse.** Generarla no lo ejecuta.
Trabajar en `/content/tiny-transformer-lab`; Drive sólo almacena bundles, métricas y checkpoints importantes.
Ruta persistente: `MyDrive/Proyectos_B/tiny-transformer-lab`.
El montaje de esa ruta aún falla; no usarlo para entrenar hasta verificarlo. Sí se guardó una [copia privada del notebook de diagnóstico](https://colab.research.google.com/drive/1VevQ4Hm1utXIhCKv8nJrxvgEnJDaNEsl) en Drive, con estado de guardado visible. Esa copia registra pruebas y no sustituye al bundle para restaurar un runtime vacío.
`output/initial-3m.safetensors` es un checkpoint aleatorio inicial, no un modelo entrenado.

Comprobación de recuperación sin entrenamiento, con directorio de salida único automático:

```powershell
.\.venv\Scripts\python.exe scripts\utils\check_checkpoint.py --device cuda
```

Exporta safetensors, restaura en un proceso nuevo y verifica pesos/logits, RNG, cursor de datos, buffers AdamW sintéticos y scaler. Rechaza una copia corrupta antes de cargarla. No llama a `optimizer.step` y no valida continuidad de entrenamiento.

AdamW + GradScaler, evaluación y retención están implementados en `src/training.py` y `scripts/train.py`. `scripts/utils/check_training.py` verifica continuidad con 10 actualizaciones reales; no ejecutarlo para una comprobación sólo de lectura. `--verify-existing --check-id adamw-check-20261001-v1` comprueba los artefactos conservados sin actualizar pesos.
La política detallada está en [plan de experimentos](docs/architecture/experimentos.md). La campaña local Muon/QK-Norm está documentada arriba; técnicas y presupuestos adicionales siguen sujetos a autorización.

Colab usa `src/colab_training.py`, `scripts/train_colab.py` y `scripts/utils/check_colab_training.py`, separados de las fuentes locales existentes. El modelo y los datos siguen compartidos. Sus controles CUDA usaron fixtures pequeñas; la reanudación efectiva del 3M recuperado sigue pendiente.
