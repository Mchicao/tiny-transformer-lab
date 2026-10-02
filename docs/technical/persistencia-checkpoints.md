# Persistencia y recuperación — continuación 2026-09-30

**Pareja GELU/SwiGLU seed44, 2026-10-01:** continuidad 5 frente a 2 + 3 en procesos nuevos para cada arquitectura; tolerancia rtol=1e-6/atol=1e-7 y hashes de pesos bit a bit iguales. 20 pasos / 81.920 tokens separados; los principales partieron de artefactos aleatorios propios. 74 checkpoints nuevos restaurados, optimizer/scaler/RNG exactos y validation/greedy/muestreo reproducidos sin updates. Fuentes anteriores y 1.714 archivos preservados por SHA-256, incluido el prefijo histórico de results.jsonl. `scripts/utils/check_ffn_seed44.py --ffn gelu|swiglu --audit-run RUN` realiza auditoría sin entrenamiento; el selector FFN debe coincidir con la identidad del checkpoint. [Reporte, evidencia y snapshots](../../output/ffn-seed44-20261001-3f197002/report.md).

**SwiGLU 2026-10-01:** 5 pasos continuos frente a 2 + 3 reanudados en procesos nuevos, rtol=1e-6 / atol=1e-7 y pesos bit a bit iguales. Se verificaron batches, cursor, contadores, optimizer/scaler/RNG y actualización de las 15 matrices FFN. Principal desde artefacto aleatorio propio, nunca desde las pruebas; 37 checkpoints completos recuperados y evaluación/greedy/muestreo finales exactos, 1.215 archivos anteriores intactos. `scripts/utils/check_ffn.py --audit-run RUN` realiza auditoría sin entrenamiento. [Reporte y evidencia](../../output/swiglu-3m-20261001-01cfc7e0/report.md). Fuentes anteriores intactas; variante, configuración y artefacto inicial tienen identidad propia.

**Réplica seed42 2026-10-01:** recuperación Muon 5 frente a 2 + 3 bit a bit igual; ampliación AdamW con estado exacto y proceso nuevo sin actualizaciones. Se recuperaron 59 checkpoints nuevos y se reprodujeron evaluación y generaciones finales. [Reporte y evidencia](../../output/replication-seed42-20261001-d523c048/report.md). El entrypoint `scripts/replicate.py` añade una identidad propia conservando las fuentes de los cargadores anteriores; `scripts/utils/check_replication.py --audit-run RUN` verifica la réplica sin entrenar.

**Campaña local 2026-10-01:** AdamW, Muon combinado con AdamW y Muon + QK-Norm fijo completaron el presupuesto común de 6.144.000 tokens. Pasó recuperación 5 frente a 2 + 3 en procesos nuevos para ambas variantes nuevas, con pesos bit a bit iguales y tolerancia declarada. Se recuperaron los 96 conjuntos nuevos y se reprodujeron evaluación y generaciones finales. [Reporte y evidencia](../../output/campaign-3m-20261001/report.md). `scripts/utils/check_experiments.py --audit-run RUN` comprueba las nuevas identidades sin actualizar pesos; el cargador anterior sigue intacto.

**Actualización 2026-10-01:** AdamW local, recuperación entrenada y latest/best/milestones ya verificados. Ver [primer entrenamiento](../../runs/adamw-3m-20261001-001/report.md). El resto de este documento conserva el estado histórico de la preparación del 30 de septiembre; los pendientes locales indicados abajo fueron resueltos por ese run. Drive/Colab no se modificaron ni se validaron de nuevo.

**Preparación parcial, sin entrenamientos.** Entorno local y checkpoints aleatorios verificados. Colab se reconectó con el MCP oficial. Hay persistencia real de un archivo remoto en el SSD local; Drive sigue bloqueado y la reanudación de entrenamiento no está implementada.

## Evidencia verificada

| Prueba | Resultado y artefacto |
| --- | --- |
| Entorno local | Windows 11 Pro 26200, driver existente `32.0.21045.5002`, Python 3.12.11, torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, RX 6750 GRE `gfx1031`, 10.720.641.024 bytes VRAM. `uv pip check` compatible. |
| Modelo y GPU | Causalidad, paridad manual/SDPA CPU, 3.000.384 parámetros. FP16 forward/backward pequeños, gradientes finitos y hashes de pesos inalterados en manual/SDPA HIP. `output/prep-continuation-20260930/local-verification.json`. |
| Datos y mediciones anteriores | Hashes de raw, tokenizer y ambos shards coinciden con manifest; BPE 4096, 2000/200 historias, 427856/39276 tokens. Runs manual-v4 siguen siendo comparables, 41934,13/50366,36 tokens/s históricos. No se repitió el benchmark. |
| MCP por proyecto | `.codex/config.toml` conserva ruta oficial y Helium. Vendor limpio en `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`; no se cambió configuración global. Sesión oficial STDIO → WebSocket → Helium: listar, añadir y ejecutar celdas con resultados recuperados. |
| Runtime de esta continuación | Colab **CPU**, Python 3.13.15, torch `2.11.0+cpu`, NumPy 2.1.3, safetensors 0.8.0, tokenizers 0.23.2. No se instalaron dependencias ni se solicitó GPU. La T4 de la primera preparación no se presupone vigente. |

Triton sigue ausente. Se conservó el fallo histórico `TritonMissing`; no se instalaron workarounds ni se repitió compile. Los outputs anteriores fueron rehasheados y permanecen iguales.

## Reconexión MCP y conservación

El primer intento venció sin conexión. En el código oficial, `await_proxy_connection` aplica `asyncio.wait_for` a un `gather` que incluye `_start_task`; el timeout cancela esa tarea. Reintentar en el mismo proceso terminó en timeout de cliente. Arrancar un proceso nuevo y aceptar **Connect** a tiempo produjo `true` y habilitó las herramientas del notebook.

Se añadió `--session-id` al cliente auxiliar y `-SessionId` al wrapper. Cada sesión nueva tiene su cola y evidencia; evita sobrescribir `output/colab-tools.json`, `output/colab-connection.json` y reproducir solicitudes de una cola anterior. La implementación oficial no fue parcheada.

Sesiones de esta continuación: `prep-20260930-drive-v1` y `prep-20260930-drive-v2`. Ambas están detenidas. El controlador Cua temporal fue detenido; no se modificaron los controladores ni MCP globales. No asumir conexión activa al retomar.

Dos lecturas finales de `get_cells` fallaron por timeout/desconexión. Se reconstruyó el notebook local desde las solicitudes y respuestas oficiales ya recibidas: `output/prep-continuation-20260930/colab-diagnostics.ipynb`. Sus seis celdas y siete eventos de ejecución están archivados en `colab-cell-evidence.json`. El notebook privado anterior no se modificó.

## Drive: qué falla y qué no se demostró

`scripts/agents/check_colab_drive.py` es la comprobación reproducible: monta, crea un archivo único sólo en `MyDrive/Proyectos_B/tiny-transformer-lab/persistence-checks`, verifica SHA-256, hace flush/unmount, remonta y relee. Falla si no se prueba la última relectura.

- Dos intentos con `timeout_ms=60000` devolvieron `ValueError: mount failed`, a los 61,349 y 60,710 segundos. El consentimiento tardío es un factor observado en esos intentos.
- Un intento con consentimiento completado dentro del plazo devolvió `MessageError` a los 33,993 segundos. La instrumentación inicial sólo registró el tipo, no el mensaje exacto; no atribuirlo a un motivo concreto.
- Un último probe limitado registró `mount failed` a los 60,967 segundos; el controlador GUI quedó sin respuesta y no se pudo completar ese consentimiento dentro del plazo.
- DriveFS contiene referencias a timeout y credenciales, sin `UNAUTHENTICATED`, `PERMISSION_DENIED`, `invalid_grant` o política de dominio. Las referencias no prueban por sí solas la causa. Tras el `MessageError`, los logs anteriores no aumentaron de tamaño: el fallo puede ser anterior al nuevo proceso DriveFS.
- Se comprobó la cuenta existente y se seleccionó sólo el permiso de documentos Drive. Photos, contactos, mensajes, configuración móvil y actividad de archivos quedaron sin seleccionar. No se extrajeron cookies, contraseñas, 2FA ni tokens.

**No se escribió ni se releyó ningún archivo en Drive.** La causa exacta sigue pendiente. No se investigó moviendo, borrando o enumerando archivos personales de la raíz de Drive.

Acción manual concreta para desbloquear: cargar `output/prep-continuation-20260930/drive-persistence-check.ipynb` en Colab, ejecutar sus celdas y completar **Connect to Google Drive → Continue** con la cuenta existente y sólo permisos Drive. Requiere terminar el consentimiento dentro de 60 segundos. Si falla de nuevo, conservar el JSON `error_type`/`error_class` que imprime la celda; no compartir URLs OAuth ni credenciales. No seleccionar permisos ajenos para intentar resolverlo.

Fuentes: [código oficial de Drive](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py) y [FAQ de timeouts](https://research.google.com/colaboratory/faq.html#drive-timeout). La FAQ contempla carpetas raíz muy grandes; no se verificó esa hipótesis en esta cuenta.

## Persistencia real disponible: MCP → SSD

Se creó un archivo de 145 bytes en el scratch de Colab, se releyó, se transfirió por el MCP oficial, se escribió con flush/fsync en NTFS y se releyó en un proceso local nuevo.

- Archivo: `output/prep-continuation-20260930/persistence-6b2146b39c1b4da5a979c0e7bde57994.json`.
- SHA-256: `5c358bd159f48a415ec5417bb4c206333394ede87c99d707c5f024e1d8bdc3f2`.
- Resultado: `output/prep-continuation-20260930/persistence-ssd.json`.

Esto demuestra persistencia del artefacto en el SSD, incluso con el cliente detenido. No demuestra montaje Drive, transferencia de checkpoints grandes, recuperación tras pérdida abrupta del runtime ni almacenamiento automático para un entrenamiento Colab. El transporte depende de que MCP y el navegador sigan conectados hasta recuperar el archivo.

## Safetensors y recuperación aleatoria

```powershell
.\.venv\Scripts\python.exe scripts/utils/check_checkpoint.py --device cuda
```

El comando crea una carpeta nueva automáticamente; `--output-dir` debe señalar una carpeta inexistente. No contiene pasos de optimizador. Exporta pesos aleatorios, estado y metadata mediante temporal + reemplazo; calcula checksums antes de cargar en un proceso nuevo.

Verificaciones realizadas:

1. Safetensors coincide tensor a tensor con `output/initial-3m.safetensors`; hash de pesos `194d4675760675037cfaef56e78f3e8aad74ca07d244068e782ac60d8ac1ef64`.
2. Proceso nuevo: carga estricta, pesos y logits exactos dentro del mismo backend, identidad config/tokenizer/shards y siguiente batch tras restaurar cursor.
3. RNG Python, NumPy, PyTorch CPU y HIP restaurados. GradScaler inicializado/restaurado en HIP.
4. Estado AdamW no vacío con buffers **sintéticos** y contador cero; verificación exacta de buffers y learning rate después de carga. No representa un optimizador entrenado.
5. Una copia con un byte corrupto es rechazada por la ruta real de recuperación antes de cargar pesos.

Pasó en HIP local (`output/checkpoint-check-20260930-hip-v1/recovery.json`, `corruption-check.json`, `logs/tests/checkpoint-check-20260930-hip-v1.log`) y Colab CPU (`output/prep-continuation-20260930/colab-checkpoint-recovery.json`). Hash inicial igual en ambos; no se exige igualdad de logits entre CPU y HIP. El bundle se extrajo en una carpeta remota nueva, sin ejecutar bootstrap/benchmark ni instalar ruedas AMD.

Validación estática: `py_compile` pasó en los scripts modificados. Ruff, basedpyright y pytest no están instalados/configurados en el proyecto; no se instalaron para esta preparación. La evidencia principal son las pruebas ejecutadas sobre los artefactos reales.

## Puerta del primer entrenamiento 3M

Todavía **no está preparado para ejecutarse**: faltan bucle AdamW, evaluación, continuidad de entrenamiento tras interrupción y retención `latest/best/milestones`. Los criterios están en [experimentos](../architecture/experimentos.md#criterios-de-la-siguiente-etapa).

La siguiente autorización debe cubrir la implementación y sus pruebas con actualización de pesos antes del sanity de 1M tokens. Para Colab también debe resolverse persistencia automática; la ruta MCP → SSD actual no satisface esa condición. No se ejecutaron entrenamientos de modelos ni tokenizers, ni commits/push, publicaciones o compras.
