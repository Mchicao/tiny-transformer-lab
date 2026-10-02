# Primer entrenamiento Colab — resultado parcial verificado

**El 3M aprendió, pero el sanity quedó incompleto.** Se entrenaron y recuperaron 409.600 tokens de los 1.003.520 previstos, con validation loss de 8,3223 a 4,8894. El checkpoint de iteración 100 está completo en D:. La reanudación real no pudo confirmarse y la siguiente solicitud T4 devolvió HTTP 503.

## Experimento ejecutado

| Campo | Valor |
| --- | --- |
| Run | `colab-adamw-3m-20261001-v1`, estado `paused` |
| Hardware | Tesla T4, 15.360 MiB por nvidia-smi, driver 580.82.07 |
| Stack | Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8; NumPy 2.2.6, safetensors 0.6.2, tokenizers 0.21.4 |
| Modelo | 3.000.384 parámetros; arquitectura y datos archivados existentes, atención manual eager |
| Batch | Contexto 256, microbatch 4, acumulación 4: 4.096 tokens/iteración |
| AdamW | LR 3e-4, betas 0.9/0.95, warmup 10 actualizaciones, decay 0.1 en matrices y cero en normas, clipping 1.0 tras unscale |
| Precisión | FP16 autocast, pesos FP32, GradScaler inicial 65536; TF32 deshabilitado |
| Inicialización | Seed 42, safetensors inicial archivado; hash de parámetros `194d4675760675037cfaef56e78f3e8aad74ca07d244068e782ac60d8ac1ef64` |

Validation contiene 39.276 tokens: se predicen los siguientes 39.275, incluyendo el fragmento final. Se suma NLL y se divide por el número real de predicciones.

| Iteración | Tokens de entrenamiento | Validation loss |
| ---: | ---: | ---: |
| 0 | 0 | 8,32230759 |
| 50 | 204.800 | 5,77380028 |
| 100 | 409.600 | 4,88938804 |

100 actualizaciones efectivas y cero omitidas. Ventanas de entrenamiento sincronizadas: 9,048 s, aproximadamente 45.269 tokens/s. Evaluación: 3,045 s. Wall-clock del proceso: 13,553 s. Pico asignado PyTorch: 218.237.952 bytes; reservado: 249.561.088 bytes.

Esos tiempos excluyen instalación, transferencias, depuración y tiempo que el runtime permaneció asignado. No equivalen a facturación ni a duración total de sesión. No comparar este throughput AdamW con el benchmark local sin optimizador ni con otra implementación instrumentada de forma distinta.

## Verificaciones reales

1. OAuth y cuenta autorizada confirmados; GPU T4 real y saldo consultados. Saldo observado: 0,00 unidades. No se compraron unidades ni se modificó el plan.
2. Checkpoint inicial de 12.004.536 bytes transferido SSD → WSL CLI → Colab → WSL CLI → SSD, con hash idéntico.
3. Controles CUDA pequeños de actualización, evaluación aislada, RNG, continuidad en proceso nuevo, AdamW/GradScaler, corrupción/identidad y retención. Logs: `logs/tests/colab-training-check-20261001-v2.log` y `...-v3.log`. El control usa 12 iteraciones de 32 tokens; dos pasadas exitosas, aparte del 3M.
4. Segmento 3M real de 100 actualizaciones. Tres checkpoints recuperados a D: y comprobados archivo por archivo. Pesos de iteración 100 cargados estrictamente en un proceso local; momentos finitos y contador AdamW 100 en todos los parámetros. Esa carga no entrenó localmente.

## Artefactos para recuperar

- Run: `runs/colab-adamw-3m-20261001-v1/`.
- Checkpoint: `runs/colab-adamw-3m-20261001-v1/checkpoints/iteration-000100/`.
- `latest.json`, `best.json` y `milestones.json` apuntan a conjuntos completos; latest/best son iteración 100.
- Archivo: `output/colab-experiment-20261001-v1/phase1-trained-checkpoints.tar.gz`, 77.341.254 bytes, SHA-256 `5db256d2e4aa4c9c8a10547c8acab659791b3d8fbec9a797024a67df4481cd68`.
- Relectura: `output/colab-experiment-20261001-v1/phase1-persistence-verification.json`.
- Bundle final: `output/colab-experiment-20261001-v1/reproducible-training-bundle-final.zip` y su manifest.

Hash de parámetros de iteración 100: `98ac94a6d02fcb1b9d374f30111135b0fbff56f7327df25406e741527dd57bcd`. Tokens/cursor: 409.600. Conserva configuración, hashes, optimizer/scaler y RNG.

## Pérdida de sesión y presupuesto incompleto

- La primera sesión dejó de ser accesible durante preparación: CLI informó 404/401 y retiró su referencia local.
- La segunda permitió controles y entrenamiento. Al intentar restaurar el archivo del SSD, Jupyter/WebSocket terminó con `RuntimeError: Connection was lost`; no se recuperaron métricas de `colab-adamw-3m-20261001-v1-resumed`.
- No se sabe si ese intento llegó a ejecutar actualizaciones. El trabajo perdido no está cuantificado; sólo se afirman los 409.600 tokens de la trayectoria recuperada.
- Al comprobarla, la sesión ya no figuraba disponible. Una nueva solicitud para recuperar respondió `503 Service Unavailable`. No se atribuye ese código al saldo cero como causa demostrada.
- La última consulta reportó saldo 0,00, rate 0,00/h y cero asignaciones. No asumir una GPU activa.

**No se completó 1M tokens ni se verificó la continuidad efectiva del 3M hasta ese objetivo.** El control pequeño de continuidad y la carga del checkpoint entrenado no sustituyen esa prueba.

La generación greedy de ambos prompts produjo un punto y repitió saltos de línea. La reducción de validation loss demuestra aprendizaje inicial, no calidad narrativa ni convergencia. Muon/QK-Norm no se probaron en esta sesión.

## Fuentes locales concurrentes conservadas

Otro proceso completó el baseline HIP local en paralelo. Se detectó una colisión de nombres con sus tres archivos de entrenamiento. El núcleo original se recuperó del bundle anterior y las tres fuentes finales se recuperaron exactamente de los parches del mismo workspace; el hash del núcleo coincide con el run local. Recibo: `output/colab-experiment-20261001-v1/source-collision-recovery.json`.

Los archivos locales originales permanecen en `src/training.py`, `scripts/train.py` y `scripts/utils/check_training.py`. Colab quedó separado en `src/colab_training.py`, `scripts/train_colab.py` y `scripts/utils/check_colab_training.py`; el núcleo Colab conserva exactamente el hash que exige su checkpoint. Se compiló y se verificó el control local existente en modo sin nuevas actualizaciones.

El run local ajeno no se modificó. Las dos implementaciones comparten modelo/datos, pero difieren en escala inicial y comprobaciones de finitud/hashing. No declarar una comparación pura de hardware sin alinear esos detalles y el presupuesto.

## Continuación

Cuando Colab vuelva a asignar T4, restaurar el bundle final en scratch nuevo, instalar el lock común y restaurar desde D: el checkpoint 100. Usar `scripts/train_colab.py`, run-ID nuevo y registrar `resume_from`; el trainer valida artefactos/fuentes y backend antes de cargar.

Quedan 145 iteraciones de 4.096 tokens para alcanzar 1.003.520 en la trayectoria aceptada. Comprobar que validation loss al cargar sea cercana a 4,88938804 y recuperar checkpoints/métricas antes de terminar. Si se pierde acceso, registrar el trabajo perdido o desconocido antes de repetir pasos.

La sincronización automática resistente a desconexiones todavía falta; la transferencia CLI exige conexión hasta descargar. Drive sigue sin superar la prueba de persistencia. No ampliar presupuesto ni cambiar técnica para ocultar ese límite.
