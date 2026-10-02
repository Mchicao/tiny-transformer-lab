# Preparación verificada — 2026-09-30

**Actualización 2026-10-01:** primer AdamW 3M local completado: validation NLL 8,322308 → 4,151172; recuperación entrenada verificada. [Reporte y límites](../../runs/adamw-3m-20261001-001/report.md). Las mediciones y pendientes originales de este documento son históricos.

**Actualización:** ver [persistencia y checkpoints](persistencia-checkpoints.md). Se verificaron de nuevo entorno, datos y comparación v4; pasó recuperación aleatoria HIP/Colab CPU. Persistencia por MCP al SSD sí está probada; Drive y reanudación de entrenamiento siguen pendientes. Los apartados originales conservan la evidencia de la primera preparación.

## Entorno y cambios

Proyecto privado: `D:\Proyectos_B\tiny-transformer-lab`, rama `main`, sin commits ni remotos.
SSD adicional: disco 0, SATA HS-SSD-WAVE(S) 2048G; D: NTFS, inicialmente 1.833.426.354.176 bytes libres.
Windows 11 Pro 64 bits, build 26200. Driver AMD existente `32.0.21045.5002`, sin modificaciones.
Python detectados: 3.10 global, 3.14 y un intérprete uv 3.12.11 existente. uv 0.12.12.
Había una carpeta ROCm 7.1 con HIPRT/rocgdb, pero no hipinfo/hipcc/offload-arch accesibles. No era evidencia de PyTorch operativo.
WSL2 Ubuntu y docker-desktop estaban detenidos. No se iniciaron ni alteraron.

Venv `.venv`: Python 3.12.11; TheRock estable desde índice oficial AMD, versiones y hashes en `configs/requirements-amd.lock.txt`.
torch `2.13.0+rocm10.0.0`, paquetes ROCm SDK `10.0.0`; torch reporta HIP `7.15.26333`.
Se distinguen las versiones del paquete y del runtime, sin afirmar que sean el mismo identificador.
Dependencias comunes bloqueadas: NumPy 2.2.6, tokenizers 0.21.4, safetensors 0.6.2.
`uv pip check`: compatible. Cachés y ruedas descargadas en `.cache/` del SSD adicional.

## GPU y capacidades reales

RX 6750 GRE 10GB, PCI device `73FF`. Detector oficial rocm-bootstrap y propiedades HIP/PyTorch coinciden en `gfx1031`.
Memoria real reportada: 10.720.641.024 bytes / 10.224 MiB. No usar `AdapterRAM` WMI: su campo de 32 bits no representa esta VRAM.

| Feature | Evidencia actual |
| --- | --- |
| FP16 GEMM | Forward/backward con gradientes finitos |
| Transformer manual eager | Forward/backward 3M y benchmark real |
| SDPA | Forward/backward y benchmark; más lento local en esta muestra |
| torch.compile/Inductor | Falla reproducida: TritonMissing |
| Triton | Ausente en el entorno; no instalado |
| FlashAttention, FP8, rocWMMA, hipBLASLt | No probados ni requeridos; no inferir compatibilidad |

PyTorch utiliza HIP con librerías ROCm del venv. No se trazó cada GEMM para atribuirlo específicamente a rocBLAS o hipBLASLt.
La matriz anterior Radeon Windows 7.2.1 no incluía esta RX; TheRock actual publica gfx1031 para Windows. La instalación fue validada sobre el hardware, no sólo por disponibilidad de ruedas.

## Colab y MCP

Implementación oficial `googlecolab/colab-mcp`, commit `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`, paquete 1.0.1.
Se instaló con `uv sync --frozen --no-dev` y el lockfile oficial. Venv independiente bajo `.cache/vendor/colab-mcp/.venv`, Python 3.14.
Configuración sólo en `.codex/config.toml` de este proyecto. No se modificaron MCP globales ni configuración de confianza.
Para cargarla en el agente, abrir el proyecto como confiable y comenzar una sesión allí; esta conversación nació en C:.

Prueba real mediante cliente FastMCP local → servidor STDIO oficial → proxy WebSocket localhost → Colab en Helium.
Cuenta visible: `el.matias.ch@gmail.com`. Se utilizó su sesión existente y el consentimiento normal; no se extrajeron cookies, passwords ni tokens.
La primera apertura superó el timeout de conexión oficial de 60 s; al reconectar dentro del plazo funcionó. No se parcheó el servidor.
Herramientas verificadas: list_tools, add_code_cell, update_cell, run_code_cell, get_cells con outputs.
Se leyó stdout y stderr, instaló el lock común en runtime y ejecutó scripts del mismo repo desde `/content/tiny-transformer-lab`.
GPU obtenida: **Tesla T4**, nvidia-smi 15.360 MiB, driver 580.82.07; torch reporta 15.637.086.208 bytes utilizables.
Python remoto 3.13.15, torch 2.11.0+cu128, CUDA 12.8.
No se verificó el saldo ni la activación de ventajas Google AI Pro. La UI sólo ofreció T4 seleccionable; no se compró ni cambió ningún plan.

## Datos

TinyStories original, API de filas pública: 2.000 train + 200 validation, sin descargar los archivos completos.
427.856 tokens train y 39.276 validation. Cero historias exactamente repetidas entre ambos subconjuntos.
BPE byte-level 4.096 entrenado sólo con train; `<eos>` entre historias. Shards uint16 y mmap con sequence packing secuencial.
Se comprobaron counts, rango de IDs, desplazamiento causal y continuidad del packing.
La API no garantiza fijar revisión histórica: conservar los archivos archivados y hashes. Dataset revision observada `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.

## Comparación final válida

Decoder-only de **3.000.384 parámetros**, 5 capas, dim 192, 6 heads, RoPE/RMSNorm/GELU, weight tying.
Vocab 4.096, contexto 256, microbatch 4, acumulación 4, batch efectivo 16, FP16 autocast con pesos FP32.
Eager manual: un warmup y tres pasos forward/backward, 12.288 tokens medidos, **cero optimizer steps**.
Se verificó igualdad exacta de pesos iniciales, tokenizer, datos y config; pesos sin cambios al terminar.
Las pérdidas evalúan el siguiente token de ejemplos train con pesos aleatorios. No son una curva de entrenamiento ni validation loss.

| Métrica | RX 6750 GRE / HIP | Tesla T4 / CUDA |
| --- | ---: | ---: |
| Tokens/s | 41.934 | 50.366 |
| Tiempo por paso | 97,68 ms | 81,32 ms |
| Tiempo total medido | 0,293 s | 0,244 s |
| Pico asignado PyTorch | 242,61 MiB | 194,11 MiB |
| Pico reservado PyTorch | 280 MiB | 236 MiB |
| Loss primer paso | 8,312599 | 8,312598 |

Diferencia máxima de loss: `2.145767e-6`. Versiones PyTorch distintas, indicadas arriba.
La muestra es corta e incluye comprobaciones de gradientes y transferencias; no extrapolar directamente a GPU-hours de entrenamiento.
SDPA local: 24.909 tokens/s, 230,54 MiB asignados, pérdidas cercanas; manual conserva el default.
Runs preliminares v1-v3 se conservaron para diagnóstico; no constituyen la comparación válida porque sus hashes de pesos diferían.

## Persistencia y pendientes

Las métricas remotas finales se recuperaron al SSD y se verificaron. Hay bundle local de código/datos y checkpoint aleatorio safetensors.
Se guardó una [copia privada del notebook](https://colab.research.google.com/drive/1VevQ4Hm1utXIhCKv8nJrxvgEnJDaNEsl) por la función **Copy to Drive**; se verificó URL `/drive/` y estado **Last saved**. Persiste las celdas/resultados de diagnóstico, no los archivos scratch de la VM.
Drive mount se intentó dos veces con consentimiento de la cuenta existente: devolvió `ValueError: mount failed` y venció el timeout MCP de 120 s.
El diagnóstico de DriveFS mostró referencias a timeout/credenciales, sin códigos PERMISSION_DENIED/UNAUTHENTICATED. **La causa exacta sigue sin demostrarse.**
Se autorizó el permiso Drive, sin seleccionar permisos de Photos/contactos/mensajes. No se conceden permisos ajenos para ocultar el fallo.
No afirmar que la retención Colab latest/best o la reanudación están listas. Resolver persistencia antes de entrenar allí.

Recomendación provisional: 3M local para iteración rápida y Colab para A/B en paralelo; 10M con benchmark propio y tokenizer 8K antes de decidir. Para 20–120M evaluar memoria real y tiempo hasta umbral; preferir Colab si la GPU asignada demuestra ventaja. No inferir rentabilidad por throughput de 3M.

## Fuentes oficiales consultadas

- [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) y [estado de targets](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md).
- [MCP oficial Google Colab](https://github.com/googlecolab/colab-mcp).
- [MCP por proyecto Codex](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
- [Colab FAQ: persistencia y mount](https://research.google.com/colaboratory/faq.html#drive-timeout).
