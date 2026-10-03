# Tiny Transformer Lab

Local research laboratory in `D:\Proyectos_B\tiny-transformer-lab`; first AdamW 3M baseline executed and restored in HIP.
A single unified PyTorch model implementation for both native HIP/AMD (Windows) and CUDA/Colab. Zero dependency on Triton or FlashAttention.

## FFN Seed 44 Replication (GELU vs. SwiGLU) — 2026-10-01

- Both AdamW 3M runs completed from random weight initializations: GELU NLL **8.346746 → 3.160184** (also its best); SwiGLU **8.342689 → 3.214367**, best **3.165126** at step 1,200. **6,144,000 tokens / 1,500 effective updates / 0 skipped per model**.
- GELU wins on both metrics in seed 44. Across seeds 42, 43, and 44, the mean paired difference SwiGLU − GELU is: final **+0.018757**, best checkpoint **−0.005979 nats/target**. GELU wins at the final budget in two seeds; SwiGLU wins by best checkpoint in two. There is no consistent advantage across all three seeds and no demonstrated statistical significance. SwiGLU seed 44 degrades after reaching its minimum.
- New main runs: 12,288,000 tokens / 3,000 steps; separate test passes: 81,920 tokens / 20 steps. Bit-for-bit continuity verified across both architectures; 74 checkpoints restored with exact evaluation and sampling output matches. 1,714 historical files and the `results.jsonl` prefix were preserved.
- GELU: 162.65 s train loop / 222.67 s wall, 37,775 tokens/s, PyTorch peak 257.5/282.0 MiB allocated/reserved. SwiGLU: 179.00 / 243.94 s, 34,323 tokens/s, 262.5/288.0 MiB. Sequential runs; runtime differences are not causally attributed to the FFN variant alone.
- [Report across three seeds, curves, generations, and checkpoints](output/ffn-seed44-20261001-3f197002/report.md). Entry point: `scripts/ffn_seed44.py --ffn gelu|swiglu`; zero-training audit: `scripts/utils/check_ffn_seed44.py --ffn gelu|swiglu --audit-run RUN`. Proposed next step: same seed 44 pair with LR decay, without scaling model or token budgets; not executed.

## FFN Seed 43 Replication — 2026-10-01

- SwiGLU AdamW 3M: NLL **8.318994 → 3.195542**, best **3.169331** at step 1,200. 6,144,000 tokens / 1,500 effective updates / 0 skipped; separate test passes 40,960 tokens / 10 steps.
- Preserved GELU seed 43 baseline: final **3.186252**, best **3.177860**. SwiGLU falls behind by **+0.009290** at the final step, although its best checkpoint improves by **−0.008530**. The final advantage observed in seed 42 did not replicate: mean paired final difference is **+0.001044**; no consistent winner at the final budget and no demonstrated statistical significance across two seeds.
- 132.32 s train loop / 183.54 s wall, 46,433 tokens/s; PyTorch peak 262.5/288.0 MiB allocated/reserved. Bit-for-bit identical state recovery; 37 checkpoints restored with exact evaluations and generations without weight updates. 1,450 previous files preserved by SHA-256.
- [Report, curves, both seeds, generations, and checkpoints](output/swiglu-seed43-20261001-b6cfe902/report.md). Entry point: `scripts/ffn_seed43.py`; zero-training audit: `scripts/utils/check_ffn_seed43.py --audit-run RUN`.

## FFN GELU / SwiGLU Seed 42 — 2026-10-01

- SwiGLU with AdamW, seed 42, 3,000,384 parameters: NLL **8.321924 → 3.185541**, best **3.178393** at 4,915,200 tokens. Full main run: 6,144,000 tokens / 1,500 effective updates / 0 skipped.
- Preserved GELU baseline finished at 3.192743: final SwiGLU advantage of **−0.007201 nats/target**, 0.72% lower perplexity. Single seed with dedicated FFN initialization: modest descriptive signal without demonstrated statistical significance.
- 122.30 s train loop / 168.34 s wall, 50,239 tokens/s; PyTorch peak 262.5/288.0 MiB allocated/reserved. Continuity (5 steps continuous vs. 2 + 3 resumed) bit-for-bit identical; 37 checkpoints restored; 40,960 test tokens separated and 1,215 prior files preserved.
- [Report, curve, generations, identities, and limits](output/swiglu-3m-20261001-01cfc7e0/report.md). `scripts/ffn_experiment.py` reuses the loop without modifying prior sources; `scripts/utils/check_ffn.py --audit-run RUN` verifies without training. [Technical explanation of NLL, perplexity, and paired t-test](docs/technical/metricas-nll-perplexity-t-pareado.md).

## AdamW / Muon Seed 42 Replication — 2026-10-01

- Completed at 6,144,000 tokens / 1,500 effective updates / 0 skipped per variant: final/best NLL AdamW **3.192743 / 3.192743**, Muon **3.302172 / 3.244917**. Shared initial NLL: 8.322308.
- AdamW achieved better NLL across both seeds 42 and 43; best checkpoint Muon − AdamW difference: **+0.052174 / +0.051871**. Two seeds support the AdamW baseline under these hyperparameters; they do not prove universal superiority.
- New main runs: 10,280,960 tokens; separate Muon test: 40,960 tokens / 10 updates. Exact inherited AdamW state, 59 new checkpoints restored, and 840 historical files/data/sources preserved.
- [Report, two-seed curves, timing, memory, checkpoints, and generations](output/replication-seed42-20261001-d523c048/report.md). `scripts/replicate.py` reuses the prior loop while preserving sources; `scripts/utils/check_replication.py --audit-run RUN` verifies without training.

## AdamW / Muon / QK-Norm Campaign — 2026-10-01

- Three 3M seed 43 variants compared at 6,144,000 tokens / 1,500 effective updates / 0 skipped each. AdamW reached an operational plateau before the authorized maximum of 8,028,160 tokens.
- Final / best NLL: AdamW **3.186252 / 3.177860**; Muon **3.271451 / 3.229731**; Muon + fixed QK-Norm **3.638630 / 3.354514**. QK-Norm showed validation degradation after its best checkpoint.
- Continuous vs. interrupted/resumed state recovery verified across new processes; AMP skipped on both branches and 96 new checkpoints verified. Separate test passes: 81,920 tokens / 20 updates. 294 historical files/data/sources conserved.
- [Comparative report, curve, timing, memory, checkpoints, and generations](output/campaign-3m-20261001/report.md). Contract and limits detailed in [experiments](docs/architecture/experimentos.md#campa\u00f1a-local-autorizada--2026-10-01).
- `scripts/experiment.py` orchestrates the campaign without modifying prior core or CLI code. `scripts/utils/check_experiments.py --audit-run RUN` verifies checkpoints and reproduces evaluation/generations without training.

## First Local Training — 2026-10-01

- AdamW 3M: validation NLL 8.322308 → 4.151172; 1,003,520 tokens, 245 effective updates, zero skipped.
- Continuity verified across new processes: 5 continuous steps vs. 2 + 3 resumed; 40,960 test tokens / 10 updates isolated.
- Initial, milestone, and final checkpoints restored; evaluation and final generations reproduced without extra training.
- [Report, curves, generations, and limits](runs/adamw-3m-20261001-001/report.md). Local experiment artifacts ignored by Git.
- A single sanity run demonstrates learning and numerical stability within that budget, not full convergence. Any new training requires explicit authorization.

## Local Continuation AdamW 3M — 2026-10-01

- Resumed from the final 1M checkpoint: 1,003,520 additional tokens, 245 effective updates / 0 skipped; cumulative 2,007,040 tokens and 490 updates.
- Validation NLL 4.151172 → 3.709261; 25.59 s train loop / 41.89 s run wall clock. Warmup, optimizer, scaler, dataset cursor, and RNG states preserved.
- Six checkpoints restored, evaluation and generations reproduced; core model and original run artifacts remain untouched. [Cumulative curve and report](runs/adamw-3m-20261001-002-continuation/report.md).
- `scripts/train.py --additional-iterations 245 --resume CHECKPOINT --run-id NEW_ID` extends only the full baseline up to 490 iterations. Does not imply authorization for other training runs.
- `scripts/utils/check_continuation.py --checkpoint CHECKPOINT` and `scripts/utils/sample_checkpoint.py --run-dir RUN` verify and sample without updating weights; they enforce the explicit 245/490 budgets.

## Local Replication AdamW 3M, Seed 43 — 2026-10-01

- From a fresh random initialization: 2,007,040 tokens, 490 effective updates / 0 skipped; validation NLL 8.355683 → 3.696174.
- Seed 42 finished at 3.709261 with identical budget; difference: −0.013087 nats/target. All 490 batches, cursor positions, LR schedules, and loss scales match; only the random seed was changed.
- 53.80 s train loop / 78.16 s run wall clock; 11 checkpoints restored, seed 42 sources and results preserved. [Report, generations, and comparative curve](runs/adamw-3m-20261001-003-seed43/report.md).
- `scripts/utils/prepare_initialization.py` creates a unique seed 43 initialization artifact in `output/initializations/`, without training or tokenizer modifications. `scripts/train.py --seed 43 --initialization DIRECTORY --run-id NEW_ID` executes 490 iterations from scratch.
- The CLI and audit tools restore both seeds with distinct identities; new trainings require authorization. Two seeds do not prove statistical significance or an optimal convergence plateau.

## First Colab Training — 2026-10-01

- AdamW 3M on Tesla T4: validation NLL `8.322308 → 4.889388` over 409,600 tokens, 100 updates, zero skipped.
- Three full checkpoints restored and verified locally on `D:`. The 1,003,520 token sanity run was interrupted when the Colab session disconnected; subsequent T4 allocation was rejected with HTTP 503.
- [Colab report and recovery status](docs/technical/primer-entrenamiento-colab.md). This short run demonstrates functionality, not convergence or narrative text quality.

## Verified Environment & Hardware

- Windows 11 Pro build 26200, Python 3.12.11 local.
- AMD Radeon RX 6750 GRE 10GB, target architecture `gfx1031`, 10,224 MiB VRAM.
- TheRock stable: PyTorch `2.13.0+rocm10.0.0`, reported HIP runtime `7.15.26333`.
- Colab: Tesla T4, Python 3.13.15, PyTorch `2.11.0+cu128`, CUDA 12.8.
- FP16, forward/backward passes, manual attention fallback, and local SDPA all operate correctly. Torch Inductor / `torch.compile` is disabled due to the lack of Windows Triton support.
- Official Colab MCP verified: connect, list tools, read, create, modify, and execute notebook cells; stdout/stderr capture, remote package install, and repository execution.
- See [readiness report](docs/technical/preparacion.md) and evidence in `output/` and `runs/`.
- Continuity: random checkpoint restoration verified on both HIP and Colab CPU; Colab → MCP → local SSD persistence validated. Google Drive mounting remains locked. See [tests and limits](docs/technical/persistencia-checkpoints.md).

## Local Usage

```powershell
Set-Location D:\Proyectos_B\tiny-transformer-lab
.\scripts\utils\setup.ps1
.\.venv\Scripts\python.exe scripts\utils\check_model.py
.\.venv\Scripts\python.exe scripts\utils\probe_gpu.py
```

The setup script installs locked package versions inside a dedicated virtual environment with caching on `D:`. It does not initiate GPU training runs or modify system drivers.
The baseline model features RoPE, RMSNorm, GELU, bias-free projections, and tied embedding/output weights.
Configurations: 3,000,384, 4,983,552, and 9,998,400 parameters. Only the 3M configuration has been benchmarked and trained.
The 10M configuration requires an 8,192-token vocabulary tokenizer; the current binary shard is formatted exclusively for the 4,096-token vocabulary.

Reprepare data from the small public source:

```powershell
.\.venv\Scripts\python.exe scripts\utils\prepare_data.py
```

Archives 2,000 training stories and 200 validation stories; byte-level BPE tokenizer (vocab size 4,096), `<eos>` tokens between documents, little-endian uint16 binary shards, and memory-mapped reads (`mmap`).
Hashes of the archived subset establish experimental identity. To reproduce exact batches, use the archived dataset bundle.
Modifying vocabulary size requires creating a new output directory/artifact and updating both local and remote environments before comparing runs.
Future example: `prepare_data.py --vocab-size 8192 --output-dir data/processed-8k`.

Optional benchmark without training (only when a new timing/memory measurement is explicitly requested):

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --run-id my-unique-run --steps 3
```

Contains no `optimizer.step`. Records GPU device, backend, dataset hashes, loss, step timing, tokens/sec, and VRAM allocation; verifies weights remain unaltered.

## Colab MCP (Project-Scoped)

Configuration is located in `.codex/config.toml`; global user settings were not modified.
Open this directory as a trusted project in Codex/T3 and launch a new session to load it.
Use `open_colab_browser_connection`; the server launches Helium. Accept **Connect** inside the notebook within 60 seconds.
The client environment must support `notifications/tools/list_changed`.

A lightweight local client tested against the official STDIO server is also available:

```powershell
.\scripts\agents\start-colab.ps1 -SessionId my-unique-session
```

From a secondary terminal:

```powershell
.\.venv\Scripts\python.exe scripts\agents\colab_call.py list --session-id my-unique-session
.\.venv\Scripts\python.exe scripts\agents\colab_call.py get_cells --session-id my-unique-session
```

For long argument payloads, use `--arguments-file` pointing to a JSON file inside `.cache/`.
The client communicates via local request/response files; it does not replace the official MCP or use extracted user credentials.
Only one session/notebook may connect to the proxy at a time. Do not run this client and the agent MCP concurrently against the same notebook.

## Colab CLI via WSL

Official `google-colab-cli` 0.7.4 installed in a dedicated Ubuntu/WSL environment, isolated from the Windows AMD `.venv` and MCP servers. Authenticated account, active T4 runtime utilization, and trained checkpoint transfers have been verified. Do not assume an active session exists.

```powershell
.\scripts\agents\colab-wsl.ps1 version
.\scripts\agents\colab-wsl.ps1 --help
```

See the [WSL Colab CLI guide](docs/guides/colab-cli-wsl.md). No cloud runtimes or training runs are launched during installation.

## Portability & Persistence

`scripts/agents/build_colab_bundle.py` generates `output/colab-bundle.zip` and a self-contained bootstrap notebook cell containing source code and dataset fixtures; it does not publish public repositories.
The bootstrap cell installs dependencies by pin hash, keeps the runtime's CUDA PyTorch build, and prints version info.
**The generated bootstrap cell includes a zero-update forward benchmark when executed.** Generating the bundle does not execute it.
Remote workspace root: `/content/tiny-transformer-lab`. Google Drive is reserved for bundles, run summaries, and critical checkpoints.
Persistent Drive path: `MyDrive/Proyectos_B/tiny-transformer-lab`.
Direct mounting of that path can be unstable; do not rely on it for active training runs until explicitly verified. A [private backup copy of the diagnostic notebook](https://colab.research.google.com/drive/1VevQ4Hm1utXIhCKv8nJrxvgEnJDaNEsl) is maintained on Drive for inspection.
`output/initial-3m.safetensors` is a randomized initialization checkpoint, not a trained model.

Zero-training checkpoint restoration check with automatic unique output folder:

```powershell
.\.venv\Scripts\python.exe scripts\utils\check_checkpoint.py --device cuda
```

Exports safetensors, restores state in a fresh process, and verifies weights, output logits, RNG state, dataset cursor, synthetic AdamW optimizer buffers, and GradScaler state. Rejects corrupted files before loading. Does not call `optimizer.step` and does not evaluate training continuity.

AdamW + GradScaler, validation evaluation, and checkpoint retention are implemented in `src/training.py` and `scripts/train.py`. `scripts/utils/check_training.py` verifies continuity with 10 real optimizer steps; do not run it for read-only audits. `--verify-existing --check-id adamw-check-20261001-v1` inspects existing checkpoint artifacts without updating model weights.
Detailed experiment protocols are documented in the [experiment log](docs/architecture/experimentos.md). Additional optimization techniques and training budgets remain strictly subject to authorization.

Colab runs utilize `src/colab_training.py`, `scripts/train_colab.py`, and `scripts/utils/check_colab_training.py`, separated from local training scripts. The underlying model architecture and dataset pipeline remain shared.
