# Tiny Transformer Lab

A hands-on research and experimentation laboratory for small language models (SLMs).  
Read the accompanying logs, key concepts, and research notes at **[matiaschicao.cl/lab](https://matiaschicao.cl/lab)**.

The primary objective of this repository is to investigate pre-training and post-training techniques, reproduce and stress-test modern deep learning papers, and explore architectural trade-offs directly on consumer hardware—specifically an AMD Radeon RX 6750 GRE (10GB VRAM, RDNA 2) running ROCm/HIP natively on Windows.

Everything is designed from first principles to deeply understand how transformers learn: from raw tokenization and optimizer dynamics to reinforcement learning from verifiable rewards (RLVR) and capacity scaling, avoiding opaque high-level frameworks or dependencies on NVIDIA-exclusive tooling (such as Triton or FlashAttention).

---

## What Is Inside This Repository

- **`src/` — Core Modeling & Training Engine**
  - **Model Architecture**: Unified PyTorch implementation with Rotary Position Embeddings (RoPE), RMSNorm, modular FFNs (GELU and SwiGLU), bias-free linear projections, and tied embedding/output weights.
  - **Optimization**: Standard AdamW and Muon optimizers with gradient clipping, linear warmup schedules, and mixed-precision support.
  - **Post-Training & RL**: Implementation of Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) with completion masking, KL penalties, and verifiable programmatic rewards.
  - **Checkpointing**: Bit-for-bit reproducible state serialization (weights, optimizer buffers, RNG state, dataset cursors, and GradScaler).

- **`scripts/` — Experiment Runners & Verification Audits**
  - Reproducible CLI scripts for pre-training sweeps, replication campaigns, and post-training interventions.
  - Read-only audit scripts (`scripts/utils/check_*.py`) that verify continuity, evaluate model checkpoints, and sample outputs across fresh processes without updating model weights.

- **`configs/` — Model Configurations**
  - Architecture specs and hyperparameters across scales (e.g., 3M parameter baseline, 10M capacity experiments) with explicit token budgets.

- **`docs/` — Technical Notes & Protocols**
  - Detailed experiment contracts (`docs/architecture/`), evaluation metrics guides (NLL, perplexity, paired t-tests in `docs/technical/`), and hardware setup notes.

- **`PAPERS.md` — Living Literature Review**
  - Curated catalogue of primary papers (Muon, SwiGLU, DeepSeekMath/GRPO, RLVR, Reasoning Boundaries, Fixed-Point Looping, Action Chunking) cross-referenced with experiments executed or planned in this lab.

- **`EXPERIMENTOS_FUTUROS.md` — Research Roadmap & Preregistered Proposals**
  - Structured backlog of experimental hypotheses, compute budgets, stop/continue criteria, and dependencies.

- **`output/` & `runs/` — Empirical Results & Artifacts**
  - Complete logs, loss curves, evaluation reports, generation samples, and execution receipts for every completed run.

---

## Hardware & Runtime Environment

- **GPU**: AMD Radeon RX 6750 GRE 10GB (`gfx1031`, 10,224 MiB VRAM).
- **Host OS**: Windows 11 Pro (build 26200), Python 3.12.
- **PyTorch Stack**: AMD TheRock build (`2.13.0+rocm10.0.0`), HIP runtime `7.15.26333`.
- **Compute Primitives**: Native FP32 and FP16 execution, manual attention fallback, and PyTorch SDPA (Scaled Dot-Product Attention) without requiring Triton or FlashAttention.

---

## Key Experiments & Empirical Findings

### Pre-Training & Architecture

#### FFN Replications: GELU vs. SwiGLU (3 Seeds)
- Evaluated parameter-matched 3M models across seeds 42, 43, and 44 on TinyStories (6,144,000 tokens / 1,500 updates per run):
  - **Seed 42**: SwiGLU held a slight advantage (−0.0072 nats/target final NLL).
  - **Seed 43**: GELU held a slight final advantage (SwiGLU +0.0093 nats/target final; SwiGLU best checkpoint −0.0085 nats/target).
  - **Seed 44**: GELU won on both metrics (GELU final NLL 3.160184 vs. SwiGLU 3.214367).
- **Conclusion**: Across all 3 seeds, the mean paired difference (SwiGLU − GELU) showed no consistent or statistically significant advantage for SwiGLU at this scale and budget, while GELU trained with higher token throughput (37.8k vs. 34.3k tokens/s).
- Detailed report: [3-seed replication analysis](output/ffn-seed44-20261001-3f197002/report.md).

#### Optimizer Comparisons: AdamW vs. Muon vs. QK-Norm
- Compared AdamW against Muon (orthogonalized momentum updates) and Muon + QK-Norm on 3M models across identical token budgets:
  - AdamW converged to lower validation NLL (3.186252) than Muon (3.271451).
  - Fixed QK-Norm with Muon exhibited late-stage validation degradation (3.638630).
- Detailed report: [Optimizer campaign report](output/campaign-3m-20261001/report.md).

#### Data Scale & Learning Curve (F01)
- Verified dense 3M pre-training scaling from 1M up to 24.576M tokens with BPE-4096: validation NLL improved from ~3.71 nats down to 2.073 nats across 3 confirmed seeds with zero skipped updates.

### Post-Training & Reasoning Interventions

#### RLVR Feasibility (GRPO on AMD ROCm)
- Implemented and validated synchronous GRPO with binary correctness rewards directly on consumer ROCm hardware without distributed frameworks:
  - Validated rollout generation, logprob tracking, completion masking, and numerical stability.
  - Programmatic verifier with strict delimiters prevented format collapse and ensured reproducible candidate evaluation.

#### SFT vs. GRPO Efficacy (Evening Campaign)
- Compared compute-matched Supervised Fine-Tuning against GRPO on a 96.2M parameter foundation model across multi-digit arithmetic and reasoning tasks:
  - Under matched active compute and tight language retention guards (+0.05 nats threshold), SFT demonstrated higher greedy accuracy and test generalization than pure GRPO from a low-capacity checkpoint.
  - High sampling temperatures increased oracle pass@k coverage but did not improve majority-voting accuracy.
  - Detailed report: [Post-training campaign report](output/posttraining-evening-20261002-v1/report.md).
