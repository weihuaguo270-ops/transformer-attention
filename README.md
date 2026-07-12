# Attention From Scratch

**NumPy/PyTorch implementation of Transformer attention mechanisms** — covering the full evolution from the original 2017 Transformer to modern LLM architectures including GQA, Llama Block, DeepSeek MLA, Speculative Decoding, and Attention Sinks.

## Architecture

The repository tracks two independent evolutionary paths from the original Transformer (2017):

![Architecture Diagram](docs/modules_overview.excalidraw)

| Directory | Coverage |
|-----------|----------|
| [`np_impl/`](np_impl/README.md) | Original Transformer (2017) in NumPy |
| [`modern_llm/`](modern_llm/README.md) | GQA, Llama Block, MLA, Spec Decoding, StreamingLLM in NumPy |
| [`pytorch/`](pytorch/README.md) | PyTorch GQA + Llama Block + GPT training pipeline with experiment tracking |
| [`experiments/`](experiments/README.md) | Side-by-side comparisons of attention variants, KV strategies, decoding methods |

## Project Structure

```
attention-from-scratch/
│
├── np_impl/                    # Original Transformer (NumPy)
│   ├── attention.py            Single-head self-attention + causal mask
│   ├── multi_head_attention.py Multi-Head Attention (MHA)
│   ├── kv_cache.py             KV cache inference optimization
│   ├── positional_encoding.py  Sinusoidal positional encoding
│   ├── rotary.py               Rotary Position Embedding (RoPE)
│   ├── transformer_block.py    Original decoder block (Post-Norm + ReLU)
│   ├── cross_attention.py      Encoder-decoder cross attention
│   ├── encoder_block.py        Encoder block
│   ├── encoder_decoder.py      Full encoder-decoder architecture
│   ├── utils.py                Shared utilities
│   └── test.py                 36+ tests
│
├── modern_llm/                 # Modern LLM architectures (2023-2024)
│   ├── gqa.py                  Grouped Query Attention
│   ├── llama_block.py          Llama decoder block (Pre-Norm + RMSNorm + SwiGLU)
│   ├── mla.py                  Multi-head Latent Attention (DeepSeek V2/V3)
│   ├── speculative_decoding.py Speculative decoding with draft model
│   ├── attention_sinks.py      StreamingLLM / attention sinks
│   ├── rotary.py               RoPE (standalone)
│   ├── utils.py                Shared utilities
│   └── test.py                 15+ tests
│
├── experiments/                # Comparative experiments
│   ├── compare_attention.py    MHA vs GQA vs MLA: cache size & parameters
│   ├── compare_cache.py        Full cache vs StreamingLLM quality/savings
│   ├── compare_decoding.py     Standard vs Speculative Decoding speedup
│   ├── compare_training.py     Hyperparameter training comparison
│   └── runs/                   Auto-logged experiments + interactive comparison tool
│
├── pytorch/                    # PyTorch training pipeline
│   ├── gqa.py                  GQA + RoPE (PyTorch nn.Module)
│   ├── llama_block.py          RMSNorm + SwiGLU + full GPT model
│   ├── train_gpt.py            Training script (interactive / CLI)
│   ├── data.py                 TinyStories data loader + vocab builder
│   └── ...                     PyTorch ports of all attention modules
│
├── test_all.py                 Unified test entry point (51+ total tests)
├── pyproject.toml
└── docs/                       Architecture diagrams
```

## Key Implementations

### MHA — Multi-Head Attention (2017)

The foundation of the Transformer. Q, K, V projected to d_model, split into n_heads, scored with scaled dot-product attention.

**KV Cache** (`np_impl/kv_cache.py`): During autoregressive decoding, cached K/V tensors eliminate redundant recomputation of previously generated tokens, reducing per-step complexity from O(n²·d) to O(n·d).

### GQA — Grouped Query Attention (2023)

Used by Llama 2/3, Mistral, Qwen. Reduces K/V heads while keeping Q heads — a practical trade-off between MHA quality and MQA efficiency.

| Variant | KV Heads | KV Cache (32h, 4096seq, FP16) | Representative Models |
|---------|----------|-------------------------------|----------------------|
| MHA | 32 | 64.0 MB | Original Transformer |
| GQA | 8 | 16.0 MB | Llama 3 70B |
| GQA | 4 | 8.0 MB | Mistral 7B |
| MQA | 1 | 2.0 MB | Falcon |

### Llama Decoder Block

Five architectural innovations over the original Transformer:

| Dimension | Original (2017) | Llama (2023) | Rationale |
|-----------|----------------|--------------|-----------|
| Normalization position | Post-Norm (after sublayer) | Pre-Norm (before sublayer) | Gradient flows through residual path directly |
| Norm type | LayerNorm (μ+σ+γ+β) | RMSNorm (σ only) | 30% faster, comparable quality |
| FFN activation | ReLU | SwiGLU (gated) | Learnable gated activation |
| Attention mechanism | MHA (32 KV heads) | GQA (4-8 KV heads) | 75-87% KV cache reduction |
| Position encoding | Sinusoidal PE (additive) | RoPE (rotary) | Supports length extrapolation |

```
# Original Transformer Block (Post-Norm)
x → MHA → +residual → LayerNorm → FFN(ReLU) → +residual → LayerNorm → output

# Llama Decoder Block (Pre-Norm)
x → RMSNorm → GQA(RoPE) → +residual → RMSNorm → SwiGLU → +residual → output
```

### MLA — Multi-head Latent Attention (2024)

**DeepSeek V2/V3's core innovation.** Compresses K/V into a low-dimensional latent space, reducing KV cache to approximately 2% of MHA.

```
MHA:   K = h · W_K,       cache K-V (d_model dim each)
MLA:   c = h · W_DKV,     cache c (d_c dim, d_c << d_model)
       K = c · W_UK,      V = c · W_UV (decompress from latent)
```

**Absorption matrix trick** — at inference time, the decompression step is elided:
```
Q · (W_UK · c) = (Q · W_UK) · c    # W_UK absorbed into Q projection
```

Real parameters (DeepSeek V2, d_model=5120):
- MHA per-step cache: 2 × 5120 = 10,240 dimensions
- MLA per-step cache: 512 + 64 = 576 dimensions
- **Compression ratio: ~18x**

### Speculative Decoding

Accelerates autoregressive generation: a smaller draft model proposes K candidate tokens, then the target model verifies them in a single forward pass. Achieves 2-3x wall-clock speedup with no quality degradation.

### Attention Sinks / StreamingLLM

Maintains a small window of recent tokens plus a few initial tokens (the "attention sink") to handle sequences much longer than the training length without full KV recomputation.

## Training & Experiments

The PyTorch pipeline provides end-to-end GPT training on TinyStories:

```bash
# Train a baseline model
python -m pytorch.train_gpt --epochs 3 --d_model 64 --num_heads 4

# Compare GQA vs MHA
python -m pytorch.train_gpt --epochs 5 --num_kv_heads 2

# Launch experiment comparison dashboard
python experiments/runs/compare.py
```

Experiment runs are automatically logged with config snapshots, loss curves, and model checkpoints.

## Test Suite

```bash
# Run all 51+ tests
python test_all.py

# Or run individually:
python -m np_impl.test         # Original Transformer: 36+ tests
python -m modern_llm.test      # Modern LLM: 15+ tests
```

## Requirements

- Python 3.10+
- NumPy (all modules)
- PyTorch 2.0+ (required for `pytorch/` and training pipeline only)

## Related Projects

- [llm-eval-engine](https://github.com/weihuaguo270-ops/llm-eval-engine) — Production-grade LLM evaluation framework with Process Reward scoring
- [handwritten-react-agent](https://github.com/weihuaguo270-ops/handwritten-react-agent) — Production-grade ReAct Agent framework

## License

MIT
