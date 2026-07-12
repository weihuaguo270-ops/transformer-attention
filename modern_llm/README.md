# 现代 LLM 架构（2023-2024）

`modern_llm/` — 纯 NumPy 实现，覆盖当前主流大模型使用的 Attention 变体。

包含 Llama 路线（GQA + RMSNorm + SwiGLU）和 DeepSeek 路线（MLA）两大类方案。
两条路线从原始 Transformer 分叉而来，解决不同层面的优化问题。

独立包，不依赖 `np_impl/` 目录。

## 文件说明

| 文件 | 内容 |
|------|------|
| `gqa.py` | Grouped Query Attention：分组机制、KV 头广播、与 RoPE 集成 |
| `llama_block.py` | Llama Decoder Block：Pre-Norm + RMSNorm + SwiGLU + GQA + RoPE |
| `mla.py` | Multi-head Latent Attention：低维 KV 压缩、吸收矩阵技巧 |
| `speculative_decoding.py` | Speculative Decoding：Draft Model 并行验证 |
| `attention_sinks.py` | StreamingLLM：Attention Sinks 长文本缓存优化 |
| `rotary.py` | RoPE 旋转位置编码（独立模块） |
| `utils.py` | 工具函数 |
| `test.py` | 15+ 项测试 |

## 对比实验

参见 [`experiments/`](../experiments/README.md) 目录。

## 运行测试

```bash
python -m modern_llm.test
```