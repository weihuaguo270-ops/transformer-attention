# 原始 Transformer 架构（2017）

> `np_impl/` — 纯 NumPy 实现，逐行手写，理解每一步计算。

覆盖从单头 Self-Attention 到完整 Encoder-Decoder 的全过程。是理解 Attention 机制的起点。

## 文件说明

| 文件 | 内容 | 面试对应题 |
|------|------|-----------|
| `attention.py` | 单头 Self-Attention + 因果掩码 | "Attention 的计算公式？" |
| `multi_head_attention.py` | 多头自注意力（支持 RoPE 切换） | "为什么用多头？头怎么拆分合并？" |
| `kv_cache.py` | KV Cache 推理加速（有/无缓存对比） | "为什么 LLM 首字生成慢？" |
| `positional_encoding.py` | Sinusoidal 位置编码 | "位置编码为什么用 sin/cos？" |
| `rotary.py` | RoPE 旋转位置编码 | "RoPE 和 Sinusoidal 的区别？" |
| `transformer_block.py` | 完整 Decoder Block（Post-Norm + ReLU） | "原始 Transformer Block 的结构？" |
| `cross_attention.py` | 编码器-解码器交叉注意力 | "Cross-Attention 的 Q/K/V 来自哪？" |
| `encoder_block.py` | Encoder Block（双向 Attention） | "Encoder 和 Decoder 的 Attention 有何不同？" |
| `encoder_decoder.py` | Encoder-Decoder 完整串联 | "翻译任务中的 Encoder-Decoder 如何衔接？" |
| `utils.py` | softmax / split_heads / combine_heads / layer_norm | — |

## 学习路径

按顺序阅读效果最佳：

```
1. utils.py               → 基础函数
2. attention.py           → QKV 计算 + 因果掩码
3. multi_head_attention.py → 多头拆分合并
4. positional_encoding.py → 位置编码
5. rotary.py              → RoPE 旋转位置编码
6. kv_cache.py            → 推理加速
7. transformer_block.py   → 组装为完整 Block
8. cross_attention.py     → 交叉注意力
9. encoder_block.py       → Encoder
10. encoder_decoder.py    → Encoder-Decoder 组合
```

## 运行

```bash
# 独立运行单个模块
python -m np_impl.attention
python -m np_impl.multi_head_attention
python -m np_impl.kv_cache
python -m np_impl.positional_encoding
python -m np_impl.rotary
python -m np_impl.transformer_block
python -m np_impl.cross_attention
python -m np_impl.encoder_block
python -m np_impl.encoder_decoder

# 运行全部测试（36+ 项）
python -m np_impl.test
```

## 模块依赖

```
utils.py
  ├── attention.py
  ├── multi_head_attention.py  ← 被 transformer/encoder 引用
  ├── kv_cache.py
  ├── positional_encoding.py
  └── rotary.py                ← 被 MHA 引用

transformer_block.py  ← 依赖 multi_head_attention.py
cross_attention.py    ← 只依赖 utils.py
encoder_block.py      ← 依赖 multi_head_attention.py
encoder_decoder.py    ← 依赖 encoder_block + cross_attention
```

---

> 🔄 框架工程实践版 → [`pytorch/README.md`](../pytorch/README.md)
