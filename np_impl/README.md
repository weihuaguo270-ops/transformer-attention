# 原始 Transformer 架构（2017）

`np_impl/` — 纯 NumPy 实现。覆盖从单头 Self-Attention 到完整 Encoder-Decoder 的全过程。

## 文件说明

| 文件 | 内容 |
|------|------|
| `attention.py` | 单头 Self-Attention：QKV 投影、缩放点积、因果掩码 |
| `multi_head_attention.py` | 多头注意力（MHA）：拆分/合并、可切换 RoPE |
| `kv_cache.py` | KV Cache：有/无缓存的计算量对比 |
| `positional_encoding.py` | Sinusoidal 位置编码：公式推导、维度周期 |
| `rotary.py` | RoPE 旋转位置编码：旋转矩阵、对角线不变性、长度外推 |
| `transformer_block.py` | Decoder Block：Post-Norm + ReLU FFN |
| `cross_attention.py` | 编码器-解码器交叉注意力 |
| `encoder_block.py` | Encoder Block |
| `encoder_decoder.py` | Encoder-Decoder 完整串联 |
| `utils.py` | 公共工具函数：softmax、head 拆分合并、LayerNorm |
| `test.py` | 36+ 项测试 |

## 运行测试

```bash
python -m np_impl.test
```

## 阅读顺序

```
想学什么 → 看哪个文件
attention 计算细节 → attention.py
多头机制 → multi_head_attention.py
位置编码 → positional_encoding.py → rotary.py
完整架构 → transformer_block.py → encoder_decoder.py
```