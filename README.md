# Attention From Scratch

> 纯 NumPy 实现 Transformer Attention 机制 — 从零手写，不依赖 PyTorch / TensorFlow。
> 覆盖 Self-Attention、多头注意力、因果掩码、KV Cache、位置编码、完整 Transformer Block。
> 目的是**彻底搞懂 Attention 的计算过程**，而不是当一个框架的搬运工。

## 项目动机

Transformer 架构的核心是 Attention 机制，但主流框架（PyTorch、TensorFlow）把它封装成了 `nn.MultiheadAttention` 这样的一行调用。这种封装虽然方便，但隐藏了关键细节：

- QKV 是怎么通过线性变换得到的？
- 因果掩码是如何遮住未来位置的？
- 多头注意力中的"头"是怎么拆分和合并的？
- KV Cache 为什么能加速推理？加速了多少？

本项目用 **纯 NumPy** 逐行实现这些过程，每一步都可以打印出中间张量的形状，直观理解 Attention 的计算本质。

## 文件说明

| 文件 | 核心内容 | 可独立运行 |
|------|---------|:---------:|
| `utils.py` | softmax、split_heads、combine_heads、layer_norm | ❌ 工具库 |
| `attention.py` | 单头 Self-Attention + 因果掩码（GPT 风格） | ✅ |
| `multi_head_attention.py` | 多头 Self-Attention（类封装，可被 import） | ✅ |
| `kv_cache.py` | KV Cache 推理加速原理 + 速度对比 | ✅ |
| `positional_encoding.py` | 正弦位置编码（Sinusoidal PE） | ✅ |
| `transformer_block.py` | 完整 Decoder Block（Attention + 残差 + LayerNorm + FFN） | ✅ |

## 安装与运行

### 依赖

- Python 3.8+
- NumPy（仅此一个依赖）

```bash
pip install numpy
```

### 按顺序运行

建议按以下顺序阅读代码，每个文件独立可运行：

```bash
# 1. Self-Attention 原理
python attention.py

# 2. 多头注意力
python multi_head_attention.py

# 3. KV Cache 推理加速（含速度对比）
python kv_cache.py

# 4. 位置编码
python positional_encoding.py

# 5. 完整 Transformer Block
python transformer_block.py
```

### 运行示例

```bash
$ python attention.py
=== Part A: 无掩码 Self-Attention ===
输入形状: (3, 4)  ← 3个词，每个4维
Q形状: (3, 3)      ← 3个query，每个3维
注意力权重形状: (3, 3)  ← 词与词之间的注意力分数
输出形状: (3, 3)

=== Part B: 因果掩码 Self-Attention ===
词"坐"的注意力分布: [0.38, 0.62, 0.00]
只看了"猫"和自己，没看"垫子"
=== 验证通过 ===
```

## 学习路线

### 1. Self-Attention 原理 → `attention.py`

实现 Attention 计算公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

每个步骤对应一段代码：
1. **Q @ K^T** → 计算词与词之间的相似度
2. **/ √d_k** → 缩放，防止 softmax 梯度消失
3. **+ 因果掩码**（Part B）→ 上三角矩阵填 -inf，遮住未来位置
4. **Softmax** → 每行归一化为概率分布
5. **@ V** → 加权求和，得到上下文感知表示

### 2. 因果掩码 — GPT 自回归生成

掩码矩阵：

```
[[0, -inf, -inf],     词0 只看自己
 [0,   0,  -inf],     词1 看词0和词1
 [0,   0,    0]]      词2 看所有词
```

-inf 经过 softmax 后会变成 0，保证未来信息不会被"偷看"。

### 3. 多头注意力 (Multi-Head Attention)

- 将 Q、K、V 拆成 `h` 个头（split_heads）
- 每个头独立计算 Attention
- 合并结果（combine_heads）后过输出投影

```python
def forward(self, Q, K, V):
    # 1. 线性投影
    # 2. 拆头: (batch, seq, d_model) → (batch, h, seq, d_k)
    # 3. 每个头独立算 Attention
    # 4. 合并: (batch, h, seq, d_k) → (batch, seq, d_model)
    # 5. 输出投影
```

### 4. KV Cache — 推理加速

自回归生成时，每步只生成一个新 token。
如果不做 KV Cache，每步都要重新算所有历史 token 的 K 和 V —— 重复计算。
KV Cache 把之前算过的 K、V 存起来，每步只算新 token 的 K、V：

```python
# 无 KV Cache: O(n²) 每步重新算
# 有 KV Cache: O(n)  每步只算新的
```

`kv_cache.py` 中有速度对比，可以看到随着序列变长，差距越来越大。

### 5. 位置编码

Transformer 没有循环结构，需要显式注入位置信息。
正弦位置编码为每个位置生成唯一向量：

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

`positional_encoding.py` 运行后会打印位置向量的热力图，能看到不同位置的编码模式。

### 6. 完整 Transformer Block

将以上所有组件组装为一个完整的 Decoder Block：

```
输入 → Positional Encoding → Multi-Head Attention → 残差 + LayerNorm → FFN → 残差 + LayerNorm → 输出
```

`transformer_block.py` 包含了完整的 Block，可堆叠 N 层。

## 架构总览

```
输入 (seq_len, d_model)
        │
  [Positional Encoding]          ← positional_encoding.py
        │
  ┌────────────────────────┐
  │ Transformer Block × N   │     ← transformer_block.py
  │  ┌──────────────────┐   │
  │  │ Multi-Head        │   │     ← multi_head_attention.py
  │  │ Self-Attention    │   │
  │  │  (含因果掩码)      │   │     ← attention.py
  │  └────────┬─────────┘   │
  │           ↓              │
  │  + 残差连接 + LayerNorm  │     ← utils.py (layer_norm)
  │           ↓              │
  │  ┌──────────────────┐   │
  │  │ FFN               │   │     ← transformer_block.py 内
  │  └────────┬─────────┘   │
  │           ↓              │
  │  + 残差连接 + LayerNorm  │
  └────────────────────────┘
        │
  KV Cache（推理时复用）       ← kv_cache.py
        │
  输出 → 预测下一个词
```

## 模块依赖关系

```
utils.py ← 所有文件从这里 import
  ├── attention.py
  ├── multi_head_attention.py
  ├── kv_cache.py
  ├── positional_encoding.py
  └── transformer_block.py ← 还 import 了 multi_head_attention.py
```

每个文件独立可运行，按顺序阅读效果最佳。

## 后续计划

- [ ] 交叉注意力（Cross-Attention）— Encoder-Decoder 架构
- [ ] KV Cache 的进一步优化：PagedAttention、MQA、GQA
- [ ] Flash Attention 原理与数值对比
- [ ] RoPE（旋转位置编码）实现
- [ ] PyTorch 版对比（相同逻辑用 PyTorch 写一遍）

## License

MIT
