# PyTorch 版 — 框架工程实践

> `pytorch/` — 与 `np_impl/` 结构对应的 PyTorch 实现，额外包含训练流程和对比实验。

`np_impl/` 的每一步计算都手动实现（适合理解原理），`pytorch/` 用 `nn.Linear`、`F.softmax`、自动微分等框架 API 重写（适合工程实践）。

## 与 NumPy 版的对应关系

| 模块 | NumPy 版 | PyTorch 版 | 差异 |
|------|---------|------------|------|
| 工具函数 | `np_impl/utils.py` | `utils.py` | `nn.LayerNorm` 替代手写 |
| 单头 Attention | `np_impl/attention.py` | `attention.py` | `F.softmax` + 自动微分 |
| 多头注意力 | `np_impl/multi_head_attention.py` | `multi_head_attention.py` | `nn.Linear` 替代手写 Wq/Wk/Wv |
| KV Cache | `np_impl/kv_cache.py` | `kv_cache.py` | PyTorch tensor 操作 |
| 位置编码 | `np_impl/positional_encoding.py` | `positional_encoding.py` | 逻辑相同，tensor 实现 |
| Decoder Block | `np_impl/transformer_block.py` | `transformer_block.py` | `nn.Module` 封装 |
| Cross-Attention | `np_impl/cross_attention.py` | `cross_attention.py` | — |
| Encoder Block | `np_impl/encoder_block.py` | `encoder_block.py` | — |
| Encoder-Decoder | `np_impl/encoder_decoder.py` | `encoder_decoder.py` | — |
| 测试 | `np_impl/test.py` | `test_all.py` | 25+ 项测试 |

## PyTorch 版独有内容

### 训练流程（`train_transformer.py`）

完整的 Encoder-Decoder 训练 pipeline：

- Token Embedding + Positional Encoding → Encoder → Decoder → LM Head → vocab 概率
- Teacher Forcing（CrossEntropyLoss + Adam）
- 200 epoch 后 Acc 100%

```bash
python pytorch/train_transformer.py
```

### 位置编码对比实验（`compare_pos_encoding.py`）

在同一训练任务上，分别用 Sinusoidal PE 和 RoPE 训练 Transformer，打印 loss/acc 曲线对比：

```bash
python pytorch/compare_pos_encoding.py
```

输出示例：

```
指标                   Sinusoidal PE       RoPE
--------------------------------------------------------
最终 Loss              0.0014             0.0013
最终 Acc               100.0%             100.0%
到达 90% Acc           epoch 20           epoch 20
```

## 运行

```bash
# 安装依赖
pip install torch

# 测试全部模块
cd pytorch && python test_all.py

# 训练
cd pytorch && python train_transformer.py

# 对比实验
cd pytorch && python compare_pos_encoding.py
```

## 跟 `np_impl/` 的对照价值

两版对比可以直观感受框架封装 vs 手写的差异：

| 维度 | NumPy 版 | PyTorch 版 |
|------|---------|------------|
| 代码量 | 每步运算手动写 | `nn.Linear` / `F.softmax` 一行搞定 |
| 自动微分 | ❌ 手动实现梯度 | ✅ 自动计算 |
| GPU 训练 | ❌ | ✅ `.to('cuda')` |
| 训练能力 | ❌ | ✅ 完整训练 pipeline |
| 理解难度 | 适合学原理 | 适合学工程 |
