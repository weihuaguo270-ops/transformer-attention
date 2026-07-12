# 对比实验

> 对 `modern_llm/` 中各模块的横向对比，以及超参数训练实验记录。

## 实验列表

| 实验 | 文件/目录 | 对比内容 | 产出 |
|------|----------|---------|------|
| Attention 变体对比 | `compare_attention.py` | MHA vs GQA(8KV) vs GQA(4KV) vs MQA vs MLA | KV Cache 大小、参数量、60层推估 |
| KV Cache 策略对比 | `compare_cache.py` | 完整缓存 vs StreamingLLM（不同窗口） | 缓存节省量、输出质量差异 |
| 解码策略对比 | `compare_decoding.py` | 标准自回归 vs Spec Decoding | 加速比、接受率、Gamma 影响 |
| 超参数对比 | `compare_training.py` | 不同 d_model / lr 组合的训练效果 | Loss 曲线、PPL、生成文本 |
| 实验记录 | `runs/compare.py` | 查看、筛选、对比所有历史实验 | 一键交互式对比 |

## 运行

```bash
cd attention-from-scratch

# Attention 变体对比
python -m experiments.compare_attention

# KV Cache 策略对比
python -m experiments.compare_cache

# 解码策略对比
python -m experiments.compare_decoding

# 超参数对比
python -m experiments.compare_training

# 查看所有历史实验记录（交互式）
python -m experiments.runs.compare
```

## 关键结论

### Attention 变体（60 层推估）

| 方案 | 总 KV Cache | 相对 MHA | K/V 参数量 |
|------|------------|---------|-----------|
| MHA (32 KV) | 3.75 GB | 1x | 41.9 M |
| GQA (8 KV) | 0.94 GB | 25% | 10.5 M |
| GQA (4 KV) | 0.06 GB | 1.6% | 5.2 M |
| MQA (1 KV) | 0.12 GB | 3.1% | 1.3 M |
| MLA | 0.26 GB | 7.0% | 8.2 M |

### StreamingLLM（seq_len=60）

| 策略 | 缓存 | 节省 | 输出差异 |
|------|------|------|---------|
| 完整缓存 | 60 tokens | — | 基准 |
| Streaming(4+30) | 34 tokens | 43% | 0.71 ✅ |
| Streaming(4+20) | 24 tokens | 60% | 1.08 ⚠️ |
| Streaming(4+12) | 16 tokens | 73% | 1.37 ⚠️ |

### Speculative Decoding

| Gamma | 加速比 | 接受率 |
|-------|--------|-------|
| 1 | 2.0x | 100% |
| 2 | 3.0x | 100% |
| 4 | 5.0x | 100% |
| 8 | 6.0x | 94% |
