# 对比实验

对 Attention 各变体的横向对比及超参数训练实验记录。

## 实验列表

| 实验 | 文件 | 对比内容 | 产出 |
|------|------|---------|------|
| Attention 变体对比 | `compare_attention.py` | MHA vs GQA(8KV) vs GQA(4KV) vs MQA vs MLA | KV Cache 大小、参数量、60 层推估 |
| KV Cache 策略对比 | `compare_cache.py` | 完整缓存 vs StreamingLLM（不同窗口大小） | 缓存节省量、输出质量差异 |
| 解码策略对比 | `compare_decoding.py` | 标准自回归 vs Speculative Decoding | 加速比、接受率 |
| 超参数对比 | `compare_training.py` | 不同 lr/head/dim 的训练曲线 | Train/Val Loss 对比图 |

## 实验记录

每次训练自动存档至 `runs/` 目录，可通过对比工具交互式查看：

```bash
python experiments/runs/compare.py
```