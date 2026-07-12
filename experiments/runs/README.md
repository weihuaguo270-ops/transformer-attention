# 实验记录系统

每次训练的结果记录在 `runs/` 下。

## 目录命名规则

| 来源 | 命名格式 | 示例 |
|------|---------|------|
| 手动记录 | `legacy_{序号}_{描述}/` | `legacy_001_baseline/` |
| 自动记录 | `{时间戳}_{tag}_{参数}_auto/` | `20260712_140332_lr-test_lr0.001_auto/` |

## 对比工具

```bash
python experiments/runs/compare.py
```

支持按 tag、参数过滤，横向对比不同实验的 loss 曲线和最终指标。