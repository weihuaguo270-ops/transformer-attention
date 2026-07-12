"""
解码策略对比实验 — 标准自回归 vs Speculative Decoding

测量不同 gamma 值和 draft model 质量下的加速比和接受率。

用法：
  python -m experiments.compare_decoding
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from modern_llm.speculative_decoding import SimpleLM, SpeculativeDecoder


def run_single_comparison(name, target, draft, gamma, prefix, max_tokens):
    """运行一次对比，返回 (speedup, accept_rate)"""
    decoder = SpeculativeDecoder(draft, target, gamma=gamma)
    _ = decoder.generate(prefix.copy(), max_new_tokens=max_tokens)

    target_forward_baseline = max_tokens
    speedup = target_forward_baseline / decoder.stats["target_calls"]
    total = decoder.stats["accepted"] + decoder.stats["rejected"]
    accept_rate = decoder.stats["accepted"] / total if total > 0 else 1.0

    return speedup, accept_rate, decoder.stats


def compare_gamma_values():
    """对比不同 gamma 值下的加速效果"""
    print("=" * 65)
    print("Speculative Decoding — Gamma 值影响")
    print("=" * 65)

    np.random.seed(42)
    target = SimpleLM(vocab_size=50, d_model=32, seed=42)
    draft = SimpleLM(vocab_size=50, d_model=16, seed=99)
    prefix = np.array([5, 12, 3])
    max_tokens = 30

    print(f"\nTarget: d_model=32, Draft: d_model=16")
    print(f"生成 {max_tokens} 个 token\n")
    print(f"{'Gamma':>8} | {'Target前向':>10} | {'加速比':>8} | {'接受率':>8} | {'接受/拒绝':>10}")
    print("-" * 50)

    for gamma in [1, 2, 3, 4, 5, 6, 8]:
        np.random.seed(42)
        t = SimpleLM(vocab_size=50, d_model=32, seed=42)
        d = SimpleLM(vocab_size=50, d_model=16, seed=99)
        speedup, accept, stats = run_single_comparison(
            "", t, d, gamma, prefix, max_tokens)
        print(f"{gamma:>8} | {stats['target_calls']:>10} | "
              f"{speedup:>6.2f}x | {accept:>7.0%} | "
              f"{stats['accepted']}/{stats['rejected']}")


def compare_draft_quality():
    """对比不同 draft model 质量下的表现"""
    print("\n" + "=" * 65)
    print("Draft Model 质量对 Spec Decoding 的影响")
    print("=" * 65)

    class StrongDraft(SimpleLM):
        def forward(self, token_ids):
            logits = super().forward(token_ids)
            return logits * 0.8  # 接近 target

    class WeakDraft(SimpleLM):
        def forward(self, token_ids):
            logits = super().forward(token_ids)
            return logits * 0.1  # 很平坦，经常猜错

    np.random.seed(42)
    target = SimpleLM(vocab_size=50, d_model=32, seed=42)
    prefix = np.array([5, 12, 3])
    gamma = 4
    max_tokens = 30

    print(f"\nTarget: d_model=32, gamma={gamma}, 生成 {max_tokens} tokens\n")
    print(f"{'Draft质量':>15} | {'Target前向':>10} | {'加速比':>8} | {'接受率':>8}")
    print("-" * 45)

    for name, draft_cls in [("Strong(d_model=32)", StrongDraft),
                             ("Weak(d_model=16)",  WeakDraft),
                             ("Random(d_model=8)", SimpleLM)]:
        np.random.seed(42)
        d = draft_cls(vocab_size=50, d_model=32 if "32" in name else
                      16 if "16" in name else 8, seed=99)
        speedup, accept, stats = run_single_comparison(
            name, target, d, gamma, prefix, max_tokens)
        print(f"{name:>15} | {stats['target_calls']:>10} | "
              f"{speedup:>6.2f}x | {accept:>7.0%}")


def theoretical_vs_actual():
    """理论加速 vs 实际加速"""
    print("\n" + "=" * 65)
    print("理论加速 vs 实际加速")
    print("=" * 65)

    print(f"""
理论公式:
  加速比 ≈ (1 + γ) / (1/α + γ)
  其中 α = draft model 的接受率, γ = 候选数

{'-'*55}
{'γ':>4} | {'α=50%':>10} | {'α=60%':>10} | {'α=70%':>10} | {'α=80%':>10} | {'α=90%':>10}
{'-'*55}""")

    for gamma in [1, 2, 3, 4, 5, 6, 8]:
        values = []
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
            speedup = (1 + gamma) / (1/alpha + gamma)
            values.append(f"{speedup:>6.2f}x")
        print(f"{gamma:>4} | " + " | ".join(values))

    print(f"""
观察:
  - γ 越大，加速比越大，但边际递减
  - 接受率 α 比 γ 更重要：80% + γ=4 > 60% + γ=8
  - 实际场景中 γ=4 是常见选择（平衡草稿成本和接受率）""")


if __name__ == "__main__":
    compare_gamma_values()
    compare_draft_quality()
    theoretical_vs_actual()
