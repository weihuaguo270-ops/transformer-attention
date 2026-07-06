"""
PyTorch 版 KV Cache — 模拟自回归生成循环

对比：
  有缓存：每步只算新 token 的 K、V，然后拼到缓存后面
  无缓存：每步重新算所有 token 的 K、V（重复浪费）

核心区别：
  有缓存 → 每步计算量恒定（只算 1 个 token）
  无缓存 → 每步计算量线性增长（算 i 个 token）
"""

import torch
from utils import softmax


def attention_with_cache(Q, K, V, K_cache=None, V_cache=None):
    """支持 KV Cache 的 Attention"""
    if K_cache is not None:
        K = torch.cat([K_cache, K], dim=0)
        V = torch.cat([V_cache, V], dim=0)

    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k ** 0.5)
    weights = softmax(scores)
    return weights @ V, K, V


def generate(num_steps, use_cache=True, d_k=4):
    """模拟自回归生成，追踪每步计算量

    参数:
        num_steps: 生成的 token 数（不包含首个 token）
        use_cache: True=用KV Cache，False=每次都重新算全部
        d_k: 向量维度
    """
    torch.manual_seed(42)

    # 首个 token（Prompt 阶段）
    Q = torch.randn(1, d_k)
    K = torch.randn(1, d_k)
    V = torch.randn(1, d_k)

    K_cache, V_cache = None, None
    total_flops_saved = 0   # 节省了多少次 K·V 计算

    print(f"{'步数':>5} | {'当前Q':>20} | {'缓存K数量':>8} | {'本次算K数量':>8} | {'总计算量(算K次数)':>15}")
    print("-" * 70)

    for step in range(num_steps):
        if use_cache:
            # 有缓存：只算当前 1 个 token 的 K、V
            k_cur = torch.randn(1, d_k)
            v_cur = torch.randn(1, d_k)
            num_k_computed = 1
            _, K_cache, V_cache = attention_with_cache(Q, k_cur, v_cur, K_cache, V_cache)
            Q = torch.randn(1, d_k)  # 下一个 token 的 Q
        else:
            # 无缓存：重新算第 0 到第 step 步的所有 K、V（共 step+1 个）
            all_K = torch.randn(step + 1 + 1, d_k)  # +1 是因为有首个 token
            all_V = torch.randn(step + 1 + 1, d_k)
            num_k_computed = step + 2    # 已经生成的token数
            _, _, _ = attention_with_cache(Q, all_K, all_V)
            Q = torch.randn(1, d_k)

        cache_size = 0 if K_cache is None else K_cache.shape[0]

        if use_cache:
            total_flops_saved += (step + 2) - 1  # 如果不用缓存，这步要算 (step+2) 个，实际只算了 1 个

        print(f"{step+1:>5} | {'[..., d_k]':>20} | {cache_size:>8} | {num_k_computed:>8} | {(step+2 if not use_cache else num_k_computed):>15}")

    print("-" * 70)
    print(f"\n生成 {num_steps} 个 token 的总结：")
    print(f"  用缓存: total K 计算次数 = {num_steps}（每步1次）")
    print(f"  不用缓存: total K 计算次数 = {sum(range(2, num_steps + 2))}（第t步算t+1次）")
    print(f"  节省: {total_flops_saved} 次 K、V 计算")


if __name__ == "__main__":
    num_steps = 5

    print("=" * 60)
    print("KV Cache 自回归生成模拟")
    print("=" * 60)

    print("\n>>> 使用 KV Cache:")
    generate(num_steps, use_cache=True)

    print("\n>>> 不使用 KV Cache:")
    generate(num_steps, use_cache=False)
