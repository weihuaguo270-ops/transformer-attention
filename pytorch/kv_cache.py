"""
PyTorch 版 KV Cache — 与 NumPy 版 kv_cache.py 对应

对比有缓存和无缓存的 Attention 输出是否一致。
"""
import torch
from utils import softmax


def attention_with_cache(Q, K, V, K_cache=None, V_cache=None):
    """
    支持 KV Cache 的 Attention — 与 NumPy 版逻辑完全一致

    - 有 cache: K = concat(K_cache, K), V = concat(V_cache, V)
    - 无 cache: 直接用 K, V
    """
    if K_cache is not None:
        K = torch.cat([K_cache, K], dim=0)
        V = torch.cat([V_cache, V], dim=0)

    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k ** 0.5)
    weights = softmax(scores)
    return weights @ V, K, V


# ============================================================
# 演示 — 与 NumPy 版 kv_cache.py 对比
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    d_k = 4

    print("=" * 50)
    print("KV Cache 数值一致性验证")
    print("=" * 50)

    # 第1步：无缓存
    q1 = torch.randn(1, d_k)
    k1 = torch.randn(1, d_k)
    v1 = torch.randn(1, d_k)
    out1, K_cache, V_cache = attention_with_cache(q1, k1, v1)

    # 第2步：用缓存
    q2 = torch.randn(1, d_k)
    k2 = torch.randn(1, d_k)
    v2 = torch.randn(1, d_k)
    out2_cached, K_cache, V_cache = attention_with_cache(q2, k2, v2, K_cache, V_cache)

    # 第2步：不用缓存（看到全部 K,V）
    out2_direct, _, _ = attention_with_cache(q2, torch.cat([k1, k2]), torch.cat([v1, v2]))

    diff = (out2_cached - out2_direct).abs().max().item()
    print(f"\n缓存 vs 非缓存 最大差异: {diff:.2e}")
    print("结论:", "✅ 输出一致" if diff < 1e-6 else "❌ 输出不一致")
