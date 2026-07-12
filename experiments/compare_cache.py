"""
KV Cache 策略对比实验 — 完整缓存 vs StreamingLLM

测量 StreamingLLM 在不同 sink_len / window_len 配置下，
相对于完整缓存的输出质量差异和缓存节省量。

用法：
  python -m experiments.compare_cache
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def generate_sequence(d_model=32, seq_len=50, seed=42):
    """生成一段模拟的 token 序列"""
    rng = np.random.RandomState(seed)
    return rng.randn(seq_len, d_model)


def compute_attention(q, k_cache, v_cache, d_k):
    """单头 attention 计算"""
    scores = (q @ k_cache.T) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ v_cache


def compare_cache_strategies():
    """对比完整缓存 vs StreamingLLM 在不同窗口大小下的表现"""
    print("=" * 65)
    print("KV Cache 策略对比实验")
    print("=" * 65)

    d_model, d_k = 32, 32
    seq_len = 60
    tokens = generate_sequence(d_model, seq_len)

    from modern_llm.attention_sinks import StreamingKVCache

    configs = [
        ("完整缓存(无淘汰)", None, None),
        ("Streaming(4+12)",  4, 12),
        ("Streaming(4+20)",  4, 20),
        ("Streaming(4+30)",  4, 30),
    ]

    print(f"\n序列长度: {seq_len}, d_model={d_model}")
    print(f"\n{'策略':>25} | {'缓存大小':>8} | {'向量差异':>10} | {'节省':>8}")
    print("-" * 55)

    for name, sink, window in configs:
        if sink is None:
            # 完整缓存：全部保留
            k_cache = tokens
            v_cache = tokens
            cache_size = seq_len
        else:
            cache = StreamingKVCache(sink_len=sink, window_len=window)
            for i, t in enumerate(tokens):
                cache.update(t.reshape(1, -1), t.reshape(1, -1),
                            positions=np.array([i]))
            k_cache, v_cache, _ = cache.get_all()
            cache_size = cache.size

        # 用最后一个 token 算 attention，比较输出差异
        q = tokens[-1:]

        # 完整 attention
        out_full = compute_attention(q, tokens, tokens, d_k)

        # 缓存 attention
        out_cached = compute_attention(q, k_cache, v_cache, d_k)

        diff = np.linalg.norm(out_full - out_cached)
        saving = (1 - cache_size / seq_len) * 100 if sink is not None else 0

        stable = "✅" if np.all(np.isfinite(out_cached)) else "❌"
        print(f"{name:>25} | {cache_size:>5d}/{seq_len:<3} | "
              f"{diff:.4f}{'  ✅' if diff < 1.0 else '  ⚠️'} | "
              f"{saving:>5.0f}%")


def analyze_sink_impact():
    """分析 sink_len 对输出质量的影响"""
    print("\n" + "=" * 65)
    print("Attention Sinks 长度对输出质量的影响")
    print("=" * 65)

    d_model, d_k = 32, 32
    seq_len = 80
    tokens = generate_sequence(d_model, seq_len, seed=123)
    window_len = 16
    q = tokens[-1:]

    out_full = compute_attention(q, tokens, tokens, d_k)

    print(f"\n序列长度={seq_len}, window_len={window_len}")
    print(f"\n{'sink_len':>10} | {'缓存大小':>8} | {'向量差异':>10} | {'节省':>8}")
    print("-" * 45)

    from modern_llm.attention_sinks import StreamingKVCache

    for sink_len in range(0, 9):
        cache = StreamingKVCache(sink_len=sink_len, window_len=window_len)
        for i, t in enumerate(tokens):
            cache.update(t.reshape(1, -1), t.reshape(1, -1),
                        positions=np.array([i]))

        k_cache, v_cache, _ = cache.get_all()
        out_cached = compute_attention(q, k_cache, v_cache, d_k)
        diff = np.linalg.norm(out_full - out_cached)
        saving = (1 - cache.size / seq_len) * 100

        print(f"{sink_len:>10} | {cache.size:>4d}/{seq_len:<3} | "
              f"{diff:.4f} | {saving:>5.0f}%")


def benchmark_cache_speed():
    """在模拟长序列下测量缓存总大小"""
    print("\n" + "=" * 65)
    print("长对话场景：缓存总量对比")
    print("=" * 65)

    from modern_llm.attention_sinks import StreamingKVCache

    d_model = 32
    total_steps = 200  # 模拟 200 步对话
    configs = [
        ("完整缓存(无淘汰)", None, None, 0),
        ("Streaming(4+12)",  4, 12, 0),
        ("Streaming(4+20)",  4, 20, 0),
    ]

    print(f"\n模拟 {total_steps} 步自回归生成\n")
    print(f"{'方案':>20} | {'最终缓存大小':>12} | {'60层推估(GB)':>13}")
    print("-" * 50)

    for name, sink, window, _ in configs:
        if sink is None:
            full_size = total_steps
        else:
            cache = StreamingKVCache(sink_len=sink, window_len=window)
            for i in range(total_steps):
                k = np.random.randn(1, d_model)
                v = np.random.randn(1, d_model)
                cache.update(k, v, positions=np.array([i]))
            full_size = cache.size

        # 按 DeepSeek V2 参数推估 60 层总缓存
        per_layer_bytes = full_size * 5120 * 2  # d_model=5120, FP16
        total_60l_gb = per_layer_bytes * 60 / (1024**3)

        print(f"{name:>20} | {full_size:>8d} tokens | {total_60l_gb:>8.2f}")


if __name__ == "__main__":
    compare_cache_strategies()
    analyze_sink_impact()
    benchmark_cache_speed()
