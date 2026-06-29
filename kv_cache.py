"""
KV Cache — 自回归生成中的推理加速
理解为什么 LLM 生成时"第一个字慢，后面越来越快"
"""
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# ============================================================
# 1. 设定场景
# ============================================================
# 模拟生成过程：逐个生成 4 个词
# 假设每个词用 4 维向量，Q/K/V 投影到 3 维
np.random.seed(42)

# 模拟的输入 embedding
embeddings = {
    0: np.array([1.0, 0.5, 0.0, 0.0]),    # 词0
    1: np.array([0.5, 1.0, 0.5, 0.0]),    # 词1
    2: np.array([0.0, 0.5, 1.0, 0.5]),    # 词2
    3: np.array([0.0, 0.0, 0.5, 1.0]),    # 词3
}

d_model = 4
d_k = 3

Wq = np.random.randn(d_model, d_k)
Wk = np.random.randn(d_model, d_k)
Wv = np.random.randn(d_model, d_k)

print("=" * 60)
print("KV Cache 演示：逐步生成 4 个词")
print("=" * 60)

# ============================================================
# 2. 无 KV Cache 版本（每次从头算全部）
# ============================================================
print("\n--- 无 KV Cache：每次重新算所有词的 Attention ---")

def generate_no_cache(num_tokens):
    """朴素方式：每步生成时，对已有的全部词重新算 Attention"""
    for step in range(num_tokens):
        # 当前已有的全部输入
        tokens = [embeddings[i] for i in range(step + 1)]
        X = np.array(tokens)  # (step+1, 4)

        # 重新算所有词的 Q/K/V
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv

        # Attention（带因果掩码）
        seq_len = step + 1
        scores = Q @ K.T / np.sqrt(d_k)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        masked_scores = scores + mask
        attn_weights = softmax(masked_scores)
        output = attn_weights @ V

        # 取最后一个词的输出作为"预测结果"
        predicted = output[-1]
        print(f"  第{step+1}步: 已有{step+1}个词, 重新算了{step+1}个Q, {step+1}个K, {step+1}个V")
        print(f"            Q shape: ({seq_len}, {d_k}), K shape: ({seq_len}, {d_k})")

generate_no_cache(4)

# ============================================================
# 3. 有 KV Cache 版本（只算新词的 Q，复用旧的 K/V）
# ============================================================
print("\n--- 有 KV Cache：只算新词的 Q，旧的 K/V 存起来复用 ---")

def generate_with_cache(num_tokens):
    """优化方式：每步只算新词的 Q，缓存所有历史 K/V"""
    cache_k = []
    cache_v = []

    for step in range(num_tokens):
        token = embeddings[step]
        x = token.reshape(1, -1)  # (1, 4)

        # 只算当前这个词的 Q/K/V
        q_new = x @ Wq  # (1, 3)
        k_new = x @ Wk  # (1, 3)
        v_new = x @ Wv  # (1, 3)

        # 把新的 K/V 追加到缓存
        cache_k.append(k_new)
        cache_v.append(v_new)

        # 拿出缓存的全部 K/V（不用重新算）
        K_all = np.concatenate(cache_k, axis=0)  # (step+1, 3)
        V_all = np.concatenate(cache_v, axis=0)  # (step+1, 3)

        # Attention：只用新词的 Q 去跟缓存的 K 做匹配
        # Q 只有一个 (1, 3)，K 有 (step+1, 3)
        scores = q_new @ K_all.T / np.sqrt(d_k)  # (1, step+1)
        # 因果掩码：最后一个位置允许看所有（因为都是过去+当前）
        attn_weights = softmax(scores)
        output = attn_weights @ V_all  # (1, 3)

        predicted = output[0]
        # 关键区别
        q_count = 1  # 只算了 1 个 Q
        k_count = step + 1  # 但匹配了全部 K（从缓存拿的）
        computed_new = 1
        reused_from_cache = step
        print(f"  第{step+1}步: 新算 {computed_new} 个Q, {computed_new} 个K, {computed_new} 个V")
        print(f"           从缓存复用 {reused_from_cache} 个K, {reused_from_cache} 个V")
        print(f"           当前Q shape: (1, {d_k}), 缓存K shape: ({k_count}, {d_k})")

generate_with_cache(4)

# ============================================================
# 4. 计算量对比
# ============================================================
print("\n" + "=" * 60)
print("计算量对比（生成长度 = N）")
print("=" * 60)

def compute_flops_no_cache(N, d):
    """无缓存：每步重新算所有词，总 QKV 计算量"""
    total_qkv = 0
    total_attn = 0
    for step in range(N):
        seq_len = step + 1
        # 每步算 seq_len 个 Q/K/V（每个是 d × d 矩阵乘）
        total_qkv += 3 * seq_len * d * d
        # 注意力分数: seq_len × d @ d × seq_len
        total_attn += seq_len * seq_len * d
    return total_qkv + total_attn

def compute_flops_with_cache(N, d):
    """有缓存：每步只算 1 个新 Q/K/V"""
    total_qkv = 0
    total_attn = 0
    for step in range(N):
        # 每步只算 1 个词的新 Q/K/V（3 个矩阵乘）
        total_qkv += 3 * d * d
        # 注意力分数: 1 × d @ d × (step+1)
        total_attn += (step + 1) * d
    return total_qkv + total_attn

for N in [4, 10, 100, 1000]:
    no_cache = compute_flops_no_cache(N, d_k)
    with_cache = compute_flops_with_cache(N, d_k)
    speedup = no_cache / with_cache
    print(f"  N={N:4d}: 无缓存={no_cache:>8,d} | 有缓存={with_cache:>8,d} | 加速比={speedup:.1f}x")
