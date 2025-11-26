import numpy as np

def mmpp2(l0=2.0, l1=8.0, alpha=0.5, beta=1.0):
    """
    MMPP-2 arrival model:
      phase0 arrival rate = l0
      phase1 arrival rate = l1
      phase0 --alpha→ phase1
      phase1 --beta → phase0
    """
    Q = np.array([
        [-alpha, alpha],
        [ beta, -beta]
    ])
    D1 = np.diag([l0, l1])
    D0 = Q - D1
    return D0, D1


def super_burst_map(k=10):
    """
    极大 burst MAP：
    phase1 非常容易连续触发到达（heavy burst）
    """
    D0 = np.array([
        [-5, 5],
        [ 1, -1-k]
    ])
    D1 = np.array([
        [0,0],
        [k,0]
    ])
    return D0, D1


def hawkes_like_map(mu=2.0, alpha=3.0, beta=5.0):
    """
    类 Hawkes 自激发流量（用二相 MAP 模拟）
      phase0: rate = mu
      phase1: rate = mu + alpha
      beta 控制 burst 衰减速度
    """
    l0 = mu
    l1 = mu + alpha

    D0 = np.array([
        [-beta, beta],
        [ beta, -beta-l1]
    ])
    D1 = np.array([
        [l0,0],
        [l1,0]
    ])
    return D0, D1

# ==============================================================
# 下面提供 "带相关性 level" 的包装函数，统一与基础 map_with_correlation 接口。
# level ∈ [0,1] 用来调节 burst 程度 / 自相关强度。
# ==============================================================

def mmpp2_with_level(level, l0=2.0, l1=8.0,
                     alpha_min=0.1, alpha_max=2.0, beta=1.0):
    """根据 level 调节相位停留时间（alpha 越小越稳定在 phase0，越大越频繁切换）。
    这里用 alpha = alpha_min + (alpha_max - alpha_min) * level 线性插值。
    beta 固定，保持从高 burst 相位向低相位的回复速度恒定。
    """
    alpha = alpha_min + (alpha_max - alpha_min) * float(level)
    return mmpp2(l0=l0, l1=l1, alpha=alpha, beta=beta)


def hawkes_like_with_level(level, mu=2.0, alpha_max=5.0, beta=5.0):
    """通过调节 alpha (额外自激发强度) 来体现相关性/聚集性。level=0 没有额外激发，level=1 达到 alpha_max。"""
    alpha = alpha_max * float(level)
    return hawkes_like_map(mu=mu, alpha=alpha, beta=beta)


def super_burst_with_level(level, k_min=1, k_max=15):
    """调节 k (连续 burst 触发率)，k 越大在 phase1 连续到达概率越高。"""
    k = int(round(k_min + (k_max - k_min) * float(level)))
    k = max(k_min, min(k_max, k))
    return super_burst_map(k=k)
