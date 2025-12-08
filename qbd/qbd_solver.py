import numpy as np


# ============================================================
# 1. 外部 MAP 分列到队列 r：构造 (D0_r, D1_r)
# ============================================================

def build_map_for_queue(D0, D1, p_r):
    """
    给定外部 MAP(D0, D1) 与队列 r 的路由概率 p_r(i),
    构造该队列的 MAP(D0_r, D1_r).

    公式:
        D1_r(i,j) = D1(i,j) * p_r(i)
        D0_r(i,j) = D0(i,j) + D1(i,j) * (1 - p_r(i))

    参数
    ----
    D0, D1 : ndarray, shape (m, m)
        外部 MAP 的矩阵
    p_r : ndarray, shape (m,)
        相位 i 被路由到队列 r 的条件概率 p_r(i)

    返回
    ----
    D0_r, D1_r : ndarray
        队列 r 的 MAP
    """
    D0 = np.asarray(D0, dtype=float)
    D1 = np.asarray(D1, dtype=float)
    p_r = np.asarray(p_r, dtype=float)

    assert D0.shape == D1.shape
    m = D0.shape[0]
    assert p_r.shape == (m,)

    # 每行按 p_r(i) 缩放得到 D1_r
    D1_r = (D1.T * p_r).T
    # 剩余 (1-p_r(i)) 部分的到达被视作“非到达相位跳转”，并入 D0_r
    D0_r = D0 + (D1.T * (1.0 - p_r)).T

    return D0_r, D1_r


# ============================================================
# 2. 计算 MAP(D0_r, D1_r) 的稳态分布与到达率 λ_r
# ============================================================

def map_stationary_and_rate(D0_r, D1_r):
    """
    对 MAP(D0_r, D1_r) 计算背景链稳态分布 alpha_r 与到达率 lambda_r.

    alpha_r 解:
        alpha_r (D0_r + D1_r) = 0, alpha_r 1 = 1

    lambda_r = alpha_r * D1_r * 1

    返回
    ----
    alpha : ndarray shape (m,)
    lambda_r : float
    """
    Q = D0_r + D1_r
    m = Q.shape[0]

    # 解左特征向量，对应特征值 0
    vals, vecs = np.linalg.eig(Q.T)
    idx = np.argmin(np.abs(vals))
    alpha = np.real(vecs[:, idx])
    alpha = alpha / np.sum(alpha)
    # 清理数值小负数
    alpha = np.maximum(alpha, 0.0)
    alpha = alpha / np.sum(alpha)

    ones = np.ones(m)
    lambda_r = float(alpha @ (D1_r @ ones))
    return alpha, lambda_r


def check_stability(D0_r, D1_r, mu, verbose=True):
    """
    检查 MAP/M/1 队列的稳定性：ρ = λ / μ < 1

    返回
    ----
    rho : float
    """
    _, lambda_r = map_stationary_and_rate(D0_r, D1_r)
    rho = lambda_r / mu
    if verbose:
        print(f"[QBD] lambda_r = {lambda_r:.6f}, mu = {mu:.6f}, rho = {rho:.6f}")
    if rho >= 1.0:
        print("[QBD][WARN] 该队列在理论上不稳定 (rho >= 1)，稳态平均队长发散。")
    return rho


# ============================================================
# 3. 构造截断的 MAP/M/1 QBD 生成矩阵 Q_trunc (levels 0..K)
# ============================================================

def build_qbd_generator_map_m1(D0_r, D1_r, mu, K_max):
    """
    构造 MAP/M/1 的 QBD 生成矩阵 Q_trunc，截断到 level=K_max.

    状态索引:
        level k = 0,1,...,K_max
        phase i = 0,...,m-1
        index(k, i) = k*m + i

    跳转结构:
        - level 0:
            (0,i) -> (0,j) 速率 D0_r[i,j]
            (0,i) -> (1,j) 速率 D1_r[i,j]
        - 1 <= k <= K_max-1:
            (k,i) -> (k,j)   速率 D0_r[i,j]
            (k,i) -> (k+1,j) 速率 D1_r[i,j]
            (k,i) -> (k-1,i) 速率 mu
        - k = K_max:
            同上，但 (k,i)->(k+1,j) 被截断（丢弃往更高 level 的到达）

    注意: K_max 应足够大，使得 level=K_max 的概率质量很小(比如 < 1e-8)。

    返回
    ----
    Q : ndarray, shape (N, N)
        截断后的 CTMC 生成矩阵
    """
    D0_r = np.asarray(D0_r, dtype=float)
    D1_r = np.asarray(D1_r, dtype=float)
    m = D0_r.shape[0]
    assert D0_r.shape == D1_r.shape

    num_levels = K_max + 1
    N = num_levels * m
    Q = np.zeros((N, N), dtype=float)

    def idx(k, i):
        return k * m + i

    # level 0
    k = 0
    for i in range(m):
        row = idx(k, i)
        # phase transitions at same level
        for j in range(m):
            if i == j:
                continue
            rate = D0_r[i, j]
            if rate != 0.0:
                Q[row, idx(k, j)] += rate
        # arrivals to level 1
        if K_max >= 1:
            for j in range(m):
                rate = D1_r[i, j]
                if rate != 0.0:
                    Q[row, idx(1, j)] += rate

    # 1 <= k <= K_max-1
    for k in range(1, K_max):
        for i in range(m):
            row = idx(k, i)
            # same level phase transitions
            for j in range(m):
                if i == j:
                    continue
                rate = D0_r[i, j]
                if rate != 0.0:
                    Q[row, idx(k, j)] += rate
            # arrivals to level k+1
            for j in range(m):
                rate = D1_r[i, j]
                if rate != 0.0:
                    Q[row, idx(k + 1, j)] += rate
            # service to level k-1, phase unchanged
            Q[row, idx(k - 1, i)] += mu

    # level K_max（截断层）
    k = K_max
    for i in range(m):
        row = idx(k, i)
        # same level phase transitions
        for j in range(m):
            if i == j:
                continue
            rate = D0_r[i, j]
            if rate != 0.0:
                Q[row, idx(k, j)] += rate
        # arrivals到更高层被丢弃（或可重定向到自身）
        # 服务：k -> k-1
        Q[row, idx(k - 1, i)] += mu

    # 设置对角线：q_ii = - sum_{j != i} q_ij
    for s in range(N):
        Q[s, s] = -np.sum(Q[s, :]) + Q[s, s]

    return Q


# ============================================================
# 4. 解截断 QBD 的稳态分布 π_trunc，并计算平均队长 L
# ============================================================

def stationary_distribution_from_Q(Q):
    """
    给定一个 CTMC 生成矩阵 Q，求其稳态分布 π，满足:
        π Q = 0, ∑ π_i = 1

    采用的方法：
        - 将 Q^T 的一行替换为全 1
        - 右侧向量 b = [0,0,...,1]^T
        - 解线性方程组 M x = b 得 x，即 π^T

    返回
    ----
    pi : ndarray, shape (N,)
    """
    Q = np.asarray(Q, dtype=float)
    N = Q.shape[0]
    # 构造线性方程组: (Q^T) π^T = 0, 再加一行和为 1
    M = Q.T.copy()
    b = np.zeros(N)
    # 用最后一行强制归一化
    M[-1, :] = 1.0
    b[-1] = 1.0

    pi = np.linalg.solve(M, b)
    # 清理数值误差（小负数）
    pi = np.real(pi)
    pi = np.maximum(pi, 0.0)
    pi = pi / np.sum(pi)
    return pi


def mean_queue_length_from_pi(pi, m, K_max):
    """
    根据稳态分布 π（按 level×phase 展开）计算平均队长 L.

    参数
    ----
    pi : ndarray, shape (N,)
    m  : MAP 相位数
    K_max : 截断最大 level

    返回
    ----
    L : float
    """
    N = (K_max + 1) * m
    assert pi.shape[0] == N

    L = 0.0
    for k in range(K_max + 1):
        # level k 的概率质量
        level_slice = pi[k * m: (k + 1) * m]
        pk = np.sum(level_slice)
        L += k * pk
    return float(L)


def mean_queue_length_map_m1(D0_r, D1_r, mu, K_max=200, verbose=True):
    """
    对单个 MAP/M/1 队列，利用截断 QBD 生成矩阵直接求稳态平均队长 L.

    步骤:
        1) 稳定性检查（ρ < 1）
        2) 构造 Q_trunc (levels 0..K_max)
        3) 求稳态分布 π
        4) 计算平均队长 L

    参数
    ----
    D0_r, D1_r : MAP 矩阵
    mu         : 服务率
    K_max      : 截断最大 level（建议 100~300，根据负载大小）

    返回
    ----
    L : float
    """
    rho = check_stability(D0_r, D1_r, mu, verbose=verbose)
    if rho >= 1.0:
        # 理论上发散，这里仍然可以给一个截断意义下的“伪”L，但要提醒
        print("[QBD] 警告：rho >= 1，平均队长无界，截断 L 仅作形式参考。")

    Q = build_qbd_generator_map_m1(D0_r, D1_r, mu, K_max=K_max)
    pi = stationary_distribution_from_Q(Q)
    m = D0_r.shape[0]
    L = mean_queue_length_from_pi(pi, m=m, K_max=K_max)

    if verbose:
        print(f"[QBD] (trunc K={K_max}) Theoretical mean queue length L ≈ {L:.6f}")
    return L


# ============================================================
# 5. 多个队列的理论平均队长向量
# ============================================================

def theoretical_L(D0, D1, mus, routing_probs, K_max=200, verbose=True):
    """
    外部 MAP(D0, D1)，多队列服务率 mus，路由概率 routing_probs[r, i] = p_r(i).
    对每个队列 r，计算 MAP/M/1 QBD 理论平均队长 L_r.

    参数
    ----
    D0, D1 : ndarray, shape (m, m)
        外部 MAP
    mus : ndarray, shape (R,)
        各队列服务率
    routing_probs : ndarray, shape (R, m)
        routing_probs[r, i] = P(到达由相位 i 路由到队列 r)
    K_max : int
        截断最大 level

    返回
    ----
    Ls : ndarray, shape (R,)
        各队列的理论平均队长
    """
    D0 = np.asarray(D0, dtype=float)
    D1 = np.asarray(D1, dtype=float)
    mus = np.asarray(mus, dtype=float)
    routing_probs = np.asarray(routing_probs, dtype=float)

    R_num = mus.shape[0]
    m = D0.shape[0]
    assert D0.shape == D1.shape
    assert routing_probs.shape == (R_num, m)

    Ls = []
    for r in range(R_num):
        mu_r = mus[r]
        p_r = routing_probs[r]
        if verbose:
            print(f"\n[QBD] ==== Queue r={r}, mu={mu_r:.4f} ====")
        D0_r, D1_r = build_map_for_queue(D0, D1, p_r)
        L_r = mean_queue_length_map_m1(D0_r, D1_r, mu_r, K_max=K_max, verbose=verbose)
        Ls.append(L_r)
    return np.array(Ls)
