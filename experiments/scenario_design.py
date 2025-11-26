import numpy as np


def map_with_correlation(level=0.0):
    """
    根据输入的相关性参数 level ∈ [0,1]
    构造不同相位相关性的 MAP
    """
    # 例子：phase1 强 burst，phase2 弱 burst
    D0 = np.array([[-5, 5],
                   [1, -6]], dtype=float)
    D1 = np.array([[0, 0],
                   [5 * level, 0]], dtype=float)
    return D0, D1


def scale_load(D0, D1, factor):
    """
    将 MAP 到达速率按 factor 缩放，用于控制系统利用率
    """
    return D0 * factor, D1 * factor
