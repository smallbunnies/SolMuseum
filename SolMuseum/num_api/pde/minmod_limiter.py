import numpy as np
from numba import njit


@njit(cache=True)
def minmod(a, b, c):
    if isinstance(a, (np.int32, np.int64, np.float64, np.float32, int, float)):
        if a * b > 0 and a * c > 0:
            return np.min(np.abs(np.array([a, b, c]))) * np.sign(a)
        return 0.0

    if isinstance(b, (np.int32, np.int64, np.float64, np.float32, int, float)):
        b = np.array([b]).reshape(-1)
    if isinstance(c, (np.int32, np.int64, np.float64, np.float32, int, float)):
        c = np.array([c]).reshape(-1)
    if not (len(a) == len(b) == len(c)):
        raise ValueError("Input length must be the same!")
    res = np.zeros_like(a)
    for i in range(len(a)):
        if a[i] * b[i] > 0 and a[i] * c[i] > 0:
            res[i] = np.min(np.abs(np.array([a[i], b[i], c[i]]))) * np.sign(a[i])
    return res


@njit(cache=True)
def minmod_flag(a, b, c):
    if isinstance(a, (np.int32, np.int64, np.float64, np.float32, int, float)):
        if a * b > 0 and a * c > 0:
            return np.int32(np.abs(np.array([a, b, c])).argmin() + 1)
        return np.int32(0)

    if isinstance(b, (np.int32, np.int64, np.float64, np.float32, int, float)):
        b = np.array([b]).reshape(-1)
    if isinstance(c, (np.int32, np.int64, np.float64, np.float32, int, float)):
        c = np.array([c]).reshape(-1)
    if not (len(a) == len(b) == len(c)):
        raise ValueError("Input length must be the same!")
    res = np.zeros_like(a).astype(np.int32)
    for i in range(len(a)):
        if a[i] * b[i] > 0 and a[i] * c[i] > 0:
            res[i] = np.abs(np.array([a[i], b[i], c[i]])).argmin() + 1
    return res


@njit(cache=True)
def switch_minmod(a, b, c, flag):
    if isinstance(flag, (np.int32, np.int64, np.float64, np.float32, int, float)):
        if flag == 1:
            return float(a)
        elif flag == 2:
            return float(b)
        elif flag == 3:
            return float(c)
        return 0.0

    if isinstance(a, (np.int32, np.int64, np.float64, np.float32, int, float)):
        a = np.array([a]).reshape(-1)
    if isinstance(b, (np.int32, np.int64, np.float64, np.float32, int, float)):
        b = np.array([b]).reshape(-1)
    if isinstance(c, (np.int32, np.int64, np.float64, np.float32, int, float)):
        c = np.array([c]).reshape(-1)
    res = np.zeros_like(flag, dtype=np.float64)
    for i in range(len(flag)):
        if flag[i] == 1:
            res[i] = a[i] if len(a) > 1 else a[0]
        elif flag[i] == 2:
            res[i] = b[i] if len(b) > 1 else b[0]
        elif flag[i] == 3:
            res[i] = c[i] if len(c) > 1 else c[0]
    return res
