import numpy as np
from numba import njit


@njit(cache=True)
def minmod(a, b, c):
    if isinstance(a, (np.int32, np.int64, np.float64, np.float32, int, float)):
        a = np.array([a])
        a = a.reshape(-1)

    if isinstance(b, (np.int32, np.int64, np.float64, np.float32, int, float)):
        b = np.array([b])
        b = b.reshape(-1)

    if isinstance(c, (np.int32, np.int64, np.float64, np.float32, int, float)):
        c = np.array([c])
        c = c.reshape(-1)

    # check the consistency of input length
    if not (len(a) == len(b) == len(c)):
        raise ValueError("Input length must be the same!")

    res = np.zeros_like(a)

    for i in range(len(a)):
        if a[i] * b[i] > 0 and a[i] * c[i] > 0:
            res[i] = np.min(np.abs(np.array([a[i], b[i], c[i]]))) * np.sign(a[i])

    return res


@njit(cache=True)
def minmod_flag(a, b, c):
    """
    Return the index of minmod_flag
    """

    if isinstance(a, (np.int32, np.int64, np.float64, np.float32, int, float)):
        a = np.array([a])
        a = a.reshape(-1)

    if isinstance(b, (np.int32, np.int64, np.float64, np.float32, int, float)):
        b = np.array([b])
        b = b.reshape(-1)

    if isinstance(c, (np.int32, np.int64, np.float64, np.float32, int, float)):
        c = np.array([c])
        c = c.reshape(-1)

    # check the consistency of input length
    if not (len(a) == len(b) == len(c)):
        raise ValueError("Input length must be the same!")

    res = np.zeros_like(a).astype(np.int32)

    for i in range(len(a)):
        if a[i] * b[i] > 0 and a[i] * c[i] > 0:
            res[i] = np.abs(np.array([a[i], b[i], c[i]])).argmin() + 1

    return res


@njit(cache=True)
def switch_minmod(a, b, c, flag):
    """
    Conditionally output the derivatives of minmod according to the flag
    """

    if isinstance(a, (np.int32, np.int64, np.float64, np.float32, int, float)):
        a = np.array([a])
        a = a.reshape(-1)

    if isinstance(b, (np.int32, np.int64, np.float64, np.float32, int, float)):
        b = np.array([b])
        b = b.reshape(-1)

    if isinstance(c, (np.int32, np.int64, np.float64, np.float32, int, float)):
        c = np.array([c])
        c = c.reshape(-1)

    if isinstance(flag, (np.int32, np.int64, np.float64, np.float32, int, float)):
        flag = np.array([flag])
        flag = flag.reshape(-1)

    # if not (len(a) == len(b) == len(c) == len(flag)):
    #     raise ValueError("Input length must be the same!")

    res = np.zeros_like(flag)

    for i in range(len(flag)):
        if flag[i] == 1:
            if len(a) > 1:
                res[i] = a[i]
            else:
                res[i] = a[0]
        elif flag[i] == 2:
            if len(b) > 1:
                res[i] = b[i]
            else:
                res[i] = b[0]
        elif flag[i] == 3:
            if len(c) > 1:
                res[i] = c[i]
            else:
                res[i] = c[0]

    return res
