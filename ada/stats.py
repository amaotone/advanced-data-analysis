import numpy as np


def mean_squared_error(Y, T):
    Y = Y.ravel()
    T = T.ravel()
    assert Y.shape == T.shape
    return np.mean(np.power(Y - T, 2))
