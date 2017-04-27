import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def gaussian_kernel(h):
    def f(X, Y=None):
        K = euclidean_distances(X, Y, squared=True)
        K *= -1 / (2 * h ** 2)
        return np.exp(K, K)
    
    return f
