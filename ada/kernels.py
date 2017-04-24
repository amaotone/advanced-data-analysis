import numpy as np
from scipy.spatial import distance_matrix


def gaussian_kernel(h):
    def f(X, Y=None):
        if Y is None:
            Y = X
        gamma = 1 / (2 * h ** 2)
        return np.exp(-gamma * euclidean_distance(X, Y, squared=True))
    
    return f


def euclidean_distance(X, Y=None, squared=False):
    if Y is None:
        Y = X
    
    if squared:
        return np.power(distance_matrix(X, Y), 2)
    else:
        return distance_matrix(X, Y)
