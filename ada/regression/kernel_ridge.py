import numpy as np

from ..kernels import gaussian_kernel
from ..stats import mean_squared_error


class KernelRidgeRegression(object):
    def __init__(self, kernel=gaussian_kernel, h=0.1, l2=0.0):
        self.kernel = kernel
        self.h = h
        self.l2 = l2
        self.theta = None
        self._X_fit = None
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        K = self._get_kernel(X)
        self.theta = np.linalg.solve(
            K.dot(K) + self.l2 * np.eye(n_samples),
            np.transpose(K).dot(y))
        self._X_fit = X
    
    def predict(self, X):
        K = self._get_kernel(X, self._X_fit)
        return K.dot(self.theta)
    
    def evaluate(self, X, y):
        return mean_squared_error(self.predict(X), y)
    
    @property
    def _get_kernel(self):
        return self.kernel(self.h)
