import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from ..kernels import gaussian_kernel


class KernelRidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, kernel=gaussian_kernel, h=0.1, l2=0.0):
        self.kernel = kernel
        self.h = h
        self.l2 = l2
        self.theta = None
        self.X_ = None
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        K = self._get_kernel(X)
        self.theta = np.linalg.solve(
            K.dot(K) + self.l2 * np.eye(n_samples),
            np.transpose(K).dot(y))
        self.X_ = X
    
    def predict(self, X):
        K = self._get_kernel(X, self.X_)
        return K.dot(self.theta)
    
    @property
    def _get_kernel(self):
        return self.kernel(self.h)
