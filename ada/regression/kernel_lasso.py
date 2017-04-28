import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from ..kernels import gaussian_kernel


class KernelLasso(BaseEstimator, RegressorMixin):
    def __init__(self, kernel=gaussian_kernel, h=0.1, l1=0.1,
                 max_iter=1000000, eps=1e-5, verbose=False):
        self.kernel = kernel
        self.h = h
        self.l1 = l1
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        
        self.theta = None
        self.X_fit = None
    
    def fit(self, X, y):
        K = self._get_kernel(X)
        theta, z, u = [np.zeros((K.shape[0], 1)) for _ in range(3)]
        zeros = np.zeros(theta.shape)
        ones = np.ones(theta.shape)
        
        # pre-computation
        KTK = K.T.dot(K)
        KTy = K.T.dot(y.reshape(-1, 1))
        Q = np.linalg.inv(KTK + np.eye(KTK.shape[0]))
        
        # alternating direction method of multipliers
        for i in range(self.max_iter):
            theta_old = theta
            theta = Q.dot(KTy - u + z)
            z = np.maximum(zeros, theta + u - self.l1 * ones) + \
                np.minimum(zeros, theta + u + self.l1 * ones)
            u += (theta - z)
            
            error = np.linalg.norm(theta - theta_old)
            if self.verbose:
                print(i, error)
            if error < self.eps:
                break
        else:
            raise RuntimeError("ADMM didn't converge.")
        
        self.theta = theta
        self.X_fit = X
        return self
    
    def predict(self, X):
        K = self._get_kernel(X, self.X_fit)
        return K.dot(self.theta)
    
    @property
    def _get_kernel(self):
        return self.kernel(self.h)
