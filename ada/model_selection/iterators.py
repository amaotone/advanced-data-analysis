import numpy as np


class KFold(object):
    def __init__(self, fold=5, shuffle=False):
        self.shuffle = shuffle
        self.fold = fold
    
    def __call__(self, X, y):
        idx = np.arange(X.shape[0])
        if self.shuffle:
            np.random.shuffle(idx)
        self.indices = np.split(idx, self.fold)
        
        for idx in self.indices:
            X_train = X[~idx]
            y_train = y[~idx]
            X_valid = X[idx]
            y_valid = y[idx]
            yield (X_train, y_train), (X_valid, y_valid)
