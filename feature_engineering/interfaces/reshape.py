import numpy as np


class Reshape:

    def reshape_x(self, X: np.ndarray):
        """
        Reshape data if 3-dimensional.
        @param X: data
        @return: reshaped data
        """
        if X.ndim == 3:
            return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        return X

    def reshape_y(self, y: np.ndarray=None):
        """
        Reshape data if 3-dimensional.
        @param y: data
        @return: reshaped data
        """
        if y is None:
            return None
        if y.ndim == 3:
            y = y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
        return y
