import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class Swapaxes(BaseEstimator, TransformerMixin):
    def __init__(self, axis1, axis2):
        self._axis1 = axis1
        self._axis2 = axis2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.swapaxes(X, self._axis1, self._axis2)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
