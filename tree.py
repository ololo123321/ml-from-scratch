import numpy as np
from sklearn.base import BaseEstimator


class Tree(BaseEstimator):
    def __init__(self, max_depth=3, min_data_in_leaf=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.min_samples_split = min_samples_split

        self.n_classes = None
        self.tree = {(): None}

    @staticmethod
    def get_entropy(y):
        values, counts = np.unique(y, return_counts=True)
        if len(values) == 1:
            return 0
        p = counts / y.shape[0]
        return -(p * np.log(p)).sum()

    def get_split(self, X, y):
        n, m = X.shape
        E_min, feat_best, val_best, mask_best = np.inf, 0, X[0, 0], np.ones(n, dtype=np.bool)
        for j in range(m):
            for t in np.unique(X[:, j]):
                mask = X[:, j] <= t
                # мало объектов в листе:
                if np.unique(mask, return_counts=True)[1].min() < self.min_data_in_leaf:
                    continue
                mask_mean = mask.mean()
                E = mask_mean * self.get_entropy(y[mask]) + (1 - mask_mean) * self.get_entropy(y[~mask])
                if E < E_min:
                    E_min, feat_best, val_best, mask_best = E, j, t, mask
        return feat_best, val_best, mask_best

    def traverse(self, X, y, path=(), depth=0):
        # мало объектов для деления
        if X.shape[0] < self.min_samples_split:
            return

        # достигнута максимальная глубина
        if depth >= self.max_depth:
            return

        col, t, mask = self.get_split(X, y)

        # все объекты принадлежат одному классу
        if len(np.unique(mask)) == 1:
            return

        del self.tree[path]
        for is_greater in range(2):
            mask_leaf = ((mask + is_greater) % 2).astype(np.bool)
            X_leaf, y_leaf = X[mask_leaf], y[mask_leaf]
            path_leaf = tuple(x for x in path if x[:2] != (col, is_greater)) + ((col, is_greater, t),)
            # path_leaf = path + ((col, is_greater, t),)
            self.tree[path_leaf] = np.unique(y_leaf, return_counts=True)
            self.traverse(X_leaf, y_leaf, path_leaf, depth+1)

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.traverse(X, y)

    def predict_proba(self, X):
        n_rows = X.shape[0]
        y_pred = np.zeros((n_rows, self.n_classes))
        for path, (classes, counts) in self.tree.items():
            probs = counts / counts.sum()
            mask = np.ones(n_rows)
            for j, is_greater, t in path:
                mask_j = X[:, j] <= t
                mask *= (mask_j + is_greater) % 2
            row_idx = np.arange(n_rows, dtype=np.int32)[mask.astype(np.bool)]
            y_pred[row_idx[:, None], classes] = probs
        return y_pred
