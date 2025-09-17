import numpy as np


class FastRidge:
    def __init__(self, alpha: float, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: (n_samples, n_features)
        Y: (n_samples,) or (n_samples, n_targets)
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        assert X.shape[0] == Y.shape[0]

        try:
            if X.shape[0] < X.shape[1]:
                self.coef_ = X.T @ np.linalg.solve(
                    X @ X.T + self.alpha * np.eye(X.shape[0]), Y
                )
            else:
                self.coef_ = np.linalg.solve(
                    X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ Y
                )
        except np.linalg.LinAlgError:
            assert self.alpha == 0, "singular matrix"
            self.coef_ = np.linalg.pinv(X) @ Y
        return self

    def predict(self, X: np.ndarray):
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return X @ self.coef_
