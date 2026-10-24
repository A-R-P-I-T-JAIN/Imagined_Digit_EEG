
import numpy as np
from scipy.linalg import eigh


class BinaryCSP:
    """Common Spatial Pattern for two EEG classes."""

    def __init__(self, n_components=4, reg=1e-6):
        self.n_components = n_components
        self.reg = reg
        self.filters_ = None

    @staticmethod
    def _covariance(trial):
        # trial shape: channels x time samples
        trial = trial - trial.mean(axis=1, keepdims=True)
        cov = trial @ trial.T
        return cov / (np.trace(cov) + 1e-12)

    def fit(self, X, y, class_a, class_b):
        # Use only the requested pair of classes.
        mask = np.isin(y, [class_a, class_b])
        X_pair = X[mask]
        y_pair = y[mask]

        if not np.any(y_pair == class_a) or not np.any(y_pair == class_b):
            raise ValueError("Both classes must have training trials.")

        cov_a = np.mean(
            [self._covariance(x) for x in X_pair[y_pair == class_a]],
            axis=0,
        )
        cov_b = np.mean(
            [self._covariance(x) for x in X_pair[y_pair == class_b]],
            axis=0,
        )

        # Regularize the composite covariance for numerical stability.
        composite = cov_a + cov_b
        composite += self.reg * np.eye(composite.shape[0])

        # Solve the generalized eigenvalue problem:
        # cov_a * w = lambda * composite * w
        eigenvalues, eigenvectors = eigh(cov_a, composite)

        # Extreme eigenvalues distinguish the two classes.
        order = np.argsort(eigenvalues)
        half = max(1, self.n_components // 2)

        selected = np.concatenate([
            order[:half],
            order[-half:],
        ])
        self.filters_ = eigenvectors[:, selected].T

        return self

    def transform(self, X):
        if self.filters_ is None:
            raise RuntimeError("Call fit() before transform().")

        features = []

        for trial in X:
            centered = trial - trial.mean(axis=1, keepdims=True)
            projected = self.filters_ @ centered

            variance = np.var(projected, axis=1)
            log_variance = np.log(
                variance / (variance.sum() + 1e-12) + 1e-12
            )
            features.append(log_variance)

        return np.asarray(features)


if __name__ == "__main__":
    # Small synthetic test to verify the implementation.
    rng = np.random.default_rng(42)

    X_demo = rng.normal(size=(20, 64, 400))
    y_demo = np.array([0] * 10 + [1] * 10)

    csp = BinaryCSP(n_components=4)
    csp.fit(X_demo, y_demo, 0, 1)

    features = csp.transform(X_demo)

    print("Filters:", csp.filters_.shape)
    print("Features:", features.shape)
