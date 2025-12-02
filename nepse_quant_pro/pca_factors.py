import numpy as np
import pandas as pd


def run_pca(returns_df: pd.DataFrame):
    """Run a covariance-based PCA on the provided returns matrix."""
    clean = returns_df.dropna()
    if clean.empty or clean.shape[1] < 2:
        raise ValueError("Need at least two assets with valid returns for PCA.")
    X = clean.values
    X = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.real(eigvals[idx])
    eigvecs = np.real(eigvecs[:, idx])
    explained_var = eigvals / eigvals.sum()
    return eigvals, eigvecs, explained_var


__all__ = ["run_pca"]
