import numpy as np
from numpy.random import RandomState
from scipy.sparse import spmatrix
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd


def svd(
    X: spmatrix, n_components: int, n_iter: int = 5, random_state: int | RandomState | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train Singular Vector Decomposition with given matrix `X`

    Args:
        X (scipy.sparse.csr_matrix) : Input matrix
        n_components (int) : Size of embedding dimension
        n_iter (int) : Maximum number of iteration. Default is 5
        random_state (random state) : Default is None

    Returns:
        U (numpy.ndarray) : Representation matrix of rows.
            shape = (n_rows, n_components)
        Sigma (numpy.ndarray) : Eigenvalue of dimension.
            shape = (n_components, n_components)
            Diagonal value are in decreasing order
        VT (numpy.ndarray) : Representation matrix of columns.
            shape = (n_components, n_cols)

    Examples::
        >>> type(X)  # scipy.sparse.csr_matrix
        >>> U, Sigma, VT = svd(X, n_components=100)
    """

    if (random_state is None) or isinstance(random_state, int):
        random_state = check_random_state(random_state)

    n_features = X.shape[1]

    if n_components >= n_features:
        raise ValueError(f"n_components must be < n_features; got {n_components} >= {n_features}")

    U, Sigma, VT = randomized_svd(X, n_components, n_iter=n_iter, random_state=random_state)  # type: ignore[arg-type]

    return U, Sigma, VT
