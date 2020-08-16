from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd


def svd(X, n_components, n_iter=5, random_state=None):
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
        raise ValueError("n_components must be < n_features;"
                         " got %d >= %d" % (n_components, n_features))

    U, Sigma, VT = randomized_svd(
        X, n_components,
        n_iter=n_iter,
        random_state=random_state)

    return U, Sigma, VT
