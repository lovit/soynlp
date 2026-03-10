"""Deprecated: soynlp.word.pmi → soynlp.utils.pmi로 이동되었습니다."""

import warnings

import numpy as np
from scipy.sparse import csr_matrix

from soynlp.utils.pmi import _as_diag, _logarithm_and_ppmi
from soynlp.utils.pmi import pmi as _pmi_impl

__all__ = ["pmi", "_as_diag", "_logarithm_and_ppmi"]


def pmi(
    X: csr_matrix,
    py: np.ndarray | None = None,
    min_pmi: float = 0,
    alpha: float = 0.0,
    beta: float = 1,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """.. deprecated:: soynlp.utils.pmi를 사용하세요."""
    warnings.warn(
        "soynlp.word.pmi is deprecated. Use soynlp.utils.pmi instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _pmi_impl(X, py=py, min_pmi=min_pmi, alpha=alpha, beta=beta)
