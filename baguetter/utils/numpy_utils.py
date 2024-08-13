from __future__ import annotations

import numpy as np

from baguetter.utils.numba_utils import get_min_max


def reversed_scale_min_max_normalization(
    scores: np.ndarray,
    min_max: tuple[float, float] | None = None,
) -> np.ndarray:
    """Perform reversed scale min-max normalization on the input scores.

    Args:
        scores: Input array of scores to normalize.
        min_max: Optional tuple of (min_score, max_score). If not provided, calculated from scores.

    Returns:
        Normalized scores array.

    """
    min_score, max_score = min_max or get_min_max(scores)
    denominator = max(max_score - min_score, 1e-9)
    return (max_score - scores) / denominator


def min_max_normalization(
    scores: np.ndarray,
    min_max: tuple[float, float] | None = None,
) -> np.ndarray:
    """Perform min-max normalization on the input scores.

    Args:
        scores: Input array of scores to normalize.
        min_max: Optional tuple of (min_score, max_score). If not provided, calculated from scores.

    Returns:
        Normalized scores array.

    """
    min_score, max_score = min_max or get_min_max(scores)
    denominator = max(max_score - min_score, 1e-9)
    return (scores - min_score) / denominator


def top_k_numpy(
    scores: np.ndarray,
    k: int,
    *,
    sort: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the top k scores and their indices in the input array.

    Args:
        scores: Input array of scores.
        k: Number of top elements to return.
        sort: Whether to sort the results in descending order.

    Returns:
        Tuple of (top_k_scores, top_k_indices).

    """
    k = min(k, len(scores))

    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_scores = scores[top_k_indices]

    if sort:
        sorted_indices = np.argsort(top_k_scores)[::-1]
        top_k_scores = top_k_scores[sorted_indices]
        top_k_indices = top_k_indices[sorted_indices]

    return top_k_scores, top_k_indices
