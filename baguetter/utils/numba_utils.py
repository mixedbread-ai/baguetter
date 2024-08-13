from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def get_min_max(x: np.ndarray) -> tuple[float, float]:
    """Find the minimum and maximum values in a numpy array.

    Args:
        x (np.array): Input array

    Returns:
        tuple[float, float]: A tuple containing the minimum and maximum values

    """
    min_val = float("inf")
    max_val = float("-inf")

    for i in range(len(x)):
        if x[i] < min_val:
            min_val = x[i]
        if x[i] > max_val:
            max_val = x[i]

    return min_val, max_val


# UNION ------------------------------------------------------------------------
@njit(cache=True)
def union_sorted(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Compute the union of two sorted numpy arrays.

    Args:
        a1 (np.array): First sorted array
        a2 (np.array): Second sorted array

    Returns:
        np.array: Sorted union of a1 and a2

    """
    result = np.empty(len(a1) + len(a2), dtype=np.int32)
    i = 0
    j = 0
    k = 0

    while i < len(a1) and j < len(a2):
        if a1[i] < a2[j]:
            result[k] = a1[i]
            i += 1
        elif a1[i] > a2[j]:
            result[k] = a2[j]
            j += 1
        else:  # a1[i] == a2[j]
            result[k] = a1[i]
            i += 1
            j += 1
        k += 1

    result = result[:k]

    if i < len(a1):
        result = np.concatenate((result, a1[i:]))
    elif j < len(a2):
        result = np.concatenate((result, a2[j:]))

    return result


@njit(cache=True)
def union_sorted_multi(arrays: list[np.ndarray]) -> np.ndarray:
    """Compute the union of multiple sorted numpy arrays.

    Args:
        arrays (list of np.array): List of sorted arrays

    Returns:
        np.array: Sorted union of all input arrays

    """
    if len(arrays) == 1:
        return arrays[0]
    if len(arrays) == 2:
        return union_sorted(arrays[0], arrays[1])
    return union_sorted(
        union_sorted_multi(arrays[:2]),
        union_sorted_multi(arrays[2:]),
    )


# INTERSECTION -----------------------------------------------------------------
@njit(cache=True)
def intersect_sorted(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Compute the intersection of two sorted numpy arrays.

    Args:
        a1 (np.array): First sorted array
        a2 (np.array): Second sorted array

    Returns:
        np.array: Sorted intersection of a1 and a2

    """
    result = np.empty(min(len(a1), len(a2)), dtype=np.int32)
    i = 0
    j = 0
    k = 0

    while i < len(a1) and j < len(a2):
        if a1[i] < a2[j]:
            i += 1
        elif a1[i] > a2[j]:
            j += 1
        else:  # a1[i] == a2[j]
            result[k] = a1[i]
            i += 1
            j += 1
            k += 1

    return result[:k]


@njit(cache=True)
def intersect_sorted_multi(arrays):
    """Compute the intersection of multiple sorted numpy arrays.

    Args:
        arrays (list of np.array): List of sorted arrays

    Returns:
        np.array: Sorted intersection of all input arrays

    """
    a = arrays[0]

    for i in range(1, len(arrays)):
        a = intersect_sorted(a, arrays[i])

    return a


# DIFFERENCE -------------------------------------------------------------------
@njit(cache=True)
def diff_sorted(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Compute the difference between two sorted numpy arrays (a1 - a2).

    Args:
        a1 (np.array): First sorted array
        a2 (np.array): Second sorted array

    Returns:
        np.array: Sorted difference of a1 and a2

    """
    result = np.empty(len(a1), dtype=np.int32)
    i = 0
    j = 0
    k = 0

    while i < len(a1) and j < len(a2):
        if a1[i] < a2[j]:
            result[k] = a1[i]
            i += 1
            k += 1
        elif a1[i] > a2[j]:
            j += 1
        else:  # a1[i] == a2[j]
            i += 1
            j += 1

    result = result[:k]

    if i < len(a1):
        result = np.concatenate((result, a1[i:]))

    return result


#  -----------------------------------------------------------------------------
@njit(cache=True)
def concat1d(arrays):
    """Concatenate multiple 1D numpy arrays.

    Args:
        arrays (list of np.array): List of 1D arrays to concatenate

    Returns:
        np.array: Concatenated 1D array

    """
    out = np.empty(sum([len(arr) for arr in arrays]), dtype=arrays[0].dtype)

    i = 0
    for arr in arrays:
        for j in range(len(arr)):
            out[i] = arr[j]
            i = i + 1

    return out


@njit(cache=True)
def get_indices(array, scores):
    """Find indices in 'array' that match the values in 'scores'.

    Args:
        array (np.array): Array to search in
        scores (np.array): Array of values to find

    Returns:
        np.array: Array of indices where values in 'scores' are found in 'array'

    """
    n_scores = len(scores)
    min_score = min(scores)
    max_score = max(scores)
    indices = np.full(n_scores, -1, dtype=np.int64)
    counter = 0

    for i in range(len(array)):
        if array[i] >= min_score and array[i] <= max_score:
            for j in range(len(scores)):
                if indices[j] == -1 and scores[j] == array[i]:
                    indices[j] = i
                    counter += 1
                    if len(indices) == counter:
                        return indices
                    break

    return indices
