"""bm25s: Fast BM25 Implementation.

This module was copied from the below github repository and only slightly modified.
It provides a fast implementation of the BM25 (Best Matching 25) algorithm,
a ranking function used by search engines to rank matching documents according to
their relevance to a given search query.

Key features:
- Efficient implementation of BM25 algorithm
- Optimized for performance in information retrieval tasks

GitHub repository: https://github.com/xhluca/bm25s
Author: Xing Han Lu
License: MIT License
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp
from numba import njit

from baguetter.utils.numpy_utils import top_k_numpy

from .scoring import (
    NON_OCCURRENCE_METHODS,
    build_idf_array,
    build_nonoccurrence_array,
    build_scores_and_indices_for_matrix,
    compute_document_frequencies_and_vocabulary,
    get_idf_fn,
    get_tfc_fn,
)


@dataclasses.dataclass
class BM25:
    """Represents the BM25 index for efficient document scoring and retrieval."""

    scores: np.ndarray
    doc_indices: np.ndarray
    col_pointers: np.ndarray
    num_documents: int
    vocabulary: dict[str, int] | None = dataclasses.field(default_factory=dict)
    nonoccurrence_array: np.ndarray | None = None


def build_index(
    *,
    corpus_tokens: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 0.5,
    method: str = "lucene",
    idf_method: str = "lucene",
    dtype: str = "float32",
    int_dtype: str = "int32",
    show_progress: bool = True,
) -> BM25:
    """Build the BM25 index from token IDs.

    Args:
        corpus_tokens (List[List[str]]): List of tokenized documents.
        k1 (float): BM25 free parameter controlling term frequency saturation.
        b (float): BM25 free parameter controlling document length normalization.
        delta (float): BM25 smoothing parameter.
        method (str): BM25 scoring method.
        dtype (str): Data type for floating-point arrays.
        int_dtype (str): Data type for integer arrays.
        show_progress (bool): Whether to display progress bar.

    Returns:
        The constructed BM25 index.

    """
    idf_fn = get_idf_fn(idf_method)
    tfc_fn = get_tfc_fn(method)

    # Step 1: Calculate document frequencies and create vocabulary
    doc_frequencies, vocabulary, corpus_token_ids = compute_document_frequencies_and_vocabulary(
        corpus_tokens=corpus_tokens,
        show_progress=show_progress,
    )

    avg_doc_len = float(np.mean([len(doc_ids) for doc_ids in corpus_token_ids]))
    n_docs = len(corpus_token_ids)
    n_vocab = len(vocabulary)

    # Step 2: Calculate non-occurrence array if needed
    nonoccurrence_array = None
    if method in NON_OCCURRENCE_METHODS:
        nonoccurrence_array = build_nonoccurrence_array(
            doc_frequencies=doc_frequencies,
            n_docs=n_docs,
            compute_idf_fn=idf_fn,
            calculate_tfc_fn=tfc_fn,
            l_d=avg_doc_len,
            l_avg=avg_doc_len,
            k1=k1,
            b=b,
            delta=delta,
            dtype=dtype,
            show_progress=show_progress,
        )

    # Step 3: Calculate IDF for each token
    idf_array = build_idf_array(
        doc_frequencies=doc_frequencies,
        n_docs=n_docs,
        compute_idf_fn=idf_fn,
        dtype=dtype,
        show_progress=show_progress,
    )

    # Step 4: Calculate BM25 scores
    scores_flat, doc_idx, vocab_idx = build_scores_and_indices_for_matrix(
        corpus_token_ids=corpus_token_ids,
        idf_array=idf_array,
        avg_doc_len=avg_doc_len,
        doc_frequencies=doc_frequencies,
        calculate_tfc=tfc_fn,
        k1=k1,
        b=b,
        delta=delta,
        dtype=dtype,
        int_dtype=int_dtype,
        nonoccurrence_array=nonoccurrence_array,
        show_progress=show_progress,
    )

    # Step 5: Build sparse matrix
    score_matrix = sp.csc_matrix(
        (scores_flat, (doc_idx, vocab_idx)),
        shape=(n_docs, n_vocab),
        dtype=dtype,
    )

    return BM25(
        scores=score_matrix.data,
        doc_indices=score_matrix.indices,
        col_pointers=score_matrix.indptr,
        num_documents=n_docs,
        nonoccurrence_array=nonoccurrence_array,
        vocabulary=vocabulary,
    )


@njit(cache=True)
def _calculate_scores_optimized(
    *,
    token_ids: np.ndarray,
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
    dtype: np.dtype = np.float32,
    token_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Optimized function to calculate BM25 scores for given token IDs.

    Args:
        token_ids (np.ndarray): Array of token IDs to score.
        data (np.ndarray): Data array of the BM25 index.
        indptr (np.ndarray): Index pointer array of the BM25 index.
        indices (np.ndarray): Indices array of the BM25 index.
        num_docs (int): Number of documents in the BM25 index.
        dtype (np.dtype): Data type for score calculation.
        token_weights (np.ndarray, optional): Array of token weights. Defaults to None.

    Returns:
        Array of BM25 scores.

    """
    indptr_starts = indptr[token_ids]
    indptr_ends = indptr[token_ids + 1]

    scores = np.zeros(num_docs, dtype=dtype)
    for i in range(len(token_ids)):
        token_weight = token_weights[i] if token_weights is not None else 1.0
        start, end = indptr_starts[i], indptr_ends[i]
        for j in range(start, end):
            scores[indices[j]] += data[j] * token_weight
    return scores


def calculate_scores(
    *,
    token_ids: np.ndarray,
    index: BM25,
    top_k: int = 100,
    dtype: str = "float32",
    token_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate BM25 scores for given token IDs and return top-k results.

    Args:
        token_ids (np.ndarray): Array of token IDs to score.
        index (BM25): The BM25 index.
        top_k (int): Number of top scores to return.
        dtype (str): Data type for score calculation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: Top-k scores.
            - np.ndarray: Corresponding document indices for the top-k scores.

    """
    query_scores = _calculate_scores_optimized(
        token_ids=token_ids,
        data=index.scores,
        indptr=index.col_pointers,
        indices=index.doc_indices,
        num_docs=index.num_documents,
        dtype=np.dtype(dtype),
        token_weights=token_weights,
    )

    top_k_scores, top_k_indices = top_k_numpy(query_scores, top_k)
    return top_k_scores, top_k_indices
