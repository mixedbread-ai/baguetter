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

# ruff: noqa
from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

DocFrequencies = dict[int, int]
Vocabulary = dict[str, int]
CorpusTokenIds = list[list[int]]

# Constants
DEFAULT_DTYPE = "float32"
DEFAULT_INT_DTYPE = "int32"


def build_idf_array(
    *,
    doc_frequencies: DocFrequencies,
    n_docs: int,
    compute_idf_fn: Callable,
    dtype: str = DEFAULT_DTYPE,
    show_progress: bool = True,
) -> np.ndarray:
    """Build the Inverse Document Frequency (IDF) array."""
    n_vocab = len(doc_frequencies)
    idf_array = np.zeros(n_vocab, dtype=dtype)

    for token_id, df in tqdm(
        doc_frequencies.items(),
        desc="Computing IDF",
        disable=not show_progress,
    ):
        idf_array[token_id] = compute_idf_fn(df, n_docs=n_docs)

    return idf_array


def build_nonoccurrence_array(
    *,
    doc_frequencies: DocFrequencies,
    n_docs: int,
    compute_idf_fn: Callable,
    calculate_tfc_fn: Callable,
    l_d: float,
    l_avg: float,
    k1: float,
    b: float,
    delta: float,
    dtype: str = DEFAULT_DTYPE,
    show_progress: bool = False,
) -> np.ndarray:
    """Build the non-occurrence array for BM25L and BM25+ variants."""
    n_vocab = len(doc_frequencies)
    nonoccurrence_array = np.zeros(n_vocab, dtype=dtype)

    for token_id, df in tqdm(
        doc_frequencies.items(),
        desc="Computing Non-Occurrence Array",
        disable=not show_progress,
    ):
        idf = compute_idf_fn(df, n_docs=n_docs)
        tfc = calculate_tfc_fn(
            tf_array=0,
            l_d=l_d,
            l_avg=l_avg,
            k1=k1,
            b=b,
            delta=delta,
        )
        nonoccurrence_array[token_id] = idf * tfc

    return nonoccurrence_array


# TFC (Term Frequency Component) scoring functions


def score_tfc_robertson(
    tf_array: np.ndarray,
    l_d: float,
    l_avg: float,
    k1: float,
    b: float,
    delta: float | None = None,
) -> np.ndarray:
    """Calculate TFC score using Robertson's method."""
    return tf_array / (k1 * ((1 - b) + b * l_d / l_avg) + tf_array)


def score_tfc_lucene(
    tf_array: np.ndarray,
    l_d: float,
    l_avg: float,
    k1: float,
    b: float,
    delta: float | None = None,
) -> np.ndarray:
    """Calculate TFC score using Lucene's method (identical to Robertson's)."""
    return score_tfc_robertson(tf_array, l_d, l_avg, k1, b)


def score_tfc_atire(
    tf_array: np.ndarray,
    l_d: float,
    l_avg: float,
    k1: float,
    b: float,
    delta: float | None = None,
) -> np.ndarray:
    """Calculate TFC score using ATIRE's method."""
    return (tf_array * (k1 + 1)) / (tf_array + k1 * (1 - b + b * l_d / l_avg))


def score_tfc_bm25l(
    tf_array: np.ndarray,
    l_d: float,
    l_avg: float,
    k1: float,
    b: float,
    delta: float,
) -> np.ndarray:
    """Calculate TFC score using BM25L method."""
    c_array = tf_array / (1 - b + b * l_d / l_avg)
    return ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)


def score_tfc_bm25plus(
    tf_array: np.ndarray,
    l_d: float,
    l_avg: float,
    k1: float,
    b: float,
    delta: float,
) -> np.ndarray:
    """Calculate TFC score using BM25+ method."""
    num = (k1 + 1) * tf_array
    den = k1 * (1 - b + b * l_d / l_avg) + tf_array
    return (num / den) + delta


# IDF (Inverse Document Frequency) scoring functions


def score_idf_robertson(df: int, n_docs: int, allow_negative: bool = False) -> float:
    """Calculate IDF score using Robertson's method."""
    inner = (n_docs - df + 0.5) / (df + 0.5)
    if not allow_negative and inner < 1:
        inner = 1
    return math.log(inner)


def score_idf_lucene(df: int, n_docs: int) -> float:
    """Calculate IDF score using Lucene's method."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def score_idf_atire(df: int, n_docs: int) -> float:
    """Calculate IDF score using ATIRE's method."""
    return math.log(n_docs / df)


def score_idf_bm25l(df: int, n_docs: int) -> float:
    """Calculate IDF score using BM25L method."""
    return math.log((n_docs + 1) / (df + 0.5))


def score_idf_bm25plus(df: int, n_docs: int) -> float:
    """Calculate IDF score using BM25+ method."""
    return math.log((n_docs + 1) / df)


def get_counts_from_token_ids(
    token_ids: list[int],
    dtype: str,
    int_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Get vocabulary indices and term frequency arrays from token IDs."""
    token_counter = Counter(token_ids)
    voc_ind = np.array(list(token_counter.keys()), dtype=int_dtype)
    tf_array = np.array(list(token_counter.values()), dtype=dtype)
    return voc_ind, tf_array


def build_scores_and_indices_for_matrix(
    corpus_token_ids: CorpusTokenIds,
    idf_array: np.ndarray,
    avg_doc_len: float,
    doc_frequencies: DocFrequencies,
    k1: float,
    b: float,
    delta: float,
    nonoccurrence_array: np.ndarray | None,
    calculate_tfc: Callable,
    dtype: str = DEFAULT_DTYPE,
    int_dtype: str = DEFAULT_INT_DTYPE,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build scores and indices for the BM25 matrix."""
    array_size = sum(doc_frequencies.values())

    # We create 3 arrays to store the scores, document indices, and vocabulary indices
    # The length is at most n_tokens, remaining elements will be truncated at the end
    scores = np.empty(array_size, dtype=dtype)
    doc_indices = np.empty(array_size, dtype=int_dtype)
    voc_indices = np.empty(array_size, dtype=int_dtype)

    i = 0
    for doc_idx, token_ids in enumerate(
        tqdm(
            corpus_token_ids,
            desc="Computing BM25 Scores",
            disable=not show_progress,
        ),
    ):
        doc_len = len(token_ids)

        # Get the term frequency array for the document
        # Note: tokens might contain duplicates, we use Counter to get the term freq
        voc_ind_doc, tf_array = get_counts_from_token_ids(
            token_ids,
            dtype=dtype,
            int_dtype=int_dtype,
        )

        # Calculate the BM25 score for each token in the document
        tfc = calculate_tfc(
            tf_array=tf_array,
            l_d=doc_len,
            l_avg=avg_doc_len,
            k1=k1,
            b=b,
            delta=delta,
        )
        idf = idf_array[voc_ind_doc]
        scores_doc = idf * tfc

        # If the method is uses a non-occurrence score array, then we need to subtract
        # the non-occurrence score from the scores
        if nonoccurrence_array is not None:
            scores_doc -= nonoccurrence_array[voc_ind_doc]

        # Update the arrays with the new scores, document indices, and vocabulary indices
        doc_len = len(scores_doc)
        start, end = i, i + doc_len
        i = end

        doc_indices[start:end] = doc_idx
        voc_indices[start:end] = voc_ind_doc
        scores[start:end] = scores_doc

    return scores, doc_indices, voc_indices


def get_unique_tokens(
    corpus_tokens: list[list[str]],
    show_progress: bool = True,
) -> set[str]:
    """Get unique tokens from the corpus."""
    unique_tokens = set()
    for doc_tokens in tqdm(
        corpus_tokens,
        disable=not show_progress,
        desc="Calculating Unique Tokens",
    ):
        unique_tokens.update(doc_tokens)
    return unique_tokens


def compute_document_frequencies_and_vocabulary(
    corpus_tokens: list[list[str]],
    doc_frequencies: DocFrequencies | None = None,
    show_progress: bool = True,
) -> tuple[DocFrequencies, Vocabulary, CorpusTokenIds]:
    """Compute document frequencies, create vocabulary, and convert corpus tokens to token IDs."""
    unique_tokens = get_unique_tokens(
        corpus_tokens,
        show_progress=show_progress,
    )

    vocabulary = {}
    unique_token_ids = set()
    for i, token in enumerate(sorted(unique_tokens)):
        vocabulary[token] = i
        unique_token_ids.add(i)

    corpus_token_ids = [
        [vocabulary[token] for token in tokens]
        for tokens in tqdm(
            corpus_tokens,
            desc="Converting tokens to token IDs",
            disable=not show_progress,
        )
    ]

    doc_frequencies = doc_frequencies or {token: 0 for token in unique_token_ids}

    for doc_token_ids in tqdm(
        corpus_token_ids,
        desc="Counting Tokens",
        disable=not show_progress,
    ):
        shared_token_ids = unique_token_ids.intersection(doc_token_ids)
        for token in shared_token_ids:
            doc_frequencies[token] += 1

    return doc_frequencies, vocabulary, corpus_token_ids


# BM25 method configurations
BM25_METHODS = {
    "robertson": (score_tfc_robertson, score_idf_robertson),
    "lucene": (score_tfc_lucene, score_idf_lucene),
    "atire": (score_tfc_atire, score_idf_atire),
    "bm25l": (score_tfc_bm25l, score_idf_bm25l),
    "bm25plus": (score_tfc_bm25plus, score_idf_bm25plus),
}

NON_OCCURRENCE_METHODS = {"bm25l", "bm25plus"}


def _get_bm25_fns(method: str) -> tuple[Callable, Callable]:
    """Get the TFC and IDF functions for the specified BM25 method."""
    if method in BM25_METHODS:
        return BM25_METHODS[method]
    msg = f"Invalid BM25 method: {method}. Choose from {', '.join(BM25_METHODS.keys())}."
    raise ValueError(msg)


def get_tfc_fn(method: str) -> Callable:
    return _get_bm25_fns(method)[0]


def get_idf_fn(method: str) -> callable:
    return _get_bm25_fns(method)[1]
