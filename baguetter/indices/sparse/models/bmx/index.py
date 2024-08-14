"""BMX: Enhanced BM25-like Implementation for Document Retrieval.

This module provides an efficient implementation of a BM25-like algorithm (BMX)
for ranking documents according to their relevance to a given search query.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from numpy import ndarray

from baguetter.utils.common import tqdm
from baguetter.utils.numpy_utils import top_k_numpy

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclasses.dataclass
class BMX:
    """Represents the BMX index for efficient document scoring and retrieval."""

    inverted_index: dict[int, dict]
    doc_lens: np.ndarray
    relative_doc_lens: np.ndarray
    avg_doc_len: float
    n_docs: int
    vocabulary: dict[str, int]


@nb.njit(cache=True)
def compute_query_doc_similarity(
    doc_ids: list[int],
    query_len: int,
    doc_count: int,
    dtype: np.dtype = np.float32,
) -> list[np.ndarray]:
    """Compute the query-document similarity.
    S(Q, D) = len(Q & D) / len(Q).
    """
    count = np.zeros((doc_count,), dtype=dtype)
    for ids in doc_ids:
        count[ids] += 1
    count /= query_len
    return [count[ids] for ids in doc_ids]


def calculate_scores(
    token_ids: np.array,
    index: BMX,
    top_k: int,
    dtype: str = "float32",
    token_weights: np.ndarray | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> tuple[ndarray, ndarray]:
    query_len = len(token_ids)
    if query_len == 0:
        top_k = min(top_k, index.n_docs)
        return np.zeros((top_k,), dtype=dtype), np.arange(top_k)

    doc_ids = nb.typed.List()
    term_freqs = nb.typed.List()
    term_idfs = nb.typed.List()
    term_entropies = nb.typed.List()

    for token_id in token_ids:
        token_id_entry = index.inverted_index[token_id]
        doc_ids.append(token_id_entry["doc_ids"])
        term_freqs.append(token_id_entry["tf"])
        term_idfs.append(token_id_entry["idf"])
        term_entropies.append(token_id_entry["entropy"])

    query_doc_similarities = compute_query_doc_similarity(
        doc_ids,
        len(token_ids),
        index.n_docs,
        dtype=np.dtype(dtype),
    )

    scores = _calculate_scores(
        query_doc_similarities=query_doc_similarities,
        term_freqs=term_freqs,
        term_idfs=term_idfs,
        term_entropies=term_entropies,
        doc_ids=doc_ids,
        relative_doc_lens=index.relative_doc_lens,
        avg_doc_len=index.avg_doc_len,
        doc_count=index.n_docs,
        token_weights=token_weights,
        alpha=alpha,
        beta=beta,
        dtype=np.dtype(dtype),
    )

    top_k_scores, top_k_indices = top_k_numpy(scores, top_k, sort=True)
    return top_k_scores, top_k_indices


@nb.njit(cache=True)
def _calculate_scores(
    *,
    query_doc_similarities: list[np.ndarray],
    term_freqs: list[np.ndarray],
    term_idfs: list[np.ndarray],
    term_entropies: list[np.ndarray],
    doc_ids: list[np.ndarray],
    relative_doc_lens: np.ndarray,
    avg_doc_len: float,
    doc_count: int,
    alpha: float | None = None,
    beta: float | None = None,
    token_weights: list[float] | None = None,
    dtype: np.dtype = np.float32,
) -> ndarray:
    """Compute BMX scores."""
    # normalize entropy
    entropy = np.asarray(term_entropies)
    entropy /= np.max(entropy)

    # set hyperparameters
    if alpha is None:
        alpha = max(min(1.5, avg_doc_len / 100), 0.5)
    if beta is None:
        beta = 1 / np.log(1 + doc_count)
    avg_entropy = np.mean(entropy)

    # Initialize scores
    scores = np.zeros((doc_count,), dtype=dtype)

    for i in range(len(term_freqs)):
        token_weight = token_weights[i] if token_weights is not None else 1.0
        sims = query_doc_similarities[i]
        indices = doc_ids[i]
        freqs = term_freqs[i]
        idf = term_idfs[i]

        scores[indices] += token_weight * (
            idf * ((freqs * (alpha + 1.0)) / (freqs + alpha * relative_doc_lens[indices] + alpha * avg_entropy))
            + sims * entropy[i] * beta
        )

    return scores


def convert_df_matrix_into_inverted_index(
    *,
    df_matrix: ndarray,
    unique_token_ids: ndarray,
    n_docs: int,
    int_dtype: str = "int32",
    show_progress: bool = True,
) -> dict:
    """Convert document-frequency matrix into an inverted index."""
    inverted_index = defaultdict(dict)

    for i, term in enumerate(
        tqdm(
            unique_token_ids,
            disable=not show_progress,
            desc="Building inverted index",
            dynamic_ncols=True,
            mininterval=0.5,
        ),
    ):
        df = np.float32(len(df_matrix[i].indices))  # noqa: PD901
        idf = np.float32(np.log(1.0 + (((n_docs - df) + 0.5) / (df + 0.5))))
        tf = np.array(df_matrix[i].data, dtype=int_dtype)

        p = 1 / (1 + np.exp(-tf))
        entropy = -np.sum(p * np.log(p))

        inverted_index[term]["doc_ids"] = df_matrix[i].indices
        inverted_index[term]["tf"] = tf
        inverted_index[term]["idf"] = idf
        inverted_index[term]["entropy"] = entropy

    return inverted_index


def build_index(
    *,
    corpus_tokens: Iterable,
    n_docs: int,
    min_df: int = 1,
    int_dtype: str = "int32",
    dtype: str = "float32",
    show_progress: bool = True,
) -> BMX:
    """Build the BMX index from corpus tokens.

    Args:
        corpus_tokens (Iterable): Iterable of tokenized documents.
        n_docs (int): Number of documents in the corpus.
        min_df (int): Minimum document frequency for terms.
        int_dtype (str): Data type for integer arrays.
        dtype (str): Data type for float arrays.
        show_progress (bool): Whether to display progress bar.

    Returns:
        BMX: The constructed BMX index.

    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        min_df=min_df,
        dtype=int_dtype,
        token_pattern=None,
        lowercase=False,
    )

    # [doc_count x n_terms]
    dt_matrix = vectorizer.fit_transform(
        tqdm(
            corpus_tokens,
            total=n_docs,
            disable=not show_progress,
            desc="Building doc-term matrix",
            dynamic_ncols=True,
            mininterval=0.5,
        ),
    )

    # [n_terms x doc_count]
    dt_matrix = dt_matrix.transpose().tocsr()

    unique_tokens = vectorizer.get_feature_names_out()
    unique_token_ids = np.arange(len(unique_tokens))

    inverted_index = convert_df_matrix_into_inverted_index(
        df_matrix=dt_matrix,
        unique_token_ids=unique_token_ids,
        n_docs=n_docs,
        int_dtype=int_dtype,
        show_progress=show_progress,
    )
    doc_lens = np.squeeze(np.asarray(dt_matrix.sum(axis=0), dtype=dtype))
    avg_doc_len = float(np.mean(doc_lens))
    relative_doc_lens = doc_lens / avg_doc_len

    return BMX(
        inverted_index=inverted_index,
        doc_lens=doc_lens,
        relative_doc_lens=relative_doc_lens,
        avg_doc_len=avg_doc_len,
        n_docs=n_docs,
        vocabulary=dict(zip(unique_tokens, unique_token_ids)),
    )
