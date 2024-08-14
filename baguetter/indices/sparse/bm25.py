from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from baguetter.indices.sparse.base import BaseSparseIndex
from baguetter.indices.sparse.models import bm25

if TYPE_CHECKING:
    from collections.abc import Callable

NORMALIZATION_METHODS: dict[str, Callable[[int], float]] = {
    "lucene": lambda n: np.log(1 + (n - 0.5) / 1.5),
    "robertson": lambda n: np.log(1 + (n - 0.5) / 1.5),
    "atire": lambda n: np.log(n),
    "bm25l": lambda n: np.log((n + 1) / 1.5),
    "bm25plus": lambda n: np.log(n + 1),
}


class BM25SparseIndex(BaseSparseIndex):
    """BM25 Sparse Index implementation.

    This class extends BaseSparseIndex to provide BM25-specific functionality
    for indexing and searching documents.
    """

    def normalize_scores(self, n_tokens: int, scores: ndarray) -> ndarray:
        """Normalize BM25 scores by the number of tokens in the query.

        Args:
            n_tokens (int): The number of tokens in the query.
            scores (ndarray): The BM25 scores.

        Returns:
            ndarray: The normalized BM25 scores.

        Raises:
            ValueError: If the normalization method is not supported.

        """
        try:
            normalization_func = NORMALIZATION_METHODS[self.config.method]
        except KeyError as e:
            msg = f"Unsupported normalization method: {self.config.method}"
            raise ValueError(msg) from e

        return scores / (n_tokens * normalization_func(n_tokens))

    def _build_index(
        self,
        corpus_tokens: list[list[str]],
        *,
        show_progress: bool = False,
    ) -> None:
        """Build the BM25 index.

        Args:
            corpus_tokens (list[list[str]]): The list of tokenized documents.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to False.

        """
        self.index = bm25.build_index(
            corpus_tokens=corpus_tokens,
            b=self.config.b,
            k1=self.config.k1,
            delta=self.config.delta,
            method=self.config.method,
            idf_method=self.config.idf_method,
            dtype=self.config.dtype,
            int_dtype=self.config.int_dtype,
            show_progress=show_progress,
        )

    def _get_top_k(
        self,
        token_ids: np.ndarray,
        *,
        token_weights: np.ndarray | None = None,
        top_k: int = 100,
    ) -> tuple[ndarray, ndarray]:
        """Get the top-k documents for a query.

        Args:
            token_ids (np.ndarray): The token IDs of the query.
            token_weights (np.ndarray | None): The token weights of the query. Defaults to None.
            top_k (int): The number of documents to return. Defaults to 100.

        Returns:
            tuple[ndarray, ndarray]: A tuple containing the document IDs and scores.

        """
        return bm25.calculate_scores(
            token_ids=token_ids,
            index=self.index,
            top_k=top_k,
            dtype=self.config.dtype,
            token_weights=token_weights,
        )
