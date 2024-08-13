from __future__ import annotations

import numpy as np
from numpy import ndarray

from baguetter.indices.sparse.base import BaseSparseIndex
from baguetter.indices.sparse.models import bmx


class BMXSparseIndex(BaseSparseIndex):
    """BMX Sparse Index implementation.

    This class extends BaseSparseIndex to provide BMX-specific functionality
    for indexing and searching documents.
    """

    def normalize_scores(self, n_tokens: int, scores: ndarray) -> ndarray:
        """Normalize BMX scores by the number of tokens in the query.

        Args:
            n_tokens (int): The number of tokens in the query.
            scores (ndarray): The BMX scores.

        Returns:
            ndarray: The normalized BMX scores.

        """
        corpus_size = len(self.corpus_tokens)
        normalization_factor = n_tokens * np.log(1 + (corpus_size - 0.5) / 1.5)
        return scores / normalization_factor

    def _build_index(
        self,
        corpus_tokens: list[list[str]],
        *,
        show_progress: bool = False,
    ) -> None:
        """Build the BMX index.

        Args:
            corpus_tokens (list[list[str]]): The list of tokenized documents.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to False.

        """
        self.index = bmx.build_index(
            corpus_tokens=corpus_tokens,
            show_progress=show_progress,
            min_df=self.config.min_df,
            n_docs=len(corpus_tokens),
            int_dtype=self.config.int_dtype,
            dtype=self.config.dtype,
        )

    def _get_top_k(
        self,
        token_ids: np.ndarray,
        *,
        token_weights: np.ndarray | None = None,
        top_k: int = 100,
    ) -> tuple[ndarray, ndarray]:
        """Get the top-k BMX scores.

        Args:
            token_ids (np.ndarray): The token IDs.
            token_weights (np.ndarray | None): The token weights. Defaults to None.
            top_k (int): The number of top documents to return. Defaults to 100.

        Returns:
            tuple[ndarray, ndarray]: A tuple containing the document IDs and BMX scores.

        """
        return bmx.calculate_scores(
            index=self.index,
            token_ids=token_ids,
            token_weights=token_weights,
            alpha=self.config.alpha,
            beta=self.config.beta,
            top_k=top_k,
            dtype=self.config.dtype,
        )
