from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import TYPE_CHECKING

from baguetter.indices.base import BaseIndex

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

_INDEX_PREFIX = "usearch_index_"
_STATE_PREFIX = "usearch_state_"


class BaseDenseIndex(BaseIndex, abc.ABC):
    NAME_PREFIX: str = "dense_"

    def __init__(
        self,
        index_name: str = "new-index",
        *,
        embed_fn: Callable[[list[str], bool], np.ndarray] | None = None,
        n_workers: int | None = None,
    ) -> None:
        super().__init__()

        self.index_name: str = index_name
        self.n_workers: int = n_workers or max(1, (os.cpu_count() or 1) - 1)
        self._embed_fn: Callable[[list[str], bool], np.ndarray] | None = embed_fn

    @property
    def name(self) -> str:
        """Get the full name of the index."""
        return f"{self.NAME_PREFIX}{self.index_name}"

    def _embed(self, query: list[str], *, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        """Embed text queries into vectors.

        Args:
            query (List[str]): List of text queries to embed.
            is_query (bool): Whether the input is a query (as opposed to a document).
            show_progress (bool): Whether to display a progress bar during embedding.

        Returns:
            np.ndarray: Embedded vectors.

        Raises:
            ValueError: If no embedding function is provided.

        """
        if self._embed_fn is None:
            msg = (
                "Embedding function not provided. "
                "Please provide an embedding function to convert text queries to vectors."
            )
            raise ValueError(msg)
        return self._embed_fn(query, is_query=is_query, show_progress=show_progress)

    @staticmethod
    def build_index_file_paths(name_or_path: str) -> tuple[str, str]:
        """Build the file paths for the index and state files.

        Args:
            name_or_path (str): Path to the index.

        Returns:
            Tuple[str, str]: File paths for the state and index files.

        """
        path = Path(name_or_path)
        state_file_name = f"{_STATE_PREFIX}{path.name}"
        index_file_name = f"{_INDEX_PREFIX}{path.name}"

        dir_name = path.parent if path.parent != Path() else Path()
        state_file_path = dir_name / state_file_name
        index_file_path = dir_name / index_file_name

        return str(state_file_path), str(index_file_path)
