from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np

from baguetter.indices.base import SearchResults
from baguetter.indices.dense.base import BaseDenseIndex
from baguetter.indices.dense.config import FaissDenseIndexConfig
from baguetter.logger import LOGGER

if TYPE_CHECKING:
    from collections.abc import Callable

    from baguetter.types import DTypeLike, Key, ScoringMetric, TextOrVector
    from baguetter.utils.file_repository import AbstractFileRepository


def _support_nprobe(index: Any) -> bool:
    """Check if the index supports nprobe."""
    return hasattr(index, "nprobe")


class FaissDenseIndex(BaseDenseIndex):
    """A dense index implementation using Faiss."""

    def __init__(
        self,
        index_name: str = "new-index",
        *,
        embedding_dim: int | None = None,
        metric: ScoringMetric | None = None,
        dtype: DTypeLike | None = None,
        faiss_string: str = "Flat",
        embed_fn: Callable[[list[str], bool], np.ndarray] | None = None,
        n_workers: int | None = None,
        normalize_score: bool = True,
        train_samples: list[TextOrVector] | None = None,
    ) -> None:
        """Initialize the FaissDenseIndex.

        Args:
            index_name (str): Name of the index. Defaults to "new-index".
            embedding_dim (int): Dimensionality of the embedding vectors.
            metric (Optional[ScoringMetric]): Scoring metric to use. Defaults to None.
            dtype (Optional[DTypeLike]): Data type for the index. Defaults to None.
            faiss_string (str): Faiss index factory string. Defaults to "Flat".
            embed_fn (Optional[Callable]): Function to embed text into vectors. Defaults to None.
            n_workers (Optional[int]): Number of worker threads. Defaults to None.
            normalize_score (bool): Whether to normalize scores. Defaults to True.
            train_samples (Optional[list[TextOrVector]]): Samples to train the index. Defaults to None.

        """
        super().__init__(
            index_name=index_name,
            embed_fn=embed_fn,
            n_workers=n_workers,
        )

        if embed_fn is not None and embedding_dim is None:
            embedding_dim = embed_fn([". "], is_query=False, show_progress=False).shape[
                1
            ]

        if embedding_dim is None:
            msg = "embedding_dim must be provided if embed_fn is not None."
            raise ValueError(msg)

        self.config = FaissDenseIndexConfig(
            index_name=index_name,
            embedding_dim=embedding_dim,
            metric=metric,
            dtype=dtype,
            faiss_string=faiss_string,
            normalize_score=normalize_score,
        )
        self.key_mapping: dict[int, Key] = {}
        self.faiss_index = faiss.index_factory(
            self.config.embedding_dim, self.config.faiss_string
        )

        if self.require_training() and train_samples:
            self.train(train_samples)
            LOGGER.info("Index trained with provided samples.")

    @property
    def size(self) -> int:
        return len(self.key_mapping)

    @classmethod
    def from_config(cls, config: FaissDenseIndexConfig) -> FaissDenseIndex:
        """Create an instance from a FaissDenseIndexConfig."""
        return cls(**asdict(config))

    def _save(
        self,
        path: str,
        repository: AbstractFileRepository,
    ) -> str:
        """Save the index state and data."""
        state = {
            "key_mapping": self.key_mapping,
            "config": asdict(self.config),
        }

        (
            state_file_path,
            index_file_path,
        ) = BaseDenseIndex.build_index_file_paths(path or self.name)

        with repository.open(state_file_path, "wb") as file:
            np.savez_compressed(file, state=state)

        with repository.open(index_file_path, "wb") as file:
            index = faiss.serialize_index(self.faiss_index)
            np.savez_compressed(file, index=index)
        return state_file_path

    @classmethod
    def _load(
        cls,
        path: str,
        *,
        repository: AbstractFileRepository,
        mmap: bool = False,
    ) -> FaissDenseIndex:
        """Load the index from saved state.

        Args:
            path (str): Name or path of the index to load.
            repository (AbstractFileRepository): File repository to use for loading.
            mmap (bool): Whether to use memory mapping. Defaults to False.

        Returns:
            FaissDenseIndex: Loaded index instance.

        Raises:
            FileNotFoundError: If the index files are not found in the repository.
        """
        state_file_path, index_file_path = BaseDenseIndex.build_index_file_paths(path)

        if not repository.exists(state_file_path):
            msg = f"Index.state {state_file_path} not found in repository."
            raise FileNotFoundError(msg)

        if not repository.exists(index_file_path):
            msg = f"Index.index {index_file_path} not found in repository."
            raise FileNotFoundError(msg)

        with repository.open(state_file_path, "rb") as file:
            state = np.load(file, allow_pickle=True)["state"][()]
            config = state["config"]
            index = cls.from_config(FaissDenseIndexConfig(**config))
            index.key_mapping = state["key_mapping"]

        mmap_mode = "r" if mmap else None
        with repository.open(index_file_path, "rb") as file:
            stored = np.load(file, allow_pickle=True, mmap_mode=mmap_mode)
            index.faiss_index = faiss.deserialize_index(stored["index"])

        return index

    def require_training(self) -> bool:
        """Check if the index requires training."""
        return (
            hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained
        )

    def train(
        self, values: list[TextOrVector], *, show_progress: bool = False, **kwargs
    ):
        """Train the index.

        Args:
            values (list[TextOrVector]): List of vectors or texts to train on.
            show_progress (bool): Whether to display a progress bar during training. Defaults to False.
            **kwargs: Additional keyword arguments for training.

        Raises:
            ValueError: If training is not required or if there are insufficient vectors for training.
        """
        if not self.require_training():
            LOGGER.warning("Index does not require training.")
            return

        if len(values) < 2:
            msg = "Training requires at least two vectors."
            raise ValueError(msg)

        if isinstance(values[0], str):
            values = self._embed(values, show_progress=show_progress)

        self.faiss_index.train(np.array(values, dtype=self.config.dtype), **kwargs)

    def search(
        self,
        query: TextOrVector,
        *,
        top_k: int = 100,
        n_probe: int | None = None,
        **kwargs: Any,
    ) -> SearchResults:
        """Search the index for similar vectors.

        Args:
            query (TextOrVector): The query vector or text.
            top_k (int): Number of top results to return. Defaults to 100.
            n_probe (int | None): Number of probes for search. Defaults to None.
            **kwargs: Additional keyword arguments for search.

        Returns:
            SearchResults: Search results containing keys and scores.
        """
        return self.search_many([query], top_k=top_k, n_probe=n_probe, **kwargs)[0]

    def search_many(
        self,
        queries: list[str | TextOrVector],
        *,
        top_k: int = 100,
        n_probe: int | None = None,
        n_workers: int | None = None,
        show_progress: bool = False,
    ) -> list[SearchResults]:
        """Search the index for multiple queries.

        Args:
            queries (list[str | TextOrVector]): List of query vectors or texts.
            top_k (int): Number of top results to return for each query. Defaults to 100.
            n_probe (int | None): Number of probes for search. Defaults to None.
            n_workers (int | None): Number of workers for parallel processing. Defaults to None.
            show_progress (bool): Whether to display a progress bar during search. Defaults to False.

        Returns:
            list[SearchResults]: List of search results for each query.
        """
        if not queries:
            return []
        if isinstance(queries[0], str):
            queries = self._embed(queries, is_query=True, show_progress=show_progress)
        if _support_nprobe(self.faiss_index) and n_probe is not None:
            self.faiss_index.nprobe = n_probe

        n_workers = n_workers if n_workers is not None else self.n_workers
        faiss.omp_set_num_threads(n_workers)

        query_vectors = np.array(queries, dtype=np.float32)

        scores, indices = self.faiss_index.search(query_vectors, top_k)

        # Metric types https://github.com/facebookresearch/faiss/blob/main/faiss/MetricType.h
        if self.faiss_index.metric_type != 0:  # IF not METRIC_INNER_PRODUCT
            scores = 1 / (1 + scores)
        return [
            SearchResults(
                keys=[self.key_mapping[idx] for idx in query_indices if idx != -1],
                scores=np.array(
                    [
                        score
                        for idx, score in zip(query_indices, query_scores)
                        if idx != -1
                    ]
                ),
                normalized=self.config.normalize_score,
            )
            for query_scores, query_indices in zip(scores, indices)
        ]

    def add(self, key: Key, value: TextOrVector) -> FaissDenseIndex:
        """Add a single item to the index.

        Args:
            key (Key): The key for the new item.
            value (TextOrVector): The vector or text to add.

        Returns:
            FaissDenseIndex: The index instance for method chaining.
        """
        return self.add_many([key], [value])

    def add_many(
        self,
        keys: list[Key],
        values: list[TextOrVector],
        *,
        show_progress: bool = False,
    ) -> FaissDenseIndex:
        """Add multiple items to the index.

        Args:
            keys (list[Key]): List of keys for the new items.
            values (list[TextOrVector]): List of vectors or texts to add.
            show_progress (bool): Whether to display a progress bar during addition. Defaults to False.

        Raises:
            ValueError: If the index requires training before adding items.

        Returns:
            FaissDenseIndex: The index instance for method chaining.
        """
        if not keys or not values:
            return self

        self.validate_key_value(keys, values)

        if self.require_training():
            msg = "Index requires training before adding items."
            raise ValueError(msg)

        self.remove_many(keys)

        if isinstance(values[0], str):
            values = self._embed(values, show_progress=show_progress)

        self.faiss_index.add(np.array(values, dtype=self.config.dtype))
        self.key_mapping.update(dict(enumerate(keys, start=len(self.key_mapping))))
        return self

    def remove(self, key: Key) -> FaissDenseIndex:
        """Delete an item from the index.

        Args:
            key (Key): The key of the item to remove.

        Returns:
            FaissDenseIndex: The index instance for method chaining.
        """
        return self.remove_many([key])

    def remove_many(self, keys: list[Key]) -> FaissDenseIndex:
        """Delete items from the index.

        Args:
            keys (list[Key]): List of keys of the items to remove.

        Returns:
            FaissDenseIndex: The index instance for method chaining.
        """
        inv_key_mapping = {v: k for k, v in self.key_mapping.items()}
        indices_to_remove = [
            inv_key_mapping[key] for key in keys if key in inv_key_mapping
        ]

        if not indices_to_remove:
            return self

        self.faiss_index.remove_ids(np.array(indices_to_remove, dtype=np.int64))

        remaining_keys = [
            v for k, v in self.key_mapping.items() if k not in indices_to_remove
        ]
        self.key_mapping = dict(enumerate(remaining_keys))
        return self
