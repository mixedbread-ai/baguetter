from __future__ import annotations

import math
import tempfile
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
from usearch.index import Index, MetricKind

from baguetter.indices.base import SearchResults
from baguetter.indices.dense.base import BaseDenseIndex
from baguetter.indices.dense.config import UsearchDenseIndexConfig
from baguetter.utils.numpy_utils import reversed_scale_min_max_normalization

if TYPE_CHECKING:
    from collections.abc import Callable

    from baguetter.types import DTypeLike, Key, ScoringMetric, TextOrVector
    from baguetter.utils.file_repository import AbstractFileRepository


def get_normalization_fn(metric: MetricKind, embedding_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    """Get the appropriate normalization function based on the metric and dimensionality.

    Args:
        metric (MetricKind): The metric used for similarity calculation.
        embedding_dim (int): The number of dimensions in the vector space.

    Returns:
        Callable[[np.ndarray], np.ndarray]: A function that normalizes the input scores.

    Raises:
        ValueError: If normalization for the given metric is not implemented.

    """
    if metric == MetricKind.Hamming:
        return lambda scores: reversed_scale_min_max_normalization(scores, (0, embedding_dim))
    if metric == MetricKind.Cos:
        return lambda scores: 1 - scores
    if metric in {MetricKind.Jaccard, MetricKind.Tanimoto, MetricKind.Sorensen}:
        return lambda scores: scores
    msg = f"Normalization for metric {metric} not implemented."
    raise ValueError(msg)


class USearchDenseIndex(BaseDenseIndex, Index):
    """A dense index implementation using USearch.

    This class combines functionality from BaseIndex and USearch's Index,
    providing methods for adding, searching, and managing dense vector data.
    """

    def __init__(
        self,
        index_name: str = "new-index",
        *,
        embedding_dim: int | None = None,
        metric: ScoringMetric | None = None,
        dtype: DTypeLike | None = None,
        connectivity: int | None = None,
        ef_construction: int | None = None,
        ef: int | None = None,
        enable_key_lookups: bool = True,
        embed_fn: Callable[[list[str], bool], np.ndarray] | None = None,
        n_workers: int | None = None,
        normalize_score: bool = True,
        exact_search: bool = True,
    ) -> None:
        """Initialize the USearchDenseIndex.

        Args:
            index_name (str): Name of the index. Defaults to "new-index".
            embedding_dim (int): Number of dimensions for the vectors.
            metric (Optional[ScoringMetric]): Metric for similarity calculation. Defaults to None.
            dtype (Optional[DTypeLike]): Data type for the vectors. Defaults to None.
            connectivity (Optional[int]): Connectivity parameter for the index. Defaults to None.
            ef_construction (Optional[int]): ef_construction parameter for index building. Defaults to None.
            ef (Optional[int]): ef parameter for search. Defaults to None.
            enable_key_lookups (bool): Whether to enable key lookups. Defaults to True.
            embed_fn (Optional[Callable]): Function to embed text queries into vectors. Defaults to None.
            n_workers (Optional[int]): Number of threads to use for parallel operations. Defaults to None.
            normalize_score (bool): Whether to normalize scores. Defaults to True.
            exact_search (bool): Whether to perform exact search. Defaults to True.

        """
        BaseDenseIndex.__init__(
            self,
            index_name=index_name,
            embed_fn=embed_fn,
            n_workers=n_workers,
        )

        if embed_fn is not None and embedding_dim is None:
            embedding_dim = embed_fn([". "], is_query=False, show_progress=False).shape[1]

        if embedding_dim is None:
            msg = "embedding_dim must be provided if embed_fn is not None."
            raise ValueError(msg)

        Index.__init__(
            self,
            ndim=embedding_dim,
            metric=metric,
            dtype=dtype,
            connectivity=connectivity,
            expansion_add=ef_construction,
            expansion_search=ef,
            enable_key_lookups=enable_key_lookups,
        )

        self.config = UsearchDenseIndexConfig(
            index_name=index_name,
            embedding_dim=embedding_dim,
            metric=metric,
            dtype=dtype,
            connectivity=connectivity,
            ef_construction=ef_construction,
            ef=ef,
            enable_key_lookups=enable_key_lookups,
            normalize_score=normalize_score,
            exact_search=exact_search,
        )

        self.key_counter: int = 0
        self.key_mapping: dict[int, Key] = {}

        self.normalization_fn: Callable[[np.ndarray], np.ndarray] = (
            get_normalization_fn(self.metric, self.config.embedding_dim) if normalize_score else lambda x: x
        )

    @property
    def size(self) -> int:
        return len(self.key_mapping)

    @classmethod
    def from_config(cls, config: UsearchDenseIndexConfig) -> USearchDenseIndex:
        """Create an instance from a UsearchDenseIndexConfig."""
        return cls(**asdict(config))

    def _save(
        self,
        path: str,
        repository: AbstractFileRepository,
    ) -> str:
        """Save the index state and data.

        Args:
            path (str): Path to save the index. If None, uses the index name.
            repository (AbstractFileRepository): File repository to use for saving.

        """
        state = {
            "key_mapping": self.key_mapping,
            "id_count": self.key_counter,
            "config": asdict(self.config),
        }

        (
            state_file_path,
            index_file_path,
        ) = BaseDenseIndex.build_index_file_paths(path or self.name)

        with repository.open(state_file_path, "wb") as file:
            np.savez_compressed(file, state=state)

        with (
            repository.open(index_file_path, "wb") as file,
            tempfile.NamedTemporaryFile() as temp_file,
        ):
            Index.save(self, temp_file.name)
            file.write(temp_file.read())

        return state_file_path

    @classmethod
    def _load(
        cls,
        path: str,
        *,
        repository: AbstractFileRepository,
        mmap: bool = False,
    ) -> USearchDenseIndex:
        """Load the index from saved state.

        Args:
            path (str): Name or path of the index to load.
            repository (AbstractFileRepository): File repository to use for loading.
            mmap (bool): Whether to use memory mapping. Defaults to False.

        Returns:
            USearchDenseIndex: Loaded index instance.

        Raises:
            FileNotFoundError: If the index files are not found in the repository.

        """
        (
            state_file_path,
            index_file_path,
        ) = BaseDenseIndex.build_index_file_paths(path)

        if not repository.exists(state_file_path):
            msg = f"Index.state {state_file_path} not found in repository."
            raise FileNotFoundError(msg)

        if not repository.exists(index_file_path):
            msg = f"Index.index {index_file_path} not found in repository."
            raise FileNotFoundError(msg)

        with repository.open(state_file_path, "rb") as file:
            state = np.load(file, allow_pickle=True)["state"][()]
            config = state["config"]
            index = cls.from_config(UsearchDenseIndexConfig(**config))
            index.key_mapping = state["key_mapping"]
            index.key_counter = state["id_count"]

        with (
            repository.open(index_file_path, "rb") as file,
            tempfile.NamedTemporaryFile(delete=not mmap) as temp,
        ):
            temp.write(file.read())
            temp.seek(0)

            if mmap:
                Index.view(index, temp.name)
            else:
                Index.load(index, temp.name)

        return index

    def search(
        self,
        query: TextOrVector,
        *,
        top_k: int = 100,
        radius: float = math.inf,
        exact_search: bool | None = None,
        **kwargs,
    ) -> SearchResults:
        """Search the index for similar vectors.

        Args:
            query (TextOrVector): The query vector or text.
            top_k (int): Number of top results to return. Defaults to 100.
            radius (float): Search radius. Defaults to infinity.
            exact_search (Optional[bool]): Whether to perform exact search. If None, uses the index's default setting.
            **kwargs: Additional keyword arguments for the search.

        Returns:
            SearchResults: Search results containing keys and scores.

        """
        if isinstance(query, str):
            query = self._embed([query], is_query=True)[0]

        results = Index.search(
            self,
            query,
            count=top_k,
            radius=radius,
            exact=self.config.exact_search if exact_search is None else exact_search,
            log=False,
            **kwargs,
        )

        return SearchResults(
            keys=[self.key_mapping[idx] for idx in results.keys],
            scores=(self.normalization_fn(results.distances) if self.config.normalize_score else results.distances),
            normalized=self.config.normalize_score,
        )

    def search_many(
        self,
        queries: list[TextOrVector],
        *,
        top_k: int = 100,
        exact_search: bool | None = None,
        radius: float = math.inf,
        n_workers: int | None = None,
        show_progress: bool = False,
        **kwargs,
    ) -> list[SearchResults]:
        """Perform multiple searches in parallel.

        Args:
            queries (List[TextOrVector]): List of query vectors or texts.
            top_k (int): Number of top results to return for each query. Defaults to 100.
            exact_search (Optional[bool]): Whether to perform exact search. If None, uses the index's default setting.
            radius (float): Search radius. Defaults to infinity.
            n_workers (Optional[int]): Number of threads to use for parallel search. If None, uses the index's default.
            show_progress (bool): Whether to show progress during search. Defaults to False.
            **kwargs: Additional keyword arguments for the search.

        Returns:
            List[SearchResults]: List of search results for each query.

        """
        if not queries:
            return []
        if isinstance(queries[0], str):
            queries = self._embed(queries, is_query=True, show_progress=show_progress)
        if not isinstance(queries, np.ndarray):
            queries = np.array(queries)

        n_workers = n_workers if n_workers is not None else self.n_workers

        results = Index.search(
            self,
            vectors=queries,
            count=top_k,
            threads=n_workers,
            log=show_progress,
            radius=radius,
            exact=self.config.exact_search if exact_search is None else exact_search,
            **kwargs,
        )

        return [
            SearchResults(
                keys=[self.key_mapping[idx] for idx in result.keys],
                scores=(self.normalization_fn(result.distances) if self.config.normalize_score else result.distances),
                normalized=self.config.normalize_score,
            )
            for result in results
        ]

    def remove(self, key: Key) -> USearchDenseIndex:
        """Remove a single item from the index.

        Args:
            key (Key): The key of the item to remove.

        Returns:
            USearchDenseIndex: The index instance for method chaining.

        """
        return self.remove_many([key])

    def remove_many(self, keys: list[Key]) -> USearchDenseIndex:
        """Remove multiple items from the index.

        Args:
            keys (List[Key]): List of keys to remove.

        Returns:
            USearchDenseIndex: The index instance for method chaining.

        """
        if not keys:
            return self

        self.validate_key_value(keys)

        inverse_key_mapping = {v: k for k, v in self.key_mapping.items()}
        keys_to_remove = [inverse_key_mapping[key] for key in keys if key in inverse_key_mapping]
        if not keys_to_remove:
            return self

        Index.remove(self, keys=keys_to_remove)

        for key in keys_to_remove:
            self.key_mapping.pop(key, None)

        return self

    def add(self, key: Key, value: TextOrVector) -> USearchDenseIndex:
        """Add a single item to the index.

        Args:
            key (Key): The key for the new item.
            value (TextOrVector): The vector or text to add.

        Returns:
            USearchDenseIndex: The index instance for method chaining.

        """
        return self.add_many([key], [value])

    def add_many(
        self,
        keys: list[Key],
        values: list[TextOrVector],
        *,
        show_progress: bool = False,
    ) -> USearchDenseIndex:
        """Add multiple items to the index.

        Args:
            keys (List[Key]): List of keys for the new items.
            values (List[TextOrVector]): List of vectors or texts to add.
            show_progress (bool): Whether to display a progress bar during the addition process.
                                  Defaults to False.

        Raises:
            ValueError: If the number of keys and values don't match.

        Returns:
            USearchDenseIndex: The index instance for method chaining.

        """
        if not keys or not values:
            return self

        self.validate_key_value(keys, values)

        if isinstance(values[0], str):
            values = self._embed(values, show_progress=show_progress)

        self.remove_many(keys)

        internal_keys = list(range(self.key_counter, self.key_counter + len(keys)))
        Index.add(self, internal_keys, np.array(values), log=show_progress)
        self.key_counter += len(keys)
        self.key_mapping.update(zip(internal_keys, keys))
        return self
