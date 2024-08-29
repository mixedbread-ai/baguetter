from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from baguetter.fuser.config import FuserConfig
from baguetter.fuser.fuser import Fuser
from baguetter.indices.base import BaseIndex, SearchResults
from baguetter.types import HybridValue, Key

if TYPE_CHECKING:
    from baguetter.utils.file_repository import AbstractFileRepository


class MultiIndex(BaseIndex[HybridValue]):
    """A Hybrid Index that combines results from multiple BaseIndex instances.

    This class implements a composite index that can search across multiple
    underlying indices and merge the results.
    """

    def __init__(
        self,
        indices: dict[str, BaseIndex] | None = None,
        *,
        fuser_config: FuserConfig | None = None,
        n_workers: int | None = None,
    ) -> None:
        """Initialize a MultiIndex instance.

        Args:
            indices (Optional[dict[str, BaseIndex]]): Dictionary of BaseIndex instances to be used.
            fuser_config (Optional[FuserConfig]): Configuration for the Fuser. Defaults to FuserConfig().
            n_workers (Optional[int]): Number of threads to use for parallel processing.
                                       Defaults to the number of CPUs.

        """
        fuser_config = fuser_config or FuserConfig()

        self.indices: dict[str, BaseIndex] = indices or {}
        self.fuser: Fuser = Fuser.from_config(fuser_config)
        self.n_workers: int = n_workers or os.cpu_count() or 1

    @property
    def name(self) -> str:
        """Get the name of the MultiIndex.

        Returns:
            str: A string representation of the keys of all contained indices.

        """
        return str(list(self.indices.keys()))

    def add_index(self, key: str, index: BaseIndex) -> MultiIndex:
        """Add a new BaseIndex to the hybrid index.

        Warning:
            This method should only be used during initialization, not on the fly.
            Indices added later will not have access to data added before their inclusion.

        Args:
            key (str): The key for the index to be added.
            index (BaseIndex): The index to be added.

        Returns:
            MultiIndex: The updated MultiIndex instance.

        """
        self.indices[key] = index
        return self

    def remove_index(self, key: str) -> MultiIndex:
        """Remove a BaseIndex from the hybrid index.

        Args:
            key (str): The key of the index to be removed.

        Returns:
            MultiIndex: The updated MultiIndex instance.

        """
        self.indices.pop(key, None)
        return self

    def add(self, key: Key, value: HybridValue) -> MultiIndex:
        """Add a single item to all indices.

        Args:
            key (Key): The key for the new item.
            value (HybridValue): The value to be added.

        Returns:
            MultiIndex: The updated MultiIndex instance.

        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            list(executor.map(lambda idx: idx.add(key, value), self.indices.values()))
        return self

    def add_many(self, keys: list[Key], values: list[HybridValue], *, show_progress: bool = False) -> MultiIndex:
        """Add multiple items to all indices.

        Args:
            keys (List[Key]): The keys for the new items.
            values (List[HybridValue]): The values to be added.
            show_progress (bool): Whether to show progress.

        Returns:
            MultiIndex: The updated MultiIndex instance.

        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            list(
                executor.map(
                    lambda idx: idx.add_many(keys, values, show_progress=show_progress),
                    self.indices.values(),
                )
            )
        return self

    def remove(self, key: Key) -> MultiIndex:
        """Remove a single item from all indices.

        Args:
            key (Key): The key of the item to be removed.

        Returns:
            MultiIndex: The updated MultiIndex instance.

        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            list(executor.map(lambda idx: idx.remove(key), self.indices.values()))
        return self

    def remove_many(self, keys: list[Key], **kwargs) -> MultiIndex:
        """Remove multiple items from all indices.

        Args:
            keys (List[Key]): The keys of the items to be removed.
            **kwargs: Additional arguments to pass to the remove_many method.
        Returns:
            MultiIndex: The updated MultiIndex instance.

        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            list(executor.map(lambda idx: idx.remove_many(keys, **kwargs), self.indices.values()))
        return self

    def _validate_query(self, query: str | dict[str, Any]) -> None:
        """Validate the search query.

        Args:
            query (Union[str, Dict[str, Any]]): The query to be validated.

        Raises:
            ValueError: If the query contains keys not present in the indices.

        """
        if isinstance(query, str):
            return

        if not set(query.keys()).issubset(self.indices.keys()):
            msg = (
                "The query contains keys that are not present in the indices. "
                "The query keys must be a subset of the index keys."
            )
            raise ValueError(msg)

    def search(
        self,
        query: str | dict[str, HybridValue],
        *,
        top_k: int = 100,
        **kwargs,
    ) -> SearchResults:
        """Perform a search across all indices and merge the results.

        Args:
            query (Union[str, Dict[str, HybridValue]]): The query to search for.
                If a string, it will be used for all indices.
                If a dict, it must contain keys that are present in the indices.
            top_k (int): Number of results to return. Defaults to 100.
            **kwargs: Additional search parameters.

        Returns:
            SearchResults: Merged results from all indices.

        """
        self._validate_query(query)
        if isinstance(query, str):
            query = {key: query for key in self.indices}

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(
                executor.map(
                    lambda key: self.indices[key].search(query=query[key], top_k=top_k, **kwargs),
                    self.indices.keys(),
                )
            )
        return self.fuser.merge(results)

    def search_many(
        self,
        queries: list[str | dict[str, HybridValue]],
        *,
        top_k: int = 100,
        show_progress: bool = False,
        **kwargs,
    ) -> list[SearchResults]:
        """Compute results for multiple queries across all indices.

        Args:
            queries (List[Union[str, Dict[str, HybridValue]]]): List of queries to search for.
                Each query can be a string or a dictionary.
            top_k (int): Number of results to return per query. Defaults to 100.
            show_progress (bool): Whether to show progress.
            **kwargs: Additional search parameters.

        Returns:
            List[SearchResults]: Results for each query, merged across all indices.

        """
        if not queries:
            return []

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            return list(
                tqdm(
                    executor.map(
                        lambda query: self.search(query, top_k=top_k, **kwargs),
                        queries,
                    ),
                    total=len(queries),
                    desc="Searching queries",
                    disable=not show_progress,
                )
            )

    @classmethod
    def from_config(cls, _config: dict[str, Any]) -> MultiIndex:
        """Create a MultiIndex instance from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            MultiIndex: New MultiIndex instance.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        msg = "MultiIndex.from_config is not implemented yet."
        raise NotImplementedError(msg)

    @classmethod
    def _load(cls, _name_or_path: str, _repository: AbstractFileRepository) -> MultiIndex:
        """Load a MultiIndex instance from a file.

        Args:
            name_or_path (str): Name or path of the MultiIndex to load.
            repository (AbstractFileRepository): The repository to load from.

        Returns:
            MultiIndex: Loaded MultiIndex instance.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        msg = "MultiIndex._load is not implemented yet."
        raise NotImplementedError(msg)

    def _save(self, _path: str | None, _repository: AbstractFileRepository) -> MultiIndex:
        """Save the MultiIndex instance to a file.

        Args:
            path (Optional[str]): Path to save the MultiIndex.
            repository (AbstractFileRepository): The repository to save to.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        msg = "MultiIndex._save is not implemented yet."
        raise NotImplementedError(msg)
