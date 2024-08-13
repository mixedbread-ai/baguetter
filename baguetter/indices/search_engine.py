from __future__ import annotations

import dataclasses
from collections.abc import Callable
from itertools import chain
from typing import TYPE_CHECKING, Any

from baguetter.indices.base import BaseIndex, SearchResults
from baguetter.indices.sparse.bmx import BMXSparseIndex
from baguetter.settings import settings
from baguetter.utils.common import ensure_dir_exists
from baguetter.utils.sqlite_key_val import KeyValueSqlite

if TYPE_CHECKING:
    from baguetter.types import Key


@dataclasses.dataclass
class EnhancedSearchResults(SearchResults):
    """Enhanced search results containing additional information."""

    query: str
    values: list[str]

    def __repr__(self) -> str:
        return (
            f"EnhancedSearchResults(\n"
            f"  query='{self.query}',\n"
            f"  keys={self.keys},\n"
            f"  values={self.values},\n"
            f"  scores={self.scores},\n"
            f"  normalized={self.normalized}\n"
            f")"
        )

    @classmethod
    def from_search_results(
        cls, *, query: str, values: list[str], search_results: SearchResults
    ) -> EnhancedSearchResults:
        """
        Create EnhancedSearchResults from SearchResults.

        Args:
            query (str): The search query.
            values (list[str]): List of result values.
            search_results (SearchResults): Original search results.

        Returns:
            EnhancedSearchResults: New instance with additional information.
        """
        return cls(query=query, values=values, **dataclasses.asdict(search_results))


PostProcessingFn = Callable[[list[EnhancedSearchResults], bool], list[EnhancedSearchResults]]


class SearchEngine(BaseIndex[str]):
    """Search engine implementation combining an index and a key-value store."""

    def __init__(
        self,
        name: str = "new-search-engine",
        *,
        index: BaseIndex | None = None,
        db_path: str | None = None,
        table_name: str = "search_engine",
        post_process_fn: PostProcessingFn | None = None,
    ):
        """
        Initialize the SearchEngine.

        Args:
            name (str): Name of the search engine.
            index (BaseIndex | None): Search index to use. Defaults to BMXSparseIndex.
            db_path (str | None): Path to the SQLite database.
            table_name (str): Name of the table in the database.
            post_process_fn (PostProcessingFn | None): Function for post-processing search results.
        """
        db_path = db_path or f"{settings.cache_dir}/sqlite/{name}.db"
        ensure_dir_exists(db_path)

        self.index = index or BMXSparseIndex()
        self.store = KeyValueSqlite(path=db_path, table_name=table_name)
        self._name = name
        self.post_process_fn = post_process_fn

    @property
    def name(self) -> str:
        """Get the name of the search engine."""
        return self._name

    def add(self, key: Key, value: str) -> SearchEngine:
        """
        Add a single item to the search engine.

        Args:
            key (Key): The key for the item.
            value (str): The value to be added.

        Returns:
            SearchEngine: The current instance for method chaining.
        """
        self.add_many([key], [value])
        return self

    def add_many(self, keys: list[Key], values: list[str], *, show_progress: bool = False) -> SearchEngine:
        """
        Add multiple items to the search engine.

        Args:
            keys (list[Key]): List of keys for the items.
            values (list[str]): List of values to be added.
            show_progress (bool): Whether to show progress during addition.

        Returns:
            SearchEngine: The current instance for method chaining.
        """
        self.index.add_many(keys, values, show_progress=show_progress)
        self.store.update(dict(zip(keys, values)))
        return self

    def remove(self, key: Key) -> SearchEngine:
        """
        Remove a single item from the search engine.

        Args:
            key (Key): The key of the item to remove.

        Returns:
            SearchEngine: The current instance for method chaining.
        """
        self.remove_many([key])
        return self

    def remove_many(self, keys: list[Key]) -> SearchEngine:
        """
        Remove multiple items from the search engine.

        Args:
            keys (list[Key]): List of keys of the items to remove.

        Returns:
            SearchEngine: The current instance for method chaining.
        """
        self.index.remove_many(keys)
        self.store.remove_many(keys)
        return self

    def search(
        self,
        query: str | Any,
        *,
        top_k: int = 100,
        apply_post_processing: bool = True,
        **kwargs,
    ) -> EnhancedSearchResults:
        """
        Perform a search query.

        Args:
            query (str | Any): The search query.
            top_k (int): Number of top results to return.
            apply_post_processing (bool): Whether or not to apply the post processor if applicable.
            **kwargs: Additional search parameters.

        Returns:
            EnhancedSearchResults: The search results.
        """
        search_results = self.index.search(query, top_k=top_k, **kwargs)
        stored = self.store.get_many(search_results.keys)

        results = EnhancedSearchResults.from_search_results(
            query=query,
            values=[stored[key] for key in search_results.keys],
            search_results=search_results,
        )

        if self.post_process_fn and apply_post_processing:
            results = self.post_process_fn([results], show_progress=False)[0]
        return results

    def search_many(
        self,
        queries: list[str | Any],
        *,
        top_k: int = 100,
        show_progress: bool = False,
        apply_post_processing: bool = True,
        **kwargs,
    ) -> list[EnhancedSearchResults]:
        """
        Perform multiple search queries.

        Args:
            queries (list[str | Any]): List of search queries.
            top_k (int): Number of top results to return for each query.
            show_progress (bool): Whether or not to show a progress bar.
            apply_post_processing (bool): Whether or not to apply the post processor if applicable.
            **kwargs: Additional search parameters.

        Returns:
            list[EnhancedSearchResults]: List of search results for each query.
        """
        search_results = self.index.search_many(queries, top_k=top_k, show_progress=show_progress, **kwargs)
        all_keys = set(chain.from_iterable(result.keys for result in search_results))
        stored = self.store.get_many(all_keys)
        results = [
            EnhancedSearchResults.from_search_results(
                query=query,
                values=[stored[key] for key in result.keys],
                search_results=result,
            )
            for query, result in zip(queries, search_results)
        ]

        if self.post_process_fn and apply_post_processing:
            results = self.post_process_fn(results, show_progress=show_progress)
        return results

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SearchEngine:  # noqa: ARG003
        """
        Create a SearchEngine instance from a configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            SearchEngine: New SearchEngine instance.
        """
        msg = "Loading from config functionality not implemented"
        raise NotImplementedError(msg)

    def _save(self, repository: Any, path: str | None = None) -> None:  # noqa: ARG002
        """
        Save the search engine state.

        Args:
            repository (Any): The repository to save to.
            path (str | None): Path to save the state.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        msg = "Saving functionality not implemented"
        raise NotImplementedError(msg)

    @classmethod
    def _load(cls, name_or_path: str, *, repository: Any, mmap: bool = False) -> SearchEngine:  # noqa: ARG003
        """
        Load a search engine state.

        Args:
            name_or_path (str): Name or path of the search engine to load.
            repository (Any): The repository to load from.
            mmap (bool): Whether to use memory mapping.

        Returns:
            SearchEngine: Loaded SearchEngine instance.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        msg = "Loading functionality not implemented"
        raise NotImplementedError(msg)
