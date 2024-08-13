from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from baguetter.utils.persistable import HuggingFacePersistable

if TYPE_CHECKING:
    import numpy as np

    from baguetter.types import Key

T = TypeVar("T")


@dataclass
class SearchResults:
    """Represents the results of a search operation."""

    keys: list[Key]
    scores: np.ndarray
    normalized: bool

    def to_dict(self) -> dict[Key, float]:
        """Convert search results to a dictionary.

        Returns:
            dict[Key, float]: A dictionary mapping keys to their corresponding scores.

        """
        return dict(zip(self.keys, self.scores))


class BaseIndex(HuggingFacePersistable, abc.ABC, Generic[T]):
    """Abstract base class for index implementations.

    This class defines the interface for index operations such as adding,
    removing, and searching for documents.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the index.

        Returns:
            str: The name of the index.

        """

    @abc.abstractmethod
    def add(self, key: Key, value: T) -> BaseIndex:
        """Index a single document.

        Args:
            key (Key): The unique identifier for the document.
            value (T): The document to be indexed.

        Returns:
            BaseIndex: The updated index.

        """

    @abc.abstractmethod
    def add_many(self, keys: list[Key], values: list[T]) -> BaseIndex:
        """Index a collection of documents.

        Args:
            keys (list[Key]): List of unique identifiers for the documents.
            values (list[T]): List of documents to be indexed.

        Returns:
            BaseIndex: The updated index.

        """

    @abc.abstractmethod
    def remove(self, key: Key) -> BaseIndex:
        """Remove a document from the index.

        Args:
            key (Key): The unique identifier of the document to be removed.

        Returns:
            BaseIndex: The updated index.

        """

    @abc.abstractmethod
    def remove_many(self, keys: list[Key]) -> BaseIndex:
        """Remove a collection of documents from the index.

        Args:
            keys (list[Key]): List of unique identifiers of the documents to be removed.

        Returns:
            BaseIndex: The updated index.

        """

    @abc.abstractmethod
    def search(self, query: T, *, top_k: int = 100, **kwargs) -> SearchResults:
        """Search for documents matching the query.

        Args:
            query (T): The search query.
            top_k (int, optional): The maximum number of documents to return. Defaults to 100.
            **kwargs: Additional search parameters.

        Returns:
            SearchResults: The list of documents matching the query.

        """

    @abc.abstractmethod
    def search_many(self, queries: list[T], *, top_k: int = 100, **kwargs) -> list[SearchResults]:
        """Search for multiple queries.

        Args:
            queries (list[T]): A list of search queries.
            top_k (int, optional): The maximum number of documents to return per query. Defaults to 100.
            **kwargs: Additional search parameters.

        Returns:
            list[SearchResults]: A list of query results.

        """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Any) -> BaseIndex:
        """Create an index from a configuration.

        Args:
            config (Any): The configuration for creating the index.

        Returns:
            BaseIndex: An instance of the index created from the given configuration.

        """

    @staticmethod
    def validate_key_value(keys: list[Key], values: list[T] | None = None) -> None:
        """Check if the keys and values are valid:
        1. All keys are unique.
        2. If values are provided, the number of keys matches the number of values.

        Args:
            keys (list[Key]): List of keys.
            values (list[T] | None, optional): List of values. Defaults to None.

        Raises:
            ValueError: If the keys are not unique or the number of keys and values do not match.

        """
        if len(keys) != len(set(keys)):
            msg = "Keys must be unique."
            raise ValueError(msg)

        if values is not None and len(keys) != len(values):
            msg = "Length of keys and values must match."
            raise ValueError(msg)
