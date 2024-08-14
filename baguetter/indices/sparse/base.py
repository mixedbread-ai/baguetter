from __future__ import annotations

import abc
import dataclasses
import os
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from baguetter.indices.base import BaseIndex, SearchResults
from baguetter.indices.sparse.config import SparseIndexConfig
from baguetter.indices.sparse.text_preprocessor.text_processor import TextPreprocessor, TextPreprocessorConfig
from baguetter.utils.common import map_in_thread

if TYPE_CHECKING:
    from collections.abc import Generator

    from baguetter.types import Key, TextOrTokens
    from baguetter.utils.file_repository import AbstractFileRepository


class BaseSparseIndex(BaseIndex, abc.ABC):
    """Base class for sparse indices. This class should not be used directly."""

    def __init__(
        self,
        index_name: str = "new-index",
        *,
        min_df: float = 1,
        b: float = 0.75,
        k1: float = 1.2,
        delta: float = 0.5,
        method: str = "lucene",
        idf_method: str = "lucene",
        dtype: str = "float32",
        int_dtype: str = "int32",
        alpha: float | None = None,
        beta: float | None = None,
        normalize_scores: bool = False,
        preprocessor_or_config: TextPreprocessorConfig | TextPreprocessor | None = None,
        n_workers: int | None = None,
    ) -> None:
        """Initialize the BaseSparseIndex.

        Args:
            index_name (str): Name of the index.
            min_df (float): Minimum document frequency for a token to be included in the vocabulary.
            b (float): Parameter for sparse index.
            k1 (float): Parameter for sparse index.
            delta (float): Parameter for sparse index.
            method (str): Method for sparse index.
            idf_method (str): IDF method for sparse index.
            dtype (str): Data type for the index.
            int_dtype (str): Integer data type for the index.
            alpha (float | None): Parameter for sparse index.
            beta (float | None): Parameter for sparse index.
            normalize_scores (bool): Whether to normalize scores.
            preprocessor_or_config (TextPreprocessorConfig | TextPreprocessor | None):
                TextPreprocessor object or config.
            n_workers (int | None): Number of threads to use for tokenization.

        """
        if preprocessor_or_config is None:
            preprocessor_or_config = TextPreprocessorConfig()

        self._pre_processor = (
            preprocessor_or_config
            if isinstance(preprocessor_or_config, TextPreprocessor)
            else TextPreprocessor.from_config(preprocessor_or_config)
        )
        self.config = SparseIndexConfig(
            index_name=index_name,
            preprocessor_config=self._pre_processor.config,
            b=b,
            k1=k1,
            delta=delta,
            min_df=min_df,
            method=method,
            idf_method=idf_method,
            dtype=dtype,
            int_dtype=int_dtype,
            alpha=alpha,
            beta=beta,
            normalize_scores=normalize_scores,
        )
        self.index: object | None = None
        self.key_mapping: dict[int, str] = {}
        self.corpus_tokens: dict[str, list[str]] = {}
        self.n_workers: int = n_workers if n_workers is not None else max(1, (os.cpu_count() or 1) - 1)

    @abc.abstractmethod
    def normalize_scores(self, n_tokens: int, scores: ndarray) -> ndarray:
        """Normalize the scores to the range [0, 1].

        Args:
            n_tokens (int): Number of tokens in the query.
            scores (ndarray): 1D numpy array of scores.

        Returns:
            ndarray: 1D numpy array of normalized scores.

        """

    @abc.abstractmethod
    def _build_index(
        self,
        corpus_tokens: list[list[str]],
        *,
        show_progress: bool = False,
    ) -> None:
        """Build the index from the given corpus of tokens.

        Args:
            corpus_tokens (list[list[str]]): List of lists of tokens.
            show_progress (bool): Whether to show a progress bar.

        """

    @abc.abstractmethod
    def _get_top_k(
        self,
        token_ids: np.ndarray,
        *,
        token_weights: np.ndarray | None = None,
        top_k: int = 100,
    ) -> tuple[ndarray, ndarray]:
        """Get the top k indices and scores for the given token ids.

        Args:
            token_ids (np.ndarray): 1D numpy array of token ids.
            token_weights (np.ndarray | None): 1D numpy array of token weights.
            top_k (int): Number of top results to return.

        Returns:
            tuple[ndarray, ndarray]: Tuple of 1D numpy arrays of top k scores and indices. (scores, indices)

        """

    @property
    def name(self) -> str:
        """Return the name of the index.

        Returns:
            str: The name of the index

        """
        return self.config.index_name

    @property
    def vocabulary(self) -> dict[str, int]:
        """Return the vocabulary of the index.

        Returns:
            dict[str, int]: The vocabulary of the index.

        """
        return self.index.vocabulary if self.index else {}

    def _save(
        self,
        path: str,
        repository: AbstractFileRepository,
    ) -> str:
        """Save the index to the given path.

        Args:
            path (str): Path to save the index to.
            repository (AbstractFileRepository): File repository to save to.

        """
        state = {
            "key_mapping": self.key_mapping,
            "index": self.index,
            "corpus_tokens": self.corpus_tokens,
            "config": dataclasses.asdict(self.config),
        }
        with repository.open(path, "wb") as f:
            np.savez_compressed(f, state=state)
        return path

    @classmethod
    def _load(
        cls,
        path: str,
        repository: AbstractFileRepository,
        *,
        mmap: bool = False,
    ) -> BaseSparseIndex:
        """Load an index from the given path or name.

        Args:
            path (str): Name or path of the index.
            repository (AbstractFileRepository): File repository to load from.
            mmap (bool): Whether to memory-map the file.

        Returns:
            BaseSparseIndex: The loaded index.

        Raises:
            FileNotFoundError: If the index file is not found.

        """
        if not repository.exists(path):
            msg = f"Index {path} not found."
            raise FileNotFoundError(msg)

        mmap_mode = "r" if mmap else None
        with repository.open(path, "rb") as f:
            stored = np.load(f, allow_pickle=True, mmap_mode=mmap_mode)
            state = stored["state"][()]
            retriever = cls.from_config(SparseIndexConfig(**state["config"]))
            retriever.key_mapping = state["key_mapping"]
            retriever.index = state["index"]
            retriever.corpus_tokens = state["corpus_tokens"]
            return retriever

    @classmethod
    def from_config(cls, config: SparseIndexConfig) -> BaseSparseIndex:
        """Create an index from the given config.

        Args:
            config (SparseIndexConfig): SparseIndexConfig object.

        Returns:
            BaseSparseIndex: The created index.

        """
        return cls(
            index_name=config.index_name,
            min_df=config.min_df,
            b=config.b,
            k1=config.k1,
            delta=config.delta,
            method=config.method,
            idf_method=config.idf_method,
            dtype=config.dtype,
            int_dtype=config.int_dtype,
            preprocessor_or_config=config.preprocessor_config,
        )

    def _update_index(self, *, show_progress: bool = False) -> None:
        """Update the index with the current corpus tokens.

        Args:
            show_progress (bool): Whether to show a progress bar.

        """
        self.key_mapping = dict(enumerate(self.corpus_tokens.keys()))

        self._build_index(
            list(self.corpus_tokens.values()),
            show_progress=show_progress,
        )

    def build_index(
        self,
        keys: list[str],
        values: list[TextOrTokens],
        *,
        n_workers: int | None = None,
        show_progress: bool = False,
    ) -> None:
        """Build the index with the given keys and values.

        Args:
            keys (list[str]): List of keys.
            values (list[TextOrTokens]): List of values, each value can be a string or a list of tokens.
            n_workers (int | None): Number of threads to use for tokenization.
            show_progress (bool): Whether to show a progress bar.

        Raises:
            ValueError: If the number of keys and values are not the same.

        """
        self.validate_key_value(keys, values)

        if not keys:
            return

        added_corpus_tokens = (
            values
            if isinstance(values[0], list)
            else self.tokenize(values, n_workers=n_workers, show_progress=show_progress)
        )

        self.corpus_tokens.update(dict(zip(keys, added_corpus_tokens, strict=True)))

        self._update_index(show_progress=show_progress)

    def tokenize(
        self,
        text: str | list[str],
        *,
        n_workers: int | None = None,
        show_progress: bool = False,
        return_generator: bool = False,
    ) -> list[str] | list[list[str]] | Generator[list[str], None, None]:
        """Tokenize the given text or list of texts.

        Args:
            text (str | list[str]): Text or list of texts to tokenize.
            n_workers (int | None): Number of threads to use for tokenization.
            show_progress (bool): Whether to show a progress bar.
            return_generator (bool): Whether to return a generator instead of a list.

        Returns:
            list[str] | list[list[str]] | Generator[list[str], None, None]: List of tokens or list of lists of tokens.

        """
        if isinstance(text, str):
            return self._pre_processor.process(text)

        n_workers = n_workers or self.n_workers

        return self._pre_processor.process_many(
            items=text,
            n_workers=n_workers,
            show_progress=show_progress,
            return_generator=return_generator,
        )

    def add_many(
        self,
        keys: list[Key],
        values: list[TextOrTokens],
        *,
        n_workers: int | None = None,
        show_progress: bool = False,
    ) -> BaseSparseIndex:
        """Add multiple documents to the index.

        Args:
            keys (list[Key]): List of keys for the documents to add.
            values (list[TextOrTokens]): List of documents to add,
            each document can be a string or a list of tokens.
            n_workers (int | None): Number of threads to use for tokenization.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            BaseSparseIndex: The index with the added documents.

        Raises:
            ValueError: If the number of keys and values are not the same.

        """
        self.validate_key_value(keys, values)

        self.build_index(
            keys,
            values,
            n_workers=n_workers,
            show_progress=show_progress,
        )
        return self

    def add(self, key: Key, value: TextOrTokens) -> BaseSparseIndex:
        """Add a document to the index.

        Args:
            key (Key): Key for the document to add.
            value (TextOrTokens): Document to add.

        Returns:
            BaseSparseIndex: The index with the added document.

        """
        return self.add_many([key], [value])

    def remove_many(self, keys: list[Key]) -> BaseSparseIndex:
        """Remove multiple documents from the index.

        Args:
            keys (list[Key]): List of keys for the documents to remove.

        Returns:
            BaseSparseIndex: The index with the removed documents.

        """
        self.validate_key_value(keys)

        for key in keys:
            self.corpus_tokens.pop(key, None)
        self._update_index(show_progress=False)

        return self

    def remove(self, key: Key) -> BaseSparseIndex:
        """Remove a document from the index.

        Args:
            key (Key): Key of the document to remove.

        Returns:
            BaseSparseIndex: The index with the removed document.

        """
        return self.remove_many([key])

    def to_token_ids(self, tokens: list[str]) -> np.ndarray:
        """Convert a list of tokens to a numpy array of token ids.

        Args:
            tokens (list[str]): List of tokens.

        Returns:
            np.ndarray: Numpy array of token ids.

        """
        return np.array(
            [self.vocabulary[t] for t in tokens if t in self.vocabulary],
            dtype=self.config.int_dtype,
        )

    def search(self, query: TextOrTokens, *, top_k: int = 100, **_kwargs) -> SearchResults:
        """Search for the given query and return the top k results.

        Args:
            query (TextOrTokens): Query string or list of tokens.
            top_k (int): Number of top results to return.
            **_kwargs: Additional keyword arguments.

        Returns:
            SearchResults: SearchResults object containing the top k results and their scores.

        Raises:
            ValueError: If the query is empty or invalid.

        """
        tokens = self.tokenize(query) if isinstance(query, str) else query
        token_ids = self.to_token_ids(tokens)

        top_k_scores, top_k_indices = self._get_top_k(token_ids=token_ids, top_k=top_k)

        keys = [self.key_mapping[doc_id] for doc_id in top_k_indices]
        scores = self.normalize_scores(len(token_ids), top_k_scores) if self.config.normalize_scores else top_k_scores

        return SearchResults(
            keys=keys,
            scores=scores,
            normalized=self.config.normalize_scores,
        )

    def search_many(
        self,
        queries: list[TextOrTokens],
        *,
        top_k: int = 100,
        n_workers: int | None = None,
        show_progress: bool = False,
        return_generator: bool = False,
        **_kwargs,
    ) -> list[SearchResults] | Generator[SearchResults, None, None]:
        """Search for the given queries and return the top k results for each.

        Args:
            queries (list[TextOrTokens]): List of query strings or lists of tokens.
            top_k (int): Number of top results to return for each query.
            n_workers (int | None): Number of threads to use for parallel processing.
            show_progress (bool): Whether to show a progress bar.
            return_generator (bool): Whether to return a generator instead of a list.
            **_kwargs: Additional keyword arguments.

        Returns:
            list[SearchResults] | Generator[SearchResults, None, None]:
            Union[List[SearchResults], Generator[SearchResults, None, None]]:
                List of SearchResults objects or a generator of SearchResults objects.

        """
        if not queries:
            return []

        n_workers = n_workers if n_workers is not None else self.n_workers
        k_search = partial(self.search, top_k=top_k)

        results = tqdm(
            map_in_thread(
                k_search,
                queries,
                n_workers=n_workers,
            ),
            total=len(queries),
            desc="Top-K Search",
            disable=not show_progress,
        )

        return list(results) if not return_generator else results

    def search_weighted(
        self,
        queries: list[TextOrTokens],
        query_weights: list[float],
        *,
        top_k: int = 100,
        **_kwargs,
    ) -> SearchResults:
        """Search for the given queries with the given query weights and return the top k results.

        Args:
            queries (List[TextOrTokens]): List of query strings or lists of tokens.
            query_weights (List[float]): List of weights corresponding to each query.
            top_k (int): Number of top results to return.

        Returns:
            SearchResults: SearchResults object containing the top k results and their scores.

        """
        if len(queries) == 0:
            return SearchResults(keys=[], scores=[], normalized=self.config.normalize_scores)

        tokens_list = self.tokenize(queries) if isinstance(queries[0], str) else queries
        token_ids_list = [self.to_token_ids(tokens) for tokens in tokens_list]

        all_token_ids = []
        all_token_weights = []
        for token_ids, query_weight in zip(token_ids_list, query_weights, strict=False):
            all_token_ids.extend(token_ids)
            all_token_weights.extend([query_weight] * len(token_ids))

        top_k_scores, top_k_indices = self._get_top_k(
            token_ids=np.array(all_token_ids, dtype=self.config.int_dtype),
            token_weights=np.array(all_token_weights, dtype=self.config.dtype),
            top_k=top_k,
        )

        keys = [self.key_mapping[doc_id] for doc_id in top_k_indices]
        scores = (
            self.normalize_scores(len(all_token_ids), top_k_scores) if self.config.normalize_scores else top_k_scores
        )
        return SearchResults(
            keys=keys,
            scores=scores,
            normalized=self.config.normalize_scores,
        )
