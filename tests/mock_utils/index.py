from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from baguetter.indices import BaseSparseIndex


class _Index:
    vocabulary: dict

    def __eq__(self, other):
        return self.vocabulary == other.vocabulary


class MockSparseIndex(BaseSparseIndex):
    def normalize_scores(self, n_tokens: int, scores: np.ndarray) -> np.ndarray:
        def normalize(n):
            return np.log(1 + (n - 0.5) / 1.5)

        return scores / (n_tokens * normalize(n_tokens))

    def _build_index(self, corpus_tokens: list, show_progress: bool = False) -> None:
        self.index = _Index()
        self.index.vocabulary = {
            token: i for i, token in enumerate(set(sum(corpus_tokens, [])))
        }

    def _get_top_k(
        self, token_ids: np.ndarray, token_weights: np.ndarray = None, top_k: int = 100
    ) -> tuple:
        top_k = min(top_k, len(self.key_mapping))
        return np.arange(top_k), np.arange(top_k)


@pytest.fixture
def mock_queries_and_results():
    queries = ["query1", "query2", "query3"]
    expected_results = [
        ["doc1", "doc3", "doc2"],  # For "embeddings in NLP"
        ["doc2", "doc1", "doc3"],  # For "benchmarking rerank models"
        ["doc3", "doc1", "doc2"],  # For "ColBERT token embeddings"
    ]
    return queries, expected_results


@pytest.fixture
def mock_sparse_index_with_predefined_results(mock_queries_and_results):
    queries, expected_results = mock_queries_and_results

    class PredefinedResultsMockSparseIndex(MockSparseIndex):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index = MagicMock()
            self.index.vocabulary = {query: i for i, query in enumerate(queries)}
            self.key_mapping = {
                i: doc_id for i, doc_id in enumerate(expected_results[0])
            }

        def tokenize(
            self,
            text: str | list[str],
            n_workers: int | None = None,
            show_progress: bool = False,
            return_generator: bool = False,
        ) -> list[str] | list[list[str]]:
            return (
                [text]
                if isinstance(text, str)
                else [self.tokenize(doc) for doc in text]
            )

        def _get_top_k(
            self,
            token_ids: np.ndarray,
            token_weights: np.ndarray = None,
            top_k: int = 100,
        ) -> tuple:
            id_2_query = {id_: query for query, id_ in self.index.vocabulary.items()}
            doc_id_2_index = {
                doc_id: index for index, doc_id in self.key_mapping.items()
            }

            query = (
                id_2_query[token_ids[0]]
                if len(token_ids) > 0 and token_ids[0] in id_2_query
                else ""
            )

            if query in queries:
                query_index = queries.index(query)
                result_ids = expected_results[query_index][:top_k]
                scores = np.linspace(1, 0.1, len(result_ids))
                indices = [doc_id_2_index[doc_id] for doc_id in result_ids]
                return scores, indices
            else:
                return [], []

    return PredefinedResultsMockSparseIndex(
        index_name="test-index-predefined",
    )


@pytest.fixture
def mock_weighted_queries_and_results():
    weighted_queries = [
        (["embeddings", "NLP"], [0.7, 0.3]),
        (["benchmarking", "rerank", "models"], [0.5, 0.3, 0.2]),
        (["ColBERT", "token", "embeddings"], [0.4, 0.3, 0.3]),
    ]
    expected_results = [
        ["doc1", "doc3", "doc2"],  # For "embeddings" and "NLP"
        ["doc2", "doc1", "doc3"],  # For "benchmarking", "rerank", and "models"
        ["doc3", "doc1", "doc2"],  # For "ColBERT", "token", and "embeddings"
    ]
    return weighted_queries, expected_results


@pytest.fixture
def mock_sparse_index_with_predefined_weighted_results(
    mock_weighted_queries_and_results,
):
    weighted_queries, expected_results = mock_weighted_queries_and_results

    class PredefinedWeightedResultsMockSparseIndex(MockSparseIndex):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index = MagicMock()
            unique_tokens = set(
                token for query, _ in weighted_queries for token in query
            )
            self.index.vocabulary = {token: i for i, token in enumerate(unique_tokens)}
            self.key_mapping = {
                i: doc_id for i, doc_id in enumerate(expected_results[0])
            }

        def tokenize(
            self,
            text: str | list[str],
            n_workers: int | None = None,
            show_progress: bool = False,
            return_generator: bool = False,
        ) -> list[str] | list[list[str]]:
            return (
                [text]
                if isinstance(text, str)
                else [self.tokenize(doc) for doc in text]
            )

        def _get_top_k(
            self,
            token_ids: np.ndarray,
            token_weights: np.ndarray | None = None,
            top_k: int = 100,
        ) -> tuple[np.ndarray, np.ndarray]:
            id_2_token = {id_: token for token, id_ in self.index.vocabulary.items()}
            doc_id_2_index = {
                doc_id: index for index, doc_id in self.key_mapping.items()
            }
            tokens = [id_2_token[id_] for id_ in token_ids if id_ in id_2_token]

            for idx, (predefined_query, predefined_weights) in enumerate(
                weighted_queries
            ):
                if set(tokens) == set(predefined_query):
                    result_ids = expected_results[idx][:top_k]
                    scores = np.linspace(1, 0.1, len(result_ids))
                    indices = [doc_id_2_index[doc_id] for doc_id in result_ids]
                    return scores, np.array(indices)

            # Fallback to random results if query not in predefined list
            return np.array([]), np.array([])

    return PredefinedWeightedResultsMockSparseIndex(
        index_name="test-index-predefined-weighted",
    )
