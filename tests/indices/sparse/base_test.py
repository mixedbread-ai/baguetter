import numpy as np
import pytest

from baguetter.indices.base import SearchResults
from baguetter.indices.sparse.config import SparseIndexConfig
from baguetter.indices.sparse.text_preprocessor.text_processor import (
    TextPreprocessor, TextPreprocessorConfig)
from tests.mock_utils.index import (
    MockSparseIndex, mock_queries_and_results,
    mock_sparse_index_with_predefined_results,
    mock_sparse_index_with_predefined_weighted_results,
    mock_weighted_queries_and_results)
from tests.mock_utils.text_preprocessor import MockTextPreprocessor


@pytest.fixture
def mock_text_preprocessor():
    return MockTextPreprocessor()


@pytest.fixture
def mock_sparse_index(mock_text_preprocessor):
    return MockSparseIndex(
        index_name="test-index",
        preprocessor_or_config=mock_text_preprocessor,
    )


@pytest.fixture
def mock_sparse_index2(mock_text_preprocessor):
    return MockSparseIndex(
        index_name="test-index2",
        preprocessor_or_config=mock_text_preprocessor,
    )


@pytest.fixture
def mock_sparse_index_from_config():
    return MockSparseIndex.from_config(
        SparseIndexConfig(
            index_name="test-index",
            preprocessor_config=TextPreprocessorConfig(),
        )
    )


@pytest.fixture
def mock_docs():
    ids = ["doc1", "doc2", "doc3"]
    docs = [
        "Embeddings are among the most adaptable tools in natural language processing",
        "Now, we benchmark our rerank models for accuracy",
        "However, instead of creating a single embedding for the entire document, "
        "ColBERT generates contextualized embeddings for each token in the document.",
    ]
    return ids, docs


def test_constructor(mock_sparse_index):
    assert mock_sparse_index.name == "test-index"
    assert isinstance(mock_sparse_index._pre_processor, TextPreprocessor)
    assert isinstance(mock_sparse_index.config, SparseIndexConfig)
    assert mock_sparse_index.index is None
    assert mock_sparse_index.corpus_tokens == {}
    assert mock_sparse_index.key_mapping == {}
    assert mock_sparse_index.vocabulary == {}


def test_constructor_from_config(mock_sparse_index_from_config):
    assert mock_sparse_index_from_config.name == "test-index"
    assert isinstance(mock_sparse_index_from_config._pre_processor, TextPreprocessor)
    assert isinstance(mock_sparse_index_from_config.config, SparseIndexConfig)
    assert mock_sparse_index_from_config.index is None
    assert mock_sparse_index_from_config.corpus_tokens == {}
    assert mock_sparse_index_from_config.key_mapping == {}
    assert mock_sparse_index_from_config.vocabulary == {}


def test_tokenize_single_text(mock_sparse_index, mock_text_preprocessor):
    tokens1 = mock_sparse_index.tokenize("text one")
    tokens2 = mock_text_preprocessor.process("text one")
    assert tokens1 == tokens2


def test_tokenize_many_texts(mock_sparse_index, mock_docs):
    _, docs = mock_docs
    tokens1 = mock_sparse_index.tokenize(docs)
    tokens2 = [mock_sparse_index.tokenize(doc) for doc in docs]
    assert tokens1 == tokens2


def test_build_index(mock_sparse_index, mock_docs):
    ids, docs = mock_docs

    # Check initial state
    assert len(mock_sparse_index.corpus_tokens) == 0
    assert len(mock_sparse_index.key_mapping) == 0
    assert mock_sparse_index.index is None
    assert len(mock_sparse_index.vocabulary) == 0

    # Build the index
    mock_sparse_index.build_index(ids, docs)

    # Check corpus tokens
    assert len(mock_sparse_index.corpus_tokens) == len(ids)
    for id_, doc in zip(ids, docs):
        assert id_ in mock_sparse_index.corpus_tokens
        assert mock_sparse_index.corpus_tokens[id_] == mock_sparse_index.tokenize(doc)

    # Check key mapping
    assert len(mock_sparse_index.key_mapping) == len(ids)
    for i, id_ in enumerate(ids):
        assert i in mock_sparse_index.key_mapping
        assert mock_sparse_index.key_mapping[i] == id_

    # Check index
    assert mock_sparse_index.index is not None

    # Check vocabulary
    all_tokens = set(token for doc in docs for token in mock_sparse_index.tokenize(doc))
    assert set(mock_sparse_index.vocabulary.keys()) == all_tokens
    assert len(mock_sparse_index.vocabulary) == len(all_tokens)
    assert all(isinstance(idx, int) for idx in mock_sparse_index.vocabulary.values())

    # Test with empty input
    empty_mock_index = type(mock_sparse_index)()
    empty_mock_index.build_index([], [])
    assert len(empty_mock_index.corpus_tokens) == 0
    assert len(empty_mock_index.key_mapping) == 0
    assert empty_mock_index.index is None
    assert len(empty_mock_index.vocabulary) == 0


def exists(index, id, doc):
    tokens = index.tokenize(doc)
    assert id in index.corpus_tokens
    assert index.corpus_tokens[id] == tokens

    assert id in index.key_mapping.values()
    assert all(token in index.vocabulary for token in tokens)


def test_add(mock_sparse_index, mock_docs):
    ids, docs = mock_docs

    # 1. Add one doc
    mock_sparse_index.add(ids[0], docs[0])
    exists(mock_sparse_index, ids[0], docs[0])

    # 2. Add doc A and doc B
    mock_sparse_index.add(ids[1], docs[1])
    mock_sparse_index.add(ids[2], docs[2])
    exists(mock_sparse_index, ids[0], docs[0])
    exists(mock_sparse_index, ids[1], docs[1])
    exists(mock_sparse_index, ids[2], docs[2])

    # 3. Add doc A and doc A again
    initial_state = (
        mock_sparse_index.corpus_tokens.copy(),
        mock_sparse_index.key_mapping.copy(),
        mock_sparse_index.vocabulary.copy(),
    )
    mock_sparse_index.add(ids[1], docs[1])
    assert mock_sparse_index.corpus_tokens == initial_state[0]
    assert mock_sparse_index.key_mapping == initial_state[1]
    assert mock_sparse_index.vocabulary == initial_state[2]


def test_add_with_tokens(mock_sparse_index, mock_sparse_index2):
    doc = "text one two three"
    # Test adding tokens as corpus
    tokens = mock_sparse_index.tokenize(doc)

    mock_sparse_index.add("doc1", tokens)
    exists(mock_sparse_index, "doc1", doc)

    # Verify that adding tokens is equivalent to adding the original doc
    mock_sparse_index_doc = mock_sparse_index2
    mock_sparse_index_doc.add("doc1", doc)

    assert mock_sparse_index.corpus_tokens == mock_sparse_index_doc.corpus_tokens
    assert mock_sparse_index.key_mapping == mock_sparse_index_doc.key_mapping
    assert mock_sparse_index.vocabulary == mock_sparse_index_doc.vocabulary


def test_add_many(mock_sparse_index, mock_sparse_index2, mock_docs):
    ids, docs = mock_docs

    # Add documents using add_many
    mock_sparse_index.add_many(ids, docs)

    # Verify all documents are added correctly
    for id, doc in zip(ids, docs):
        exists(mock_sparse_index, id, doc)

    # Create a new index and add documents one by one
    individual_index = mock_sparse_index2
    for id, doc in zip(ids, docs):
        individual_index.add(id, doc)

    # Verify that add_many produces the same result as multiple add calls
    assert mock_sparse_index.corpus_tokens == individual_index.corpus_tokens
    assert mock_sparse_index.key_mapping == individual_index.key_mapping
    assert mock_sparse_index.vocabulary == individual_index.vocabulary

    # Test adding empty list (should not raise an error)
    mock_sparse_index.add_many([], [])

    # Test adding with mismatched ids and docs lengths
    with pytest.raises(ValueError):
        mock_sparse_index.add_many(ids, docs[:-1])


def test_add_many_with_tokens(mock_sparse_index, mock_sparse_index2, mock_docs):
    ids, docs = mock_docs

    # Tokenize the documents
    tokenized_docs = [mock_sparse_index.tokenize(doc) for doc in docs]

    # Add documents using add_many with tokens
    mock_sparse_index.add_many(ids, tokenized_docs)

    # Create a new index and add documents one by one using strings
    string_index = mock_sparse_index2
    for id, doc in zip(ids, docs):
        string_index.add(id, doc)

    # Verify that add_many with tokens produces the same result as multiple add calls with strings
    assert mock_sparse_index.corpus_tokens == string_index.corpus_tokens
    assert mock_sparse_index.key_mapping == string_index.key_mapping
    assert mock_sparse_index.vocabulary == string_index.vocabulary

    # Test adding empty list of tokens (should not raise an error)
    mock_sparse_index.add_many([], [])

    # Test adding with mismatched ids and tokenized_docs lengths
    with pytest.raises(ValueError):
        mock_sparse_index.add_many(ids, tokenized_docs[:-1])


def test_remove(mock_sparse_index, mock_docs):
    ids, docs = mock_docs

    # Add all documents to the index
    mock_sparse_index.add_many(ids, docs)

    # Verify all documents are in the index
    for id in ids:
        assert id in mock_sparse_index.corpus_tokens
        assert id in mock_sparse_index.key_mapping.values()

    # Remove the first document
    mock_sparse_index.remove(ids[0])

    # Verify the first document is removed
    assert ids[0] not in mock_sparse_index.corpus_tokens
    assert ids[0] not in mock_sparse_index.key_mapping.values()

    # Verify other documents are still in the index
    for id in ids[1:]:
        assert id in mock_sparse_index.corpus_tokens
        assert id in mock_sparse_index.key_mapping.values()

    # Remove a non-existent document (should not raise an error)
    mock_sparse_index.remove("non_existent_doc")

    # Remove all remaining documents
    for id in ids[1:]:
        mock_sparse_index.remove(id)

    # Verify all documents are removed
    assert len(mock_sparse_index.corpus_tokens) == 0
    assert len(mock_sparse_index.key_mapping) == 0
    assert len(mock_sparse_index.vocabulary) == 0


def test_remove_many(mock_sparse_index, mock_docs):
    ids, docs = mock_docs

    # Add all documents to the index
    mock_sparse_index.add_many(ids, docs)

    # Verify all documents are in the index
    for id in ids:
        assert id in mock_sparse_index.corpus_tokens
        assert id in mock_sparse_index.key_mapping.values()

    # Remove half of the documents
    half_ids = ids[: len(ids) // 2]
    mock_sparse_index.remove_many(half_ids)

    # Verify removed documents are not in the index
    for id in half_ids:
        assert id not in mock_sparse_index.corpus_tokens
        assert id not in mock_sparse_index.key_mapping.values()

    # Verify remaining documents are still in the index
    for id in ids[len(ids) // 2 :]:
        assert id in mock_sparse_index.corpus_tokens
        assert id in mock_sparse_index.key_mapping.values()

    # Remove remaining documents
    mock_sparse_index.remove_many(ids[len(ids) // 2 :])

    # Verify all documents are removed
    assert len(mock_sparse_index.corpus_tokens) == 0
    assert len(mock_sparse_index.key_mapping) == 0
    assert len(mock_sparse_index.vocabulary) == 0

    # Test removing non-existent documents (should not raise an error)
    mock_sparse_index.remove_many(["non_existent_doc1", "non_existent_doc2"])


def test_search(mock_sparse_index_with_predefined_results, mock_queries_and_results):
    queries, expected_results = mock_queries_and_results
    mock_index = mock_sparse_index_with_predefined_results

    for query, expected_result in zip(queries, expected_results):
        results = mock_index.search(query)

        # Assert that the results are of the correct type
        assert isinstance(results, SearchResults)
        assert isinstance(results.keys, list)
        assert isinstance(results.scores, np.ndarray)

        # Assert that the results are not empty
        assert len(results.keys) > 0
        assert len(results.scores) > 0

        # Assert that the number of keys and scores match
        assert len(results.keys) == len(results.scores)

        # Assert that the results are sorted by score in descending order
        assert np.all(results.scores[:-1] >= results.scores[1:])

        # Assert that the results match the expected order
        assert results.keys == expected_result

    # Test search with non-existent query
    non_existent_query = "nonexistent"
    non_existent_results = mock_index.search(non_existent_query)
    assert len(non_existent_results.keys) == 0
    assert len(non_existent_results.scores) == 0


def test_search_with_tokens(
    mock_sparse_index_with_predefined_results, mock_queries_and_results
):
    queries, _ = mock_queries_and_results
    mock_index = mock_sparse_index_with_predefined_results

    for query in queries:
        # Perform a search using a string query
        string_results = mock_index.search(query)

        # Perform a search using tokenized query
        query_tokens = mock_index.tokenize(query)
        token_results = mock_index.search(query_tokens)

        # Assert that searching with tokens gives the same result as searching with a string
        assert token_results.keys == string_results.keys
        np.testing.assert_array_almost_equal(
            token_results.scores, string_results.scores
        )


def test_search_many(
    mock_sparse_index_with_predefined_results, mock_queries_and_results
):
    queries, expected_results = mock_queries_and_results
    mock_index = mock_sparse_index_with_predefined_results

    # Perform search_many
    many_results = mock_index.search_many(queries)

    # Perform individual searches
    individual_results = [mock_index.search(query) for query in queries]

    # Assert that the number of results from search_many matches the number of queries
    assert len(many_results) == len(queries)

    # Compare results from search_many with individual searches
    for many_result, individual_result, expected_result in zip(
        many_results, individual_results, expected_results
    ):
        # Assert that the keys are the same
        assert many_result.keys == individual_result.keys == expected_result

        # Assert that the scores are almost equal (allowing for small floating-point differences)
        np.testing.assert_array_almost_equal(
            many_result.scores, individual_result.scores
        )

        # Assert that the normalized flag is the same
        assert many_result.normalized == individual_result.normalized

    # Test with empty query list
    empty_results = mock_index.search_many([])
    assert len(empty_results) == 0

    # Test with a mix of string and tokenized queries
    mixed_queries = [queries[0], mock_index.tokenize(queries[1])]
    mixed_results = mock_index.search_many(mixed_queries)
    assert len(mixed_results) == len(mixed_queries)
    assert mixed_results[0].keys == expected_results[0]
    assert mixed_results[1].keys == expected_results[1]


def test_search_many_with_tokens(
    mock_sparse_index_with_predefined_results, mock_queries_and_results
):
    queries, expected_results = mock_queries_and_results
    mock_index = mock_sparse_index_with_predefined_results

    # Tokenize the queries
    tokenized_queries = [mock_index.tokenize(query) for query in queries]

    # Perform search_many with tokenized queries
    token_results = mock_index.search_many(tokenized_queries)

    # Perform search_many with string queries
    string_results = mock_index.search_many(queries)

    # Assert that the number of results matches the number of queries
    assert len(token_results) == len(queries)
    assert len(string_results) == len(queries)

    # Compare results from tokenized queries with string queries
    for token_result, string_result, expected_result in zip(
        token_results, string_results, expected_results
    ):
        # Assert that the keys are the same
        assert token_result.keys == string_result.keys == expected_result

        # Assert that the scores are almost equal (allowing for small floating-point differences)
        np.testing.assert_array_almost_equal(token_result.scores, string_result.scores)

        # Assert that the normalized flag is the same
        assert token_result.normalized == string_result.normalized

    # Test with a mix of string and tokenized queries
    mixed_queries = [queries[0], mock_index.tokenize(queries[1]), queries[2]]
    mixed_results = mock_index.search_many(mixed_queries)
    assert len(mixed_results) == len(mixed_queries)

    # Compare mixed results with expected results
    for result, expected_result in zip(mixed_results, expected_results):
        assert result.keys == expected_result


def test_search_weighted(
    mock_sparse_index_with_predefined_weighted_results,
    mock_weighted_queries_and_results,
):
    index = mock_sparse_index_with_predefined_weighted_results
    weighted_queries, expected_results = mock_weighted_queries_and_results

    for (queries, weights), expected in zip(weighted_queries, expected_results):
        results = index.search_weighted(queries, weights, top_k=3)
        assert results.keys == expected[:3]
        assert len(results.scores) == 3
        assert np.all(
            np.diff(results.scores) <= 0
        )  # Ensure scores are in descending order

    # Test with a query not in the predefined list
    random_query = ["random", "query"]
    random_weights = [0.5, 0.5]
    random_results = index.search_weighted(random_query, random_weights, top_k=3)
    assert len(random_results.keys) == 0
    assert len(random_results.scores) == 0


def test_normalize_scores_and_top_k(mock_sparse_index):
    # Create a mock index with predefined scores
    mock_sparse_index.build_index(
        ["doc1", "doc2", "doc3", "doc4"],
        ["text one", "text two", "text three", "text four"],
    )

    mock_scores, mock_ranks = np.array([1.0, 0.8, 0.6, 0.4]), np.array([0, 1, 2, 3])
    mock_sparse_index._get_top_k = lambda token_ids, token_weights=None, top_k=100: (
        mock_scores[: min(top_k, 4)],
        mock_ranks[: min(top_k, 4)],
    )

    # Test search with normalization
    mock_sparse_index.config.normalize_scores = True
    results = mock_sparse_index.search("text one", top_k=3)

    assert len(results.keys) == 3
    assert len(results.scores) == 3
    assert results.normalized

    expected_scores = mock_sparse_index.normalize_scores(2, mock_scores[:3])
    np.testing.assert_array_almost_equal(results.scores, expected_scores, decimal=4)

    # Test search without normalization
    mock_sparse_index.config.normalize_scores = False
    results = mock_sparse_index.search("text one", top_k=3)

    assert len(results.keys) == 3
    assert len(results.scores) == 3
    assert not results.normalized
    np.testing.assert_array_almost_equal(results.scores, mock_scores[:3], decimal=4)

    # Test top_k parameter
    results = mock_sparse_index.search("text one", top_k=2)
    assert len(results.keys) == 2
    assert len(results.scores) == 2

    # Test when top_k is larger than the number of documents
    results = mock_sparse_index.search("text one", top_k=10)
    assert len(results.keys) == 4
    assert len(results.scores) == 4


def test_empty_query(mock_sparse_index):
    results = mock_sparse_index.search("")
    assert len(results.keys) == 0
    assert len(results.scores) == 0
    assert results.normalized == mock_sparse_index.config.normalize_scores


def test_save_and_load(mock_sparse_index):
    import os
    import tempfile

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_sparse_index.build_index(["doc1", "doc2"], ["text one", "text two"])

        # Save the index
        save_path = os.path.join(tmp_dir, "test-index")
        mock_sparse_index.save(save_path)

        # Verify that the file was written
        assert os.path.exists(save_path)

        # Load the index
        loaded_index = MockSparseIndex.load(save_path)

        # Verify the loaded index
        assert loaded_index.name == mock_sparse_index.name
        assert loaded_index.key_mapping == mock_sparse_index.key_mapping
        assert loaded_index.corpus_tokens == mock_sparse_index.corpus_tokens
        assert loaded_index.vocabulary == mock_sparse_index.vocabulary
        assert loaded_index.index == mock_sparse_index.index
        assert loaded_index.config == mock_sparse_index.config
