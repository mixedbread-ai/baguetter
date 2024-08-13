import numpy as np
import pytest

from baguetter.indices import BM25SparseIndex, BMXSparseIndex
from tests.mock_utils.text_preprocessor import MockTextPreprocessor


@pytest.fixture
def mock_text_preprocessor():
    return MockTextPreprocessor()


@pytest.fixture
def sample_docs():
    ids = ["doc1", "doc2", "doc3"]
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be, that is the question",
    ]
    return ids, docs


@pytest.mark.parametrize("index_class", [BMXSparseIndex, BM25SparseIndex])
def test_sparse_index_build_index(index_class, sample_docs, mock_text_preprocessor):
    index = index_class(
        index_name="test-index",
        preprocessor_or_config=mock_text_preprocessor,
    )
    ids, docs = sample_docs
    index.add_many(ids, docs)

    assert index.index is not None
    assert len(index.index.vocabulary) > 0
    assert len(index.key_mapping) == len(ids)


@pytest.mark.parametrize("index_class", [BMXSparseIndex, BM25SparseIndex])
def test_sparse_index_get_top_k(index_class, sample_docs, mock_text_preprocessor):
    index = index_class(
        index_name="test-index",
        preprocessor_or_config=mock_text_preprocessor,
    )

    ids, docs = sample_docs
    index.add_many(ids, docs)

    query = "journey miles"
    tokens = index.tokenize(query)
    token_ids = index.to_token_ids(tokens)

    scores, indices = index._get_top_k(token_ids, top_k=2)

    assert len(scores) == 2
    assert len(indices) == 2
    assert scores[0] >= scores[1]
    assert index.key_mapping[indices[0]] == "doc2"


@pytest.mark.parametrize("index_class", [BMXSparseIndex, BM25SparseIndex])
@pytest.mark.parametrize(
    "method", ["lucene", "robertson", "atire", "bm25l", "bm25plus"]
)
def test_sparse_index_normalize_scores(index_class, method, mock_text_preprocessor):
    index = index_class(
        index_name="test-index",
        preprocessor_or_config=mock_text_preprocessor,
        method=method,
    )

    index.add_many(["doc1"], ["The quick brown fox jumps over the lazy dog"])

    n_tokens = 5
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    normalized_scores = index.normalize_scores(n_tokens, scores)

    assert len(normalized_scores) == len(scores)
    assert np.all(normalized_scores <= scores)
    assert np.all(normalized_scores >= 0)
    assert np.argmax(normalized_scores) == np.argmax(scores)

    # Check relative rank
    assert np.all(np.argsort(normalized_scores) == np.argsort(scores))
