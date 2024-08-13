from __future__ import annotations

import tempfile

import numpy as np
import pytest

from baguetter.indices.dense.base import _INDEX_PREFIX
from baguetter.indices.dense.config import DenseIndexConfig
from baguetter.indices.dense.usearch import USearchDenseIndex
from baguetter.utils.file_repository import LocalFileRepository


@pytest.fixture
def sample_data():
    keys = [f"key_{i}" for i in range(10)]
    values = [np.random.rand(128).astype(np.float32) for _ in range(10)]
    return keys, values


def test_usearch_index_creation():
    index = USearchDenseIndex(embedding_dim=128)
    assert index.config.embedding_dim == 128


def test_usearch_add_and_search(sample_data):
    keys, values = sample_data
    index = USearchDenseIndex(embedding_dim=128)

    # Test adding single item
    index.add(keys[0], values[0])
    assert index.key_counter == 1

    # Test adding multiple items
    index.add_many(keys[1:], values[1:])
    assert index.key_counter == 10

    # Test search
    query = np.random.rand(128).astype(np.float32)
    results = index.search(query, top_k=5)
    assert len(results.keys) == 5
    assert len(results.scores) == 5
    assert results.normalized is True


def test_usearch_remove(sample_data):
    keys, values = sample_data
    index = USearchDenseIndex(embedding_dim=128)
    index.add_many(keys, values)

    # Remove a single item
    index.remove(keys[0])
    assert index.key_counter == 10  # key_count doesn't decrease on removal

    # Attempt to remove non-existent key
    index.remove("non_existent_key")  # Should not raise an error


def test_usearch_search_many(sample_data):
    keys, values = sample_data
    index = USearchDenseIndex(embedding_dim=128)
    index.add_many(keys, values)

    queries = [np.random.rand(128).astype(np.float32) for _ in range(3)]
    results = index.search_many(queries, top_k=5)
    assert len(results) == 3
    for result in results:
        print(result)
        print(len(result.keys))
        print(len(result.scores))
        print(result.normalized)
        assert len(result.keys) == 5
        assert len(result.scores) == 5
        assert result.normalized is True


def test_usearch_from_config():
    config = DenseIndexConfig(
        index_name="test_index",
        embedding_dim=256,
        metric="cosine",
        normalize_score=True,
    )
    index = USearchDenseIndex.from_config(config)
    assert index.config.embedding_dim == 256
    assert index.config.metric == "cosine"
    assert index.config.normalize_score is True


def test_usearch_save_load(sample_data):
    keys, values = sample_data
    index = USearchDenseIndex(embedding_dim=128)
    index.add_many(keys, values)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = "usearch_new-index"
        repository = LocalFileRepository(tmp_dir)

        # Save the index
        index._save(path=save_path, repository=repository)

        # Load the index
        loaded_index = USearchDenseIndex._load(path=save_path, repository=repository)

        assert loaded_index.config.embedding_dim == 128
        assert loaded_index.key_counter == 10

        # Verify search results are the same
        query = np.random.rand(128).astype(np.float32)
        original_results = index.search(query, top_k=5)
        loaded_results = loaded_index.search(query, top_k=5)

        assert original_results.keys == loaded_results.keys
        np.testing.assert_array_almost_equal(
            original_results.scores, loaded_results.scores
        )

        # Verify file names
        assert repository.exists(save_path)
        assert repository.exists(f"{_INDEX_PREFIX}{save_path}")


def test_usearch_embed_function():
    def mock_embed_fn(texts, is_query, show_progress=False):
        return np.random.rand(len(texts), 128).astype(np.float32)

    index = USearchDenseIndex(embedding_dim=128, embed_fn=mock_embed_fn)

    # Test adding text
    index.add("key1", "This is a test document")
    assert index.key_counter == 1

    # Test searching with text query
    results = index.search("This is a test query", top_k=5)
    assert len(results.keys) == 1
    assert len(results.scores) == 1


def test_usearch_exact_search(sample_data):
    keys, values = sample_data
    index = USearchDenseIndex(embedding_dim=128, exact_search=True)
    index.add_many(keys, values)

    query = np.random.rand(128).astype(np.float32)
    results = index.search(query, top_k=5, exact_search=True)
    assert len(results.keys) == 5
    assert len(results.scores) == 5


def test_usearch_duplicate_keys(sample_data):
    all_keys, all_values = sample_data
    keys, values = all_keys[:5], all_values[:5]  # Use only first 5 items for this test
    index = USearchDenseIndex(embedding_dim=128)

    # Add initial data
    index.add_many(keys, values)
    assert index.size == 5

    for key, value in zip(keys, values):
        results = index.search(value, top_k=1)
        assert results.keys[0] == key

    # Try to add duplicate keys with new values
    new_values = all_values[5:]
    index.add_many(keys, new_values)
    assert index.size == 5  # Count should remain the same

    # Verify that the values were overwritten
    for key, new_value in zip(keys, new_values):
        results = index.search(new_value, top_k=1)
        assert results.keys[0] == key


def test_usearch_remove_nonexistent_keys(sample_data):
    keys, values = sample_data
    keys, values = keys[:5], values[:5]  # Use only first 5 items for this test
    index = USearchDenseIndex(embedding_dim=128)

    # Add initial data
    index.add_many(keys, values)
    assert index.key_counter == 5

    # Try to remove non-existent keys
    non_existent_keys = ["non_existent_1", "non_existent_2"]
    original_count = index.key_counter
    index.remove_many(non_existent_keys)
    assert index.key_counter == original_count  # Count should remain the same

    # Verify all original keys are still in the index
    for key, value in zip(keys, values):
        results = index.search(value, top_k=1)
        assert results.keys[0] == key
