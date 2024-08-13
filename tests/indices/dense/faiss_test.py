from operator import index
import tempfile

import numpy as np
import pytest

from baguetter.indices.base import SearchResults
from baguetter.indices.dense.faiss import FaissDenseIndex
from baguetter.utils.file_repository import LocalFileRepository


@pytest.fixture
def sample_data():
    np.random.seed(42)
    keys = [f"key_{i}" for i in range(100)]
    values = [np.random.rand(128).astype(np.float32) for _ in range(100)]
    return keys, values


def test_faiss_add_and_search(sample_data):
    keys, values = sample_data
    index = FaissDenseIndex(embedding_dim=128)
    index.add_many(keys, values)

    assert index.size == 100

    query = np.random.rand(128).astype(np.float32)
    results = index.search(query, top_k=5)
    assert isinstance(results, SearchResults)
    assert len(results.keys) == 5
    assert len(results.scores) == 5


def test_faiss_exact_search(sample_data):
    keys, values = sample_data
    index = FaissDenseIndex(embedding_dim=128)
    index.add_many(keys, values)

    query = np.random.rand(128).astype(np.float32)
    results = index.search(query, top_k=5)
    assert len(results.keys) == 5
    assert len(results.scores) == 5


def test_faiss_duplicate_keys(sample_data):
    all_keys, all_values = sample_data
    keys, values = all_keys[:5], all_values[:5]  # Use only first 5 items for this test
    index = FaissDenseIndex(embedding_dim=128)

    # Add initial data
    index.add_many(keys, values)
    assert index.size == 5

    for key, value in zip(keys, values):
        results = index.search(value, top_k=1)
        assert results.keys[0] == key

    # Try to add duplicate keys with new values
    new_values = all_values[5:10]
    index.add_many(keys, new_values)
    assert (
        index.size == 5
    )  # Size should remain the same as duplicate keys are overwritten

    # Verify that new values are searchable
    for key, new_value in zip(keys, new_values):
        results_new = index.search(new_value, top_k=1)
        assert results_new.keys[0] == key  # New value should be found


def test_faiss_delete(sample_data):
    keys, values = sample_data
    keys, values = keys[:10], values[:10]  # Use only first 10 items for this test
    index = FaissDenseIndex(embedding_dim=128)

    # Add initial data
    index.add_many(keys, values)
    assert index.size == 10

    # Delete some keys
    keys_to_delete = keys[:5]
    index.remove_many(keys_to_delete)
    assert index.size == 5

    # Verify deleted keys are no longer in the index
    for key, value in zip(keys_to_delete, values[:5]):
        results = index.search(value, top_k=1)
        assert key not in results.keys

    # Verify remaining keys are still in the index
    for key, value in zip(keys[5:], values[5:]):
        results = index.search(value, top_k=1)
        assert key in results.keys


def test_faiss_save_and_load(sample_data):
    keys, values = sample_data
    index = FaissDenseIndex(embedding_dim=128)
    index.add_many(keys, values)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = "faiss_index"
        repo = LocalFileRepository(tmpdir)
        index._save(repository=repo, path=save_path)

        loaded_index = FaissDenseIndex._load(save_path, repository=repo)

        assert loaded_index.size == index.size
        assert loaded_index.config.__dict__ == index.config.__dict__

        query = np.random.rand(128).astype(np.float32)
        original_results = index.search(query, top_k=5)
        loaded_results = loaded_index.search(query, top_k=5)

        assert original_results.keys == loaded_results.keys
        np.testing.assert_allclose(original_results.scores, loaded_results.scores)


def test_faiss_add_and_search_many(sample_data):
    keys, values = sample_data
    index = FaissDenseIndex(embedding_dim=128)
    index.add_many(keys[:50], values[:50])

    # Add more items
    index.add_many(keys[50:], values[50:])
    assert index.size == 100

    # Search many
    queries = [np.random.rand(128).astype(np.float32) for _ in range(5)]
    results = index.search_many(queries, top_k=3)

    assert len(results) == 5
    for result in results:
        assert len(result.keys) == 3
        assert len(result.scores) == 3


def test_faiss_remove_nonexistent_keys(sample_data):
    keys, values = sample_data
    index = FaissDenseIndex(embedding_dim=128)
    index.add_many(keys[:50], values[:50])

    initial_size = index.size
    non_existent_keys = ["non_existent_1", "non_existent_2"]
    index.remove_many(non_existent_keys)

    assert index.size == initial_size  # Size should not change


def test_faiss_add_with_text_input():
    def mock_embed_fn(texts, is_query, show_progress=False):
        return np.random.rand(len(texts), 128).astype(np.float32)

    index = FaissDenseIndex(embedding_dim=128, embed_fn=mock_embed_fn)
    keys = ["key1", "key2", "key3"]
    text_values = ["This is a test", "Another test", "Yet another test"]

    index.add_many(keys, text_values)
    assert index.size == 3

    results = index.search("This is a query", top_k=2)
    assert len(results.keys) == 2
    assert len(results.scores) == 2


def test_faiss_initialization_options():
    # Test different initialization options for FaissDenseIndex

    # Test Flat index (default)
    flat_index = FaissDenseIndex(embedding_dim=128)
    assert flat_index.config.faiss_string == "Flat"

    # Test IVF index
    ivf_index = FaissDenseIndex(embedding_dim=128, faiss_string="IVF50,Flat")
    assert ivf_index.config.faiss_string == "IVF50,Flat"

    # Test IVFPQ index
    ivfpq_index = FaissDenseIndex(embedding_dim=128, faiss_string="IVF50,PQ8x8")
    assert ivfpq_index.config.faiss_string == "IVF50,PQ8x8"

    # Test PQ index
    pq_index = FaissDenseIndex(embedding_dim=128, faiss_string="PQ16x8")
    assert pq_index.config.faiss_string == "PQ16x8"

    # Test HNSW index
    hnsw_index = FaissDenseIndex(embedding_dim=128, faiss_string="HNSW32")
    assert hnsw_index.config.faiss_string == "HNSW32"

    # Test LSH index
    lsh_index = FaissDenseIndex(embedding_dim=128, faiss_string="LSH")
    assert lsh_index.config.faiss_string == "LSH"

    # Test if indices are properly initialized and can perform basic operations
    keys = [f"key_{i}" for i in range(300)]
    values = [np.random.rand(128).astype(np.float32) for _ in range(300)]

    for index in [flat_index, ivf_index, ivfpq_index, pq_index, hnsw_index, lsh_index]:
        if index.require_training():
            index.train(values)

        index.add_many(keys, values)
        assert index.size == 300

        query = np.random.rand(128).astype(np.float32)
        results = index.search(query, top_k=5)
        assert len(results.keys) == 5
        assert len(results.scores) == 5


def test_faiss_invalid_initialization():
    # Test invalid faiss_string
    with pytest.raises(RuntimeError):
        FaissDenseIndex(embedding_dim=128, faiss_string="InvalidString")


def test_l2_metric(sample_data):
    index = FaissDenseIndex(embedding_dim=128)
    query_vector = sample_data[1][0]

    index.add_many(sample_data[0][1:], sample_data[1][1:])
    results = index.search(query_vector)

    l2_distances = (
        np.linalg.norm(np.array(sample_data[1][1:]) - query_vector, axis=1)
    ) ** 2
    l2_distances = 1 / (1 + l2_distances)
    l2_distances = np.sort(-l2_distances)

    np.testing.assert_allclose(results.scores, -l2_distances, atol=1e-5)


def test_inner_product_metric(sample_data):
    index = FaissDenseIndex(embedding_dim=128)
    index.faiss_index.metric_type = 0  # Inner product

    query_vector = sample_data[1][0]

    index.add_many(sample_data[0][1:], sample_data[1][1:])
    results = index.search(query_vector)

    inner_products = np.dot(sample_data[1][1:], query_vector)
    inner_products = np.sort(-inner_products)

    np.testing.assert_allclose(results.scores, -inner_products, atol=1e-5)
