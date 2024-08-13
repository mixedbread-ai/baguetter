import pytest
from ranx.data_structures import Report

from baguetter.evaluation.datasets import BaseDataset
from baguetter.evaluation.eval import (EvalResult, EvalResults, create_metrics,
                                       evaluate_retrievers)
from tests.mock_utils.index import MockSparseIndex


class MockDataset(BaseDataset):
    def __init__(self, queries, relevant_docs):
        super().__init__()
        self._query_ids = [f"q{i}" for i in range(len(queries))]
        self._queries = queries
        self._relevant_docs = relevant_docs
        self._corpus = list(set(doc for docs in relevant_docs for doc in docs))
        self._doc_ids = [f"doc{i}" for i in range(len(self._corpus))]
        self._qrels = self._create_qrels()

    def _create_qrels(self):
        qrels = {}
        for q_id, rel_docs in zip(self._query_ids, self._relevant_docs):
            qrels[q_id] = {doc_id: 1 for doc_id in rel_docs}
        return qrels

    @property
    def name(self) -> str:
        return "MockDataset"

    def get_corpus(self) -> tuple[list[str], list[str]]:
        return self._doc_ids, self._corpus

    def get_queries(self) -> tuple[list[str], list[str]]:
        return self._query_ids, self._queries

    def get_qrels(self) -> dict[str, dict[str, int]]:
        return self._qrels


@pytest.fixture
def sample_data():
    queries = ["query1", "query2"]
    relevant_docs = [["doc1", "doc2"], ["doc2", "doc3"]]
    retriever_results = {
        "query1": [("doc1", 0.9), ("doc3", 0.7), ("doc2", 0.5)],
        "query2": [("doc2", 0.8), ("doc3", 0.6), ("doc1", 0.4)],
    }
    return queries, relevant_docs, retriever_results


def test_create_metrics():
    metrics = create_metrics(["ndcg", "precision"], [1, 5])
    assert set(metrics) == {"ndcg@1", "ndcg@5", "precision@1", "precision@5"}


def test_evaluate_retrievers(sample_data):
    queries, relevant_docs, retriever_results = sample_data
    dataset = MockDataset(queries, relevant_docs)

    results = evaluate_retrievers(
        [dataset],
        {"mock_retriever": lambda: MockSparseIndex()},
        metrics=create_metrics(["ndcg", "precision"], [1, 3]),
        top_k=3,
    )

    assert isinstance(results, EvalResults)
    assert len(results.results) == 1
    assert "MockDataset" in results.results

    dataset_result = results.results["MockDataset"]
    assert isinstance(dataset_result, EvalResult)
    assert isinstance(dataset_result.report, Report)
    assert "mock_retriever" in dataset_result.report.to_dict()["model_names"]

    model_scores = dataset_result.report.to_dict()["mock_retriever"]["scores"]
    assert "ndcg@1" in model_scores
    assert "ndcg@3" in model_scores
    assert "precision@1" in model_scores
    assert "precision@3" in model_scores

    assert "index_time" in dataset_result.timings["mock_retriever"]
    assert "search_time" in dataset_result.timings["mock_retriever"]
