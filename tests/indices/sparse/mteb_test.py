import pytest
from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset
from tqdm import tqdm

from baguetter.indices import BM25SparseIndex, BMXSparseIndex


def mteb_eval(model_class, dataset, split_name, top_k=100):
    searcher = model_class(f"{dataset}-bmx-index")

    # 1. index
    ds_corpus = load_dataset(dataset, "corpus", split="corpus")
    ids, docs = [], []
    for obj in tqdm(ds_corpus):
        ids.append(obj["_id"])
        docs.append(obj["title"] + "\n" + obj["text"])
    print("doc size:", len(ids))

    searcher.add_many(ids, docs)

    # 2. prepare
    ds_queries = load_dataset(dataset, "queries", split="queries")
    id2query = {}
    for obj in ds_queries:
        id2query[obj["_id"]] = obj["text"]

    # 3. search
    ds_default = load_dataset(
        dataset, "default", split=split_name, trust_remote_code=True
    )
    queries = set()
    qrels = {}
    for obj in ds_default:
        query_id = obj["query-id"]
        query = id2query[query_id]
        queries.add((query_id, query))
        if obj["score"] > 0:
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][obj["corpus-id"]] = int(obj["score"])

    pred_results = {}
    queries = sorted(list(queries), key=lambda x: x[0])
    for query_id, query in tqdm(queries):
        ret = searcher.search(query, top_k=top_k)
        pred_results[query_id] = {
            doc_id: float(score) for doc_id, score in zip(ret.keys, ret.scores)
        }

    # 4. evaluate
    ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels, pred_results, k_values=[10])
    return ndcg["NDCG@10"]


@pytest.mark.parametrize(
    "model_cls,dataset,split_name,lower_bound",
    [
        (BMXSparseIndex, "mteb/scifact", "test", 0.694),
        (BM25SparseIndex, "mteb/scifact", "test", 0.686),
    ],
)
def test_mteb(model_cls, dataset, split_name, lower_bound):
    assert mteb_eval(model_cls, dataset, split_name) >= lower_bound
