import sys
import time

from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset
from tqdm import tqdm

from baguetter.enums import FusionAlgorithm
from baguetter.fuser.fuser import Fuser
from baguetter.indices import BM25SparseIndex, BMXSparseIndex

if len(sys.argv) != 3:
    print("usage: python %.py model dataset")
    sys.exit()


def jaccard(query: list, aug_query: list):
    A = set(query)
    B = set(aug_query)
    return len(A.intersection(B)) / len(A)


topk = 100
model_name = sys.argv[1]
dataset_name = f"mteb/{sys.argv[2]}"
split_name = "dev" if dataset_name in ["mteb/msmarco"] else "test"

print(f"model_name={model_name}")
print(f"dataset_name={dataset_name}")
print(f"split_name={split_name}")

ds_qug = load_dataset(
    "mixedbread-ai/augmented-queries", f"mteb_{sys.argv[2]}", split="queries"
)
aug_map = {}
for obj in ds_qug:
    aug_map[obj["text"]] = obj["augmented_queries"]

print("aug size:", len(aug_map))

MODEL = BMXSparseIndex if model_name == "bmx" else BM25SparseIndex
print("MODEL:", MODEL)
try:
    searcher = MODEL._load(f"{dataset_name}-index")
except Exception:
    searcher = MODEL(f"{dataset_name}-bm25x-index")

    # 1. index
    ds_corpus = load_dataset(dataset_name, "corpus", split="corpus")
    ids, docs = [], []
    for obj in tqdm(ds_corpus):
        ids.append(obj["_id"])
        docs.append(obj["title"] + "\n" + obj["text"])
    print("doc size:", len(ids))

    s_time = time.time()
    searcher.add_many(ids, docs)
    print("index time:", time.time() - s_time)

# 2. prepare
ds_queries = load_dataset(dataset_name, "queries", split="queries")
id2query = {}
for obj in ds_queries:
    id2query[obj["_id"]] = obj["text"]

# 3. search
ds_default = load_dataset(dataset_name, "default", split=split_name)
queries = set()
qrels = {}
for obj in tqdm(ds_default):
    query_id = obj["query-id"]
    query = id2query[query_id]
    queries.add((query_id, query))
    if obj["score"] > 0:
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][obj["corpus-id"]] = int(obj["score"])


print("searching...")
s_time = time.time()
pred_results = {}
queries = sorted(list(queries), key=lambda x: x[0])
for i, (query_id, query) in enumerate(queries):
    """
    if i % 10 == 0:
        print()
    else:
        print(query)
    """
    rets = []
    ret = searcher.search(query, top_k=topk)
    rets.append(ret)
    weights = [1.0]
    if query in aug_map:
        # print('augmenting...')
        assert isinstance(aug_map[query], list)
        for aug_query in aug_map[query][:5]:
            rets.append(searcher.search(aug_query, top_k=topk))
            # weights.append(jaccard(query.lower().split(), aug_query.lower().split()))
            weights.append(0.3)
    if len(rets) > 1:
        # print(weights)
        ret = Fuser(weights=weights, algorithm=FusionAlgorithm.WEIGHTED).merge(
            rets, top_k=topk
        )
    pred_results[query_id] = {
        doc_id: float(score) for doc_id, score in zip(ret.keys, ret.scores)
    }
print("search time:", time.time() - s_time)

# 4. evaluate
ndcg, map_score, recall, precision = EvaluateRetrieval.evaluate(
    qrels, pred_results, k_values=[1, 3, 5, 10, 100]
)
print("ndcg:", ndcg)
print("map:", map_score)
print("recall:", recall)
print("precision:", precision)
acc = EvaluateRetrieval.evaluate_custom(qrels, pred_results, [3, 5, 10], metric="acc")
print("acc:", acc)
