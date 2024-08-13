# Benchmark whole BEIR dataset with BM25 and BMX retrievers

from transformers import AutoTokenizer

from baguetter.evaluation import evaluate_retrievers
from baguetter.evaluation.datasets import mteb_datasets
from baguetter.indices import BM25SparseIndex, BMXSparseIndex
from baguetter.indices.sparse.text_preprocessor import TextPreprocessorConfig

text_process_config = TextPreprocessorConfig(
    custom_tokenizer=AutoTokenizer.from_pretrained(
        "mixedbread-ai/mxbai-embed-large-v1"
    ).tokenize,
)

bmx_idx = BMXSparseIndex(preprocessor_or_config=text_process_config)
bm25_idx = BM25SparseIndex(preprocessor_or_config=text_process_config)

res = evaluate_retrievers(
    mteb_datasets, {"BM25": lambda: bm25_idx, "BMX": lambda: bmx_idx}
)

res.save("res_bert", "csv")
