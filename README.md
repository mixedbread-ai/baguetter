# Baguetter

Baguetter is a flexible, efficient, and hackable search engine library implemented in Python. It's designed for quickly benchmarking, implementing, and testing new search methods. Baguetter supports sparse (traditional), dense (semantic), and hybrid retrieval methods.

**Note:** Baguetter is not built for production use-cases or scale. For such use-cases, please check out other search engine projects.

Paper: https://arxiv.org/abs/2408.06643

## Features

- Sparse retrieval using BM25 and BMX algorithms
- Dense retrieval using embeddings
- Hybrid retrieval combining sparse and dense methods
- Customizable text preprocessing pipeline
- Multi-threaded indexing and searching
- Evaluation tools for benchmarking
- Easy integration with HuggingFace datasets and models for sharing
- Hackable interface to quickly implement new methods

## Installation

```bash
pip install baguetter
```

## Quick Start

```python
from baguetter.indices import BMXSparseIndex

# Create an index
idx = BMXSparseIndex()

# Add documents
docs = [
  "We all love baguette and cheese",
  "Baguette is a great bread",
  "Cheese is a great source of protein",
  "Baguette is a great source of carbs",
]
doc_ids = ["1", "2", "3", "4"]

idx.add_many(doc_ids, docs, show_progress=True)

# Search
results = idx.search("quick fox")
print(results)

# Search many
results = idx.search_many(["quick fox", "baguette is great"])
print(results)
```

## Evaluation

Baguetter includes tools for evaluating search performance on standard benchmarks:

```python
from baguetter.evaluation import datasets, evaluate_retrievers
from baguetter.indices import BM25SparseIndex, BMXSparseIndex

results = evaluate_retrievers(datasets.mteb_datasets_small, {"bm25": BM25SparseIndex, "bmx": BMXSparseIndex})
results.save("eval_results")
```

## Documentation

For more detailed usage instructions and API documentation, please refer to the [full documentation](https://github.com/mixedbread-ai/baguetter/docs).

## Contributing

Contributions are welcome! We are using the GitHub Pull Request workflow. Either open an issue first and create a PR or include a comprehensive commit message when opening a PR.

To get started, please create a clone of the repo (or a fork). We recommend working in a virtual environment.

```sh
python -m pip install -e ".[dev]"

pre-commit install
```

To test your changes, run:

```sh
pytest
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Baguetter builds upon the work of several open-source projects:

1. [retriv](https://github.com/AmenRa/retriv) by [AmenRa](https://github.com/AmenRa):
   Baguetter is a fork of retriv, adjusting it to our needs.

2. [bm25s](https://github.com/xhluca/bm25s) by [xhluca](https://github.com/xhluca):
   Our BM25 implementation is based on this project, which provides an efficient and effective implementation of the BM25 algorithm with different scoring functions.

3. [USearch](https://github.com/unum-cloud/usearch) by [unum-cloud](https://github.com/unum-cloud) for dense retrival.

4. [ranx](https://github.com/AmenRa/ranx) by [AmenRa](https://github.com/AmenRa) for evaluation.

Please check out the respective repositories and show some appreciation to the authors.

## Citing

```
@article{li2024bmx,
      title={BMX: Entropy-weighted Similarity and Semantic-enhanced Lexical Search},
      author={Xianming Li and Julius Lipp and Aamir Shakir and Rui Huang and Jing Li},
      year={2024},
      eprint={2408.06643},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2408.06643},
}
```
