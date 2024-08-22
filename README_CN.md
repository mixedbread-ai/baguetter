# Baguetter
- [English](README.md)

Baguetter是一个Python实现的灵活、高效、可定制的搜索引擎库。它旨在快速对新搜索方法进行基准测试(Benchmark)、实现和测试。Baguetter支持稀疏(Sparse Retrieval)、密集(Dense Retrieval)和混合检索(Hybrid Retrieval)方法。

**注意：** Baguetter并非为生产环境或大规模使用而设计。如需此类使用场景，请查看其他搜索引擎项目。

BMX论文: https://arxiv.org/abs/2408.06643

## 功能特点

- 基于BM25和BMX算法的稀疏检索
- 基于向量的密集检索
- 融合稀疏和密集方法的混合检索
- 自定义文本预处理
- 支持高性能的并发检索
- 提供丰富的基准测试评估工具
- 集成HuggingFace数据集和模型API
- 面向扩展的接口，便于快速实现新检索方法

## 安装

```bash
pip install baguetter
```

## 快速入门

```python
from baguetter.indices import BMXSparseIndex

# 创建索引
idx = BMXSparseIndex()

# 添加文档
docs = [
  "我们都爱法棍和奶酪",
  "法棍是一种很棒的面包",
  "奶酪是很好的蛋白质来源",
  "法棍是很好的碳水化合物来源",
]
doc_ids = ["1", "2", "3", "4"]

idx.add_many(doc_ids, docs, show_progress=True)

# 搜索
results = idx.search("敏捷的狐狸")
print(results)

# 批量搜索
results = idx.search_many(["敏捷的狐狸", "法棍很棒"])
print(results)
```

## 评估

Baguetter包含用于在标准基准上评估搜索性能的工具：

```python
from baguetter.evaluation import datasets, evaluate_retrievers
from baguetter.indices import BM25SparseIndex, BMXSparseIndex

results = evaluate_retrievers(datasets.mteb_datasets_small, {"bm25": BM25SparseIndex, "bmx": BMXSparseIndex})
results.save("eval_results")
```

## 文档

有关更详细的使用说明和API文档，请参阅[完整文档](https://github.com/mixedbread-ai/baguetter/docs)。

## 贡献

欢迎贡献代码！我们使用 GitHub Pull Request 工作流。您可以先打开一个 issue，然后创建 PR，或者在打开 PR 时包含详细的提交信息。

要开始，请先克隆（或 fork）该仓库。我们建议在Python虚拟环境中进行开发。

```sh
python -m pip install -e ".[dev]"

pre-commit install
```

要测试您的更改，运行：

```sh
pytest
```

## 许可证

本项目采用 Apache 2.0 许可证 - 详情请参见 [LICENSE](LICENSE) 文件。

## 致谢

Baguetter 基于几个开源项目的工作：

1. [retriv](https://github.com/AmenRa/retriv) by [AmenRa](https://github.com/AmenRa):
   Baguetter是retriv的一个分支项目。

2. [bm25s](https://github.com/xhluca/bm25s) by [xhluca](https://github.com/xhluca):
   我们的 BM25 实现基于该项目，它提供了 BM25 算法的高效实现和不同的评分函数。

3. [USearch](https://github.com/unum-cloud/usearch) by [unum-cloud](https://github.com/unum-cloud) 和 [Faiss](https://github.com/facebookresearch/faiss) by [facebook research](https://github.com/facebookresearch) 用于密集检索。

4. [ranx](https://github.com/AmenRa/ranx) by [AmenRa](https://github.com/AmenRa) 用于评估。

请访问相关仓库并向作者表示感谢。

## 引用
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
