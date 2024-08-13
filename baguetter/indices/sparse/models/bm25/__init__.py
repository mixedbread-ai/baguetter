"""bm25s: Fast BM25 Implementation.

This module was copied from the below github repository and only sligthly modified.
It provides a fast implementation of the BM25 (Best Matching 25) algorithm,
a ranking function used by search engines to rank matching documents according to
their relevance to a given search query.

Key features:
- Efficient implementation of BM25 algorithm
- Optimized for performance in information retrieval tasks

GitHub repository: https://github.com/xhluca/bm25s
Author: Xing Han Lu
License: MIT License
"""

from .index import build_index, calculate_scores

__all__ = ["build_index", "calculate_scores"]
