from __future__ import annotations

import dataclasses
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from baguetter.enums import FusionAlgorithm
from baguetter.indices.base import SearchResults
from baguetter.utils.numpy_utils import min_max_normalization

if TYPE_CHECKING:
    from collections.abc import Callable

    from baguetter.fuser.config import FuserConfig
    from baguetter.types import Key


def _reciprocal_rank_fusion(
    results: list[SearchResults],
    _: list[float] | None = None,
) -> dict[Key, float]:
    fused_run = defaultdict(float)
    for search_results in results:
        for rank, (doc_id, score) in enumerate(
            zip(search_results.keys, search_results.scores),
        ):
            fused_run[doc_id] += 1 / (rank + score)
    return fused_run


def _weighted_fusion(
    results: list[SearchResults],
    weights: list[float] | None = None,
) -> dict[Key, float]:
    weights = weights or [1.0] * len(results)
    fused_run = defaultdict(float)
    for weight, search_results in zip(weights, results):
        for doc_id, score in zip(search_results.keys, search_results.scores):
            fused_run[doc_id] += weight * score
    return fused_run


def _weighted_reciprocal_rank_fusion(
    results: list[SearchResults],
    weights: list[float] | None = None,
) -> dict[Key, float]:
    weights = weights or [1.0] * len(results)
    fused_run = defaultdict(float)
    for weight, search_results in zip(weights, results):
        for i, (doc_id, score) in enumerate(
            zip(search_results.keys, search_results.scores),
        ):
            fused_run[doc_id] += (1 / (i + score)) * weight
    return fused_run


def _combsum(
    results: list[SearchResults],
    _: list[float] | None = None,
) -> dict[Key, float]:
    fused_run = defaultdict(float)
    for search_results in results:
        for doc_id, score in zip(search_results.keys, search_results.scores):
            fused_run[doc_id] += score
    return fused_run


def _combmnz(
    results: list[SearchResults],
    _: list[float] | None = None,
) -> dict[Key, float]:
    fused_run = defaultdict(float)
    doc_count = defaultdict(int)
    for search_results in results:
        for doc_id, score in zip(search_results.keys, search_results.scores):
            fused_run[doc_id] += score
            doc_count[doc_id] += 1
    for doc_id in fused_run:
        fused_run[doc_id] *= doc_count[doc_id]
    return fused_run


def _borda_count(
    results: list[SearchResults],
    _: list[float] | None = None,
) -> dict[Key, float]:
    fused_run = defaultdict(float)
    for search_results in results:
        n = len(search_results.keys)
        for rank, doc_id in enumerate(search_results.keys):
            fused_run[doc_id] += n - rank
    return fused_run


def _z_score(
    results: list[SearchResults],
    _: list[float] | None = None,
) -> dict[Key, float]:
    fused_run = defaultdict(float)
    for search_results in results:
        scores = np.array(search_results.scores)
        z_scores = (scores - np.mean(scores)) / np.std(scores)
        for doc_id, z_score in zip(search_results.keys, z_scores):
            fused_run[doc_id] += z_score
    return fused_run


def _isr(
    results: list[SearchResults],
    _: list[float] | None = None,
) -> dict[Key, float]:
    fused_run = defaultdict(float)
    for search_results in results:
        for rank, doc_id in enumerate(search_results.keys, start=1):
            fused_run[doc_id] += 1 / (rank**2)
    return fused_run


def _median_rank(
    results: list[SearchResults],
    _: list[float] | None = None,
) -> dict[Key, float]:
    all_docs = set().union(*[set(sr.keys) for sr in results])
    ranks = defaultdict(list)
    for search_results in results:
        for rank, doc_id in enumerate(search_results.keys, start=1):
            ranks[doc_id].append(rank)
    return {doc_id: np.median(ranks[doc_id]) for doc_id in all_docs}


_fusion_algorithms: dict[FusionAlgorithm, Callable] = {
    FusionAlgorithm.WEIGHTED: _weighted_fusion,
    FusionAlgorithm.WEIGHTED_RECIPROCAL_RANK: _weighted_reciprocal_rank_fusion,
    FusionAlgorithm.RECIPROCAL_RANK: _reciprocal_rank_fusion,
    FusionAlgorithm.COMB_SUM: _combsum,
    FusionAlgorithm.COMB_MNZ: _combmnz,
    FusionAlgorithm.BORDA_COUNT: _borda_count,
    FusionAlgorithm.Z_SCORE: _z_score,
    FusionAlgorithm.ISR: _isr,
    FusionAlgorithm.MEDIAN_RANK: _median_rank,
}


class Fuser:
    def __init__(
        self,
        weights: list[float] | None = None,
        algorithm: str | FusionAlgorithm = FusionAlgorithm.RECIPROCAL_RANK,
    ) -> None:
        self.weights = weights
        self.algorithm = FusionAlgorithm(algorithm)
        self.fusion_func = _fusion_algorithms[self.algorithm]

    @staticmethod
    def _normalize_scores(results: list[SearchResults]) -> None:
        for search_results in results:
            if not search_results.normalized:
                search_results.scores = min_max_normalization(search_results.scores)
                search_results.normalized = True

    def merge(self, results: list[SearchResults], top_k: int = 100) -> SearchResults:
        self._normalize_scores(results)
        fused_run = self.fusion_func(results, self.weights)
        sorted_fused_run = sorted(
            fused_run.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        keys, scores = zip(*sorted_fused_run)
        normalized_scores = min_max_normalization(
            np.array(scores, dtype=np.float32),
            (0, len(results)),
        )
        return SearchResults(keys=list(keys), scores=normalized_scores, normalized=True)

    def merge_many(
        self,
        runs: list[list[SearchResults]],
        top_k: int = 100,
        n_workers: int | None = None,
    ) -> list[SearchResults]:
        n_workers = n_workers or len(runs)
        merge_func = partial(self.merge, top_k=top_k)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            return list(executor.map(merge_func, runs))

    @classmethod
    def from_config(cls, config: FuserConfig) -> Fuser:
        return cls(**dataclasses.asdict(config))
