from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baguetter.types import DTypeLike, ScoringMetric


@dataclasses.dataclass
class DenseIndexConfig:
    """Configuration class for Dense Index.

    This class holds the configuration parameters for a dense index,
    including dimensions, metric type, data type, and various search parameters.
    """

    index_name: str = "new-index"
    embedding_dim: int = -1
    metric: ScoringMetric | None = None
    dtype: DTypeLike | None = None
    normalize_score: bool = True


@dataclasses.dataclass
class UsearchDenseIndexConfig(DenseIndexConfig):
    connectivity: int | None = None
    ef_construction: int | None = None
    ef: int | None = None
    enable_key_lookups: bool = True
    exact_search: bool = False


@dataclasses.dataclass
class FaissDenseIndexConfig(DenseIndexConfig):
    faiss_string: str = "Flat"
