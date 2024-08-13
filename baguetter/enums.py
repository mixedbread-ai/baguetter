from enum import Enum


class NormalizerType(str, Enum):
    """Normalizer type for sparse indices."""

    MIN_MAX = "min-max"
    MAX = "max"
    SUM = "sum"


class FusionAlgorithm(str, Enum):
    """Fusion algorithm for sparse indices."""

    WEIGHTED = "weighted"
    RECIPROCAL_RANK = "reciprocal_rank"
    WEIGHTED_RECIPROCAL_RANK = "weighted_reciprocal_rank"
    COMB_SUM = "comb_sum"
    COMB_MNZ = "comb_mnz"
    BORDA_COUNT = "borda_count"
    Z_SCORE = "z_score"
    ISR = "isr"
    MEDIAN_RANK = "median_rank"
