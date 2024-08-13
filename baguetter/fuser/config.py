from __future__ import annotations

import dataclasses

from baguetter.enums import FusionAlgorithm


@dataclasses.dataclass
class FuserConfig:
    weights: list[float] | None = None
    algorithm: FusionAlgorithm = FusionAlgorithm.RECIPROCAL_RANK
