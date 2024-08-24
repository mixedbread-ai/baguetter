from __future__ import annotations

import dataclasses

from baguetter.enums import FusionAlgorithm


@dataclasses.dataclass
class FuserConfig:
    weights: list[float] | None = None
    algorithm: str | FusionAlgorithm = FusionAlgorithm.RECIPROCAL_RANK

    def __post_init__(self):
        if isinstance(self.algorithm, str):
            self.algorithm = FusionAlgorithm(self.algorithm)
