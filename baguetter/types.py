"""Type definitions for the baguetter package."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike as NumpyDTypeLike

# Basic types
Key = str | int
ScoringMetric = str
DTypeLike = NumpyDTypeLike

KeyScore = dict[Key, float]
QRels = dict[Key, KeyScore]

TextOrTokens = str | list[str]
TextOrVector = str | np.ndarray
HybridValue = TextOrTokens | TextOrVector
