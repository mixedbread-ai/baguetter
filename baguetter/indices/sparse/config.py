from __future__ import annotations

import dataclasses
from typing import Any

from baguetter.indices.sparse.text_preprocessor.text_processor import TextPreprocessorConfig


@dataclasses.dataclass
class SparseIndexConfig:
    """Configuration for Sparse Index."""

    index_name: str = "new-index"
    preprocessor_config: TextPreprocessorConfig = dataclasses.field(default_factory=TextPreprocessorConfig)
    min_df: int = 1
    b: float = 0.75
    k1: float = 1.2
    delta: float = 0.5
    method: str = "lucene"
    idf_method: str = "lucene"
    dtype: str = "float32"
    int_dtype: str = "int32"
    alpha: float | None = None
    beta: float | None = None
    normalize_scores: bool = False

    def __post_init__(self) -> None:
        """Post-initialization method to ensure preprocessor_config is of the correct type."""
        if isinstance(self.preprocessor_config, dict):
            self.preprocessor_config = TextPreprocessorConfig(**self.preprocessor_config)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> SparseIndexConfig:
        """Create a SparseIndexConfig instance from a dictionary.

        Args:
            config_dict (dict[str, Any]): Dictionary containing configuration parameters.

        Returns:
            SparseIndexConfig: An instance of SparseIndexConfig.
        """
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the SparseIndexConfig instance to a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the configuration.
        """
        return dataclasses.asdict(self)
