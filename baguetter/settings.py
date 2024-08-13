from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Configuration settings for the Baguetter application."""

    base_path: str = field(
        default_factory=partial(
            os.environ.get,
            "BAGUETTER_BASE_PATH",
            str(Path.home() / ".cache" / "baguetter"),
        ),
    )
    cache_dir: str = field(
        default_factory=partial(
            os.environ.get,
            "BAGUETTER_CACHE_DIR",
            str(Path.home() / ".cache" / "baguetter"),
        ),
    )

    def __post_init__(self) -> None:
        """Ensure directories exist and have proper permissions."""
        for path in [self.base_path, self.cache_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
            Path(path).chmod(0o700)  # Restrict access to owner only

    @classmethod
    def from_env(cls, env_file: str | None = None) -> Settings:
        """Create Settings from environment variables, optionally loading from a file."""
        if env_file:
            load_dotenv(env_file)
        return cls()


settings = Settings.from_env()
