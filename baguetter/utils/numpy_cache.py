from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from baguetter.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable


class NumpyCache:
    """A cache for NumPy arrays with disk and memory storage."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        hash_func: Callable[[Any], str] | None = None,
        use_mmap: bool = False,
    ) -> None:
        """Initialize the NumpyCache.

        Args:
            cache_dir: Directory to store cached NumPy arrays.
            hash_func: Custom hash function for keys. Defaults to SHA-512.
            use_mmap: Whether to use memory-mapping when loading arrays.

        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hash_func = hash_func or self._default_hash_key
        self.use_mmap = use_mmap
        self.memory_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _default_hash_key(key: Any) -> str:
        """Default hash function using SHA-512."""
        return hashlib.sha512(str(key).encode()).hexdigest()

    def get(self, key: Any) -> np.ndarray | None:
        """Retrieve a NumPy array from the cache.

        Args:
            key: The key associated with the array.

        Returns:
            The cached NumPy array if found, else None.

        """
        hashed_key = self.hash_func(key)
        if hashed_key in self.memory_cache:
            return self.memory_cache[hashed_key]

        file_path = self.cache_dir / f"{hashed_key}.npy"
        if file_path.exists():
            array = np.load(file_path, mmap_mode="r" if self.use_mmap else None)
            self.memory_cache[hashed_key] = array
            return array
        return None

    def set(self, key: Any, value: np.ndarray) -> None:
        """Store a NumPy array in the cache.

        Args:
            key: The key to associate with the array.
            value: The NumPy array to cache.

        """
        hashed_key = self.hash_func(key)
        file_path = self.cache_dir / f"{hashed_key}.npy"
        np.save(file_path, value)
        self.memory_cache[hashed_key] = value

    def clear(self) -> None:
        """Clear both disk and memory caches."""
        for file in self.cache_dir.glob("*.npy"):
            file.unlink()
        self.memory_cache.clear()


def numpy_cache(
    cache_dir: str | Path = f"{settings.cache_dir}/numpy_cache",
    *,
    hash_func: Callable[[Any], str] | None = None,
    use_mmap: bool = False,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Decorator for caching NumPy array results of a function.

    Args:
        cache_dir: Directory to store cached NumPy arrays.
        hash_func: Custom hash function for keys. Defaults to SHA-512.
        use_mmap: Whether to use memory-mapping when loading arrays.

    Returns:
        A decorator function.

    """
    cache = NumpyCache(cache_dir=cache_dir, hash_func=hash_func, use_mmap=use_mmap)

    def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
            key = (
                tuple(tuple(arg) if isinstance(arg, list) else arg for arg in args),
                frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items()),
            )
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result)
            return result

        return wrapper

    return decorator
