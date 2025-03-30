from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from baguetter.utils.file_repository import HuggingFaceFileRepository

class Persistable(ABC):
    """Abstract base class for objects that can be persisted to and loaded from storage.

    This class defines the interface for saving and loading objects, providing
    a consistent way to handle persistence across different storage mechanisms.
    """

    @classmethod
    @abstractmethod
    def _load(
        cls,
        path: str,
        *,
        allow_pickle: bool = True,
        mmap: bool = False,
    ) -> Any:
        """Load an object from storage.

        Args:
            path (str): Path of the object to load.
            allow_pickle (bool, optional): Whether to allow loading pickled objects. Defaults to True.
            mmap (bool, optional): Whether to memory-map the file. Defaults to False.

        Returns:
            Any: The loaded object.

        """

    @abstractmethod
    def _save(self, path: str) -> str:
        """Save the object to storage.

        Args:
            path (str): Path to save the object to.

        Returns:
            str: Path to the saved object.

        """

    def save(self, path: str) -> str:
        """Save the object to a local file repository.

        Args:
            path (str): Path to save the object to.

        Returns:
            str: Path to the saved object.

        """
        path = self._save(path=path)
        return os.path.basename(path)

    @classmethod
    def load(cls, path: str, *, mmap: bool = False) -> Any:
        """Load an object from a local file repository.

        Args:
            path (str): Path of the object to load.
            mmap (bool, optional): Whether to memory-map the file. Defaults to False.

        Returns:
            Any: The loaded object.

        """
        return cls._load(
            path=path,
            mmap=mmap,
        )


class HuggingFacePersistable(Persistable, ABC):
    """Abstract base class for objects that can be persisted to and loaded from the Hugging Face Hub.

    This class extends the Persistable ABC and provides additional methods for interacting
    with the Hugging Face Hub, such as loading from and pushing to repositories.
    """

    @classmethod
    def load_from_hub(
        cls,
        repo_id: str,
        path: str,
        *,
        repo_type: str | None = None,
        token: str | None = None,
        mmap: bool = False,
        **kwargs,
    ) -> Any:
        """Load an object from the Hugging Face Hub.

        Args:
            repo_id (str): Repository ID.
            path (str): Path of the object.
            repo_type (str, optional): Repository type. Defaults to None.
            token (str, optional): Hugging Face API token. Defaults to None.
            mmap (bool, optional): Whether to memory-map the file. Defaults to False.
            **kwargs: Additional arguments for HuggingFaceFileRepository.

        Returns:
            Any: Loaded object.

        """
        repository = HuggingFaceFileRepository(
            repo_id=repo_id,
            repo_type=repo_type,
            create=False,
            token=token,
            **kwargs,
        )

        return cls._load(path=path, repository=repository, mmap=mmap)

    def push_to_hub(
        self,
        repo_id: str,
        path: str,
        *,
        private: bool = True,
        repo_type: str | None = None,
        token: str | None = None,
        **kwargs,
    ) -> str:
        """Save an object to the Hugging Face Hub.

        Args:
            repo_id (str): Repository ID.
            path (str): Path of the object.
            private (bool, optional): Whether the repository is private. Defaults to True.
            repo_type (str, optional): Repository type. Defaults to None.
            token (str, optional): Hugging Face API token. Defaults to None.
            **kwargs: Additional arguments for HuggingFaceFileRepository.

        Returns:
            str: Path to the saved object.

        """
        repository = HuggingFaceFileRepository(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            token=token,
            create=True,
            **kwargs,
        )

        path = self._save(path=path, repository=repository)
        return repository.info(path)["name"]
