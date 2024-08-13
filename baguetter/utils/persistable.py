from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from baguetter.utils.file_repository import AbstractFileRepository, HuggingFaceFileRepository, LocalFileRepository


class Persistable(ABC):
    """Abstract base class for objects that can be persisted to and loaded from storage.

    This class defines the interface for saving and loading objects, providing
    a consistent way to handle persistence across different storage mechanisms.
    """

    @classmethod
    @abstractmethod
    def _load(
        cls,
        name_or_path: str,
        repository: AbstractFileRepository,
        *,
        allow_pickle: bool = True,
        mmap: bool = False,
    ) -> Any:
        """Load an object from storage.

        Args:
            name_or_path (str): Name or path of the object to load.
            repository (AbstractFileRepository): File repository to load from.
            allow_pickle (bool, optional): Whether to allow loading pickled objects. Defaults to True.
            mmap (bool, optional): Whether to memory-map the file. Defaults to False.

        Returns:
            Any: The loaded object.

        """

    @abstractmethod
    def _save(self, repository: AbstractFileRepository, path: str | None) -> None:
        """Save the object to storage.

        Args:
            repository (AbstractFileRepository): File repository to save to.
            path (str | None): Path to save the object to.

        """

    def save(self, path: str | None = None) -> None:
        """Save the object to a local file repository.

        Args:
            path (str, optional): Path to save the object to.

        """
        repository = LocalFileRepository()
        if path:
            directory = path.rsplit("/", 1)[0]
            repository.mkdirs(directory, exist_ok=True)
        self._save(repository=repository, path=path)

    @classmethod
    def load(cls, name_or_path: str, *, mmap: bool = False) -> Any:
        """Load an object from a local file repository.

        Args:
            name_or_path (str): Name or path of the object to load.
            mmap (bool, optional): Whether to memory-map the file. Defaults to False.

        Returns:
            Any: The loaded object.

        """
        repository = LocalFileRepository()
        return cls._load(
            name_or_path=name_or_path,
            repository=repository,
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
        name_or_path: str,
        *,
        repo_type: str | None = None,
        token: str | None = None,
        mmap: bool = False,
        **kwargs,
    ) -> Any:
        """Load an object from the Hugging Face Hub.

        Args:
            repo_id (str): Repository ID.
            name_or_path (str): Name or path of the object.
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

        return cls._load(name_or_path, repository, mmap=mmap)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        path_in_repo: str | None = None,
        private: bool = True,
        repo_type: str | None = None,
        token: str | None = None,
        **kwargs,
    ) -> None:
        """Save an object to the Hugging Face Hub.

        Args:
            repo_id (str): Repository ID.
            path_in_repo (str, optional): Custom path within the repository. Defaults to None.
            private (bool, optional): Whether the repository is private. Defaults to True.
            repo_type (str, optional): Repository type. Defaults to None.
            token (str, optional): Hugging Face API token. Defaults to None.
            **kwargs: Additional arguments for HuggingFaceFileRepository.

        """
        repository = HuggingFaceFileRepository(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            token=token,
            create=True,
            **kwargs,
        )

        self._save(repository, path_in_repo)
