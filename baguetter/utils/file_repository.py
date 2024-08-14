from __future__ import annotations

from pathlib import Path

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import HfFileSystem, HfFileSystemResolvedPath, create_repo, repo_exists

from baguetter.settings import settings


class AbstractFileRepository(AbstractFileSystem):
    """Abstract base class for file repositories."""


class HuggingFaceFileRepository(AbstractFileRepository, HfFileSystem):
    """File repository for interacting with Hugging Face datasets."""

    def __init__(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        private: bool = True,
        create: bool = True,
        token: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize a HuggingFaceFileRepository.

        Args:
            repo_id (str): The repository ID.
            repo_type (Optional[str], optional): The repository type. Defaults to "datasets".
            private (bool, optional): Whether the repository is private. Defaults to True.
            create (bool, optional): Whether to create the repository if it doesn't exist. Defaults to True.
            token (Optional[str], optional): The authentication token. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the repository doesn't exist and create is False.

        """
        super().__init__(token=token, **kwargs)

        if repo_type is None:
            repo_type = "datasets"

        if create:
            res = create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                exist_ok=True,
                repo_type="dataset" if repo_type == "datasets" else repo_type,
                **kwargs,
            )
            repo_id = res.repo_id
        elif not repo_exists(
            repo_id=repo_id,
            repo_type="dataset" if repo_type == "datasets" else repo_type,
            token=token,
            **kwargs,
        ):
            msg = f"Repository not found: {repo_id}"
            raise ValueError(msg)
        self.repo_id = repo_id
        self.repo_type = repo_type

    def resolve_path(
        self,
        path: str,
        revision: str | None = None,
    ) -> HfFileSystemResolvedPath:
        """Resolve a path within the repository.

        Args:
            path (str): The path to resolve.
            revision (Optional[str], optional): The revision to use. Defaults to None.

        Returns:
            HfFileSystemResolvedPath: The resolved path.

        """
        full_path = f"{self.repo_type}/{self.repo_id}/{path}"
        res = super().resolve_path(full_path, revision=revision)
        res.unresolve = lambda: path
        res.path_in_repo = path
        return res


class LocalFileRepository(AbstractFileRepository, LocalFileSystem):
    """File repository for interacting with the local file system."""

    def __init__(self, path: str | None = None, **kwargs) -> None:
        """Initialize a LocalFileRepository.

        Args:
            path (Optional[str], optional): The base path for the repository. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the specified path exists but is not a directory.

        """
        super().__init__(**kwargs)
        self._base_path = str(Path(path or f"{settings.base_path}/repository").resolve())
        if not self.isdir(self._base_path):
            if self.exists(self._base_path):
                msg = f"Path '{self._base_path}' exists but is not a directory."
                raise ValueError(msg)
            self.mkdir(self._base_path, parents=True, exist_ok=True)

    def _strip_protocol(self, path: str) -> str:
        """Strip the protocol from the given path.

        Args:
            path (str): The path to strip the protocol from.

        Returns:
            str: The path with the protocol stripped.

        """
        if path.startswith((".", "/")):
            return super()._strip_protocol(path)
        return f"{self._base_path}/{path}"
