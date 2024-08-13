from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, TypeVar

from tqdm import tqdm

from baguetter.logger import LOGGER

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

T = TypeVar("T")


@contextmanager
def ensure_import(install_name: str | None = None) -> Generator[None, None, None]:
    """Ensure a module is imported, raising a meaningful exception if it's not available.

    Args:
        install_name (Optional[str]): The name of the module to install if import fails.

    Raises:
        ImportError: If the module cannot be imported, with instructions for installation.

    """
    try:
        yield
    except ImportError as e:
        import re

        module_name = re.search(r"'(.*?)'", str(e))[1]
        install_name = install_name or module_name
        msg = (
            f"Failed to import {module_name}. This is required to use this feature. "
            f"Please install the module using: 'pip install {install_name}'"
        )
        raise ImportError(msg) from e


def batch_iter(
    data: list[T],
    batch_size: int,
    *,
    show_progress: bool = False,
    **kwargs,
) -> Generator[list[T], None, None]:
    """Generate batches of data.

    Args:
        data (List[T]): The data to batch.
        batch_size (int): The size of each batch.
        show_progress (bool): Whether to show a progress bar.
        **kwargs: Additional arguments for tqdm if show_progress is True.

    Yields:
        List[T]: Batches of data.

    """
    for i in tqdm(range(0, len(data), batch_size), disable=not show_progress, **kwargs):
        yield data[i : i + batch_size]


@contextmanager
def try_import(install_name: str | None = None) -> Generator[None, None, None]:
    """Attempt to import a module, logging a message if import fails.

    Args:
        install_name (Optional[str]): The name of the module to install if import fails.

    """
    try:
        yield
    except ImportError as e:
        import re

        module_name = re.search(r"'(.*?)'", str(e))[1]
        install_name = install_name or module_name
        LOGGER.info(
            f"Failed to import {module_name}. This is optional for improved functionality. "
            f"Consider installing it using: 'pip install {install_name}'",
        )


def ensure_dir_exists(file_path: str | pathlib.Path) -> None:
    """Ensure that the directory for the given path exists, creating it if necessary.

    Args:
        file_path (Union[str, pathlib.Path]): The path for which to ensure the directory exists.

    """
    path = pathlib.Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)


def identity_function(x: T) -> T:
    """Return the input unchanged.

    Args:
        x (T): Any input value.

    Returns:
        T: The same input value.

    """
    return x


def map_in_process(
    func: Callable[[T], Any],
    iterable: Iterable[T],
    n_workers: int = 1,
    chunksize: int = 128,
) -> Generator[Any, None, None]:
    """Apply a function to an iterable using a process pool.

    Args:
        func (Callable[[T], Any]): The function to apply.
        iterable (Iterable[T]): The input iterable.
        n_workers (int): The number of worker processes.
        chunksize (int): The size of chunks sent to each worker.

    Yields:
        Any: The results of applying the function to the iterable.

    """
    with Pool(n_workers) as pool:
        yield from pool.imap(func, iterable, chunksize=chunksize)


def map_in_thread(
    func: Callable[[T], Any],
    iterable: Iterable[T],
    n_workers: int = 1,
) -> Generator[Any, None, None]:
    """Apply a function to an iterable using a thread pool executor.

    Args:
        func (Callable[[T], Any]): The function to apply.
        iterable (Iterable[T]): The input iterable.
        n_workers (int): The number of worker threads.

    Yields:
        Any: The results of applying the function to the iterable.

    """
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        yield from pool.map(func, iterable)
