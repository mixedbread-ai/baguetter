from __future__ import annotations

import dataclasses
import warnings
from functools import partial, reduce
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from baguetter.indices.sparse.text_preprocessor.normalization import (
    apply_stemmer,
    lowercasing,
    normalize_acronyms,
    normalize_ampersand,
    normalize_special_chars,
    remove_empty,
    remove_punctuation,
    remove_stopwords,
    strip_whitespaces,
)
from baguetter.indices.sparse.text_preprocessor.stemmer import get_stemmer
from baguetter.indices.sparse.text_preprocessor.stopwords import get_stopwords
from baguetter.indices.sparse.text_preprocessor.tokenizer import get_tokenizer
from baguetter.utils.common import map_in_process

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable


def create_preprocessing_steps(
    *,
    tokenizer: Callable[[str], list[str]],
    stopwords: set[str] | None = None,
    stemmer: Callable[[str], str] | None = None,
    do_lowercasing: bool = True,
    do_ampersand_normalization: bool = True,
    do_special_chars_normalization: bool = True,
    do_acronyms_normalization: bool = True,
    do_punctuation_removal: bool = True,
) -> list[Callable]:
    """Creates a list of preprocessing steps based on the given parameters.

    Args:
        tokenizer: The tokenizer function.
        stopwords: A set of stopwords to remove.
        stemmer: The stemmer function.
        do_lowercasing: Whether to apply lowercasing.
        do_ampersand_normalization: Whether to normalize ampersands.
        do_special_chars_normalization: Whether to normalize special characters.
        do_acronyms_normalization: Whether to normalize acronyms.
        do_punctuation_removal: Whether to remove punctuation.

    Returns:
        A list of preprocessing functions.

    """
    steps = []

    if do_lowercasing:
        steps.append(lowercasing)
    if do_ampersand_normalization:
        steps.append(normalize_ampersand)
    if do_special_chars_normalization:
        steps.append(normalize_special_chars)
    if do_acronyms_normalization:
        steps.append(normalize_acronyms)
    if do_punctuation_removal:
        steps.append(remove_punctuation)
        if tokenizer == str.split:
            steps.append(strip_whitespaces)

    steps.append(tokenizer)

    if stopwords:
        steps.append(partial(remove_stopwords, stopwords=stopwords))

    if stemmer:
        steps.append(partial(apply_stemmer, stemmer=stemmer))

    steps.append(remove_empty)
    return steps


@dataclasses.dataclass
class TextPreprocessorConfig:
    tokenizer: str = "whitespace"
    stemmer: str | None = "english"
    stopwords: str | set[str] = "english"
    do_lowercasing: bool = True
    do_ampersand_normalization: bool = True
    do_special_chars_normalization: bool = True
    do_acronyms_normalization: bool = True
    do_punctuation_removal: bool = True
    custom_tokenizer: Callable | None = None
    custom_stemmer: Callable | None = None

    def get_tokenizer(self) -> Callable:
        """Returns the tokenizer function, either custom or from predefined options."""
        return self.custom_tokenizer or get_tokenizer(self.tokenizer)

    def get_stemmer(self) -> Callable | None:
        """Returns the stemmer function, either custom or from predefined options."""
        return self.custom_stemmer or get_stemmer(self.stemmer)

    def get_stopwords(self) -> set[str]:
        """Returns the set of stopwords, either from a predefined list or a custom set."""
        if isinstance(self.stopwords, str):
            return set(get_stopwords(self.stopwords))
        return self.stopwords

    def to_dict(self) -> dict[str, Any]:
        """Converts the config to a dictionary for serialization.
        Warns if custom tokenizer or stemmer are present as they can't be serialized.
        """
        if self.custom_tokenizer or self.custom_stemmer:
            warnings.warn("Custom tokenizer or stemmer can not be serialized.", stacklevel=2)

        return {
            "tokenizer": self.tokenizer,
            "stemmer": self.stemmer,
            "stopwords": self.stopwords,
            "do_lowercasing": self.do_lowercasing,
            "do_ampersand_normalization": self.do_ampersand_normalization,
            "do_special_chars_normalization": self.do_special_chars_normalization,
            "do_acronyms_normalization": self.do_acronyms_normalization,
            "do_punctuation_removal": self.do_punctuation_removal,
            "custom_tokenizer": (self.custom_tokenizer.__name__ if self.custom_tokenizer else None),
            "custom_stemmer": (self.custom_stemmer.__name__ if self.custom_stemmer else None),
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> TextPreprocessorConfig:
        """Creates a TextPreprocessorConfig instance from a dictionary."""
        return cls(**config_dict)


class TextPreprocessor:
    def __init__(
        self,
        *,
        tokenizer: str | Callable = "whitespace",
        stemmer: str | Callable | None = "english",
        stopwords: str | set[str] = "english",
        do_lowercasing: bool = True,
        do_ampersand_normalization: bool = True,
        do_special_chars_normalization: bool = True,
        do_acronyms_normalization: bool = True,
        do_punctuation_removal: bool = True,
    ) -> None:
        """Initializes the TextPreprocessor with the given parameters."""
        custom_tokenizer = tokenizer if callable(tokenizer) else None
        custom_stemmer = stemmer if callable(stemmer) else None

        self.config = TextPreprocessorConfig(
            tokenizer=tokenizer,
            stemmer=stemmer,
            stopwords=stopwords,
            do_lowercasing=do_lowercasing,
            do_ampersand_normalization=do_ampersand_normalization,
            do_special_chars_normalization=do_special_chars_normalization,
            do_acronyms_normalization=do_acronyms_normalization,
            do_punctuation_removal=do_punctuation_removal,
            custom_tokenizer=custom_tokenizer,
            custom_stemmer=custom_stemmer,
        )
        self.steps = create_preprocessing_steps(
            tokenizer=self.config.get_tokenizer(),
            stemmer=self.config.get_stemmer(),
            stopwords=self.config.get_stopwords(),
            do_lowercasing=self.config.do_lowercasing,
            do_ampersand_normalization=self.config.do_ampersand_normalization,
            do_special_chars_normalization=self.config.do_special_chars_normalization,
            do_acronyms_normalization=self.config.do_acronyms_normalization,
            do_punctuation_removal=self.config.do_punctuation_removal,
        )

    @classmethod
    def from_config(cls, config: TextPreprocessorConfig) -> TextPreprocessor:
        """Creates a TextPreprocessor instance from a TextPreprocessorConfig."""
        return cls(
            tokenizer=config.custom_tokenizer or config.tokenizer,
            stemmer=config.custom_stemmer or config.stemmer,
            stopwords=config.stopwords,
            do_lowercasing=config.do_lowercasing,
            do_ampersand_normalization=config.do_ampersand_normalization,
            do_special_chars_normalization=config.do_special_chars_normalization,
            do_acronyms_normalization=config.do_acronyms_normalization,
            do_punctuation_removal=config.do_punctuation_removal,
        )

    def _call_steps(self, item: str) -> list[str]:
        """Applies all preprocessing steps to a single text item.

        This method sequentially applies each preprocessing step defined in self.steps
        to the input text item. It uses the reduce function to chain the steps together.

        Args:
            item (str): The input text to preprocess.

        Returns:
            List[str]: The preprocessed text as a list of tokens.

        """
        return reduce(lambda x, step: step(x), self.steps, item)

    def process(self, item: str) -> list[str]:
        """Processes a single text item through all preprocessing steps.

        This method is a wrapper around _call_steps, providing a more intuitive name
        for external use.

        Args:
            item (str): The input text to process.

        Returns:
            List[str]: The processed text as a list of tokens.

        """
        return self._call_steps(item)

    def process_many(
        self,
        items: Iterable[str],
        *,
        n_workers: int = 0,
        chunksize: int = 1_000,
        show_progress: bool = False,
        return_generator: bool = False,
    ) -> Generator[list[str], None, None] | list[list[str]]:
        """Processes multiple text items, optionally using multiple processes.

        Args:
            items (Iterable[str]): An iterable of text items to process.
            n_workers (int): Number of processes to use. Default is 0 (single-threaded).
            chunksize (int): Size of chunks for multiprocessing. Default is 128.
            show_progress (bool): Whether to show a progress bar. Default is False.
            return_generator (bool): Whether to return a generator. Default is False.

        Returns:
            Generator[list[str], None, None] | list[list[str]]: Processed text items as lists of tokens.
        """
        if n_workers <= 0:
            processor = map(self._call_steps, items)
        else:
            processor = map_in_process(
                self._call_steps,
                items,
                n_workers=n_workers,
                chunksize=chunksize,
            )

        if show_progress:
            processor = tqdm(
                processor,
                total=len(items) if hasattr(items, "__len__") else None,
                desc="Tokenization",
            )

        return processor if return_generator else list(processor)

    def __call__(self, item: str) -> list[str]:
        """Allows the TextPreprocessor to be called as a function.

        This method enables the TextPreprocessor instance to be used directly as a
        callable object, providing a convenient shorthand for the process method.

        Args:
            item (str): The input text to process.

        Returns:
            List[str]: The processed text as a list of tokens.

        Example:
            preprocessor = TextPreprocessor(...)
            processed_text = preprocessor("Some input text")

        """
        return self.process(item)
