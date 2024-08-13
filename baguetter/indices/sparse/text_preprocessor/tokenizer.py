from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import nltk

from baguetter.utils.common import identity_function

tokenizers_dict = {
    "whitespace": str.split,
    "word": nltk.tokenize.word_tokenize,
    "wordpunct": nltk.tokenize.wordpunct_tokenize,
    "sent": nltk.tokenize.sent_tokenize,
}


def _get_tokenizer(tokenizer: str) -> Callable:
    if tokenizer.lower() not in tokenizers_dict:
        msg = f"Tokenizer {tokenizer} not supported."
        raise ValueError(msg)
    if tokenizer == "punkt":
        nltk.download("punkt", quiet=True)
    return tokenizers_dict[tokenizer.lower()]


def get_tokenizer(tokenizer: str | Callable | bool) -> Callable:
    """Get a tokenizer function based on the provided input.

    Args:
        tokenizer (Union[str, Callable, bool]): The tokenizer specification.
            If str: Name of the tokenizer to use.
            If callable: A custom tokenizer function.
            If None: Returns the identity function.

    Returns:
        callable: The tokenizer function.

    Raises:
        NotImplementedError: If the input is not a string, callable, or None.

    """
    if isinstance(tokenizer, str):
        return _get_tokenizer(tokenizer)
    if callable(tokenizer):
        return tokenizer
    if tokenizer is None:
        return identity_function
    raise NotImplementedError
