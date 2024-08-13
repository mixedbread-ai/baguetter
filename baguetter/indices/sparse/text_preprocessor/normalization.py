from __future__ import annotations

import re
import string
from typing import TYPE_CHECKING

from unidecode import unidecode

if TYPE_CHECKING:
    from collections.abc import Callable

# Translation tables
_SPECIAL_CHARS_TRANS = str.maketrans("‘’´“”–-", "'''\"\"--")  # noqa: RUF001
_PUNCTUATION_TRANSLATION = str.maketrans(
    string.punctuation,
    " " * len(string.punctuation),
)


def lowercasing(text: str) -> str:
    """Convert text to lowercase.

    Args:
        text (str): The input text.

    Returns:
        The normalized text.

    """
    return text.lower()


def normalize_ampersand(text: str) -> str:
    """Replace '&' with 'and'.

    Args:
        text (str): The input text.

    Returns:
        The normalized text.

    """
    return text.replace("&", " and ")


def normalize_diacritics(text: str) -> str:
    """Convert diacritics to ASCII characters.

    Args:
        text (str): The input text.

    Returns:
        The normalized text.

    """
    return unidecode(text)


def normalize_special_chars(text: str) -> str:
    """Normalize special characters.

    Args:
        text (str): The input text.

    Returns:
        The normalized text.

    """
    return text.translate(_SPECIAL_CHARS_TRANS)


def normalize_acronyms(text: str) -> str:
    """Remove periods from acronyms unless followed by a non-space non-period or a digit.

    Args:
        text (str): The input text.

    Returns:
        The normalized text.

    """
    return re.sub(r"\.(?!(\S[^. ])|\d)", "", text)


def remove_punctuation(text: str) -> str:
    """Replace punctuation with spaces.

    Args:
        text (str): The input text.

    Returns:
        The normalized text.

    """
    return text.translate(_PUNCTUATION_TRANSLATION)


def strip_whitespaces(text: str) -> str:
    """Remove extra whitespaces.

    Args:
        text (str): The input text.

    Returns:
        The normalized text.

    """
    return " ".join(text.split())


def remove_empty_tokens(tokens: list[str]) -> list[str]:
    """Remove empty tokens.

    Args:
        tokens (List[str]): The input tokens.

    Returns:
        The normalized tokens.

    """
    return [t for t in tokens if t]


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Remove stopwords.

    Args:
        tokens (List[str]): The input tokens.
        stopwords (Set[str]): The stopwords.

    Returns:
        The normalized tokens.

    """
    return [t for t in tokens if t not in stopwords]


def apply_stemmer(tokens: list[str], stemmer: Callable[[str], str]) -> list[str]:
    """Apply stemmer.

    Args:
        tokens (List[str]): The input tokens.
        stemmer (Callable[[str], str]): The stemmer.

    Returns:
        The normalized tokens.

    """
    return list(map(stemmer, tokens))


def remove_empty(tokens: list[str]) -> list[str]:
    """Remove empty tokens.

    Args:
        tokens (List[str]): The input tokens.

    Returns:
        The normalized tokens.

    """
    return [t for t in tokens if t]
