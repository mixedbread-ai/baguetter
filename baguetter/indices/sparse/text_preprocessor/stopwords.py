from __future__ import annotations

import nltk

supported_languages = {
    "arabic",
    "azerbaijani",
    "basque",
    "bengali",
    "catalan",
    "chinese",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hebrew",
    "hinglish",
    "hungarian",
    "indonesian",
    "italian",
    "kazakh",
    "nepali",
    "norwegian",
    "portuguese",
    "romanian",
    "russian",
    "slovene",
    "spanish",
    "swedish",
    "tajik",
    "turkish",
}


def _get_stopwords(lang: str) -> list[str]:
    nltk.download("stopwords", quiet=True)
    if lang.lower() not in supported_languages:
        msg = f"Stop-words for {lang.capitalize()} are not available."
        raise ValueError(msg)
    return nltk.corpus.stopwords.words(lang)


def get_stopwords(sw_list: str | list[str] | set[str] | bool) -> list[str]:
    if isinstance(sw_list, str):
        return _get_stopwords(sw_list)
    if isinstance(sw_list, list) and all(isinstance(x, str) for x in sw_list):
        return sw_list
    if isinstance(sw_list, set):
        return list(sw_list)
    if sw_list is None:
        return []
    raise NotImplementedError
