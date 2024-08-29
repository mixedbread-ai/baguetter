from __future__ import annotations

from typing import Literal

import numpy as np

from baguetter.indices.search_engine import EnhancedSearchResults
from baguetter.utils.numpy_cache import numpy_cache


def create_embed_fn(
    embedding_model,
    *,
    query_prompt: str | None = None,
    document_prompt: str | None = None,
    truncation_dim: int | None = None,
    use_caching: bool = True,
    batch_size: int = 32,
):
    """
    Wraps an embedding model into an embed function that can be directly used for dense index.
    This function depends on the sentence-transformers library.

    Args:
        embedding_model: The embedding model to be wrapped.
        query_prompt (str, optional): A prompt to be prepended to queries. Defaults to None.
        document_prompt (str, optional): A prompt to be prepended to documents. Defaults to None.
        truncation_dim (int, optional): The dimension to truncate embeddings to. Defaults to None.
        use_caching (bool): Whether to use caching for embeddings. Defaults to True.
        batch_size (int): The batch size for encoding. Defaults to 32.

    Returns:
        function: An encoding function that can be used for dense indexing.
    """
    from sentence_transformers.quantization import quantize_embeddings

    def embed_fn(text: list[str], *, is_query: bool = False, show_progress: bool = False):
        if is_query and query_prompt:
            text = [f"{query_prompt}{query}" for query in text]
        elif document_prompt:
            text = [f"{document_prompt}{document}" for document in text]
        return embedding_model.encode(text, batch_size=batch_size, show_progress_bar=show_progress)

    if use_caching:
        embed_fn = numpy_cache(cache_postfix=embedding_model._first_module().auto_model.name_or_path)(embed_fn)  # noqa: SLF001

    def encode_fn(
        text: list[str],
        *,
        is_query: bool = False,
        show_progress: bool = False,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    ):
        embeddings = embed_fn(text=text, is_query=is_query, show_progress=show_progress)
        if truncation_dim:
            embeddings = embeddings[:, :truncation_dim]

        return quantize_embeddings(embeddings, precision=precision)

    return encode_fn


def create_embed_fn_ofen(
    embedding_model,
    *,
    query_prompt: str | None = None,
    document_prompt: str | None = None,
    truncation_dim: int | None = None,
    use_caching: bool = True,
    batch_size: int = 32,
):
    """
    Wraps an embedding model into an embed function that can be directly used for dense index.
    This function depends on ofen, an experimental project. Use with caution.

    Args:
        embedding_model: The embedding model to be wrapped.
        query_prompt (str, optional): A prompt to be prepended to queries. Defaults to None.
        document_prompt (str, optional): A prompt to be prepended to documents. Defaults to None.
        truncation_dim (int, optional): The dimension to truncate embeddings to. Defaults to None.
        use_caching (bool): Whether to use caching for embeddings. Defaults to True.
        batch_size (int): The batch size for encoding. Defaults to 32.

    Returns:
        function: An encoding function that can be used for dense indexing.
    """
    from ofen.common.tensor_utils import quantize_embeddings
    from ofen.enums import EncodingFormat

    def embed_fn(text: list[str], *, is_query: bool = False, show_progress: bool = False):
        if is_query and query_prompt:
            text = [f"{query_prompt}{query}" for query in text]
        elif document_prompt:
            text = [f"{document_prompt}{document}" for document in text]
        return embedding_model.encode(text, batch_size=batch_size, show_progress=show_progress).embeddings

    if use_caching:
        embed_fn = numpy_cache(cache_postfix=embedding_model.name_or_path)(embed_fn)

    def encode_fn(
        text: list[str],
        *,
        is_query: bool = False,
        show_progress: bool = False,
        encoding_format: EncodingFormat = EncodingFormat.FLOAT,
    ):
        embeddings = embed_fn(text=text, is_query=is_query, show_progress=show_progress)
        if truncation_dim:
            embeddings = embeddings[:, :truncation_dim]

        return quantize_embeddings(embeddings, encoding_format=encoding_format)

    return encode_fn


def create_post_processing_fn(reranking_model, batch_size: int = 32):
    """
    Creates a post-processing function that wraps a reranking model to rerank search candidates.

    This function can be used directly with a search engine to improve the ranking of search results.
    It takes a reranking model and returns a function that can be applied to a list of EnhancedSearchResults.

    Args:
        reranking_model: The reranking model to be used for post-processing.
        batch_size (int): The batch size for reranking. Defaults to 32.

    Returns:
        function: A post-processing function that takes a list of EnhancedSearchResults and returns
                  a list of reranked EnhancedSearchResults.
    """

    def post_process_fn(results: list[EnhancedSearchResults], *, show_progress: bool = False):
        if len(results) == 0:
            return []

        flat_queries = []
        flat_documents = []
        for result in results:
            flat_queries.extend([result.query] * len(result.values))
            flat_documents.extend(result.values)

        reranked = reranking_model.rank(
            flat_queries,
            flat_documents,
            top_k=len(flat_documents),
            sort=False,
            show_progress=show_progress,
            batch_size=batch_size,
        )

        reranked_results = []
        index = 0
        for result in results:
            reranked_values = reranked.results[index : index + len(result.values)]
            sorted_ranked_values = sorted(reranked_values, key=lambda x: x.score, reverse=True)
            new_keys = []
            new_values = []
            new_scores = []
            for i in sorted_ranked_values:
                new_keys.append(result.keys[i.index - index])
                new_values.append(result.values[i.index - index])  # noqa: PD011
                new_scores.append(i.score)
            reranked_result = EnhancedSearchResults(
                keys=new_keys,
                values=new_values,
                scores=np.array(new_scores),
                query=result.query,
                normalized=True,
            )
            reranked_results.append(reranked_result)
            index += len(result.values)
        return reranked_results

    return post_process_fn
