from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from baguetter.indices.search_engine import EnhancedSearchResults
from baguetter.utils.common import ensure_import
from baguetter.utils.numpy_cache import numpy_cache

with ensure_import("ofen"):
    from ofen.enums import EncodingFormat

if TYPE_CHECKING:
    from ofen.models import CrossEncoder, TextEncoder


def create_embed_fn(
    embedding_model: TextEncoder,
    *,
    query_prompt: str | None = None,
    document_prompt: str | None = None,
    use_caching: bool = True,
    batch_size: int = 32,
):
    from ofen.common.tensor_utils import quantize_embeddings

    def embed_fn(text: list[str], *, is_query: bool = False, show_progress: bool = False):
        if is_query and query_prompt:
            text = [f"{query_prompt}{query}" for query in text]
        elif document_prompt:
            text = [f"{document_prompt}{document}" for document in text]
        return embedding_model.encode_text(text, batch_size=batch_size, show_progress=show_progress).embeddings

    if use_caching:
        embed_fn = numpy_cache()(embed_fn)

    def enocde_fn(
        text: list[str],
        *,
        is_query: bool = False,
        show_progress: bool = False,
        encoding_format: EncodingFormat = EncodingFormat.FLOAT,
    ):
        embeddings = embed_fn(text=text, is_query=is_query, show_progress=show_progress)
        return quantize_embeddings(embeddings, encoding_format=encoding_format)

    return enocde_fn


def create_post_processing_fn(reranking_model: CrossEncoder, batch_size: int = 32):
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
