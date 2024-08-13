from __future__ import annotations

from datasets import Dataset, load_dataset

from .base import BaseDataset


class HFDataset(BaseDataset):
    """A class to handle HuggingFace datasets for evaluation purposes."""

    def __init__(self, name: str, split: str = "test") -> None:
        """Initialize the HFDataset.

        Args:
            name (str): The name of the dataset.
            split (str, optional): The split of the dataset to use. Defaults to "test".

        """
        self.dataset_name: str = name
        self.split_name: str = split

    @property
    def name(self) -> str:
        """Get the name of the dataset."""
        return self.dataset_name

    def _load_dataset(self, config: str, split: str) -> Dataset:
        """Load a dataset from HuggingFace.

        Args:
            config (str): The configuration of the dataset.
            split (str): The split of the dataset to load.

        Returns:
            Dataset: The loaded dataset.

        """
        return load_dataset(self.dataset_name, config, split=split)

    def get_corpus(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Get the corpus from the dataset.

        Returns
        -------
            Tuple[Tuple[str, ...], Tuple[str, ...]]: A tuple containing document IDs and texts.

        """
        ds_corpus = self._load_dataset("corpus", "corpus")
        df_queries = ds_corpus.to_pandas()
        if "title" in df_queries.columns:
            df_queries["combined_text"] = df_queries["title"] + " " + df_queries["text"]
            return tuple(df_queries["_id"]), tuple(df_queries["combined_text"])
        return tuple(df_queries["_id"]), tuple(df_queries["text"])

    def get_queries(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Get the queries from the dataset.

        Returns
        -------
            Tuple[Tuple[str, ...], Tuple[str, ...]]: A tuple containing query IDs and texts.

        """
        ds_queries = self._load_dataset("queries", "queries")
        df_queries = ds_queries.to_pandas()
        if "title" in df_queries.columns:
            df_queries["combined_text"] = df_queries["title"] + " " + df_queries["text"]
            return tuple(df_queries["_id"]), tuple(df_queries["combined_text"])
        return tuple(df_queries["_id"]), tuple(df_queries["text"])

    def get_qrels(self) -> dict[str, dict[str, int]]:
        """Get the query relevance judgments (qrels) from the dataset.

        Returns
        -------
            Dict[str, Dict[str, int]]: A dictionary of query relevance judgments.

        """
        ds_default = self._load_dataset("default", self.split_name)
        merged_qrels: dict[str, dict[str, int]] = {}

        for item in ds_default:
            query_id, corpus_id, score = (
                item["query-id"],
                item["corpus-id"],
                int(item["score"]),
            )
            if score > 0:
                merged_qrels.setdefault(query_id, {})[corpus_id] = score

        return merged_qrels
