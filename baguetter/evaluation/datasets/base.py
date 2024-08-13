from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Abstract base class for dataset implementations.

    This class defines the interface for dataset classes used in the evaluation process.
    Concrete implementations should inherit from this class and implement all abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the dataset.

        Returns
        -------
            str: The name of the dataset.

        """

    @abstractmethod
    def get_corpus(self) -> tuple[list[str], list[str]]:
        """Retrieve the corpus of documents.

        Returns
        -------
            Tuple[List[str], List[str]]: A tuple containing two lists:
                1. List of document IDs
                2. List of document contents

        """

    @abstractmethod
    def get_queries(self) -> tuple[list[str], list[str]]:
        """Retrieve the queries associated with the dataset.

        Returns
        -------
            Tuple[List[str], List[str]]: A tuple containing two lists:
                1. List of query IDs
                2. List of query texts

        """

    @abstractmethod
    def get_qrels(self) -> dict[str, dict[str, int]]:
        """Retrieve the relevance judgments (qrels) for the dataset.

        Returns
        -------
            Dict[str, Dict[str, int]]: A nested dictionary where:
                - The outer key is the query ID
                - The inner key is the document ID
                - The value is the relevance score

        """
