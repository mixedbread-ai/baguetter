# ruff: noqa
from __future__ import annotations

import dataclasses
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import ranx
from openpyxl import Workbook
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

from baguetter.indices import USearchDenseIndex
from baguetter.utils.common import ensure_dir_exists
from baguetter.utils.numpy_cache import numpy_cache

if TYPE_CHECKING:
    from baguetter.evaluation.datasets.base import BaseDataset
    from baguetter.indices.base import BaseIndex


def create_metrics(metrics: list[str], steps: list[int]) -> list[str]:
    """Create a list of metrics with specified steps.

    Args:
        metrics (List[str]): List of metric names.
        steps (List[int]): List of step values.

    Returns:
        List[str]: Combined list of metrics with steps.

    """
    return [f"{metric}@{step}" for metric in metrics for step in steps]


@contextmanager
def timer(message: str):
    """Context manager to time execution of a block of code.

    Args:
        message (str): Message to display before and after timing.

    Yields:
    ------
        Callable[[], float]: Function to get elapsed time.

    """
    print(f"Starting {message}...")
    start = time.time()
    yield lambda: time.time() - start
    print(f"{message} took {time.time() - start:.2f} seconds")


@dataclasses.dataclass
class EvalResult:
    """Stores evaluation results for a single dataset.

    Attributes
    ----------
        qrels (ranx.Qrels): Query relevance judgments.
        runs (List[ranx.Run]): List of evaluation runs.
        report (ranx.data_structures.Report): Evaluation report.
        timings (Dict[str, Dict[str, float]]): Timing information.

    """

    qrels: ranx.Qrels
    runs: list[ranx.Run]
    report: ranx.data_structures.Report
    timings: dict[str, dict[str, float]]

    def save(self, result_dir: str) -> None:
        """Save evaluation results to files.

        Args:
            result_dir (str): Directory to save results.

        """
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        for run in self.runs:
            run.save(f"{result_dir}/{run.name}.txt")
        self.report.save(f"{result_dir}/report.json")
        with open(f"{result_dir}/timings.json", "w") as f:
            json.dump(self.timings, f, indent=4)


@dataclasses.dataclass
class EvalResults:
    """Stores evaluation results for multiple datasets.

    Attributes
    ----------
        results (Dict[str, EvalResult]): Mapping of dataset names to evaluation results.

    """

    results: dict[str, EvalResult] = dataclasses.field(default_factory=dict)

    def add(self, dataset_name: str, eval_result: EvalResult) -> None:
        """Add evaluation result for a dataset.

        Args:
            dataset_name (str): Name of the dataset.
            eval_result (EvalResult): Evaluation result for the dataset.

        """
        self.results[dataset_name] = eval_result

    def save(self, result_dir: str = "eval_results", fmt: Literal["csv", "excel"] = "excel") -> None:
        """Save evaluation results to files.

        Args:
            result_dir (str): Directory to save results. Defaults to "eval_results".
            fmt (Literal["csv", "excel"]): Format to save results. Defaults to "excel".

        Raises:
            ValueError: If an unsupported format is specified.

        """
        if fmt not in ["csv", "excel"]:
            msg = "Unsupported format. Use 'csv' or 'excel'."
            raise ValueError(msg)

        for ds, result in self.results.items():
            result.save(f"{result_dir}/{ds}")

        file_extension = "xlsx" if fmt == "excel" else "csv"
        full_path = f"{result_dir}/report.{file_extension}"
        save_method = self._save_excel if fmt == "excel" else self._save_csv
        save_method(full_path)

        self._save_json(f"{result_dir}/result.json")

    def _save_csv(self, path: str) -> None:
        """Save results to a CSV file."""
        import csv

        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            self._write_data(writer.writerow)

    def _save_excel(self, path: str) -> None:
        """Save results to an Excel file."""
        wb = Workbook()
        ws = wb.active
        self._write_data(ws.append)
        wb.save(path)

    def _save_json(self, path: str) -> None:
        """Save results to a JSON file."""
        data = {
            "metrics": self._get_metrics_data(),
            "timings": self._get_timings_data(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def _write_data(self, write_func: Callable[[list], None]) -> None:
        """Write data using the provided write function."""
        metrics = self._get_all_metrics()
        self._write_metrics(write_func, metrics)
        self._write_timings(write_func)

    def _write_metrics(self, write_func: Callable[[list], None], metrics: list[str]) -> None:
        """Write metric data using the provided write function."""
        write_func(["Dataset", "Model", *metrics])
        for dataset_name, eval_result in self.results.items():
            report = eval_result.report.to_dict()
            for model in report["model_names"]:
                row = [dataset_name, model]
                row.extend([f"{report[model]['scores'].get(metric, 'N/A'):.4f}" for metric in metrics])
                write_func(row)
        write_func([])
        write_func([])

    def _write_timings(self, write_func: Callable[[list], None]) -> None:
        """Write timing data using the provided write function."""
        write_func(["Dataset", "Model", "Index Time", "Search Time"])
        for dataset_name, eval_result in self.results.items():
            for model, timing in eval_result.timings.items():
                row = [
                    dataset_name,
                    model,
                    f"{timing.get('index_time', 'N/A'):.4f}",
                    f"{timing.get('search_time', 'N/A'):.4f}",
                ]
                write_func(row)

    def _get_all_metrics(self) -> list[str]:
        """Get a sorted list of all unique metrics across all results."""
        return sorted(metric for result in self.results.values() for metric in result.report.to_dict()["metrics"])

    def _get_metrics_data(self) -> list[dict[str, str]]:
        """Get metrics data for all datasets and models."""
        metrics = self._get_all_metrics()
        data = []
        for dataset_name, eval_result in self.results.items():
            report = eval_result.report.to_dict()
            for model in report["model_names"]:
                row = {"Dataset": dataset_name, "Model": model}
                row.update({metric: f"{report[model]['scores'].get(metric, 'N/A'):.4f}" for metric in metrics})
                data.append(row)
        return data

    def _get_timings_data(self) -> list[dict[str, str]]:
        """Get timing data for all datasets and models."""
        return [
            {
                "dataset": dataset_name,
                "model": model,
                "index_time": f"{timing.get('index_time', 'N/A'):.4f}",
                "search_time": f"{timing.get('search_time', 'N/A'):.4f}",
            }
            for dataset_name, eval_result in self.results.items()
            for model, timing in eval_result.timings.items()
        ]


def evalaute_retriever(
    datasets: list[BaseDataset],
    retriever: BaseIndex,
    metrics: list[str] = create_metrics(["ndcg", "precision", "mrr"], [1, 5, 10]),
    top_k: int = 100,
    ignore_identical_ids: bool = True,
) -> EvalResult:
    """Evaluate a retriever on a dataset.

    Args:
        datasets (List[BaseDataset]): List of datasets to evaluate on.
        retriever (BaseIndex): Retriever to evaluate.
        metrics (List[str]): List of metrics to evaluate.
        top_k (int): Number of top results to consider.
        ignore_identical_ids (bool): Whether to ignore identical document IDs.

    Returns:
        EvalResult: Evaluation result for the retriever.

    """
    return evaluate_retrievers(
        datasets=datasets,
        retriever_factories={retriever.name: lambda: retriever},
        metrics=metrics,
        top_k=top_k,
        ignore_identical_ids=ignore_identical_ids,
    )


def evaluate_retrievers(
    datasets: list[BaseDataset],
    retriever_factories: dict[str, Callable[[], BaseIndex]],
    metrics: list[str] = create_metrics(["ndcg", "precision", "mrr"], [1, 5, 10]),
    top_k: int = 100,
    ignore_identical_ids: bool = True,
) -> EvalResults:
    """Evaluate multiple retrievers on multiple datasets.

    Args:
        datasets (List[BaseDataset]): List of datasets to evaluate on.
        retriever_factories (Dict[str, Callable[[], BaseIndex]]): Dictionary of retriever factory functions.
        metrics (List[str]): List of metrics to evaluate. Defaults to NDCG, Precision, and MRR at 1, 5, and 10.
        top_k (int): Number of top results to consider. Defaults to 100.
        ignore_identical_ids (bool): Whether to ignore identical document IDs. Defaults to True.
        post_process_fn (Optional[Callable[[List[SearchResults]], List[SearchResults]]]): Optional post-processing function for search results.

    Returns:
        EvalResults: Evaluation results for all datasets and retrievers.

    """
    print("Evaluating ", len(retriever_factories), "retrievers...")
    print("---------------------------------------------------------------")
    print("Datasets: ", [dataset.name for dataset in datasets])
    print("Top K: ", top_k)
    print("Metrics: ", metrics)
    print("Ignore identical IDs: ", ignore_identical_ids)
    results = EvalResults()
    for dataset in datasets:
        print("\nEvaluating Dataset:", dataset.name)
        print("---------------------------------------------------------------")
        qrels_dict = dataset.get_qrels()
        doc_ids, doc_texts = dataset.get_corpus()
        query_ids, queries = dataset.get_queries()

        qrels = ranx.Qrels.from_dict(qrels_dict)
        runs = []
        timings = {}

        # Filter out queries that aren't in qrels
        for retriever_name, retriever_factory in retriever_factories.items():
            run, timing = _evaluate_single_retriever(
                retriever_name=retriever_name,
                retriever_factory=retriever_factory,
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                query_ids=query_ids,
                queries=queries,
                top_k=top_k,
                ignore_identical_ids=ignore_identical_ids,
            )
            runs.append(run)
            timings[retriever_name] = timing
        report = ranx.compare(qrels, runs, metrics, make_comparable=True)
        print("\nReport (rounded):")
        print("---------------------------------------------------------------")
        print(f"{report}")
        results.add(dataset.name, EvalResult(qrels, runs, report, timings))

    return results


def _evaluate_single_retriever(
    retriever_name: str,
    retriever_factory: Callable[[], BaseIndex],
    doc_ids: list[str],
    doc_texts: list[str],
    query_ids: list[str],
    queries: list[str],
    top_k: int,
    ignore_identical_ids: bool = True,
) -> tuple[ranx.Run, dict[str, float]]:
    """Evaluate a single retriever on a dataset.

    Args:
        retriever_name (str): Name of the retriever.
        retriever_factory (Callable[[], BaseIndex]): Factory function to create the retriever.
        doc_ids (List[str]): List of document IDs.
        doc_texts (List[str]): List of document texts.
        query_ids (List[str]): List of query IDs.
        queries (List[str]): List of query texts.
        top_k (int): Number of top results to consider.
        ignore_identical_ids (bool): Whether to ignore identical document IDs. Defaults to True.

    Returns:
        Tuple[ranx.Run, Dict[str, float]]: Evaluation run and timing information.

    """
    run = ranx.Run(name=retriever_name)
    timing = {}

    retriever = retriever_factory()

    with timer(f"Adding {len(doc_ids)} documents to {retriever_name}") as index_time:
        retriever.add_many(doc_ids, doc_texts, show_progress=True)
        timing["index_time"] = index_time()

    with timer(f"Searching {len(queries)} queries with {retriever_name}") as search_time:
        search_results = retriever.search_many(queries, top_k=top_k, show_progress=True)
        timing["search_time"] = search_time()

    res_doc_ids = []
    scores = []
    for sr, qid in zip(search_results, query_ids):
        if ignore_identical_ids:
            filtered_indices = [i for i, pid in enumerate(sr.keys) if pid != qid]
            sr.keys = [sr.keys[i] for i in filtered_indices]
            sr.scores = sr.scores[filtered_indices]

        res_doc_ids.append(sr.keys)
        scores.append(sr.scores)

    run.add_multi(query_ids, res_doc_ids, scores)
    return run, timing
