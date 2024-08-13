# ruff: noqa
from __future__ import annotations

import typer
from typer import Option

from baguetter.evaluation.datasets import mteb_datasets_big, mteb_datasets_small
from baguetter.evaluation.datasets.hf_dataset import HFDataset
from baguetter.evaluation.eval import create_embedding_retriever_factories, create_metrics, evaluate_retrievers
from baguetter.indices import BM25SparseIndex, BMXSparseIndex
from baguetter.logger import LOGGER

app = typer.Typer()


@app.command("eval_embeddings")
def eval_embeddings(
    models: list[str] = Option(..., help="List of model names to evaluate"),
    device_ids: list[int] = Option([], help="List of GPU device IDs to use"),
    encoding_formats: list[str] = Option(["float32"], help="List of encoding formats to use"),
    top_k: int = Option(100, help="Number of top results to retrieve"),
    datasets: list[str] | None = Option(None, help="List of dataset names to evaluate on"),
    mteb_size: list[str] = Option(["small"], help="Use MTEB datasets: 'small' or 'big'"),
    metric_names: list[str] = Option(["ndcg", "precision", "mrr"], help="List of metric names to use"),
    metric_k_values: list[int] = Option([1, 5, 10], help="List of k values for metrics"),
) -> None:
    """Evaluate embedding models on specified datasets using various metrics.

    This function evaluates the performance of embedding models on given datasets
    using specified metrics and parameters.
    """
    model_list = [model.strip() for model in models]
    device_list = device_ids

    if "small" in mteb_size:
        eval_datasets = mteb_datasets_small
    elif "big" in mteb_size:
        eval_datasets = mteb_datasets_big
    elif datasets:
        eval_datasets = [HFDataset(dataset) for dataset in datasets]
    else:
        msg = "Please specify either --mteb-size or --datasets"
        raise ValueError(msg)

    metrics = create_metrics(metric_names, metric_k_values)

    for model in model_list:
        LOGGER.info(f"Evaluating model: {model}")

        retriever_factories, clean_up = create_embedding_retriever_factories(
            model,
            encoding_formats=encoding_formats,
            device_ids=device_list,
        )

        results = evaluate_retrievers(
            datasets=eval_datasets,
            retriever_factories=retriever_factories,
            metrics=metrics,
            top_k=top_k,
        )

        results.save(f"eval_results/dense/{model.replace('/', '_')}")
        clean_up()

    LOGGER.info("Evaluation complete!")


@app.command("eval_sparse")
def eval_sparse(
    models: list[str] = Option(..., help="Comma-separated list of model names to evaluate (bm25, bmx)"),
    top_k: int = Option(100, help="Number of top results to retrieve"),
    datasets: list[str] | None = Option(None, help="List of dataset names to evaluate on"),
    mteb_size: str = Option("small", help="Use MTEB datasets: 'small' or 'big'"),
    metric_names: list[str] = Option(["ndcg", "precision", "mrr"], help="List of metric names to use"),
    metric_k_values: list[int] = Option([1, 5, 10], help="List of k values for metrics"),
) -> None:
    """Evaluate sparse retrieval models on specified datasets using various metrics.

    This function evaluates the performance of sparse retrieval models (BM25, BMX)
    on given datasets using specified metrics and parameters.
    """
    model_list = [model.strip().lower() for model in models]

    valid_models = {"bm25", "bmx"}
    if not all(model in valid_models for model in model_list):
        msg = f"Invalid model(s) specified. Choose from: {', '.join(valid_models)}"
        raise ValueError(msg)

    if mteb_size == "small":
        eval_datasets = mteb_datasets_small
    elif mteb_size == "big":
        eval_datasets = mteb_datasets_big
    elif datasets:
        eval_datasets = [HFDataset(dataset) for dataset in datasets]
    else:
        msg = "Please specify either --mteb-size or --datasets"
        raise ValueError(msg)

    metrics = create_metrics(metric_names, metric_k_values)

    retriever_factories = {
        "bm25": BM25SparseIndex,
        "bmx": BMXSparseIndex,
    }

    results = evaluate_retrievers(
        datasets=eval_datasets,
        retriever_factories={model: retriever_factories[model] for model in model_list},
        metrics=metrics,
        top_k=top_k,
    )

    results.save("eval_results/sparse")

    LOGGER.info("Evaluation complete!")


if __name__ == "__main__":
    app()
