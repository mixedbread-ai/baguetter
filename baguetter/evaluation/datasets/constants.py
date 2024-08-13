from __future__ import annotations

from baguetter.evaluation.datasets.hf_dataset import HFDataset

TASKS_SMALL: list[str] = [
    "arguana",
    "cqadupstack-android",
    "cqadupstack-english",
    "cqadupstack-gaming",
    "cqadupstack-gis",
    "cqadupstack-mathematica",
    "cqadupstack-physics",
    "cqadupstack-programmers",
    "cqadupstack-stats",
    "cqadupstack-tex",
    "cqadupstack-unix",
    "cqadupstack-webmasters",
    "cqadupstack-wordpress",
    "fiqa",
    "trec-covid",
    "nfcorpus",
    "scifact",
    "scidocs",
    "touche2020",
]

TASKS_BIG: list[str] = [
    "dbpedia",
    "msmarco",
    "nq",
    "climate-fever",
    "fever",
    "hotpotqa",
    "quora",
    "touche2020",
]


def create_mteb_datasets(tasks: list[str], prefix: str = "mteb/") -> list[HFDataset]:
    """Create a list of HFDataset objects for given tasks.

    Args:
        tasks (List[str]): List of task names.
        prefix (str): Prefix to be added to each task name. Defaults to "mteb/".

    Returns:
        List[HFDataset]: List of HFDataset objects.

    """
    return [HFDataset(f"{prefix}{task}") for task in tasks]


mteb_datasets_small: list[HFDataset] = create_mteb_datasets(TASKS_SMALL)
mteb_datasets_big: list[HFDataset] = [
    *create_mteb_datasets(TASKS_BIG),
    HFDataset("mteb/msmarco", split="dev"),
]

mteb_datasets: list[HFDataset] = mteb_datasets_small + mteb_datasets_big
