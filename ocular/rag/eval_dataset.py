"""The evaluation questions as an MLflow evaluation dataset.

The question generator registers the set it writes here, and the experiments read
it back from here, so the dataset on the MLflow server is the one source the
evaluation runs on. Each record holds the question as its input, the reference
answer and the source as its expectations, and the split, subspecialty and
category as tags.
"""
from __future__ import annotations

from ocular.rag import questions

NAME = "ocular-questions-full"
EXPERIMENTS = ("ocular-retrieval", "ocular-rag-pipelines")


def records(qs: list[questions.Question]) -> list[dict]:
    """The questions as MLflow evaluation records."""
    return [
        {
            "inputs": {"question": q.question},
            "expectations": {
                "expected_response": q.ground_truth,
                "source_doc_id": q.source_doc_id,
                "evidence": q.evidence,
                "answer_sentences": q.answer_sentences,
                "gold_chunks": q.gold_chunks,
                "gold_match": q.gold_match,
            },
            "tags": {"id": q.id, "split": q.split, "subspecialty": q.subspecialty, "category": q.category},
        }
        for q in qs
    ]


def register(qs: list[questions.Question], tracking_uri: str, replace: bool = False, name: str = NAME):
    """Register the questions as an evaluation dataset, linked to the experiments.

    An existing dataset of the same name is kept unless ``replace`` is set, so a
    new generation run cannot silently change the set past experiments used.

    Returns
    -------
    mlflow.genai.datasets.EvaluationDataset
        The registered dataset.
    """
    import mlflow
    from mlflow.genai import datasets

    mlflow.set_tracking_uri(tracking_uri)
    ids = [mlflow.set_experiment(e).experiment_id for e in EXPERIMENTS]
    existing = datasets.search_datasets(filter_string=f"name = '{name}'")
    if existing:
        if not replace:
            raise FileExistsError(
                f"MLflow dataset {name!r} already exists; pass replace=True (--replace) to overwrite it"
            )
        datasets.delete_dataset(dataset_id=existing[0].dataset_id)
    ds = datasets.create_dataset(
        name=name, experiment_id=ids,
        tags={"questions": len(qs), "writer": questions.QUESTION_MODEL},
    )
    ds.merge_records(records(qs))
    return ds
