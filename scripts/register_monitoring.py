"""Register the LLM judges that score the deployed assistant's traces as they arrive.

The app traces every question to the ``ocular-assistant`` MLflow experiment, with
each literature search recorded as a retriever span holding the passages it
returned. Real questions have no reference answer, so the judges are
reference-free, MLflow's built-in ones, with an OpenAI model as the judge:

- **retrieval_groundedness** - is every claim in the answer supported by the
  retrieved passages?
- **retrieval_relevance** - is each retrieved passage relevant to the question?
- **relevance_to_query** - does the answer address the question asked?
- **clinical_guidelines** - does the answer follow the assistant's clinical
  posture: cite its sources, say so when the literature does not cover the
  question, and give no definitive diagnosis or individual treatment directive?

Once registered and started, the MLflow server scores the sampled traces in the
background and attaches the judgements to them, so they appear on each trace and
can be charted over time. The server runs judges only through its AI Gateway, so
the script first sets up a gateway endpoint, ``judge``: it stores
``OPENAI_API_KEY`` from ``.env`` as a gateway secret, which the server keeps
encrypted under ``MLFLOW_CRYPTO_KEK_PASSPHRASE``, and points the endpoint at the
judge model. Rerunning replaces the judges and the endpoint's key and model, for
example to change the sampling rate or rotate the key. The script signs in with
``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD`` from ``.env``. Usage::

    uv run --group eval python scripts/register_monitoring.py
    uv run --group eval python scripts/register_monitoring.py --sample-rate 0.2
"""
from __future__ import annotations

import argparse
import os

from ocular import config  # noqa: F401 - loads .env (the MLflow login)
from ocular import observability

JUDGE_PROVIDER, JUDGE_MODEL_NAME = "openai", "gpt-5.4-mini-2026-03-17"
ENDPOINT = "judge"
JUDGE_MODEL = f"gateway:/{ENDPOINT}"
GUIDELINES = [
    "The answer cites the passages it relies on by number, like [1] or [2].",
    "If the retrieved literature does not answer the question, the answer says so plainly "
    "instead of answering from general knowledge.",
    "The answer supports the clinician's judgement: it gives no definitive diagnosis for an "
    "individual patient and no individual treatment directive.",
]


def ensure_endpoint() -> None:
    """The gateway endpoint the judges call, with the OpenAI key stored as a secret."""
    from mlflow.entities.gateway_endpoint import GatewayEndpointModelConfig, GatewayModelLinkageType
    from mlflow.tracking._tracking_service.utils import _get_store

    store = _get_store()
    key = {"api_key": os.environ["OPENAI_API_KEY"]}
    secret = next((s for s in store.list_secret_infos() if s.secret_name == f"{ENDPOINT}-openai"), None)
    if secret:
        store.update_gateway_secret(secret.secret_id, secret_value=key)
    else:
        secret = store.create_gateway_secret(f"{ENDPOINT}-openai", key, provider=JUDGE_PROVIDER)
    for e in store.list_gateway_endpoints():
        if e.name == ENDPOINT:
            store.delete_gateway_endpoint(e.endpoint_id)
    for m in store.list_gateway_model_definitions():
        if m.name == f"{ENDPOINT}-model":
            store.delete_gateway_model_definition(m.model_definition_id)
    model = store.create_gateway_model_definition(
        name=f"{ENDPOINT}-model", secret_id=secret.secret_id, provider=JUDGE_PROVIDER, model_name=JUDGE_MODEL_NAME
    )
    store.create_gateway_endpoint(
        name=ENDPOINT,
        model_configs=[GatewayEndpointModelConfig(model_definition_id=model.model_definition_id,
                                                  linkage_type=GatewayModelLinkageType.PRIMARY)],
    )


def judges():
    from mlflow.genai.scorers import Guidelines, RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance

    return [
        RetrievalGroundedness(model=JUDGE_MODEL),
        RetrievalRelevance(model=JUDGE_MODEL),
        RelevanceToQuery(model=JUDGE_MODEL),
        Guidelines(name="clinical_guidelines", guidelines=GUIDELINES, model=JUDGE_MODEL),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Share of new traces to score.")
    args = parser.parse_args()

    import mlflow
    from mlflow.genai.scorers import ScorerSamplingConfig, delete_scorer, list_scorers

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment_id = mlflow.set_experiment(observability.EXPERIMENT).experiment_id
    ensure_endpoint()
    names = {s.name for s in judges()}
    for s in list_scorers(experiment_id=experiment_id):
        if s.name in names:
            delete_scorer(name=s.name, experiment_id=experiment_id, version="all")
    for s in judges():
        s.register(experiment_id=experiment_id).start(
            experiment_id=experiment_id, sampling_config=ScorerSamplingConfig(sample_rate=args.sample_rate)
        )
        print(f"  {s.name}: scoring {args.sample_rate:.0%} of new traces")
    print(f"registered {len(names)} judges on {observability.EXPERIMENT!r} through {JUDGE_MODEL} "
          f"({JUDGE_PROVIDER}:{JUDGE_MODEL_NAME})")


if __name__ == "__main__":
    main()
