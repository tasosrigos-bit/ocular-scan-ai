"""Register the LLM judges that score the deployed assistant's traces as they arrive.

The app traces every question to the ``ocular-assistant`` MLflow experiment, with
each literature search recorded as a retriever span holding the passages it
returned. Real questions have no reference answer, so the judges are
reference-free, MLflow's built-in ones, with an OpenAI model as the judge:

- **retrieval_groundedness** - is every claim in the answer supported by the
  retrieved passages?
- **retrieval_relevance** - is each retrieved passage relevant to the question?
- **relevance_to_query** - does the answer address the question asked?
- **guideline_citations**, **guideline_literature_gaps** and
  **guideline_clinician_judgement** - the assistant's clinical posture, one judge
  per guideline so each is scored on its own: does the answer cite its sources,
  say so when the literature does not cover the question, and give no definitive
  diagnosis or individual treatment directive?

The two retrieval judges run on answers that searched the literature, which the
app tags ``searched``; the others run on every answer. Each answer's trace records
the earlier turns of the conversation and the numbered passages the model saw, so
the judges can follow a follow-up question and check the citations.

Once registered and started, the MLflow server scores the sampled traces in the
background and attaches the judgements to them, so they appear on each trace and
can be charted over time. The server runs judges only through its AI Gateway, so
the script first sets up a gateway endpoint, ``judge``: it stores
``OPENAI_API_KEY`` from ``.env`` as a gateway secret, which the server keeps
encrypted under ``MLFLOW_CRYPTO_KEK_PASSPHRASE``, and points the endpoint at the
judge model. It also registers the assistant's system prompt, from
``ocular/service.py``, in the Prompt Registry as ``ocular-assistant-system``, adding
a version only when the text has changed, and lets the app's MLflow user read it:
the app tags every trace with the version it runs, so scores can be compared
across prompt changes. Rerunning replaces the judges and the endpoint's key and model, for
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
# Only the assistant's answers: a scan analysis has no question or passages to judge.
ANSWERS = "trace.name = 'answer_question'"
SEARCHED = f"{ANSWERS} AND tags.searched = 'true'"
GUIDELINES = {
    "guideline_citations":
        "Every claim the answer takes from the retrieved passages cites them by number in "
        "brackets, such as [1] or [2, 5]. An answer that makes no claims from the literature, "
        "such as a greeting, a thank-you or a refusal, needs no citations.",
    "guideline_literature_gaps":
        "Compare the answer with the retrieved passages in the response. If the passages contain "
        "what the question asks, using them meets this guideline. Only when they do not must the "
        "answer say so plainly instead of answering from general knowledge. A refusal of an "
        "out-of-scope question meets it, and so does a reply that needs no literature, such as "
        "to a greeting or thanks.",
    "guideline_clinician_judgement":
        "The answer gives no definitive diagnosis for an individual patient and no instruction to "
        "treat a specific patient a particular way. Summarising treatment options, schedules or "
        "standard practice from the literature is allowed, as is reporting the scan classifier's "
        "prediction with its confidence and a recommendation for clinical confirmation.",
}


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


def ensure_prompt(tracking_uri: str) -> str:
    """Register the system prompt if its text is new, and let the app read it. Returns its version."""
    import mlflow
    import requests

    from ocular import service

    name, text = service.PROMPT_NAME, service.SYSTEM
    versions = []
    try:
        versions = mlflow.MlflowClient().search_prompt_versions(name)
    except Exception:  # the prompt does not exist yet
        pass
    version = next((v for v in versions if v.template == text), None)
    if version is None:
        version = mlflow.genai.register_prompt(
            name=name, template=text, commit_message="System prompt of ocular/service.py"
        )
    res = requests.post(
        f"{tracking_uri}/api/3.0/mlflow/users/permissions/grant",
        auth=(os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"]),
        json={"username": "app", "resource_type": "prompt", "resource_id": name, "permission": "READ"},
        timeout=30,
    )
    if not res.ok and "RESOURCE_ALREADY_EXISTS" not in res.text:
        print(f"  could not let the app read the prompt ({res.status_code}): {res.text[:200]}")
    return str(version.version)


def judges():
    """Each judge with the traces it scores."""
    from mlflow.genai.scorers import Guidelines, RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance

    return [
        (RetrievalGroundedness(model=JUDGE_MODEL), SEARCHED),
        (RetrievalRelevance(model=JUDGE_MODEL), SEARCHED),
        (RelevanceToQuery(model=JUDGE_MODEL), ANSWERS),
        *((Guidelines(name=name, guidelines=[text], model=JUDGE_MODEL), ANSWERS)
          for name, text in GUIDELINES.items()),
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
    print(f"  system prompt: version {ensure_prompt(args.tracking_uri)}")
    # Every judge on the experiment is this script's, so all are replaced, including
    # any it no longer registers.
    for s in list_scorers(experiment_id=experiment_id):
        delete_scorer(name=s.name, experiment_id=experiment_id, version="all")
    registered = judges()
    for s, traces in registered:
        s.register(experiment_id=experiment_id).start(
            experiment_id=experiment_id,
            sampling_config=ScorerSamplingConfig(sample_rate=args.sample_rate, filter_string=traces),
        )
        print(f"  {s.name}: scoring {args.sample_rate:.0%} of {'searched answers' if traces == SEARCHED else 'answers'}")
    print(f"registered {len(registered)} judges on {observability.EXPERIMENT!r} through {JUDGE_MODEL} "
          f"({JUDGE_PROVIDER}:{JUDGE_MODEL_NAME})")


if __name__ == "__main__":
    main()
