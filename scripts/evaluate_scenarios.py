"""Run the assistant on the clinical scenarios and score its answers in MLflow.

Each scenario of the ``ocular-clinical-scenarios`` dataset, registered from
``data/eval/scenarios.jsonl`` by :mod:`ocular.rag.scenarios`, is answered by
:func:`ocular.service.reply` with its conversation, scan and patient details, and
the answer is scored two ways:

- **checks**, which need no judge: the tools called, in order, read from the
  answer's trace; the closing "Recommended actions" list; an opening that advises
  urgent assessment; text that must not appear; no drug dose; and no text from
  the system prompt. A check
  that does not apply to a scenario is left out rather than counted as a pass.
- **expectations_guidelines**, MLflow's ``ExpectationsGuidelines`` judge, which
  reads the scenario's own guidelines and passes only if the answer meets all of
  them.

The system prompt is the one in ``ocular/service.py`` unless ``--prompt-version``
names a version of ``ocular-assistant-system`` in the Prompt Registry, so prompt
versions can be compared on the same cases without switching branches. Every run
is an MLflow run in the ``ocular-clinical-scenarios`` experiment, tagged with the
prompt version, and ``--runs`` repeats the evaluation, since the model ignores
temperature and answers vary from run to run. The script prints the pass rate of
each check per category. The Qdrant index and MLflow server are the Docker
ones, reached from the host::

    QDRANT_URL=http://localhost:6333 \\
    QDRANT_COLLECTION=section__embeddinggemma-300m__corpus_full \\
    MLFLOW_TRACKING_URI=http://localhost:5001 \\
    uv run --group eval python scripts/evaluate_scenarios.py --prompt-version 6

    ... scripts/evaluate_scenarios.py --ids emergency-01 dose-03   # a few cases
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

from ocular import config  # noqa: F401 - loads .env (the API keys and MLflow login)
from ocular import observability, service
from ocular.rag import scenarios as sc
from ocular.rag import tools

BSCANS = config.ROOT / "data" / "raw" / "bscans"
# The model the monitoring judges use, called directly rather than through the
# server's gateway, since these judges run in this process.
JUDGE_MODEL = "openai:/gpt-5.4-mini-2026-03-17"
CHECKS = ("no_dose_amount", "no_prompt_text", "required_tools", "no_tools", "recommended_actions",
          "urgent_first", "absent_text")
# Answers run a few at a time, as the Gemini API turns away bursts with 503s.
WORKERS = 4


def answer(messages, scan, patient):
    """The assistant's answer to one scenario, retried while the model is overloaded."""
    for attempt in range(5):
        try:
            text, _ = service.reply(messages, scan_path=BSCANS / scan if scan else None, patient=patient)
            return text
        except Exception as exc:
            if not any(code in str(exc) for code in ("503", "429")) or attempt == 4:
                raise
            time.sleep(30 * (attempt + 1))


def make_checks():
    """The checks of :func:`ocular.rag.scenarios.check` as one scorer with a value per check."""
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    @scorer(name="checks")
    def checks(outputs, expectations, trace):
        spans = sorted(trace.data.spans, key=lambda s: s.start_time_ns)
        called = [s.name for s in spans if s.name in sc.TOOLS]
        # A null or empty expectation is dropped before evaluation, as MLflow stores no empty value.
        e = {"recommended_actions": None, "urgent_first": None, "absent_text": [], **expectations}
        result = sc.check(e, outputs or "", called, service.SYSTEM)
        return [Feedback(name=name, value=ok) for name, ok in result.items() if ok is not None]

    return checks


def summarise(experiment_id: str, run_id: str) -> None:
    """Print the pass rate of each check and of the judge per category, then what each case failed."""
    import mlflow

    traces = mlflow.search_traces(locations=[experiment_id], run_id=run_id)
    passes: dict = defaultdict(lambda: defaultdict(list))
    failed: dict = defaultdict(list)
    for _, row in traces.iterrows():
        case, category = row["tags"].get("id", "?"), row["tags"].get("category", "?")
        for a in row["assessments"]:
            if "feedback" not in a:  # the scenario's expectations, logged beside the scores
                continue
            ok = a["feedback"].get("value") in (True, "yes")
            for group in (category, "all"):
                passes[a["assessment_name"]][group].append(ok)
            if not ok:
                failed[case].append(a["assessment_name"])
    categories = [*sc.CATEGORIES, "all"]
    print(f"{'':24}" + "".join(f"{c[:11]:>13}" for c in categories))
    for name in (*CHECKS, "expectations_guidelines"):
        if name in passes:
            cells = [f"{sum(g)}/{len(g)}" if (g := passes[name].get(c)) else "-" for c in categories]
            print(f"{name:24}" + "".join(f"{x:>13}" for x in cells))
    for case in sorted(failed):
        print(f"  {case}: {', '.join(failed[case])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt-version", help="A Prompt Registry version; default the prompt in the code.")
    parser.add_argument("--runs", type=int, default=1, help="How many times to answer every scenario.")
    parser.add_argument("--ids", nargs="+", help="Only these scenario ids.")
    args = parser.parse_args()

    assert observability.setup(), "set MLFLOW_TRACKING_URI to a reachable MLflow server"
    import mlflow
    from mlflow.genai import datasets
    from mlflow.genai.scorers import ExpectationsGuidelines

    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", str(WORKERS))
    experiment_id = mlflow.set_experiment(sc.EXPERIMENT).experiment_id
    if args.prompt_version:
        service.SYSTEM = mlflow.genai.load_prompt(f"prompts:/{service.PROMPT_NAME}/{args.prompt_version}").template
    version = observability.prompt_version(service.PROMPT_NAME, service.SYSTEM)

    dataset = datasets.get_dataset(name=sc.NAME)
    data = dataset.to_df()
    if args.ids:
        data = data[data["tags"].map(lambda t: t["id"] in args.ids)]
        if len(data) != len(set(args.ids)):
            raise SystemExit(f"unknown ids among {args.ids}")
    data = data[["inputs", "expectations", "tags"]].assign(
        expectations=lambda d: d["expectations"].map(lambda e: {k: v for k, v in e.items() if v not in (None, [])}))

    # Loaded once here rather than by the first few answers at the same time.
    service.load_index()
    tools.load_classifier()
    scorers = [make_checks(), ExpectationsGuidelines(model=JUDGE_MODEL)]
    for run in range(args.runs):
        with mlflow.start_run(run_name=f"prompt v{version} run {run + 1}"):
            mlflow.set_tags({"prompt_version": version, "repeat": run + 1, "scenarios": len(data),
                             "dataset": sc.NAME, "dataset_id": dataset.dataset_id})
            result = mlflow.genai.evaluate(data=data, predict_fn=answer, scorers=scorers)
        print(f"\nprompt version {version}, run {run + 1} of {args.runs} ({result.run_id})")
        summarise(experiment_id, result.run_id)


if __name__ == "__main__":
    main()
