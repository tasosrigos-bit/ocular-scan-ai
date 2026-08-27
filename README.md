# ocular-scan-ai

Retinal OCT B-scan classification into CNV, DME, DRUSEN and NORMAL, trained on one
acquisition device and measured on devices that were absent from training. The
central question is not accuracy on a held-out split of the training set, which is
inflated by shared patients and a shared device, but whether the model still works
on a scanner it has never seen. The drusen class is the hardest, so it is tracked
throughout as the primary signal.

Alongside the classifier, the project builds a retrieval-augmented assistant that
answers ophthalmology questions from a curated journal corpus, and a Streamlit
application that puts the two together in front of a clinician.

## Datasets and their roles

The three datasets play different roles and are never mixed. None of them is stored
in this repository. They are downloaded into `data/`, which is not tracked.

| Dataset | Role | Device | Notes |
| --- | --- | --- | --- |
| Kermany OCT2017 | training | one vendor | Public. Split is rebuilt at the patient level in notebook 01. |
| OCTDL | selection | an unseen vendor | Public. Held out. Configurations are ranked on it. |
| Clinic | confirmation | OPTOPOL | Private. 37 de-identified B-scans. Confirms, never selects. |

The Kermany test split is not used for reporting because 89 percent of its patients
also appear in training. Notebook 01 shows this and builds a clean patient-level
split instead.

## How the project is organised

The code is split so that computation and presentation never share a file. Scripts
compute and write a small results file. Notebooks read those files and explain the
outcome. No notebook trains a model.

```
ocular/         the package
  config.py       paths, class order, seed (shared by both subsystems)
  classifier/     the image classifier
    preprocess.py   B-scan to a fixed frame, one configurable knob per step
    data.py         caches, loaders, OCTDL and clinic mapping, class weights
    model.py        build_model over several backbones, plus a small custom network
    train.py        a fixed-length training loop, no selection on validation
    eval.py         accuracy, macro-F1, per-class recall
    explain.py      Grad-CAM over the classifier
  rag/            the retrieval assistant: corpus, chunking, index, retrieval,
                  rerank, generation, the agent tools and the two pipelines
experiments/    the compute scripts and the result files they write
scripts/        the data-build steps for the retrieval corpus and its indexes
notebooks/      the presentation, read the result files and explain them
app/            the Streamlit application over the classifier and the assistant
data/           local only, not tracked: raw images, caches, split.csv, corpus, indexes
```

## What runs on what

Each stage is one script that writes one small file or artifact. The notebook for
that stage reads it and reports it. The order below is the order of the work.

The image classifier:

| Stage | Command | Writes | Read by |
| --- | --- | --- | --- |
| Clean split | `notebooks/01_classifier_data.ipynb` | `data/split.csv` | notebook 01 |
| Preprocessing search | `python experiments/run_preprocessing.py` | `experiments/results/preprocessing_results.csv` | notebook 02 |
| Model grid | `python experiments/run_models.py` | `experiments/results/model_results.csv` | notebook 03 |
| Final training | `python experiments/train_final.py --ckpt experiments/convnext_final.pt` | `experiments/convnext_final_e*.pt` | (the delivered weights) |
| Per-epoch eval | `python experiments/eval_checkpoints.py --ckpt experiments/convnext_final_e1.pt` | `experiments/results/checkpoint_eval.csv` | notebook 03 |
| Clinic per-scan | `python experiments/eval_clinic_scan.py` | `experiments/results/clinic_scan_e1.csv` | notebook 03 |
| Clinic Grad-CAM | `python experiments/explain_clinic.py` | `experiments/results/clinic_explanations.csv` | notebook 04 |

The retrieval assistant:

| Stage | Command | Writes | Read by |
| --- | --- | --- | --- |
| Corpus | `python scripts/build_corpus.py` | `data/corpus/` | notebook 05 |
| Indexes | `python scripts/build_all_indexes.py` | `data/index/` | notebooks 06, 07 |
| Eval questions | `python scripts/build_questions.py --n 50` | `data/eval/questions.jsonl` | notebook 08 |
| Retrieval eval | `python experiments/run_rag_eval.py` | `experiments/results/rag_eval.csv` | notebook 08 |
| Answering eval | `python experiments/run_phaseb.py` | `experiments/results/rag_eval_phaseb.csv` | notebook 08 |

The preprocessing search and the model grid are screens. They run on a class-balanced
subsample of the training set to rank configurations against one another, so their
numbers are relative and lower than the final model. Only `train_final.py` runs on
the full training set. Local training on Apple silicon is memory bound, which is why
the screens use a subsample rather than the full data.

The result files are committed, so the notebooks can be read and re-run without
repeating any training, search or evaluation.

## The notebooks

The notebooks are meant to be read in order. The first four cover the image
classifier, the last four cover the retrieval assistant. Each one names the functions
that do the work and states the exact command that produced the file it reads.

1. `01_classifier_data.ipynb`. The datasets and their roles, the patient leakage in
   the provided split, and the construction of the clean split that everything else
   depends on.
2. `02_classifier_preprocessing.ipynb`. The preprocessing pipeline step by step, followed by the
   preprocessing search results. The finding is that curvature correction is the change
   that moves the drusen class the most, and that a square frame with squishing transfers
   best.
3. `03_classifier_modeling.ipynb`. The backbone grid, the final model trained on full data
   examined epoch by epoch, the clinic evaluation, and a diagnosis of the drusen
   misses. The finding is that transfer peaks at the first epoch and that the
   remaining drusen errors are a resolution and volume limit rather than a threshold.
4. `04_classifier_explainability.ipynb`. Grad-CAM over the delivered checkpoint, showing where
   the model looks on one scan per class and on the drusen misses, and a diagnosis of
   the off-tissue attention on a few of those misses.
5. `05_rag_corpus.ipynb`. The ophthalmology corpus, drawn from a whitelist of PubMed
   Central journals and filtered to a reuse-permitting licence.
6. `06_rag_chunking.ipynb`. The three strategies that cut each article into the short
   passages retrieval searches over.
7. `07_rag_retrieval.ipynb`. Embedding the passages, the dense and lexical indexes,
   the three retrieval modes and the reranker.
8. `08_rag_evaluation.ipynb`. The reference-free evaluation that selects the retrieval
   configuration and compares basic against agentic answering.

## Reproducing

```
uv sync
```

For the retrieval assistant, put an LLM API key in a git-ignored `.env` at the
repository root, as `GEMINI_API_KEY=...`.

Place the datasets under `data/raw` so that `data/raw/OCT2017`, `data/raw/octdl_dl`
and `data/raw/bscans` exist. Run notebook 01 to write `data/split.csv`. From there,
either re-run the experiment scripts to regenerate the result files and the weights,
or use the committed result files and go straight to the notebooks. The retrieval
corpus and its indexes are built once with `scripts/build_corpus.py` and
`scripts/build_all_indexes.py`, and the application is started with
`uv run streamlit run app/streamlit_app.py`.

## The delivered model

The delivered model is ConvNeXt-Tiny at 384 by 256 under the cropped,
curvature-corrected, squished pipeline, trained for one epoch on the full training
set. It reaches a macro-F1 of about 0.93 on the unseen OCTDL set and is correct on
27 of the 37 clinic scans. The one epoch schedule is chosen in advance because
transfer degrades after the first epoch, not by reading the best epoch off the
evaluation sets.

The weights are about 110 MB, over the file size that git accepts, so they are not
in the repository. The delivered checkpoint `convnext_final_e1.pt` is attached to
the latest release of this repository, on the Releases page, and can be loaded with
`build_model("convnext_tiny", pretrained=False)` followed by `load_state_dict`.
