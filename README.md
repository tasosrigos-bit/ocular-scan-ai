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
in this repository. How each one is obtained and where it belongs is described under
*The data* below.

| Dataset | Role | Device | Notes |
| --- | --- | --- | --- |
| Kermany OCT2017 | training | Heidelberg Spectralis | Public. Split is rebuilt at the patient level in notebook 01. |
| OCTDL | selection | Optovue Avanti | Public. Held out. Configurations are ranked on it. |
| Clinic | confirmation | OPTOPOL | 37 de-identified B-scans, shared separately. Confirms, never selects. |

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
| Preprocessing search | `python experiments/run_preprocessing.py --per-class 3000 --epochs 5` | `experiments/results/preprocessing_results.csv` | notebook 02 |
| Model grid | `python experiments/run_models.py --per-class 3000 --epochs 5` | `experiments/results/model_results.csv` | notebook 03 |
| Final training | `python experiments/train_final.py --ckpt experiments/convnext_final.pt` | `experiments/convnext_final_e*.pt` | (the delivered weights) |
| Per-epoch eval | `python experiments/eval_checkpoints.py --ckpt experiments/convnext_final_e1.pt` | `experiments/results/checkpoint_eval.csv` | notebooks 03 and 04 |
| Clinic per-scan | `python experiments/eval_clinic_scan.py` | `experiments/results/clinic_scan_e1.csv` | notebook 04 |

The retrieval assistant:

| Stage | Command | Writes | Read by |
| --- | --- | --- | --- |
| Corpus | `python scripts/build_corpus.py` | `data/corpus/` | notebook 06 |
| Indexes | `python scripts/build_all_indexes.py` | `data/index/` | notebooks 07, 08 |
| Eval questions | `python scripts/build_questions.py --n 50` | `data/eval/questions.jsonl` | notebook 09 |
| Retrieval eval | `python experiments/run_rag_eval.py` | `experiments/results/rag_eval.csv` | notebook 09 |
| Answering eval | `python experiments/run_phaseb.py` | `experiments/results/rag_eval_phaseb.csv` | notebook 09 |

The preprocessing search and the model grid are screens. They run on a class-balanced
subsample of the training set to rank configurations against one another, so their
numbers are relative and lower than the final model. Only `train_final.py` runs on
the full training set. Local training on Apple silicon is memory bound, which is why
the screens use a subsample rather than the full data.

The result files are committed, so the notebooks can be read and re-run without
repeating any training, search or evaluation.

## The notebooks

The notebooks are meant to be read in order. The first five cover the image
classifier, the last four cover the retrieval assistant. Each one names the functions
that do the work and states the exact command that produced the file it reads.

1. `01_classifier_data.ipynb`. The datasets and their roles, the patient leakage in
   the provided split, and the construction of the clean split that everything else
   depends on.
2. `02_classifier_preprocessing.ipynb`. The preprocessing pipeline step by step, followed by the
   preprocessing search results. The finding is that curvature correction is the change
   that moves the drusen class the most, and that a square frame with squishing transfers
   best.
3. `03_classifier_training.ipynb`. The backbone grid and the final model trained on full
   data, examined epoch by epoch. The finding is that transfer peaks at the first epoch,
   which fixes the training schedule at one epoch.
4. `04_classifier_transfer.ipynb`. The delivered model scored on all three devices. The
   finding is that only drusen fails to transfer, falling from 0.996 on the training
   scanner to 0.545 on the clinic while the other three classes hold, which places the
   weakness in the change of scanner rather than in the class.
5. `05_classifier_explainability.ipynb`. Grad-CAM over the delivered checkpoint, showing
   where the model looks on one scan per class and on the drusen misses.
6. `06_rag_corpus.ipynb`. The ophthalmology corpus, drawn from a whitelist of PubMed
   Central journals and filtered to a reuse-permitting licence.
7. `07_rag_chunking.ipynb`. The three strategies that cut each article into the short
   passages retrieval searches over.
8. `08_rag_retrieval.ipynb`. Embedding the passages, the dense and lexical indexes,
   the three retrieval modes and the reranker.
9. `09_rag_evaluation.ipynb`. The reference-free evaluation that selects the retrieval
   configuration and compares basic against agentic answering.

## Reproducing

Install the environment, then fetch the two release assets from the repository root.

```
uv sync
gh release download v0.1.0 --dir experiments --pattern convnext_final_e1.pt
gh release download v0.1.0 --pattern rag-artifacts.tar.gz && tar xzf rag-artifacts.tar.gz
```

That is enough for notebooks 03 to 09 and for the application, since the results, the
clinic scans and the patient-level split are all in the repository. For the assistant,
add an LLM API key to a git-ignored `.env` at the repository root as
`GEMINI_API_KEY=...`, and start the application with

```
uv run streamlit run app/streamlit_app.py
```

Notebooks 01 and 02 additionally need the two public datasets, described under *The
data* below. Every reported number can be regenerated rather than read, by re-running
the scripts listed under *What runs on what*, but nothing has to be.

## The data

Most of `data/` is not tracked, so the tree has to be assembled once. What is missing,
what is already here and what is rebuilt are three different things.

**The two public datasets have to be downloaded.** They are the only things a fresh
clone is missing.

| Dataset | Source | Size |
| --- | --- | --- |
| Kermany OCT2017 | [data.mendeley.com/datasets/rscbjbr9sj/2](https://data.mendeley.com/datasets/rscbjbr9sj/2), DOI [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2) | 5.6 GB |
| OCTDL | [data.mendeley.com/datasets/sncdhf53xc/4](https://data.mendeley.com/datasets/sncdhf53xc/4), DOI [10.17632/sncdhf53xc](https://doi.org/10.17632/sncdhf53xc) | 210 MB |

Version 2 of the Kermany record is the one used here. It carries the `OCT2017`
directory of 84,484 images, 83,484 for training and 1,000 for testing, which are the
counts notebook 01 reports. Unpack it so that `data/raw/OCT2017` holds the `train` and
`test` directories with one subdirectory per class.

Unpack OCTDL so that `data/raw/octdl_dl` holds the `OCTDL` directory. Its label file
is already in this repository, so it does not need to be copied over. The full record
covers seven disease groups, of which only AMD, DME and NO have a counterpart among the
four Kermany classes. `_octdl_frame` in `ocular/classifier/data.py` selects them by the
disease and condition columns and ignores the rest, which reduces the record to the
1,137 images every OCTDL number in this project is measured on. Downloading the whole
record is therefore correct, and the extra groups are simply never read.

**The clinic scans are shared separately.** The 37 de-identified B-scans are
patient-derived, so they are not kept in the repository. They are provided as
`clinic_bscans.zip` through a private link and are unpacked by hand so that
`data/raw/bscans` holds the 37 PNGs and the manifest recording where each was cut from
its report. The source reports, which carry patient name, identifier and date of birth,
are never shared. The scans are checked to carry no burned-in identifier, and they are
needed only for the clinic evaluation in notebooks 04 and 05; the application and every
other notebook run without them. From the repository root,

```
unzip clinic_bscans.zip -d data/raw/
```

puts them in place, so that `data/raw/bscans/` holds the 37 `CLASS_index__laterality.png`
files and `manifest.json`.

**The retrieval artifacts are downloaded or rebuilt.** The corpus, the chunk caches
and the retrieval index are build artifacts, so they are attached to the release rather
than tracked. From the repository root,

```
gh release download v0.1.0 --pattern rag-artifacts.tar.gz
tar xzf rag-artifacts.tar.gz
```

unpacks them into `data/` at 134 MB. That archive carries the corpus, the three chunk
caches that notebook 07 compares, the question set, and the one index the deployed
system uses, `fixed__MedEmbed-base-v0.1`. The other eight indexes of the Phase A sweep
are omitted deliberately, because nothing reads them. Notebook 08 reports that sweep
from the committed `rag_eval.csv`, and rebuilding them is only necessary to repeat the
sweep itself, with `scripts/build_all_indexes.py`.

**The image caches are always rebuilt.** The preprocessed `.npy` stacks are about
6.1 GB and are written on demand by `build_cache`, so they never travel.

**Three small inputs are tracked despite living under `data/`.** These are `split.csv`
at 3 MB, which pins the exact patient-level division every reported number was measured
on, `OCTDL_labels.csv` at 53 KB, which maps the OCTDL vocabulary onto the four classes,
and the corpus manifest at 99 KB, which notebook 06 reads. Carrying them costs little
and removes a regeneration step that could silently diverge.

### Reading the results without the data

Every number and figure this project reports is committed under
`experiments/results/`, so the notebooks that present those results run on a fresh
clone. Notebooks 03 and 09 need nothing beyond the clone. Notebook 05 needs the
delivered checkpoint and the separately provided clinic scans, and notebook 04 needs them
too for the one cell that rescores the clinic set. Notebooks 06 to 08 need the
retrieval archive. Notebooks 01 and 02 build and illustrate the classifier pipeline
itself and are the only two that need the 5.8 GB of downloaded imagery.

### Running the application

The application needs three things, none of which is in the repository. They are the
delivered checkpoint, the retrieval archive above, and an LLM API key in a git-ignored
`.env` at the repository root as `GEMINI_API_KEY=...`. With those in place,

```
uv run streamlit run app/streamlit_app.py
```

starts it. The embedding model and the reranker are pulled from Hugging Face on first
use and cached there, so the first question is slower than the rest. If the index is
missing the application says so and repeats the download command rather than failing
with a bare path.

## The delivered model

The delivered model is ConvNeXt-Tiny at 384 by 256 under the cropped,
curvature-corrected, squished pipeline, trained for one epoch on the full training
set. It reaches a macro-F1 of about 0.93 on the unseen OCTDL set and is correct on
27 of the 37 clinic scans. The one epoch schedule is chosen in advance because
transfer degrades after the first epoch, not by reading the best epoch off the
evaluation sets.

The weights are about 110 MB, over the file size that git accepts, so they are not
in the repository. The delivered checkpoint `convnext_final_e1.pt` is attached to
release `v0.1.0`, and everything that loads it expects it at
`experiments/convnext_final_e1.pt`. Download it into place with

```
gh release download v0.1.0 --pattern convnext_final_e1.pt --dir experiments
```

or save it from the Releases page into the `experiments` directory by hand. That
one file is all that is needed. The per-epoch checkpoints behind the training curve
in notebook 03 are not required, because that curve is read from the committed
`experiments/results/checkpoint_eval.csv`.

The path is defined once as `config.CKPT`, so the scripts, the notebooks and the
application all resolve it from there. Calling `config.require_ckpt()` returns it if
present and otherwise raises with the download command above. The weights load with
`build_model("convnext_tiny", pretrained=False)` followed by `load_state_dict`.
