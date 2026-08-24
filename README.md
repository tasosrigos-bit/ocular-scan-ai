# ocular-scan-ai

Retinal OCT B-scan classification into CNV, DME, DRUSEN and NORMAL, trained on one
acquisition device and measured on devices that were absent from training. The
central question is not accuracy on a held-out split of the training set, which is
inflated by shared patients and a shared device, but whether the model still works
on a scanner it has never seen. The drusen class is the hardest, so it is tracked
throughout as the primary signal.

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
ocular/         the package: one preprocessing pipeline and one model interface
  config.py       paths, class order, seed
  preprocess.py   B-scan to a fixed frame, one configurable knob per step
  data.py         caches, loaders, OCTDL and clinic mapping, class weights
  model.py        build_model over several backbones, plus a small custom network
  train.py        a fixed-length training loop, no selection on validation
  eval.py         accuracy, macro-F1, per-class recall
experiments/    the compute scripts and the result files they write
notebooks/      the presentation, read the result files and explain them
data/           local only, not tracked: raw images, caches, split.csv, bscans
```

## What runs on what

Each stage is one script that writes one small file. The notebook for that stage
reads the file and reports it. The order below is the order of the work.

| Stage | Command | Writes | Read by |
| --- | --- | --- | --- |
| Clean split | `notebooks/01_data_exploration.ipynb` | `data/split.csv` | notebook 01 |
| Preprocessing search | `python experiments/run_preprocessing.py` | `experiments/results/preprocessing_results.csv` | notebook 02 |
| Model grid | `python experiments/run_models.py` | `experiments/results/model_results.csv` | notebook 03 |
| Final training | `python experiments/train_final.py --ckpt experiments/convnext_final.pt` | `experiments/convnext_final_e*.pt` | (the delivered weights) |
| Per-epoch eval | `python experiments/eval_checkpoints.py --ckpt experiments/convnext_final.pt` | `experiments/results/checkpoint_eval.csv` | notebook 03 |
| Clinic per-scan | `python experiments/eval_clinic_scan.py` | `experiments/results/clinic_scan_e1.csv` | notebook 03 |

The preprocessing search and the model grid are screens. They run on a class-balanced
subsample of the training set to rank configurations against one another, so their
numbers are relative and lower than the final model. Only `train_final.py` runs on
the full training set. Local training on Apple silicon is memory bound, so the size
search can be run on a Colab GPU instead with `experiments/colab_run.ipynb`.

The result files are committed, so the three notebooks can be read and re-run
without repeating any training.

## The three notebooks

The notebooks are meant to be read in order.

1. `01_data_exploration.ipynb`. The datasets and their roles, the patient leakage in
   the provided split, and the construction of the clean split that everything else
   depends on.
2. `02_preprocessing.ipynb`. The preprocessing pipeline step by step, followed by the
   preprocessing search results. The finding is that curvature correction is the change that
   moves the drusen class the most, and that a square frame with squishing transfers
   best.
3. `03_modeling.ipynb`. The backbone grid, the final model trained on full data
   examined epoch by epoch, the clinic evaluation, and a diagnosis of the drusen
   misses. The finding is that transfer peaks at the first epoch and that the
   remaining drusen errors are a resolution and volume limit rather than a threshold.

## Reproducing

```
uv sync
```

Place the datasets under `data/raw` so that `data/raw/OCT2017`, `data/raw/octdl_dl`
and `data/raw/bscans` exist. Run notebook 01 to write `data/split.csv`. From there,
either re-run the experiment scripts to regenerate the result files and the weights,
or use the committed result files and go straight to notebooks 02 and 03.

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

