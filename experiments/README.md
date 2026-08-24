# experiments

The scripts that compute the results. Each one trains or evaluates a model and
writes a small CSV to `results/`, which the notebooks read and present. Nothing
here is imported by the `ocular` package. The scripts import the package, not the
other way round.

## Scripts

| Script | What it does | Writes |
| --- | --- | --- |
| `run_preprocessing.py` | Trains a ResNet50 on each of the eleven preprocessing configurations and scores it on OCTDL and the clinic. | `results/preprocessing_results.csv` |
| `run_models.py` | Holds the preprocessing fixed and varies the backbone over the two carried frames. | `results/model_results.csv` |
| `train_final.py` | Trains the chosen configuration on the full training set and saves one checkpoint per epoch. | `convnext_final_e*.pt` |
| `eval_checkpoints.py` | Scores every saved checkpoint on the Kermany validation split, OCTDL and the clinic. | `results/checkpoint_eval.csv` |
| `eval_clinic_scan.py` | Writes the per-scan clinic prediction and class probabilities for one checkpoint. | `results/clinic_scan_e1.csv` |

Which notebook reads which file is listed in the top-level README.

## Notes

The preprocessing search and the model grid are screens. They run on a class-balanced
subsample of the training set so that many configurations fit in a time budget,
so their numbers rank configurations against one another and are lower than the
full-data model. Only `train_final.py` uses the full training set.

`run_preprocessing.py`, `run_models.py` and `eval_checkpoints.py` are resumable. A row that
is already in the results CSV is skipped, so an interrupted run continues where it
stopped. Caches are reused across runs unless `--delete-cache` is given.

Local training on Apple silicon is memory bound, so the preprocessing search can be run on
a Colab GPU with `colab_run.ipynb`.

The checkpoints are about 110 MB each and are not tracked. The delivered checkpoint
is attached to the repository release, as described in the top-level README.
