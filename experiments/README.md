# experiments

The scripts that compute the results. Each one runs a compute stage, for the image
classifier or for the retrieval assistant, and writes a small CSV to `results/`,
which the notebooks read and present. Nothing here is imported by the `ocular`
package. The scripts import the package, not the other way round.

## Scripts

| Script | What it does | Writes |
| --- | --- | --- |
| `run_preprocessing.py` | Trains a ResNet50 on each of the eleven preprocessing configurations and scores it on OCTDL and the clinic. The reported rows were screened with `--per-class 3000 --epochs 5`. | `results/preprocessing_results.csv` |
| `run_models.py` | Holds the preprocessing fixed and varies the backbone over the two carried frames. The reported rows were screened with `--per-class 3000 --epochs 5`. | `results/model_results.csv` |
| `train_final.py` | Trains the chosen configuration on the full training set and saves one checkpoint per epoch. | `convnext_final_e*.pt` |
| `eval_checkpoints.py` | Scores every saved checkpoint on the Kermany validation split, OCTDL and the clinic. | `results/checkpoint_eval.csv` |
| `eval_clinic_scan.py` | Writes the per-scan clinic prediction and class probabilities for one checkpoint. | `results/clinic_scan_e1.csv` |
| `run_rag_eval.py` | Scores retrieval configurations of the assistant on the synthetic question set with reference-free metrics. | `results/rag_eval.csv` |
| `run_phaseb.py` | Compares the basic and agentic answering pipelines on the selected configuration. | `results/rag_eval_phaseb.csv` |

Which notebook reads which file is listed in the top-level README.

## Notes

The preprocessing search and the model grid are screens. Both were run on a
class-balanced subsample of three thousand training images per class, so that many
configurations fit in a time budget, which means their numbers rank configurations
against one another and sit below the full-data model. That subsample is requested
with `--per-class` and recorded both in the `per_class` column of the results CSV
and in the name of the cache, so a subsampled cache is never mistaken for a full
one. It is not the default of either script, so reproducing a reported row means
passing the flag. Only `train_final.py` always uses the full training set.

`run_preprocessing.py`, `run_models.py`, `eval_checkpoints.py` and `run_rag_eval.py` are
resumable. A row that is already in the results CSV is skipped, so an interrupted run
continues where it stopped. Caches are reused across runs unless `--delete-cache` is given.

The checkpoints are about 110 MB each and are not tracked. The delivered checkpoint
is attached to the repository release, as described in the top-level README.
