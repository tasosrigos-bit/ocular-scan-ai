# Figures used in the README

Static images committed so the README renders on GitHub. The Grad-CAM strips come
from (git-ignored) `explain.py` output; `case_browser.png` is a screenshot.

## `case_browser.png` — the Part 2 app

Screenshot of [`app/case_browser.py`](../../app/case_browser.py) running (light
mode), showing a **verified Drusen case**: the OCT scan, Part 1's frozen decision
(Drusen, `routine`), and the grounded narrative with `[amd#overview]`-style
citation markers highlighted and the `✓ VERIFIED` badge. Used in the README
"Part 2" subsection (§8).

Regenerate: run the app (see [`app/README.md`](../../app/README.md)), open a
verified case, take a full-width screenshot, save as `docs/figures/case_browser.png`.

## What the overlays show

Reviewing all 4 Drusen misclassification pairs (`DRUSEN_3`, `DRUSEN_4`,
`DRUSEN_7`, `DRUSEN_9`) with `explain.target=both`: in **3 of 4** the
predicted-class heatmap and the true-class heatmap land in **nearly the same
location** — the model localises the pathological region correctly but misreads
*what it is*. Only `DRUSEN_9__L` (predicted DME) showed genuinely divergent
attention. The dominant pattern is **"correct localisation, wrong
classification,"** not "looking in the wrong place." This matters for the CDS
discussion: the failure is semantic, not spatial, so a saliency check would not
catch it. **8 of 14** misclassified Drusen scans went to `Normal`.

## The two figures in the README

Committed as **`gradcam_drusen_wrong_1.png`** and **`_2.png`** (452 × 224 =
`[ raw B-scan | Grad-CAM for the predicted, wrong class ]` — the "pred heatmap"
panel of the `explain.target=both` output). They are representative of the
dominant pattern: the model's heatmap for the *wrong* class still lands on the
drusen region.

| README figure | scan | true → pred | pred conf | shows |
|---|---|---|---|---|
| `gradcam_drusen_wrong_1.png` | `DRUSEN_3__L` | Drusen → Normal | 0.88 | "Normal" heatmap sits on the drusen |
| `gradcam_drusen_wrong_2.png` | `DRUSEN_4__L` | Drusen → CSR | 0.54 | "CSR" heatmap sits on the drusen |

Counter-example, not used as a figure but worth a sentence in the write-up:
`DRUSEN_9__L`, Drusen → DME — the one of four reviewed pairs with genuinely
divergent predicted-vs-true attention.

To regenerate / swap in different strips:

```bash
python explain.py paths=<env> explain.split=external_test \
    'explain.classes=[Drusen]' explain.only_errors=true explain.target=both
cp "<output_dir>/explain/external_test/wrong/Drusen/<file>.png" docs/figures/gradcam_drusen_wrong_1.png
```
