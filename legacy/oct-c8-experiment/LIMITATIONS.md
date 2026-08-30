# Limitations

This project trains and internally validates well on Retinal OCT-C8, but the
external validation and CDS results come with substantial caveats. Read this
before drawing conclusions from any number in [README.md](README.md#results).

## External validation set

The OPTOPOL REVO clinic set is **37 B-scans from 27 unique eyes/patients** — a
small, single-center, single-device convenience sample.

| True class | n | Notes |
|---|---|---|
| Drusen | 22 | the only class with enough scans to say anything |
| Normal | 9 | |
| DME | 5 | underpowered |
| CNV | 1 | **not statistically interpretable** — a single scan |
| AMD, CSR, DR, Macular Hole | 0 | **never externally validated** |

Consequences:

- **Half the label space (AMD, CSR, DR, Macular Hole) has zero external
  evidence.** The internal test numbers for those classes should not be assumed
  to transfer to real clinic data.
- **CNV (n = 1)** and **DME (n = 5)** external results are anecdotes, not
  estimates. Confidence intervals on per-class metrics for these classes are
  effectively undefined.
- Only **Drusen** (n = 22) supports any quantitative statement, and even that is
  a wide interval.
- No patient demographics, acquisition metadata, scan-quality grading, or
  test–retest scans are available, so we cannot analyse failure by subgroup or
  separate model error from acquisition variability.
- The de-identified filenames encode only class, a patient index, and laterality
  (`L`/`R`/`p0`); `p0` means laterality was not recorded.

## Domain shift

Internal test accuracy **96.46%** → external clinic accuracy **59.46%**
(a ~37-point drop).

Likely cause: **device / vendor domain shift.** Retinal OCT-C8 aggregates scans
from its source acquisition devices; the clinic scans were all captured on an
**OPTOPOL REVO**, which differs in optics, axial/lateral resolution, speckle
characteristics, post-processing, contrast, and native image dimensions. The
model was trained only on OCT-C8's distribution and sees OPTOPOL scans as
out-of-distribution. The preprocessing pipeline (ROI crop → resize → ImageNet
normalise) is applied identically to both sets and the OCT-C8 normalisation
constants are reused for the clinic set on purpose, so the measured gap reflects
genuine distribution shift rather than a renormalisation artefact.

This is an expected, well-documented failure mode for OCT classifiers evaluated
across scanner types — but it means **this model is not validated for OPTOPOL
REVO images** and should not be used on them without domain adaptation and a
proper external study.

## CDS safety gap (MSP-based confidence)

The CDS layer abstains ("insufficient confidence — refer to specialist") when the
calibrated top-1 probability is below `min_confidence` or the top-1/top-2 margin
is below `min_margin`, and it rejects inputs whose OOD score exceeds
`ood_reject_score`. The OOD score currently in use is **MSP** (`1 − max softmax
probability`).

**Finding:** MSP-based confidence does **not** reliably catch confident-but-wrong
predictions, which is the dominant error mode under domain shift.

| Split | Misclassifications | Confident wrong triage | Deferred to specialist |
|---|---|---|---|
| OCT-C8 test | 99 | 63 | 36 |
| Clinic — Drusen only | 14 | 10 (7 triaged as **`none`** / "no referral indicated") | 4 |

Seven misclassified clinic Drusen scans were confidently triaged as needing **no
referral** — the worst-case direction for a screening aid. Under domain shift the
model is often *confidently* wrong: max softmax stays high, margin stays wide,
MSP stays low, so neither the abstention rule nor the MSP OOD gate fires.

Grad-CAM review of the Drusen misclassifications makes the mechanism concrete: in
3 of 4 reviewed pairs the model **localises the pathological region correctly but
misclassifies what it represents** (8 of 14 misclassified Drusen scans were
called `Normal`). The error is *semantic*, not spatial — so a saliency-overlap or
"is it looking at the retina?" check would not catch it either. What fails is the
model's mapping from a correctly-attended region to a label, which is exactly
what a feature-space distance measures.

### Proposed fix (TODO)

Replace / augment MSP with a **feature-space out-of-distribution detector** —
specifically a **Mahalanobis distance** in the penultimate-layer feature space,
with class-conditional means and a shared covariance estimated on the OCT-C8
**training** features. Distance-based scores separate "unlike anything in
training" from "confidently classified" far better than max-softmax and would
route OPTOPOL scans to abstention instead of a wrong urgency.

Status: **not implemented.** `src/oct_cds/cds/ood_detection.py` currently provides
only `msp_ood_score` and `energy_ood_score`; the Mahalanobis detector and the
training-feature statistics it needs are a documented TODO in that module.

## Other limitations

- **No prospective evaluation.** All results are retrospective on fixed image
  sets. There is no reader study, no comparison against clinician performance,
  and no measurement of clinical impact.
- **Frozen label taxonomy.** The 8 OCT-C8 classes do not cover the full range of
  macular pathology; anything outside them (e.g. epiretinal membrane, vitreo-
  macular traction, retinal vein occlusion) will be forced into one of the 8.
- **Grad-CAM is a coarse, class-discriminative saliency method**, sensitive to
  the chosen target layer; overlays indicate rough spatial attention, not a
  faithful account of the model's reasoning, and should not be over-interpreted.
- **The CDS rule thresholds and urgency mapping** (`configs/cds/rules_v1.yaml`)
  are reasonable defaults, not clinically validated cut-offs, and the guideline
  references attached to recommendations are placeholders.
- **Not a medical device.** Research and educational use only.
