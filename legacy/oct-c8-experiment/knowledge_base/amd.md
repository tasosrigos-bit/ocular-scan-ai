---
covers: [AMD, Drusen]
title: Age-Related Macular Degeneration and Drusen
kb_version: 1
sources:
  - "National Eye Institute — Age-Related Macular Degeneration (AMD), nei.nih.gov, last updated Jun 22 2021"
  - "National Eye Institute — Complications of AMD Prevention Trial (CAPT), ClinicalTrials.gov NCT00000167"
---

## Overview
Drusen are small yellowish deposits that build up beneath the retina and are typically the earliest visible sign that an eye may go on to develop age-related macular degeneration (AMD). Eyes with larger or more numerous drusen carry a higher risk of progressing to vision-threatening complications of AMD.

AMD affects central vision and results from age-related damage to the macula — the part of the retina responsible for sharp, straight-ahead sight. It has two forms: dry AMD, which progresses gradually through early, intermediate, and late stages, and wet AMD, a less common but faster-progressing form caused by abnormal blood vessel growth in the eye (see the CNV entry for detail on wet AMD specifically).

## Symptoms
The drusen/early-AMD stage is typically asymptomatic. As disease progresses toward intermediate or late stages, patients may notice mild central blurriness, trouble seeing in low light, or — a key warning sign for late-stage disease — straight lines appearing wavy or distorted.

## Risk Factors
Age over 55, family history of AMD, and smoking are associated with higher risk. Risk increases further with age.

## Management
There is currently no treatment at the early/drusen stage — management is regular monitoring via dilated eye exams. At the intermediate stage, certain dietary supplements have been shown to reduce risk of progression. If wet AMD develops, see the CNV entry for treatment options.

## Referral
Presence of drusen alone is not an emergency finding, but warrants routine ophthalmologic monitoring given the associated risk of AMD progression.

## Model Behavior Note
On external validation against real clinic scans, this project's classifier's Drusen predictions were most frequently confused with Normal (8 of 14 misclassifications in testing), and less often with CSR, CNV, or DME. Clinicians reviewing a Drusen flag — particularly one with borderline confidence — should be aware that this model has a documented tendency to under-call Drusen as Normal under certain imaging conditions, rather than over-call it.
