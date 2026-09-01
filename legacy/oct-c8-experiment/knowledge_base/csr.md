---
covers: [CSR]
title: Central Serous Retinopathy
kb_version: 1
sources:
  - "Varghese J, Kesharwani D, Parashar S, Agrawal P. A Review of Central Serous Chorioretinopathy: Clinical Presentation and Management. Cureus. 2022. (open access, CC-BY)"
---

## Overview
Central serous retinopathy (CSR, also called central serous chorioretinopathy) is a condition in which fluid leaks from the choroid through a defect in the retinal pigment epithelium, causing a small detachment of the retina at the macula. It is most often diagnosed in working-age adults and is disproportionately more common in men than women.

## Symptoms
Blurred or distorted central vision, a gray or blurred spot in central vision, and reduced visual sharpness are typical. Symptoms are often temporary, usually affecting one eye, and can resolve spontaneously — though reduced visual acuity may persist even after the fluid has cleared.

## Risk Factors
CSR has a well-documented association with corticosteroid use (oral, inhaled, or topical) and with psychological stress. It occurs more frequently in middle-aged men. The underlying mechanism is linked to changes in choroidal blood flow and pressure.

## Management
Many cases resolve on their own without treatment ("watchful waiting"), particularly for a first acute episode. For persistent, recurrent, or chronic cases, treatment options include laser photocoagulation and photodynamic therapy, guided by imaging findings.

## Referral
An isolated, first-time acute CSR finding is often reasonably managed with observation and routine follow-up, given the high rate of spontaneous resolution — but recurrent or chronic presentations warrant more active ophthalmologic management.

## Model Behavior Note
On the internal OCT-C8 held-out test set, this classifier achieved perfect performance on CSR (100% across all metrics). Like DR and Macular Hole, CSR was not represented in the 37-scan external clinic validation set. Notably, CSR was one of the classes this model mistakenly predicted for a real, domain-shifted Drusen scan during Grad-CAM analysis (DRUSEN_4__L, predicted CSR at 54% confidence) — illustrating that even classes with perfect internal test performance can be implicated in cross-class confusion once real, out-of-distribution images are involved.
