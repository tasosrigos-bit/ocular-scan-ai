---
covers: [DME]
title: Diabetic Macular Edema
kb_version: 1
sources:
  - "National Eye Institute — Macular Edema, nei.nih.gov, last updated Aug 6 2025"
---

## Overview
Macular edema is swelling of the macula caused by fluid leaking from blood vessels into the retina. Diabetic macular edema (DME) is the most common cause and occurs as a complication of diabetic retinopathy — an eye condition caused by diabetes-related damage to retinal blood vessels. Other, less common causes of macular edema include wet AMD, retinal vein occlusion, uveitis, certain medications, and post-surgical inflammation.

## Symptoms
Blurry vision that may worsen over time is the primary symptom. Patients may also report straight lines looking wavy, objects appearing different sizes between eyes, or duller/faded colors. Severity varies — some patients experience only mild blurriness, while others develop significant central vision loss affecting reading and driving.

## Risk Factors
Diabetes (particularly with poor blood sugar control) is the dominant risk factor via diabetic retinopathy. Other contributing conditions include retinal vein occlusion, uveitis, wet AMD, recent eye surgery, and certain glaucoma medications.

## Management
Treatment targets the underlying cause. For DME specifically, managing blood sugar is an important preventive step. Direct treatment options include anti-VEGF injections, corticosteroid injections, NSAID eye drops (particularly for surgery-related edema), laser treatment, and — when other treatments fail — vitrectomy surgery. OCT imaging is used clinically to assess the degree of swelling.

## Referral
DME represents active, potentially vision-threatening pathology requiring timely ophthalmologic evaluation rather than routine monitoring alone, given its association with progressive diabetic retinopathy.

## Model Behavior Note
In external validation on real clinic scans, this classifier correctly identified 4 of 5 DME cases (80% sensitivity). The one misclassified case was confidently triaged as "urgent" despite being wrong — meaning the CDS abstention mechanism did not flag it as uncertain. Given the small sample size (n=5), this single case should not be over-interpreted, but it illustrates the same confidence-reliability gap documented for Drusen.
