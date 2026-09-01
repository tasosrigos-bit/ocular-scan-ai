---
covers: [CNV]
title: Choroidal Neovascularization (Wet AMD)
kb_version: 1
sources:
  - "National Eye Institute — Age-Related Macular Degeneration (AMD), nei.nih.gov, last updated Jun 22 2021"
---

## Overview
Choroidal neovascularization (CNV) refers to abnormal growth of new blood vessels beneath the retina. In the context of age-related eye disease, CNV is what defines "wet" AMD (also called neovascular AMD) — a less common but faster-progressing form of AMD than the dry/drusen stage. These abnormal vessels are fragile and prone to leaking blood and fluid, damaging the macula and causing rapid vision loss. Wet AMD can develop from any stage of dry AMD, but is always classified as late-stage disease once present.

## Symptoms
Because wet AMD is a late-stage form of AMD, it typically produces more pronounced symptoms than earlier stages: central blurry vision, blank or dark spots in central vision, and straight lines appearing wavy or crooked — a symptom significant enough that patients experiencing it are advised to see an eye doctor immediately.

## Risk Factors
Same broad risk profile as AMD generally: age 55+, family history of AMD, and smoking. Existing dry AMD (particularly intermediate or late-stage) is itself a risk factor for later CNV development.

## Management
Unlike early/dry AMD, wet AMD (CNV) has active, effective treatment options. Anti-VEGF injections directly into the eye are the primary treatment, working to slow or stop abnormal vessel growth and leakage. Photodynamic therapy (a combination of injections and laser treatment) is another available option. Early treatment initiation is associated with better visual outcomes.

## Referral
CNV represents active, vision-threatening pathology with a narrow window where treatment is most effective — this finding warrants urgent rather than routine referral.

## Model Behavior Note
On the internal OCT-C8 held-out test set, this classifier performed strongly on CNV (93.4% sensitivity, 88.1% precision, 90.7% F1) — among the harder classes for the model, alongside DME and Drusen, likely reflecting genuine visual overlap between these conditions on OCT. In external clinic validation, only one CNV scan was available (n=1) — far too small a sample to draw any meaningful conclusion about real-world performance on this class; this result should be treated as illustrative only, not evidence of reliability.
