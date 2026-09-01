---
title: <Human-readable disease name>
covers: [<ClassKey>, ...]        # exact Part 1 class keys, e.g. [AMD, Drusen]
                                 # valid keys: AMD CNV CSR DME DR Drusen "Macular Hole"
                                 # NOT "Normal". Use [] for a general reference entry.
kb_version: 1                    # bump when the content changes
sources:                         # list of citation strings (a NEI page, a CC-BY article)
  - "National Eye Institute — <page title>, nei.nih.gov, last updated <date>"
  - "<Author> et al. (<year>). <Title>. <Journal>. (open access, CC-BY)"
---

<!--
  - id is the filename stem (amd.md -> "amd"); no `id:` field needed.
  - Paraphrased / summarised only — never copied verbatim.
  - Sources: NEI (public domain) or CC-BY / CC-BY-SA only. No AAO PPP, no
    EyeWiki, no StatPearls (CC-BY-NC-ND).
  - One self-contained thought per `##` section; the heading is the citation
    label. Keep each section ~60-150 words.
  - Section titles are flexible. The set below matches the existing entries.
-->

## Overview

<What the condition is, who it affects, how it presents. Plain clinical language.
If the source supports it, a sentence on the OCT appearance here helps the
narrator explain an OCT-based prediction.>

## Symptoms

<What the patient notices.>

## Risk Factors

<Who is at higher risk.>

## Management

<Typical treatment / monitoring options, at a general level.>

## Referral

<General referral consideration for this finding. Do NOT write anything that
reads as a per-patient instruction — the CDS urgency comes from Part 1's rule
engine, not from this text.>

## Model Behavior Note

<This project's own evidence for this class: internal test performance, external
clinic performance (or "not represented in the external set"), and any
documented confusion pattern. Describes PAST validation results, not the current
case. The narrator may cite this to add a grounded caveat.>
