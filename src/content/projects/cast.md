---
title: "CAST: Closed-form Analytic Semantic Transfer for Zero-Shot Classifier Extension"
type: paper
status: "Preprint"
venue: "arXiv preprint"
year: 2026
authors: "William Heyden, Habib Ullah, M. Salman Siddiqui, Fadi Al Machot"
order: 4
links:
  arxiv: "https://arxiv.org/abs/2608.13751"
summary: "A training-free, image-free framework that extends a pretrained classifier to unseen classes by directly injecting weights derived analytically from semantic embeddings."
---

## Summary

Every method above still needs some training to bridge the seen/unseen gap. CAST asks a more extreme question: can a classifier be extended to a brand-new class *without any training at all*, and without a single example image of that class?

CAST derives a closed-form analytic transfer that computes new classifier weights directly from semantic embeddings and injects them into a frozen, pretrained classifier — no fine-tuning, no synthesized images, no gradient steps. The paper backs this with a theoretical error decomposition that identifies the *semantic extrapolation residual*: the part of the extension error that comes specifically from how far a new class's semantic embedding sits from the classes the model already knows, giving a principled way to reason about when the approach will and won't work well.

## Method overview

![CAST method overview: semantic embeddings for new classes are transformed by a closed-form analytic transfer and injected as weights into a frozen pretrained classifier](/diagrams/cast.svg)

## Result

CAST extends pretrained classifiers to new classes purely through weight injection, with a theoretical error bound that explains extension quality as a function of semantic distance from known classes — competitive with trained approaches at a fraction of the cost.
