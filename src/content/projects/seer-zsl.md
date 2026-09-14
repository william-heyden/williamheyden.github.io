---
title: "SEER-ZSL: Semantic Encoder-Enhanced Representations for Generalized Zero-Shot Learning"
type: paper
status: "Published"
venue: "IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops"
year: 2025
authors: "William Heyden, Habib Ullah, M. Salman Siddiqui, Fadi Al Machot"
order: 3
links:
  paper: "https://openaccess.thecvf.com/content/WACV2025W/CV4Small/papers/Heyden_SEER-ZSL_Semantic_Encoder-Enhanced_Representations_for_Generalized_Zero-Shot_Learning_WACVW_2025_paper.pdf"
  arxiv: "https://arxiv.org/abs/2312.13100"
  github: "https://github.com/william-heyden/SEER-ZeroShotLearning"
summary: "A hybrid strategy that distills noisy, real-world semantic descriptions through a probabilistic encoder to close the generalization gap in generalized zero-shot learning."
---

## Summary

Generalized zero-shot learning (GZSL) is harder than plain zero-shot learning: the model must classify correctly across *both* seen and unseen classes at test time, not just unseen ones, which makes it easy for a classifier to overfit toward the seen classes it actually has data for. A second, underappreciated issue is that semantic class descriptions used in practice are rarely as clean as curated benchmark attributes — they're noisy, inconsistent, or automatically generated.

SEER-ZSL tackles both problems with a probabilistic semantic encoder that distills meaningful structure out of noisy semantic representations before they're used to bridge visual and semantic spaces, improving robustness and semantic consistency. The result is a representation that generalizes better across the seen/unseen boundary instead of just fitting the semantic space it was handed.

## Method overview

![SEER-ZSL method overview: a visual encoder and a probabilistic semantic encoder feed an alignment and fusion stage for generalized zero-shot classification](/diagrams/seer-zsl.svg)

## Result

SEER-ZSL surpasses prior state-of-the-art unseen-class accuracy on three of four standard GZSL benchmarks, with gains of roughly 11% on AWA2, 0.3% on CUB, and 13.5% on SUN.
