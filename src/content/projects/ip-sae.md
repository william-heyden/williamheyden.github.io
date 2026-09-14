---
title: "An Integral Projection-Based Semantic Autoencoder for Zero-Shot Learning"
type: paper
status: "Published"
venue: "IEEE Access"
year: 2023
authors: "William Heyden, Habib Ullah, M. Salman Siddiqui, Fadi Al Machot"
order: 4
links:
  paper: "https://ieeexplore.ieee.org/document/10213991/"
  arxiv: "https://arxiv.org/abs/2306.14628"
  github: "https://github.com/william-heyden/IP-SAE"
summary: "An autoencoder that projects visual and semantic features into a shared, domain-invariant latent space to make zero-shot classification more robust to domain shift and hubness."
---

## Summary

Zero-shot learning asks a model to recognize classes it never saw a single labeled example of, using only a semantic description (attributes, embeddings) to bridge from seen to unseen categories. Two problems dominate: **domain shift** — the mapping learned on seen classes doesn't transfer cleanly to unseen ones — and the **hubness problem** — a few points in the embedding space end up as nearest neighbors for a disproportionate number of queries, degrading retrieval-based classification.

IP-SAE addresses both with an integral projection-based autoencoder: an encoder projects a concatenation of the visual feature space and the semantic space into a shared latent representation, and a decoder reconstructs from it. Because the reconstruction is tied to a domain-invariant manifold rather than the visual space alone, samples from unseen domains stay closer to where the model expects them. Suitable regularization on the transformation function further discourages the encoder from collapsing many classes onto a small number of hub points.

## Method overview

![IP-SAE method overview: visual features and semantic attributes are projected by an integral projection encoder into a domain-invariant latent space](/diagrams/ip-sae.svg)

## Result

Evaluated on standard zero-shot benchmarks, the approach improves unseen-class accuracy by reducing both domain-shift error and hubness-driven misclassification compared to prior semantic autoencoder baselines.
