---
title: "RevCD: Reversed Conditional Diffusion for Generalized Zero-Shot Learning"
type: paper
status: "Published"
venue: "International Conference on Deep Learning Theory and Applications (DeLTA)"
year: 2025
authors: "William Heyden, Habib Ullah, M. Salman Siddiqui, Fadi Al Machot"
order: 3
links:
  arxiv: "https://arxiv.org/abs/2409.00511"
summary: "A diffusion model that runs the usual generation direction in reverse — synthesizing semantic embeddings from visual input instead of visual features from semantic descriptions."
---

## Summary

Most generative approaches to zero-shot learning synthesize *visual* features from a class's semantic description, then train an ordinary classifier on the synthesized examples. RevCD inverts this: it uses a diffusion model to generate the *semantic* embedding conditioned on the visual input.

The architecture combines a multi-headed visual transformer, which produces attention-guided visual embeddings, with a reversed conditional diffusion process — a cross Hadamard-addition embedding of a sinusoidal time schedule conditions the denoising steps on the visual signal. Instead of requiring a one-to-one mapping between a fixed semantic space and every class, the model learns to reconstruct the semantic target directly from what it sees, which relaxes some of the rigid assumptions earlier generative ZSL methods depend on.

## Method overview

![RevCD method overview: a visual transformer feeds a reversed conditional diffusion process that synthesizes a semantic embedding from visual input](/diagrams/revcd.svg)

## Result

By generating semantic embeddings conditioned on visual attention rather than assuming a fixed semantic-to-visual mapping, RevCD improves generalization to unseen classes under the standard GZSL evaluation protocol.
