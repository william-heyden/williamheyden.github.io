---
title: "Diffusion Models as Hierarchical Variational Autoencoders"
date: 2026-09-14
draft: false
tags: ["Diffusion Models", "Generative Models", "Probability", "Computer Vision"]
summary: "A derivation-first walkthrough of diffusion models from the ELBO up — through score matching and the continuous-time SDE view — with notes on where Flow Matching departs from the picture, and how RevCD fits into it."
---

When I was building [RevCD](/projects/revcd/), I needed diffusion to do something most tutorials don't cover: generate a *semantic* embedding conditioned on a *visual* input, rather than generate an image out of pure noise. Convincing myself that this was still "just diffusion" meant going back to where diffusion models actually come from — not U-Nets and image demos, but the same latent-variable machinery behind an ordinary variational autoencoder (VAE). This post is that derivation, written the way I originally worked through it.

## What this covers

- How the evidence lower bound (ELBO) that trains a plain VAE generalizes to a *hierarchy* of latents
- Why a variational diffusion model (VDM) is just a hierarchical VAE with three specific restrictions on its encoder
- How the diffusion ELBO decomposes into a reconstruction term, a prior-matching term, and a consistency term — and how that consistency term collapses into the familiar "predict the noise" objective
- Three mathematically equivalent things a diffusion network can be trained to predict, and how Tweedie's formula ties them together
- Where score matching, Langevin dynamics, and the continuous-time SDE view fit in
- How conditioning ("guidance") generalizes this picture to RevCD, and where Flow Matching leaves it entirely

This follows the derivation structure of Calvin Luo's ["Understanding Diffusion Models: A Unified Perspective"](https://arxiv.org/abs/2208.11970) and the deep generative models material from the [ProbAI 2025](https://github.com/probabilisticai/probai-2025) summer school — both worth reading in full if you want every algebra step spelled out.

## Latent variables and the ELBO

Say we believe our data $\boldsymbol{x}$ is generated via some unobserved latent $\boldsymbol{z}$, jointly described by $p(\boldsymbol{x}, \boldsymbol{z})$. We'd like to maximize the data likelihood $p(\boldsymbol{x}) = \int p(\boldsymbol{x}, \boldsymbol{z})\, d\boldsymbol{z}$, but that integral is intractable for any interesting model. Instead we introduce a tractable approximate posterior $q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$ and derive a bound we *can* optimize.

Multiply and divide by $q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$ inside the log, then apply Jensen's inequality:

$$
\log p(\boldsymbol{x}) = \log \mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\!\left[\frac{p(\boldsymbol{x},\boldsymbol{z})}{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\right] \;\geq\; \mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\!\left[\log \frac{p(\boldsymbol{x},\boldsymbol{z})}{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\right] =: \mathrm{ELBO}(\boldsymbol{x})
$$

That derivation is correct but doesn't say much about *why* it's a good objective. A second derivation, expanding $\log p(\boldsymbol{x})$ as an expectation under $q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$ and using the chain rule $p(\boldsymbol{x},\boldsymbol{z}) = p(\boldsymbol{z}\mid\boldsymbol{x})\,p(\boldsymbol{x})$ instead, gives:

$$
\log p(\boldsymbol{x}) = \underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\!\left[\log \frac{p(\boldsymbol{x},\boldsymbol{z})}{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\right]}_{\mathrm{ELBO}(\boldsymbol{x})} \;+\; D_{\mathrm{KL}}\big(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \,\|\, p(\boldsymbol{z}\mid\boldsymbol{x})\big)
$$

Since $\log p(\boldsymbol{x})$ doesn't depend on $\phi$ at all, this says the ELBO and the KL term trade off exactly: pushing the ELBO up necessarily pushes $q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$ closer to the true (intractable) posterior $p(\boldsymbol{z}\mid\boldsymbol{x})$. Maximizing the ELBO is not a hack around the intractable likelihood — it's *exactly* the objective of learning the correct posterior.

For a plain VAE, splitting the ELBO via the chain rule again gives a reconstruction term and a prior-matching term:

$$
\mathrm{ELBO}(\boldsymbol{x}) = \underbrace{\mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\big[\log p_\theta(\boldsymbol{x}\mid\boldsymbol{z})\big]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\big(q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) \,\|\, p(\boldsymbol{z})\big)}_{\text{prior matching}}
$$

This is the encoder/decoder picture everyone knows: reconstruct $\boldsymbol{x}$ from the latent, while keeping the encoded distribution close to a prior (typically $\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$).

## From one latent to a hierarchy

A Markovian Hierarchical VAE (MHVAE) stacks $T$ of these: latents $\boldsymbol{z}_1, \dots, \boldsymbol{z}_T$, where each one is generated only from the one before it — a Markov chain in both directions:

$$
p(\boldsymbol{x}, \boldsymbol{z}_{1:T}) = p(\boldsymbol{z}_T)\, p_\theta(\boldsymbol{x}\mid\boldsymbol{z}_1) \prod_{t=2}^{T} p_\theta(\boldsymbol{z}_{t-1}\mid\boldsymbol{z}_t), \qquad q_\phi(\boldsymbol{z}_{1:T}\mid\boldsymbol{x}) = q_\phi(\boldsymbol{z}_1\mid\boldsymbol{x}) \prod_{t=2}^{T} q_\phi(\boldsymbol{z}_t\mid\boldsymbol{z}_{t-1})
$$

The same Jensen's-inequality argument as before extends directly to give an ELBO over the whole chain. Think of it as stacking VAEs on top of each other — literally a "recursive VAE."

## The variational diffusion model

A variational diffusion model (VDM) is a Markovian HVAE with three extra restrictions:

1. The latent dimension equals the data dimension — $\boldsymbol{x}_t$ *is* the $t$-th latent
2. The encoder at each step is not learned; it's a fixed linear Gaussian centered on the previous step
3. The Gaussian schedule is tuned so the final latent $\boldsymbol{x}_T$ is (indistinguishable from) standard Gaussian noise

<img src="/diagrams/vdm-chain.svg" alt="A variational diffusion model as a Markov chain: forward process q adds noise step by step from data x0 to pure noise xT, while the learned reverse process p_theta denoises step by step back from xT to x0" />

Concretely, the fixed forward ("noising") transition is

$$
q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) = \mathcal{N}\big(\boldsymbol{x}_t;\; \sqrt{\alpha_t}\,\boldsymbol{x}_{t-1},\; (1-\alpha_t)\boldsymbol{I}\big)
$$

where $\alpha_t \in (0,1)$ is a schedule that decreases toward zero. Because Gaussians compose nicely, you can integrate out all the intermediate steps and get a closed form straight from $\boldsymbol{x}_0$ to any $\boldsymbol{x}_t$:

$$
q(\boldsymbol{x}_t \mid \boldsymbol{x}_0) = \mathcal{N}\big(\boldsymbol{x}_t;\; \sqrt{\bar\alpha_t}\,\boldsymbol{x}_0,\; (1-\bar\alpha_t)\boldsymbol{I}\big), \qquad \bar\alpha_t := \prod_{s=1}^{t} \alpha_s
$$

which is the identity that makes training practical — you can jump straight to a noisy $\boldsymbol{x}_t$ at any timestep without simulating the chain.

Because the encoder here has no learnable parameters, the *only* thing left to learn is the reverse process $p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)$. Plugging the VDM's joint and posterior into the hierarchical ELBO and regrouping terms (the algebra is mechanical, just tedious) gives a three-way decomposition:

$$
\log p(\boldsymbol{x}) \geq \underbrace{\mathbb{E}_{q(\boldsymbol{x}_1\mid\boldsymbol{x}_0)}\big[\log p_\theta(\boldsymbol{x}_0\mid\boldsymbol{x}_1)\big]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\big(q(\boldsymbol{x}_T\mid\boldsymbol{x}_0)\,\|\,p(\boldsymbol{x}_T)\big)}_{\text{prior matching}} - \sum_{t=1}^{T-1}\underbrace{\mathbb{E}\big[D_{\mathrm{KL}}\big(q(\boldsymbol{x}_t\mid\boldsymbol{x}_{t-1})\,\|\,p_\theta(\boldsymbol{x}_t\mid\boldsymbol{x}_{t+1})\big)\big]}_{\text{consistency}}
$$

The prior-matching term has no learnable parameters and, given a long enough schedule, is essentially zero. The reconstruction term is a single VAE-style decoder step. All the real work happens in the **consistency term**: it asks that a denoising step from a noisier $\boldsymbol{x}_t$ agree with the corresponding noising step from a cleaner $\boldsymbol{x}_{t-1}$, at every intermediate timestep, simultaneously.

## What should the network predict?

Because $q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0)$ is itself Gaussian with a mean that's a known function of $\boldsymbol{x}_t$ and $\boldsymbol{x}_0$, minimizing the consistency term's KL divergence between two Gaussians of matched variance reduces to a plain squared error between means:

$$
\arg\min_\theta \, D_{\mathrm{KL}}\big(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0) \,\|\, p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)\big) = \arg\min_\theta \, \frac{1}{2\sigma_q^2(t)} \big\| \boldsymbol{\mu}_\theta(\boldsymbol{x}_t,t) - \boldsymbol{\mu}_q(\boldsymbol{x}_t,\boldsymbol{x}_0)\big\|_2^2
$$

The target mean $\boldsymbol{\mu}_q$ is a known linear combination of $\boldsymbol{x}_t$ and $\boldsymbol{x}_0$. What's interesting is that $\boldsymbol{x}_0$ itself has three equivalent reparameterizations, and each one turns this into a differently-flavored (but mathematically identical) training objective:

**Predict the clean data $\boldsymbol{x}_0$ directly.** A network $\hat{\boldsymbol{x}}_\theta(\boldsymbol{x}_t,t)$ regresses the original signal from a noisy observation and a noise level.

**Predict the noise $\boldsymbol{\epsilon}$.** Since $\boldsymbol{x}_0 = \big(\boldsymbol{x}_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}_0\big)/\sqrt{\bar\alpha_t}$ by the reparameterization used to sample $\boldsymbol{x}_t$, substituting this into $\boldsymbol{\mu}_q$ and simplifying collapses the loss to

$$
\arg\min_\theta \, \frac{(1-\alpha_t)^2}{2\sigma_q^2(t)(1-\bar\alpha_t)\alpha_t} \big\|\boldsymbol{\epsilon}_0 - \hat{\boldsymbol{\epsilon}}_\theta(\boldsymbol{x}_t,t)\big\|_2^2
$$

— the familiar "train a network to predict the noise that was added" objective (this is what DDPM actually trains, dropping the leading weight).

**Predict the score $\nabla \log p(\boldsymbol{x}_t)$.** Tweedie's formula relates the true posterior mean of a Gaussian to its score: $\mathbb{E}[\boldsymbol{\mu}_{\boldsymbol{x}_t}\mid\boldsymbol{x}_t] = \boldsymbol{x}_t + (1-\bar\alpha_t)\nabla\log p(\boldsymbol{x}_t)$. Applying it to $q(\boldsymbol{x}_t\mid\boldsymbol{x}_0)$ and solving for $\boldsymbol{x}_0$ gives yet another substitution into $\boldsymbol{\mu}_q$, and yet another equivalent loss — this time regressing a score network $\boldsymbol{s}_\theta(\boldsymbol{x}_t,t)$ against $\nabla\log p(\boldsymbol{x}_t)$.

These three targets aren't approximately related — combining the noise and score substitutions shows $\nabla\log p(\boldsymbol{x}_t) = -\boldsymbol{\epsilon}_0/\sqrt{1-\bar\alpha_t}$ *exactly*, off only by a time-dependent scale factor. In practice, noise-prediction tends to train best, but all three are the same model wearing different clothes.

## Score matching, Langevin dynamics, and the SDE limit

The score-prediction view connects directly to a separate line of work: **score-based generative models**. The score function $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})$ is a vector field over data space that points toward higher-density regions — learn it, and you can *sample* by following it, using Langevin dynamics:

$$
\boldsymbol{x}_{i+1} \leftarrow \boldsymbol{x}_i + c\,\nabla\log p(\boldsymbol{x}_i) + \sqrt{2c}\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})
$$

Vanilla score matching has real problems — the score is ill-defined off a low-dimensional data manifold, poorly estimated in low-density regions, and can misjudge the relative weight of separated modes. Perturbing the data with **multiple levels of Gaussian noise** fixes all three at once (noise gives the score full support, fills in low-density regions, and preserves relative mode weights) — and training a single network to denoise across every noise level simultaneously is, up to notation, the exact same objective as training a VDM.

Push the number of noise levels $T\to\infty$ and the discrete chain becomes a continuous-time diffusion process described by a stochastic differential equation (SDE); sampling means solving the corresponding *reverse-time* SDE. DDPM, score-based models, and the continuous SDE formulation are three views of one object.

## Guidance: conditioning the reverse process

Everything above models the unconditional density $p(\boldsymbol{x})$. Adding conditioning information $\boldsymbol{y}$ — a class label, a text embedding, anything — is a small change to the joint:

$$
p_\theta(\boldsymbol{x}_{0:T}\mid\boldsymbol{y}) = p(\boldsymbol{x}_T)\prod_{t=1}^{T} p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{y})
$$

and the network simply learns $\hat{\boldsymbol{x}}_\theta(\boldsymbol{x}_t,t,\boldsymbol{y})$, $\hat{\boldsymbol{\epsilon}}_\theta(\boldsymbol{x}_t,t,\boldsymbol{y})$, or $\boldsymbol{s}_\theta(\boldsymbol{x}_t,t,\boldsymbol{y})$ instead. This is the hook that everything from text-to-image diffusion to RevCD hangs off of.

## Where this shows up in RevCD

Most generative zero-shot learning methods run this conditioning in one direction: $\boldsymbol{y}$ is a fixed semantic description of a class (an attribute vector, a text embedding), and the diffused variable $\boldsymbol{x}_0$ is a *visual* feature — synthesize plausible visual features for a class you've never seen an image of, then train an ordinary classifier on the synthesized examples.

[RevCD](/projects/revcd/) reverses which variable is which. The diffused quantity $\boldsymbol{x}_0$ is the *semantic* embedding, and the conditioning signal $\boldsymbol{y}$ is a *visual* embedding, produced by a multi-headed visual transformer attending over the input image. In the notation above, RevCD trains $\hat{\boldsymbol{\epsilon}}_\theta(\boldsymbol{z}_t, t, \boldsymbol{y}_{\text{visual}})$ (a "cross Hadamard-addition" of a sinusoidal time embedding with the visual conditioning embedding is how $t$ and $\boldsymbol{y}$ both get injected into the network at each step — the same role a timestep embedding plays in a standard DDPM U-Net, just fused with a conditioning vector rather than used alone).

Why bother reversing it? A fixed semantic-to-visual mapping assumes every instance of a class visually corresponds to *one* predetermined point in semantic space — a rigid assumption that doesn't hold well for real, noisy, unevenly-specified semantic attributes. By instead learning to reconstruct the semantic target *from* what the model visually attends to, RevCD lets the semantic embedding be inferred rather than assumed fixed, which is exactly the flexibility that guided diffusion's $p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{y})$ was already built to provide — RevCD just points $\boldsymbol{x}$ and $\boldsymbol{y}$ in the direction the problem actually needs.

## Where Flow Matching leaves this picture

Everything above is built from one specific choice: a fixed *stochastic* Gaussian forward process, reversed via an ELBO. **Flow Matching** ([Lipman et al., 2023](https://arxiv.org/abs/2210.02747)) makes a different choice from the start. Rather than defining a noising SDE and deriving a bound to reverse it, it directly specifies a probability path $p_t(\boldsymbol{x})$ between noise and data and a *deterministic* velocity field $v_t(\boldsymbol{x})$ that transports samples along it — trained with a simple, simulation-free regression, no ELBO or score-matching detour required:

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\, \boldsymbol{x}_0\sim p_0,\, \boldsymbol{x}_1\sim p_1}\Big[\big\| v_\theta(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\big\|^2\Big], \qquad \boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\,\boldsymbol{x}_1
$$

<img src="/diagrams/diffusion-vs-flow-matching.svg" alt="Diffusion follows a curved, stochastic multi-step path from noise to data via a reverse SDE, while flow matching follows a straight, deterministic path via an ODE, generally requiring fewer sampling steps" />

for the straight-line conditional path shown above, in contrast to diffusion's curved Gaussian path. The two aren't opposed: a diffusion model's *probability-flow ODE* — the deterministic companion to its reverse SDE — is itself one particular instance of a flow-matching path. Flow Matching generalizes the picture, and in particular lets you choose a straighter, more sample-efficient path than the one a Gaussian forward process happens to imply.

That matters practically. Sampling from a diffusion model needs many small steps because the reverse SDE is only locally accurate; integrating a near-straight ODE path can need far fewer. If I were revisiting RevCD's sampling cost today, reformulating the same visually-conditioned generation as conditional flow matching over the semantic embedding — a conditional velocity field in place of the reversed conditional diffusion — is the first thing I'd try.

## Key takeaway

Diffusion models aren't a separate species of generative model — they're what falls out of a hierarchical VAE once you fix its encoder to a specific Gaussian noising schedule. Once you see that, the three equivalent training targets, the score-based view, and conditioning-as-guidance all fall out of a few pages of the same algebra. It's that same algebra that made it natural, in RevCD, to simply point the conditioning arrow the other way.
