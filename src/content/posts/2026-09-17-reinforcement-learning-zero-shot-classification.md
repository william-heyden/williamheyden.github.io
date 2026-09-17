---
title: "Reinforcement Learning and Zero-Shot Classification: Where the Math Actually Overlaps"
date: 2026-09-17
draft: false
tags: ["Reinforcement Learning", "Zero-Shot Learning", "Representation Learning", "Computer Vision"]
summary: "A mathematical tour of where reinforcement learning genuinely intersects zero-shot classification — linear bandits, successor features, and policy gradients — and where the connection is closer to a shared trick than a shared field."
---

Reinforcement learning isn't my area — zero-shot classification is — so take this as a research diary rather than a survey. What pulled me in is a suspicion that kept resurfacing while writing about [the geometry of representation learning](/posts/2026-09-16-geometry-of-representation-learning/): the central problem of zero-shot classification — *assign a value to something the model has never been trained on* — is also, almost word for word, the central problem reinforcement learning solves every time it evaluates an action it's never taken. If that's right, the overlap isn't a vague analogy; it should show up as the same equations. This post is me checking.

## What this covers

- Two genuinely different ways "RL" and "zero-shot" show up together in the literature — and why conflating them muddies both
- Classification reframed as a one-step decision problem, and what linear contextual bandits say about assigning value to an arm you've never pulled
- The precise algebraic parallel between a linear bandit's ridge-regression solution and [CAST](/projects/cast/)'s closed-form weight injection — including why their respective "how far outside my training data am I" diagnostics have the same shape
- Successor features and universal value functions: how RL solves *zero-shot task generalization*, and why the solution is structurally the same bilinear trick as a ZSL classifier
- Where RL is actually used as an optimization tool inside recognition pipelines (policy gradients through non-differentiable choices) — a completely different role from the one above
- Seen/unseen bias in generalized ZSL, read as an exploration–exploitation problem, and how that connects to [SEER-ZSL](/projects/seer-zsl/)

## Two different questions hiding under one name

Before going further, it's worth separating two things that both get called "RL for zero-shot learning" but aren't the same claim:

1. **RL as an optimization tool.** Some part of a recognition pipeline is non-differentiable — a hard attention decision, a discrete attribute query — and policy-gradient methods are used to train through it anyway. RL shows up as *machinery*, not as a source of insight about zero-shot generalization itself.
2. **RL as a structurally parallel problem.** A large chunk of RL is about *zero-shot generalization to a new task or goal* — evaluate a policy against a reward function it was never optimized for, without further training. This is a different problem from image classification, but it turns out to be solved by the same mathematical move zero-shot classifiers use.

Most of the genuinely interesting content is in (2), so that's where this post spends most of its time — but (1) is real and worth knowing about, so it gets a section too, clearly marked as a different kind of claim.

## Classification as a one-step decision problem

A contextual bandit is the simplest object in RL: no state transitions, no long-horizon credit assignment — just a context $x$, a choice of action $a$ from a set of arms, and a reward $r(x,a)$ revealed after the choice. Formally, it's a single-step Markov Decision Process. An ordinary classifier already fits this shape: $x$ is the input, the arms are the classes, and reward is $\mathbb{1}[a = y]$. The reason nobody calls softmax classification "a bandit algorithm" is that classification training gets the true label *every time* — there's no exploration problem, because the reward signal for every action is available during training, not just the one taken.

Zero-shot classification breaks that assumption in one specific way. For seen classes, you still have full supervision. For unseen classes, you have *never observed a reward at all* — not even once, for any input. This is exactly the situation a bandit algorithm faces with an arm it has never pulled: no data, so no empirical estimate of value. The bandit-theoretic question "how should you assign value to an arm you've never tried" and the ZSL question "how should you assign a classifier score to a class you've never seen an image of" are the same question.

### Linear bandits solve this with side information

A bandit with no structure between arms is stuck: an unpulled arm's value is simply unknown. But a **linear contextual bandit** (Li, Chu, Langford & Schapire, 2010, ["A Contextual-Bandit Approach to Personalized News Article Recommendation"](https://arxiv.org/abs/1003.0146)) assumes the reward is linear in a shared feature space:

$$
r(x, a) \approx \theta_a^\top x
$$

Given a design matrix $Z \in \mathbb{R}^{d \times n}$ of $n$ observed contexts (as columns) and their rewards $r \in \mathbb{R}^n$ for a given arm, the ridge-regularized least-squares estimate of $\theta_a$ has the standard closed form:

$$
\hat{\theta}_a = \big(ZZ^\top + \lambda I\big)^{-1} Z r
$$

The point of writing it this way: **this closed form requires no gradient descent, no retraining, and generalizes immediately to a new context vector once you have $\hat\theta_a$.** But it still needs $Z$ and $r$ — observed pulls of that specific arm. For an arm that's never been pulled at all, this doesn't help on its own. What *does* help — and is the standard move in bandits with large or structured action spaces — is when arms themselves carry a feature vector $z_a$, and reward is linear jointly in $x$ and $z_a$. Then a shared weight matrix, fit only on arms you've pulled, can score an arm defined purely by its feature vector, never pulled at all.

## The CAST connection

That's precisely [CAST](/projects/cast/)'s move, just in classification language instead of bandit language. CAST fits a ridge-regularized closed-form bridge from a class's semantic embedding $z$ to its classifier weight row $w$, using only the classes a pretrained model was already trained on:

$$
\hat{P} = W Z^\top (ZZ^\top + \lambda I)^{-1}
$$

then synthesizes a new class's weight row as $\hat{w}_u = \hat{P} z_u$ — no gradient steps, no training images, for a class that has *never* been pulled, in bandit language. Line up the two equations and the structure is identical: both are the ridge-regression normal equations, both regularized the same way, both used to extrapolate to something with zero observed reward using only its feature vector. CAST isn't *inspired by* bandit theory — nobody involved was thinking about bandits — but it's solving the identical linear-algebra problem linear bandits solve for a brand-new arm, because "assign value to something unobserved, given its features and a shared linear model" doesn't care what field is asking.

This parallel isn't just decorative — it predicts something. Linear bandit algorithms don't just estimate $\hat\theta_a^\top x$ for a new context; they also carry an uncertainty term, because a value extrapolated far outside the data you fit on shouldn't be trusted as much as one close to it. The standard confidence width for a linear bandit's estimate at a candidate feature vector $z$ is (Li et al., 2010; Abbasi-Yadkori et al., 2011):

$$
\text{width}(z) \;\propto\; \sqrt{z^\top (ZZ^\top + \lambda I)^{-1}\, z}
$$

CAST's own diagnostic — the **semantic extrapolation residual** — is

$$
\rho_u = \big\| (I - ZZ^{+})\, z_u \big\|
$$

the norm of $z_u$'s component orthogonal to the span of the seen classes' embeddings. These are not the same formula — one is a quadratic form under a ridge-regularized inverse covariance, the other is a projection-residual norm — but they blow up on *exactly the same directions*: both are large precisely when $z_u$ has substantial energy outside the subspace the bridge was fit on, and both are small when $z_u$ sits well within it. A linear bandit's exploration bonus and CAST's extrapolation residual are answering the same underlying question — *how far outside what I've actually fit is this new point?* — with two different but qualitatively matched instruments.

## Successor features: RL's own zero-shot problem

The bandit picture generalizes to full RL through **successor features** (Barreto et al., 2017, ["Successor Features for Transfer in Reinforcement Learning"](https://arxiv.org/abs/1606.05312)), and this is where the "zero-shot" in "zero-shot RL" is a literal, not borrowed, term — it long predates its recent use in vision-language models.

In a standard MDP the action-value function $Q^\pi(s,a)$ is the expected discounted return of taking action $a$ in state $s$ under policy $\pi$. Suppose the reward for a whole *family* of tasks decomposes linearly in some shared features $\phi(s,a,s')$ and a task-specific weight vector $w_g$ (indexed by a goal or task $g$):

$$
r_g(s,a,s') = \phi(s,a,s')^\top w_g
$$

Then, by linearity of expectation, the value function for task $g$ under a fixed policy $\pi$ decomposes the same way:

$$
Q^\pi_g(s,a) = \psi^\pi(s,a)^\top w_g, \qquad \psi^\pi(s,a) := \mathbb{E}^\pi\Big[\sum_{t=0}^{\infty}\gamma^t \phi(s_t,a_t,s_{t+1}) \,\Big|\, s_0{=}s,\, a_0{=}a\Big]
$$

$\psi^\pi$ — the **successor features** — are the discounted expected sum of future $\phi$, exactly analogous to the successor representation in earlier RL theory, just generalized from a state-visitation count to arbitrary features. Once you've learned $\psi^\pi$ for a policy, evaluating it against a *brand-new* task $g'$ (a task never trained on, defined purely by its weight vector $w_{g'}$) costs one inner product: $Q^\pi_{g'}(s,a) = \psi^\pi(s,a)^\top w_{g'}$. [Universal Successor Feature Approximators](https://arxiv.org/abs/1812.07626) (Borsa et al., 2018) extend this to condition $\psi$ on the goal too, giving zero-shot value estimates for goals never encountered during training, across states never encountered either.

<img src="/diagrams/bilinear-zero-shot.svg" alt="Three parallel bilinear compatibility functions: a linear bandit's context dotted with arm weights giving a value, successor features dotted with a goal embedding giving a Q-value in RL, and an image representation dotted with a class embedding giving a zero-shot classification score, all converging on one unifying idea: compatibility as an inner product between a shared representation and a target embedding generalizes for free to targets never trained on" />

Set $\psi^\pi(s,a)$ next to $f(\text{image})$ and $w_g$ next to $g(\text{class})$, and a bilinear ZSL classifier — $\text{score}(x,y) = f(x)^\top g(y)$, the compatibility function behind classic embedding models like DeViSE ([Frome et al., 2013](https://arxiv.org/abs/1312.5650)) and behind CAST's synthesized weight rows — is the same object with the labels changed. Both fields hit the same wall (a compatibility function needs to be evaluated against something with zero training signal) and reached for the same fix (make the function bilinear in a shared representation, so a new embedding is all you need). Neither field owes the other credit here; it's convergent design, which is exactly what makes it worth noticing.

## Where RL is actually used as a training tool

The successor-feature story is a structural parallel — RL and ZSL solving the same kind of problem. Separately, and for an unrelated reason, RL machinery gets used *inside* recognition pipelines whenever a step in the pipeline is discrete or otherwise non-differentiable and backpropagation can't reach through it.

The **REINFORCE** estimator (Williams, 1992) is the standard tool: for a stochastic policy $\pi_\theta(a\mid s)$ and a reward $R(a)$, the score-function identity

$$
\nabla_\theta \, \mathbb{E}_{a \sim \pi_\theta}[R(a)] = \mathbb{E}_{a \sim \pi_\theta}\big[R(a)\, \nabla_\theta \log \pi_\theta(a\mid s)\big]
$$

lets you get a gradient signal through a sampling step without needing $R$ itself to be differentiable — you only need to be able to evaluate it. This shows up in vision pipelines wherever a *hard*, discrete decision sits in the middle of an otherwise differentiable network: hard visual attention that looks at one region instead of a soft-attended blend over all of them ([Xu et al., 2015](https://arxiv.org/abs/1502.03044)), or an agent that actively decides which semantic attribute to query next before committing to a zero-shot label, trading query cost against classification accuracy. In both cases "RL" means specifically the REINFORCE gradient estimator, doing a job that has nothing conceptually to do with zero-shot generalization — it would show up identically in a fully-supervised pipeline with a hard attention step.

It's also, incidentally, the same tool now used to fine-tune diffusion models directly against a downstream reward rather than the training-time likelihood, by treating the reverse diffusion chain as a multi-step policy and each denoising step as an action ([Black et al., 2023, "DDPO"](https://arxiv.org/abs/2305.13301)). [RevCD](/projects/revcd/)'s reversed conditional diffusion process — trained to maximize likelihood of the correct semantic embedding, as derived in [the diffusion models post](/posts/2026-09-14-diffusion-models-hierarchical-vaes/) — is exactly the kind of model DDPO-style fine-tuning was built for: nothing about RevCD's architecture would need to change to swap its training objective from "maximize likelihood of the ground-truth semantic embedding" to "maximize expected downstream classification reward," using policy gradients through the sampling chain instead of the ELBO. I haven't tried this — it's a genuinely open question whether the reward signal would be informative enough to beat plain likelihood training here — but it's a concrete, well-defined next experiment rather than a hand-wave.

## Seen/unseen bias as an exploration problem

One more parallel, less about the math and more about the failure mode. Generalized zero-shot learning has a well-known pathology: a classifier trained with full supervision on seen classes and only semantic side-information on unseen ones systematically over-predicts seen classes at test time, even when an unseen class is the correct answer. This is exactly **exploitation bias** in bandit and RL terms — an agent (or classifier) that has accumulated confident, well-calibrated value estimates for a few options will keep choosing them over options it has less certainty about, even when those less-certain options are, in expectation, better. A bandit algorithm without an exploration bonus behaves identically: it locks onto whichever arm looked best early and stops seriously reconsidering the others.

[SEER-ZSL](/projects/seer-zsl/)'s motivation reads almost directly as an answer to this, from the classification side rather than the bandit side: the problem it targets is that noisy, real-world semantic descriptions make the unseen-class "value estimate" unreliable, which — same as an under-explored arm with a badly estimated reward — gets systematically discounted relative to the confidently-known seen classes. Distilling more robust semantic structure before it's used to bridge visual and semantic space is, in bandit language, tightening the estimate for an under-explored option so a decision rule stops unfairly penalizing it. Nobody needs bandit theory to arrive at that fix, and SEER-ZSL certainly didn't — but it's a real instance of the same underlying tension between confidently-known options and honestly-uncertain ones that exploration algorithms in RL exist specifically to manage.

## Key takeaway

"RL for zero-shot learning" turns out to name at least two different things, and it's worth keeping them apart. As an optimization tool — REINFORCE through hard attention, RL fine-tuning of a diffusion sampler — RL machinery solves a differentiability problem that has nothing essential to do with zero-shot generalization; it would matter just as much in a fully-supervised setting. As a structural parallel, the match is much deeper than an analogy: linear bandits, successor features, and bilinear zero-shot classifiers are three names for the same move — factor a compatibility function into a shared representation and a target-specific embedding, and a target with literally zero training signal still gets a usable answer, for the price of one inner product or one closed-form solve. CAST's ridge bridge and its extrapolation residual sit exactly on that line, whether or not anyone building it was thinking in bandit terms. Given how directly the equations line up once you write them side by side, I'd guess this cross-section has more left in it than I found in one post — which was the whole reason for writing it.
