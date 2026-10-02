# SighExtraImage v0 — Physics-Guided Outpainting Design

**Date:** 2026-10-02  
**Status:** approved conversational design, written specification for review

## Goal

Test a narrow claim before building a full diffusion product:

> Weak optical evidence inside a visible crop can reduce uncertainty about a scene that lies outside the image boundary, and that evidence can constrain an outpainting prior.

The first version is a synthetic inverse-problem benchmark, not a claim that arbitrary photographs reveal a unique off-frame scene.

## Scientific question

Let `x_outer` denote a hidden scene outside the camera crop and let `y` be a weak boundary measurement produced by that hidden scene through a known light-transport operator `A`:

```text
y = A(x_outer) + noise
```

Given the same prior over plausible hidden scenes, compare:

1. **Prior-only expansion** — sample plausible `x_outer` without using `y`.
2. **Physics-guided expansion** — sample from the posterior induced by the same prior and the likelihood of `y`.

The v0 success criterion is not visual attractiveness. It is whether the physics-guided condition reduces held-out error or posterior uncertainty for hidden attributes.

## Scope

### Hidden scene

Use a small 2-D synthetic world rendered onto an expanded canvas. The visible crop occupies the left portion; a single hidden object lies entirely to the right of the crop.

V0 hidden attributes:

- horizontal/vertical position outside the frame;
- coarse size;
- RGB color/intensity;
- shape class from a small declared set (for example disk vs rectangle).

The object itself must never enter the known crop.

### Visible evidence

The known crop contains a simple visible wall/floor region. The hidden object influences that region only through a weak differentiable transport model, for example a distance/angle-dependent blurred irradiance field.

The transport should be deliberately simple enough to audit analytically. It is an inverse-problem toy, not a photorealistic renderer.

The benchmark records separately:

- the clean visible background;
- the hidden object;
- the noiseless leakage field;
- the noisy measurement `y`;
- the final visible crop.

This separation prevents us from confusing scene generation with the measurement operator.

## Three implementation approaches considered

### A. Exact synthetic posterior over object attributes — **v0 choice**

Sample candidate hidden scenes from the declared prior, render their predicted leakage, and compute likelihood weights from the measurement residual. Use importance sampling / resampling (or a small grid where feasible) to obtain posterior samples.

**Advantages:** cheap, transparent, deterministic receipts, no external model weights, direct information-gain measurement.  
**Disadvantage:** the prior is not yet a natural-image diffusion model.

### B. Small learned score model trained only on the synthetic scene family

Train a tiny diffusion/score network on the hidden-image extensions and add likelihood guidance during reverse diffusion.

**Advantages:** exercises the actual score-guidance machinery.  
**Disadvantages:** training noise, more compute, and a negative result can be blamed on the learned prior rather than the physical hypothesis.

### C. Large pretrained image diffusion prior

Use a general image diffusion model and a differentiable renderer / measurement likelihood to guide outpainting.

**Advantages:** closest to the eventual application.  
**Disadvantages:** heavy dependencies/weights, harder reproducibility, and much weaker scientific attribution in the first gate.

V0 chooses **A**. The prior and measurement interfaces are intentionally separated so B/C can replace A later without changing the evaluation protocol.

## Data generation

For each deterministic seed:

1. sample hidden attributes from a declared prior;
2. render the full expanded scene;
3. crop the visible field so the hidden object itself is absent;
4. render the leakage measurement through `A`;
5. add declared Gaussian measurement noise;
6. save truth and measurement metadata.

Use train/fit language only if a learned prior is later introduced. V0 itself requires no training split, but evaluation seeds must remain held out from any tuning of noise levels, proposal counts, or success thresholds.

## Inference arms

### Prior-only

Draw `N` hidden-scene candidates from the same declared prior. Do not evaluate the measurement likelihood. Summaries come from these samples alone.

### Physics-guided

Draw the same number of candidates from the same prior, compute

```text
E_i = || y - A(x_outer_i) ||^2
w_i ∝ exp(-E_i / (2 sigma^2))
```

and form a weighted/resampled posterior.

For numerical stability use log weights and normalize with log-sum-exp.

### Oracle sanity check

Evaluate `A` on the true hidden scene. The true measurement residual should be consistent with the injected noise. This is an implementation check, not a comparison arm.

### Wrong-physics negative control

Run the same posterior procedure with a deliberately mismatched transport operator (for example flipped horizontal geometry or incorrect blur scale). If this performs as well as the correct operator, the claimed physical constraint is not specific.

## Outputs

For every evaluated scene save:

- observed crop;
- true expanded scene;
- leakage measurement / diagnostic map;
- several prior-only expansion samples;
- several physics-guided expansion samples;
- posterior mean / MAP-style representative only as a summary, never as proof of uniqueness;
- machine-readable receipt with all seeds and parameters.

A small static report should place the prior-only and guided expansions beside the truth and visualize uncertainty across samples.

## Metrics

Primary metrics are hidden-attribute metrics, not pixel prettiness:

- position error;
- size error;
- color error;
- shape-class posterior probability / accuracy;
- posterior entropy or credible-interval width for continuous attributes;
- measurement residual on held-out truth.

Secondary image-space metrics may be reported for the hidden region, but they must not replace attribute-level evaluation.

## Predeclared v0 gate

Across a fixed multi-seed evaluation set, the physics-guided arm should show all of the following:

1. lower median hidden-position error than prior-only;
2. lower median color/intensity error than prior-only;
3. reduced posterior uncertainty for at least one continuous hidden attribute;
4. correct-physics performance better than the wrong-physics control.

If only one or two attributes improve, report the result as partial and identify which information the measurement actually carries.

No gate is based on a cherry-picked image.

## Claim boundary

A positive v0 means:

> In this declared synthetic transport model, weak in-frame optical leakage contains recoverable information about an out-of-frame scene, and an explicit likelihood can use that information to constrain a shared prior.

It does **not** mean:

- a unique real scene can be recovered from an arbitrary JPEG;
- the synthetic transport accurately models a real room;
- a diffusion model has recovered ground truth;
- generated details unsupported by the likelihood are measurements.

The correct interpretation of uncertain regions is a posterior family of compatible expansions, not one authoritative panorama.

## Architecture

Proposed package layout:

```text
src/sighextraimage/
  scene.py          # hidden-scene parameterization and rendering
  transport.py      # differentiable/auditable leakage operator A
  inference.py      # prior sampling, likelihood, posterior resampling
  metrics.py        # attribute and uncertainty metrics
  receipts.py       # deterministic run metadata
  cli.py            # generate / infer / benchmark commands

tests/
  test_scene.py
  test_transport.py
  test_inference.py
  test_metrics.py

results/
  receipts/

site/
  index.html         # generated/static experiment summary after results exist
```

Keep NumPy as the default numerical dependency unless a later diffusion adapter requires PyTorch. The first benchmark should run on CPU in seconds to minutes.

## Interfaces reserved for later diffusion work

The posterior code should depend on a minimal prior interface:

```text
sample_prior(n, rng) -> hidden candidates
render_hidden(candidate) -> hidden image / attributes
```

A later score/diffusion implementation can add something like:

```text
score(z_t, t, known_crop) -> score estimate
```

while reusing the same `transport`, metrics, receipts, and benchmark scenes.

That follow-up should only happen after the exact-posterior v0 establishes whether the synthetic boundary measurement carries enough information to be worth guiding a generative model.

## Verification strategy

Use test-driven development.

Required implementation checks:

- hidden object is never visible in the known crop;
- transport is deterministic before noise;
- zero hidden intensity produces zero leakage;
- moving/changing the hidden object changes the measurement in the expected direction;
- posterior weights normalize and remain finite;
- with vanishing noise and an identifiable toy case, the posterior concentrates near truth;
- with an uninformative transport (`A=0`), physics-guided inference collapses to the prior;
- wrong-physics control is genuinely mismatched;
- fixed seeds reproduce receipts exactly within floating-point tolerance.

## First milestone

The first useful repository state is complete when:

1. the CPU benchmark is runnable from one command;
2. tests pass;
3. a multi-seed receipt compares prior-only, correct-physics, and wrong-physics arms;
4. the README states the measured result, including a negative result if the gate fails;
5. no diffusion dependency is added merely to make the demo look impressive.
