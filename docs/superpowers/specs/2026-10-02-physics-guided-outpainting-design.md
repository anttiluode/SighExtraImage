# SighExtraImage v0 — Physics-Guided Outpainting Design

**Date:** 2026-10-02  
**Status:** revised after design review; ready for written-spec approval

## Goal

Test a narrow claim before building a full diffusion product:

> Weak optical evidence inside a visible crop can reduce uncertainty about a scene that lies outside the image boundary, and that evidence can constrain a shared world prior.

The first version is a synthetic inverse-problem benchmark. It is not a claim that arbitrary photographs reveal a unique off-frame scene.

## Core architectural change

The hidden region is represented first as a **low-dimensional physical latent** rather than as raw generated RGB pixels.

Let

```text
z_physical = {
  position,
  size,
  shape,
  albedo / coarse RGB,
  occupancy,
  coarse depth / layout,
  light parameters
}
```

and let the observed boundary signal be

```text
y = A(z_physical) + noise
```

where `A` is a deliberately cheap, differentiable, auditable proxy for indirect light transport onto an informative visible boundary region `R`.

This separates two jobs:

- **physics:** `z_physical -> A(z_physical) -> boundary evidence`;
- **appearance prior:** `z_physical -> plausible hidden image / scene appearance`.

A later diffusion model can supply appearance and semantic plausibility without pretending that every generated pixel is physically measured.

## Scientific questions and gates

The project is staged so each claim can fail independently.

### Gate 0 — Is there recoverable information in the boundary signal?

Given a declared prior over `z_physical`, compare:

1. **prior-only inference** — draw hidden hypotheses from the prior without using `y`;
2. **correct-physics posterior** — weight the same prior hypotheses by `p(y | z, A)`;
3. **wrong-physics posterior** — use a mismatched transport operator.

A positive Gate 0 means only that the synthetic boundary measurement reduces uncertainty about some hidden attributes.

### Gate 1 — Does the physics gradient point toward truth?

Before any diffusion model, test the actual likelihood gradient.

Starting from deliberately wrong hidden hypotheses, optimize

```text
L(z) = || y - A(z) ||^2
```

and measure whether `-grad_z L` moves recoverable attributes toward the true hidden state more often than matched controls.

Primary checks:

- horizontal/vertical position moves toward truth;
- coarse color/albedo moves toward truth;
- size moves toward truth where identifiable;
- correct physics beats wrong-physics gradients;
- gradient norm is finite and does not vanish trivially.

This gate prevents us from assuming that an approximate forward model produces useful guidance.

### Gate 2 — Posterior narrowing in the physical latent

Use the correct likelihood with the same prior and quantify whether posterior uncertainty contracts for the attributes the measurement actually carries.

The goal is not uniqueness. The expected output is a family of compatible hidden states.

### Gate 3 — Generative / diffusion guidance

Only after Gates 0–2 survive, attach a pretrained outpainting prior.

The intended pattern is:

```text
x_t
  -> denoiser predicts x0_hat
  -> map / decode hidden part to z_physical or a differentiable proxy
  -> A(z_physical) predicts y_hat on region R
  -> physics residual ||y - y_hat||^2
  -> guidance gradient nudges the reverse process
```

The diffusion model remains responsible for natural-image plausibility. The physics term constrains only properties supported by the measured boundary evidence.

### Gate 4 — Varjoluotain measurement extraction

Replace the perfect synthetic measurement with

```text
y = Varjoluotain.extract_illumination(I_visible)
```

or an equivalent exported descriptor.

Only at this stage does the project claim to test whether real-pixel leakage extracted by Varjoluotain can constrain outpainting.

## Scene model

Use a small synthetic world rendered onto an expanded canvas. The visible crop occupies the left portion; the hidden object / structure lies entirely outside the crop.

V0 physical latent fields should include at minimum:

- horizontal/vertical position outside the frame;
- coarse size;
- RGB albedo/intensity;
- shape class (`disk`, `rectangle` initially);
- occupancy mask;
- one or two simple light parameters.

Optional coarse depth/layout fields may be added only if the first transport requires them.

The object itself must never enter the known crop.

## Boundary measurement region

Physics is evaluated only where the camera actually has evidence.

Define an informative visible region `R`, initially a narrow wall/floor strip next to the crop boundary. Store:

- clean visible background;
- hidden truth;
- noiseless leakage field on `R`;
- noisy measurement `y`;
- final visible crop;
- mask identifying `R`.

The likelihood is

```text
L(z) = || y_R - A_R(z) ||^2
```

not a loss over the entire hallucinated panorama.

This preserves the distinction:

> stable structure across posterior samples may be measurement-supported; varying fine detail is prior-filled uncertainty.

## Forward model A

Do **not** start with a full path tracer.

Use a fast differentiable proxy capturing first-order effects such as:

- inverse-distance or smooth distance falloff;
- orientation / edge sensitivity;
- blurred first-bounce color contribution;
- soft penumbra from an extended light source;
- ambient term.

A simple form is sufficient for the first gate:

```text
y_hat_i = b + sum_j L_j * rho_j * f(distance(i,j), geometry)
```

The proxy is intentionally approximate. Its adequacy is itself tested by Gate 1 and the wrong-physics control.

## Data generation

For each deterministic seed:

1. sample `z_true` from the declared prior;
2. render the full expanded scene;
3. crop the visible field so hidden geometry is absent;
4. compute `y_clean = A(z_true)` on region `R`;
5. add declared Gaussian noise;
6. save truth, visible crop, region mask, clean/noisy measurement, and all transport parameters.

Evaluation seeds must remain held out from any tuning of noise, proposal count, or success thresholds.

## Gate 0 inference arms

### Prior-only

Draw `N` hypotheses from the same declared prior. Do not evaluate the measurement likelihood.

### Correct physics

For hypotheses `z_i`, compute

```text
E_i = || y - A(z_i) ||^2
log w_i = -E_i / (2 sigma^2)
```

normalize with log-sum-exp, and form weighted posterior summaries/resamples.

### Wrong physics

Use the same hypotheses and posterior machinery but deliberately mismatch the operator, for example:

- horizontal mirror;
- wrong blur width;
- incorrect light direction.

If wrong physics performs as well as correct physics, the physical constraint is not specific.

## Gate 1 gradient controls

For a declared set of initialization offsets:

- run one-step and short multi-step optimization of `L(z)`;
- record cosine between gradient direction and truth direction for continuous attributes;
- record absolute error before/after each step;
- compare correct operator, wrong operator, and random direction controls;
- normalize or separately report each latent component so a large-scale parameter cannot dominate merely by units.

This is the first check that a future DPS gradient would be useful rather than merely differentiable.

## Outputs

For every evaluated scene save:

- observed crop;
- true expanded scene;
- boundary measurement diagnostic;
- prior hypotheses/samples;
- correct-physics posterior samples;
- wrong-physics posterior samples;
- gradient trajectories for Gate 1;
- posterior mean / MAP-style representative only as summaries;
- machine-readable receipt with all seeds and parameters.

A later static report may visualize representative samples, but no visual cherry-pick can define success.

## Metrics

Primary metrics:

- position error;
- size error;
- color/albedo error;
- shape posterior probability / accuracy;
- posterior entropy or credible-interval width;
- measurement residual;
- gradient-to-truth cosine / error reduction for Gate 1.

Secondary image-space metrics may be added later for the hidden region, but they do not replace latent-level evaluation.

## Predeclared success gates

### Gate 0

Across a fixed multi-seed evaluation set, correct-physics posterior should show:

1. lower median hidden-position error than prior-only;
2. lower median color/intensity error than prior-only;
3. reduced posterior uncertainty for at least one continuous attribute;
4. better performance than wrong physics.

Partial improvement is reported as partial, with the informative attributes named explicitly.

### Gate 1

Across fixed perturbed initializations:

1. correct-physics gradient improves at least one declared recoverable latent attribute more often than chance/random-direction controls;
2. correct physics outperforms wrong physics;
3. multi-step optimization does not merely reduce measurement loss while systematically moving latent truth error upward.

Gate 1 may fail even if Gate 0 passes; that would mean the measurement contains information but the chosen differentiable proxy/parameterization provides poor local guidance.

## Claim boundary

A positive Gates 0–1 result means:

> In this declared synthetic transport model, weak in-frame optical leakage contains recoverable information about an out-of-frame physical latent, and the specified likelihood both narrows a shared prior and supplies locally useful guidance for at least some hidden attributes.

It does **not** mean:

- a unique real scene can be recovered from an arbitrary JPEG;
- the proxy transport accurately models a real room;
- a diffusion model has recovered ground truth;
- generated detail unsupported by the likelihood is measured;
- a wrong forward model can be trusted simply because its optimization converges.

## Architecture

```text
src/sighextraimage/
  latent.py         # physical latent z and parameter transforms
  scene.py          # synthetic full-scene and crop rendering
  transport.py      # differentiable/auditable A(z) on boundary region R
  prior.py          # declared latent prior
  inference.py      # prior, posterior weighting/resampling
  guidance.py       # Gate 1 gradient tests / optimization
  metrics.py        # attribute, uncertainty, gradient metrics
  receipts.py       # deterministic run metadata
  cli.py            # generate / infer / gradient / benchmark commands

tests/
  test_latent.py
  test_scene.py
  test_transport.py
  test_inference.py
  test_guidance.py
  test_metrics.py

results/
  receipts/

site/
  index.html         # only after measured results exist
```

## Numerical stack

Use **PyTorch CPU** from the beginning for `z` and `A(z)` so Gate 1 uses the exact same differentiable operator that later guidance will need. Keep the problem small enough to run on CPU in seconds to minutes.

Do not add a pretrained diffusion dependency until the exact posterior and gradient gates have been run and recorded.

## Interfaces reserved for generative work

The physics side should expose:

```text
sample_prior(n, rng) -> z candidates
render_hidden(z) -> hidden image / attributes
predict_boundary(z, scene_context) -> y_hat
physics_loss(z, y, scene_context) -> scalar
```

A later generative adapter can provide:

```text
denoise(x_t, t, known_crop) -> x0_hat
latent_from_hidden(x0_hat) -> z_proxy
```

or directly implement a differentiable boundary predictor from the generated hidden image, while reusing the same region mask, likelihood definition, metrics, and receipts.

## Verification strategy

Use test-driven development.

Required checks:

- hidden object never appears in known crop;
- zero albedo/intensity produces zero object leakage;
- transport deterministic before noise;
- moving/changing hidden object changes `y` in expected direction;
- posterior weights finite and normalized;
- low-noise identifiable toy posterior concentrates near truth;
- uninformative transport collapses guided inference to prior;
- wrong physics is genuinely mismatched;
- autograd gradient matches finite differences on selected latent coordinates;
- an easy controlled Gate 1 case moves toward truth;
- fixed seeds reproduce receipts within floating-point tolerance.

## First milestone

The first useful repository state is complete when:

1. PyTorch CPU package and CLI run from one command;
2. tests pass;
3. Gate 0 receipt compares prior-only, correct-physics, and wrong-physics arms;
4. Gate 1 receipt records gradient direction/error changes under correct, wrong, and random controls;
5. README states the measured outcome, including negative results;
6. no diffusion dependency is added merely to make the demo look impressive.

## Roadmap after v0

```text
Gate 0: information in boundary signal
       -> Gate 1: useful physics gradient
       -> Gate 2: posterior narrowing / ambiguity map
       -> Gate 3: diffusion-guided outpainting
       -> Gate 4: real Varjoluotain-extracted measurement
```

The guiding distinction is:

> Plain outpainting asks what could plausibly be outside the crop. SighExtraImage asks which plausible hidden worlds remain compatible with the photons that leaked into the visible boundary.
