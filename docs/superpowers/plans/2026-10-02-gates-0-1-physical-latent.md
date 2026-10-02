# SighExtraImage Gates 0–1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest falsifiable CPU benchmark showing whether weak in-frame boundary light carries recoverable information about an out-of-frame physical latent, and whether the correct physics gradient points hidden hypotheses toward truth.

**Architecture:** A low-dimensional physical latent `PhysicalLatent` generates both a hidden scene rendering and a weak boundary measurement through a cheap differentiable `BoundaryTransport`. Gate 0 compares prior-only, correct-physics, and wrong-physics posteriors over the same candidate set. Gate 1 differentiates the same boundary loss with respect to continuous latent fields and tests whether correct-physics gradients reduce truth error better than wrong-physics and random controls.

**Tech Stack:** Python 3.11+, PyTorch CPU, NumPy only where convenient, pytest, standard-library JSON/argparse/pathlib/dataclasses.

**Spec:** `docs/superpowers/specs/2026-10-02-physics-guided-outpainting-design.md`

## Global Constraints

- No diffusion model or external model weights in Gates 0–1.
- Hidden geometry must remain entirely outside the known crop.
- Physics loss is evaluated only on the explicit observed boundary region `R`.
- Prior-only, correct-physics, and wrong-physics arms must use the same candidate hypotheses.
- The transport must be deterministic before declared Gaussian noise.
- Wrong physics must be intentionally mismatched and must remain available in receipts.
- Results are attribute/posterior metrics, not aesthetic image scores.
- All benchmark runs are seed-controlled and JSON-receipted.
- A negative Gate 0 or Gate 1 result is preserved as the scientific outcome.

## Review Focus

1. **Latent-unit imbalance:** position/color/size gradients should be normalized or reported per component so one parameter scale cannot dominate Gate 1. Task 4 pins this with componentwise cosine/error tests.
2. **Boundary leakage accidentally revealing the object:** the visible crop must never include hidden occupancy. Task 2 tests zero overlap explicitly.
3. **Wrong-physics control too weak:** mirrored/incorrect transport must measurably differ on an asymmetric test case. Task 2 tests this directly.
4. **Posterior underflow/degeneracy:** log weights must remain finite and normalized under low-noise settings. Task 3 tests log-sum-exp normalization and effective sample size.
5. **Apparent improvement caused only by candidate reuse or RNG mismatch:** all Gate 0 arms must evaluate the exact same candidate tensor per scene; Task 3 tests candidate identity across arms.

---

### Task 1: Package scaffold and public interfaces

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/sighextraimage/__init__.py`
- Create: `src/sighextraimage/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: CLI entry point `sighextraimage = sighextraimage.cli:main`.
- Produces subcommands `generate`, `gate0`, `gate1`, `benchmark` with shared `--seed`, `--output`, and CPU-only execution.

- [ ] **Step 1: Write failing CLI parser tests** asserting `--help` succeeds and each subcommand parses its required arguments without importing model weights.
- [ ] **Step 2: Run** `pytest tests/test_cli.py -q` and verify failure because the package/CLI does not exist.
- [ ] **Step 3: Implement minimal package metadata and `build_parser() -> argparse.ArgumentParser` plus `main(argv: list[str] | None = None) -> int`; command bodies may raise a clear `NotImplementedError` until later tasks wire them.
- [ ] **Step 4: Run** `pytest tests/test_cli.py -q` and verify PASS.
- [ ] **Step 5: Commit** scaffold and parser as `chore: scaffold SighExtraImage benchmark`.

### Task 2: Physical latent, rendering, boundary mask, and differentiable transport

**Files:**
- Create: `src/sighextraimage/scene.py`
- Create: `src/sighextraimage/transport.py`
- Test: `tests/test_scene.py`
- Test: `tests/test_transport.py`

**Interfaces:**
- Produces dataclass `PhysicalLatent(position_x: Tensor, position_y: Tensor, size: Tensor, rgb: Tensor, shape_code: Tensor, light_gain: Tensor)` with a batch dimension.
- Produces `sample_prior(n: int, *, generator: torch.Generator, device: str = "cpu") -> PhysicalLatent`.
- Produces `render_hidden(latent: PhysicalLatent, config: SceneConfig) -> torch.Tensor` returning `[N,H,W,3]`.
- Produces `visible_crop_mask(config: SceneConfig) -> torch.Tensor` and `boundary_region_mask(config: SceneConfig) -> torch.Tensor`.
- Produces `BoundaryTransport(config: TransportConfig)` with `forward(latent: PhysicalLatent, scene: SceneConfig) -> torch.Tensor` returning `[N,B,3]` boundary measurements.
- Produces `WrongBoundaryTransport` with an intentional horizontal mirror or wrong kernel scale.

- [ ] **Step 1: Write failing scene tests** for deterministic seeded prior sampling, tensor shapes, and zero hidden-object overlap with the known crop mask.
- [ ] **Step 2: Run** `pytest tests/test_scene.py -q` and verify FAIL.
- [ ] **Step 3: Implement `SceneConfig`, `PhysicalLatent`, `sample_prior`, and `render_hidden` using differentiable PyTorch operations; start with disk and soft-rectangle occupancy masks and keep the object center strictly outside the crop boundary.
- [ ] **Step 4: Run** `pytest tests/test_scene.py -q` and verify PASS.
- [ ] **Step 5: Write failing transport tests** asserting: zero RGB or zero light gain gives zero leakage; transport is deterministic; moving the object changes the boundary measurement centroid; RGB changes the matching measurement channel; wrong physics differs on an asymmetric case; output only covers region `R`.
- [ ] **Step 6: Run** `pytest tests/test_transport.py -q` and verify FAIL.
- [ ] **Step 7: Implement `BoundaryTransport.forward` as a fast first-order proxy: sample/aggregate hidden occupancy/albedo through smooth inverse-distance falloff plus a Gaussian/soft-penumbra kernel onto the right-edge boundary strip. Implement wrong physics by mirroring horizontal hidden geometry before transport.
- [ ] **Step 8: Run** `pytest tests/test_transport.py -q` and verify PASS.
- [ ] **Step 9: Commit** as `feat: add physical latent and boundary transport`.

### Task 3: Gate 0 exact posterior and controls

**Files:**
- Create: `src/sighextraimage/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: `PhysicalLatent`, `sample_prior`, `BoundaryTransport`, `WrongBoundaryTransport`.
- Produces `gaussian_log_likelihood(observed: Tensor, predicted: Tensor, sigma: float) -> Tensor`.
- Produces `normalize_log_weights(log_w: Tensor) -> Tensor`.
- Produces dataclass `PosteriorSummary(weights, mean_position, mean_size, mean_rgb, shape_probability, entropy, ess)`.
- Produces `evaluate_gate0(z_true, observed_y, candidates, correct_transport, wrong_transport, scene, sigma) -> Gate0Result` where every arm references the exact same `candidates` object/value tensor.

- [ ] **Step 1: Write failing numerical tests** for finite normalized weights, invariant normalization to additive constants, and posterior = prior under an explicit zero-information transport.
- [ ] **Step 2: Run** `pytest tests/test_inference.py::test_normalize_log_weights tests/test_inference.py::test_zero_information_transport_returns_prior -q` and verify FAIL.
- [ ] **Step 3: Implement Gaussian log likelihood, log-sum-exp normalization, entropy, ESS, and weighted summary helpers.
- [ ] **Step 4: Run** those tests and verify PASS.
- [ ] **Step 5: Write failing Gate 0 tests** asserting the same candidate values are used by all arms; low-noise correct physics lowers position/color error versus prior-only in a declared identifiable case; correct physics beats mirrored wrong physics on that case.
- [ ] **Step 6: Run** `pytest tests/test_inference.py -q` and verify FAIL for missing Gate 0 implementation.
- [ ] **Step 7: Implement `evaluate_gate0` without resampling candidates: prior-only uses uniform weights, correct/wrong arms use likelihood weights over the same candidate batch.
- [ ] **Step 8: Run** `pytest tests/test_inference.py -q` and verify PASS.
- [ ] **Step 9: Commit** as `feat: add Gate 0 posterior benchmark`.

### Task 4: Gate 1 gradient-direction falsifier

**Files:**
- Create: `src/sighextraimage/gradient_gate.py`
- Test: `tests/test_gradient_gate.py`

**Interfaces:**
- Consumes: `BoundaryTransport`, `WrongBoundaryTransport`, `PhysicalLatent`, `SceneConfig`.
- Produces `continuous_vector(latent: PhysicalLatent) -> Tensor` for `[position_x, position_y, size, r, g, b, light_gain]` only; discrete `shape_code` is held fixed during Gate 1.
- Produces `boundary_loss(latent, observed_y, transport, scene) -> Tensor`.
- Produces dataclass `GradientStepResult(component_cosines, error_before, error_after, gradient_norm, finite)`.
- Produces `evaluate_gate1(z_true, z_init, observed_y, correct_transport, wrong_transport, scene, step_size, random_generator) -> Gate1Result` with correct, wrong, and norm-matched random directions.

- [ ] **Step 1: Write failing gradient tests** on a simple asymmetric scene: gradients are finite/nonzero; autograd reaches position, size, RGB, and light gain; componentwise truth-direction cosine is reported rather than one unit-sensitive aggregate.
- [ ] **Step 2: Run** `pytest tests/test_gradient_gate.py -q` and verify FAIL.
- [ ] **Step 3: Implement continuous latent packing/unpacking and differentiable boundary loss without detaching tensors.
- [ ] **Step 4: Run** gradient plumbing tests and verify PASS.
- [ ] **Step 5: Add failing Gate 1 tests** asserting one small correct-physics step reduces declared continuous-latent error in an identifiable case; wrong physics does worse; norm-matched random direction does not systematically match correct-physics improvement.
- [ ] **Step 6: Run** `pytest tests/test_gradient_gate.py -q` and verify FAIL for missing evaluator.
- [ ] **Step 7: Implement `evaluate_gate1`; scale each continuous component by its prior range before cosine/error comparisons, and use the same normalized coordinate system for correct/wrong/random controls.
- [ ] **Step 8: Run** `pytest tests/test_gradient_gate.py -q` and verify PASS.
- [ ] **Step 9: Commit** as `feat: add Gate 1 gradient-direction test`.

### Task 5: Metrics, receipts, and deterministic benchmark runner

**Files:**
- Create: `src/sighextraimage/metrics.py`
- Create: `src/sighextraimage/receipts.py`
- Create: `src/sighextraimage/benchmark.py`
- Modify: `src/sighextraimage/cli.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces per-arm metrics: position error, size error, RGB error, shape probability/accuracy, entropy/credible-width proxy, measurement residual, ESS.
- Produces Gate 1 metrics: component cosine, normalized latent error before/after, error reduction, gradient norm.
- Produces `run_benchmark(seeds: Sequence[int], *, candidates: int, sigma: float, output: Path) -> dict`.
- Produces JSON receipt with schema id, seed list, scene/transport config, Gate 0 per-scene and aggregate metrics, Gate 1 per-scene and aggregate metrics, and instability count.

- [ ] **Step 1: Write failing metric tests** asserting zero error at truth, positive error after controlled offsets, and entropy/ESS finite for normalized weights.
- [ ] **Step 2: Run** `pytest tests/test_metrics.py -q` and verify FAIL.
- [ ] **Step 3: Implement focused metric helpers and JSON-safe serialization.
- [ ] **Step 4: Run** metric tests and verify PASS.
- [ ] **Step 5: Write failing benchmark tests** using 3 tiny deterministic seeds: receipt schema is stable; rerunning with the same seed/config yields numerically identical metrics; correct/wrong/prior arms share candidate count; output JSON exists.
- [ ] **Step 6: Run** `pytest tests/test_benchmark.py -q` and verify FAIL.
- [ ] **Step 7: Implement benchmark orchestration and wire CLI `gate0`, `gate1`, and `benchmark` commands.
- [ ] **Step 8: Run** `pytest tests/test_benchmark.py -q` and verify PASS.
- [ ] **Step 9: Run full suite** `pytest -q` and require all tests PASS before scientific execution.
- [ ] **Step 10: Commit** as `feat: add deterministic Gates 0-1 benchmark receipts`.

### Task 6: First declared scientific run and result boundary

**Files:**
- Create: `results/receipts/gates-0-1-v0.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: verified benchmark command.
- Produces: first fixed multi-seed receipt and concise measured claim.

- [ ] **Step 1: Freeze first-run settings before execution** in README or command block: seed set, candidate count, measurement noise, scene/transport configs, Gate 1 initialization offsets, step size.
- [ ] **Step 2: Run** `sighextraimage benchmark ... --output results/receipts/gates-0-1-v0.json`.
- [ ] **Step 3: Inspect receipt for non-finite values, ESS collapse, or geometry leakage; if any implementation fault appears, stop and use systematic debugging before interpreting results.
- [ ] **Step 4: Evaluate Gate 0 only against the predeclared criteria: correct physics must improve at least position or color uncertainty/error versus uniform prior and outperform wrong physics on that recovered attribute.
- [ ] **Step 5: Evaluate Gate 1 only against the predeclared criteria: correct-physics gradients must show positive truth-directed/error-reducing behavior above wrong/random controls for at least one identifiable continuous attribute; report null components explicitly.
- [ ] **Step 6: Update README with the exact receipt values and narrow claim; do not mention diffusion success because Gate 3 has not run.
- [ ] **Step 7: Run** `pytest -q` again and verify PASS after README/result changes.
- [ ] **Step 8: Commit** as `results: record SighExtraImage Gates 0-1 v0`.

## Self-review outcome

- Spec coverage for Gates 0–1 is complete: physical latent, observed-region-only transport, prior-only/correct/wrong posteriors, gradient-direction falsifier, normalized latent coordinates, deterministic receipts, and negative-result preservation all have owning tasks.
- Diffusion guidance and Varjoluotain extraction are intentionally excluded from this plan and require their own follow-up plan only after this receipt exists.
- Function/type names are consistent across tasks.
- Review-focus failure modes each have an explicit test in the owning task.
