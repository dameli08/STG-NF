# Paper Specification — STG-NF

Reference: Or Hirschorn and Shai Avidan, *Normalizing Flows for Human Pose
Anomaly Detection*, ICCV 2023 (arXiv:2211.10946). This is the architectural
source of truth for the reproduction.

## 1. Problem

Skeleton-based **video anomaly detection**: given per-person 2D pose tracks, learn
the distribution of *normal* human motion and flag low-likelihood motion as
anomalous. Training uses **normal data only** (or normal + a few labelled
abnormal in the supervised variant). Evaluation is **frame-level** via AUC-ROC.

## 2. Representation

- A sample is a spatio-temporal skeleton window `x ∈ R^{C×T×V}`.
  - `V = 18` joints — COCO-17 plus a synthesized **neck** (mean of the two
    shoulders), reordered to the OpenPose layout.
  - `T` = temporal window length (`24` in the main pose experiments).
  - `C = 2` for `(x, y)`; the confidence channel can be added (`C = 3`) and used
    to weight the likelihood.
- The skeleton graph adjacency `A` (uniform partition, `max_hop` neighbourhood) is
  shared by every ST-GCN coupling block.

## 3. Model: spatio-temporal normalizing flow (STG-NF)

A **Glow**-style flow adapted to skeleton graphs. The exact invertible transform
`z = f(x)` gives the change-of-variables likelihood

$$\log p_X(x) = \log p_Z(f(x)) + \log\left|\det \tfrac{\partial f}{\partial x}\right|.$$

The flow is a stack of `L` levels × `K` steps. Each **flow step** composes:

1. **ActNorm** — per-channel affine normalization (data-dependent init to zero
   mean / unit variance on the first batch), log-det = `sum(logs)·T·V`.
2. **Invertible permutation** — 1×1 invertible convolution (LU-parameterised for a
   cheap, stable log-det) or a fixed/shuffled permutation.
3. **Affine coupling** — split channels `z = [z1, z2]`; a network `g(z1)` produces
   `(shift, scale)` and `z2 ← (z2 + shift)·scale`, with
   `scale = sigmoid(s + 2) + ε`. The coupling network `g` is a **spatio-temporal
   graph convolution (ST-GCN)** over `A`, which is what makes the flow
   skeleton-aware. Log-det = `sum(log scale)`.

For `L > 1`, a **squeeze** (`T → T/2`, `C → 2C`) precedes the steps at deeper
levels (trading temporal resolution for channels), matching Glow's multi-scale.

### Prior and supervision

The latent prior is a diagonal Gaussian `N(μ, σ)`. In the **unsupervised** setting
`μ = 0`. In the **supervised** setting a scalar `R` shifts the prior mean to `+R`
for normal and `−R` for abnormal training samples, pushing the two classes apart
in latent space. RetailS trains on normal data only → unsupervised, `R = 0`.

### Objective

Maximize log-likelihood of normal data ⇔ minimize **negative log-likelihood**,
reported in **bits per dimension**:

$$\text{nll} = \frac{-\big(\log p_Z(z) + \log|\det J|\big)}{\log 2 \cdot C \cdot T \cdot V}.$$

Optionally weighted by mean keypoint confidence when `model_confidence` is set.

## 4. Anomaly score & aggregation

- **Window score** = `−nll` (normality; higher ⇒ more normal). Equivalently the
  anomaly score is `nll`.
- Each window score is assigned to the frame at its **temporal centre**
  (`start + T/2`) for that person.
- **Frame score** = the **minimum normality across all people** present in the
  frame (i.e. the most anomalous person dominates).
- Per-video frame scores are **gaussian-smoothed** over time before metrics.

## 5. Training recipe

- Optimizer **Adamax**, `lr = 5·10⁻⁴`, `weight_decay = 5·10⁻⁵`.
- Exponential LR decay (`lr · 0.99^epoch`).
- Gradient clipping (norm `100`).
- `K = 8`, `L = 1`, `hidden = 0` (single ST-GCN coupling layer), permutation
  `permute`, uniform adjacency, `max_hop = 8`.
- Pose augmentation: horizontal flip and small shear (train only).

## 6. Reported results (context, not targets on RetailS)

STG-NF reports AUC-ROC ≈ **85.9%** on ShanghaiTech and strong UBnormal numbers.
On the shoplifting **PoseLift** benchmark (same protocol family as RetailS) STG-NF
is competitive (~AUC-ROC 0.67). RetailS numbers are therefore expected to land in
a similar regime rather than at the ShanghaiTech level — the benchmark, not the
model, sets the ceiling.

## 7. Under-specified details → engineering choices

See `assumptions.md`. Where the paper/repo leave a detail open (missing-joint
handling, small-gap interpolation, threshold selection, EER, PR/EER reporting), we
choose sound defaults and document them, without altering the core methodology.
