# Mathematical checks before a scale-conditioned method

All propositions below are elementary consequences of dimensions, deterministic transformations or the proposed objective. They are not novelty claims or empirical carrier results.

## 1. Physical support is not pixel support

For square views with n pixels per side and spacing g metres/pixel, physical width is L=ng, and a pixel-area threshold a_px corresponds to a_m2=a_px g². Thus input dimensions, spacing and physical field of view cannot all stay fixed when one is varied.

At n=1024 and g=0.045, L=46.08m and a30=0.06075m². For BAM's header-verified range about0.01612–0.01793m/pixel, L is about16.51–18.36m and a30 about0.00780–0.00965m². These are dimensional calculations, not thresholds chosen from VAL. Different view contexts and physical floors can affect inference; their causal contribution has not been measured.

Resolution experiments must distinguish delivered grid spacing, effective image detail, field of view, and postprocessing units. Synthetic blur/decimation does not reproduce an actual sensor, orthomosaic formation, noise, occlusion or acquisition change.

## 2. A limitation of "metric-scale" deployment from f(I) alone

Suppose the admissible acquisition family permits the same RGB array I at different spacings g1 and g2, with no observable calibration cue or uniquely identifiable physical-size prior. A deterministic f(I) returns the same output for both. If the desired policy depends on a physical quantity that differs, such as a_px g² crossing a fixed area threshold, no such f can implement that policy exactly in both cases.

This follows immediately from identical input and different required decisions. It does not rule out statistical scale estimation under a learned population prior. It does rule out an unconditional exact metric guarantee from arbitrary RGB. Training with physical metadata and deployment without it may learn a useful prior; it must be described as such. RGB-only sensing can be compatible with supplied geospatial metadata, but that option changes the proposed inference contract and is not currently assumed.

## 3. Lossless re-encoding gives a valid information-equivalence control

Let U_r repeat each pixel into an r×r block for integer r≥2, without smoothing, clipping, quantization change or JPEG compression. Let A_r pick the same designated pixel in every block. Then A_r U_r I=I exactly. Therefore U_r is injective.

For random image I and latent label variable Z, sigma(I)=sigma(U_r I), because each is a measurable deterministic function of the other. Hence

\[
P(Z\mid I)=P(Z\mid U_r I)
\]

up to versions of conditional probabilities. This is a standard information-equivalence observation. A unique Bayes decision under an appropriately transported loss would preserve its meaning. It does not force an arbitrary trained network to be equivariant or exclude ties between equivalent decisions.

Consequently, if the actual tensor reaching the carrier is U_r I and its predictions change after exact coordinate transport, discarded RGB information cannot explain that particular change. Internal receptive field, encoder-grid sampling, initialization, learned scale priors, normalization, numerical details or output decoding remain competing explanations. The manipulation does not isolate any one of them.

If preprocessing first downsizes U_r I, this argument does not apply to the tensor actually reaching the network. Record that tensor and verify the left-inverse property there, not merely on a saved input PNG.

## 4. Downsampling is a different operation

For illustration define D_2 I as the average of each nonoverlapping2×2 block. A constant block and a nonconstant block with the same channel mean have identical D_2 output. D_2 is noninjective. The averaging filter is a fixed synthetic low-pass/integration operator, not a complete alias-free or sensor model.

Upsampling D_2 I to the original size does not restore the deleted detail. Use a coordinate pullback P_s for masks; do not denote it as a genuine inverse of the image observation operator. A large GT support ratio, high teacher confidence, a low-frequency image or agreement between models does not independently prove instance identity remains observable after D_2.

An exact replication U_2 D_2 I, however, contains precisely the same information as D_2 I. Their comparison isolates raster encoding conditional on that degraded observation.

Physical tree count does not change when the image is resampled. The uncertainty of an RGB-only decision can change. Do not silently redefine the benchmark target or delete small reference masks after degradation to make the method appear consistent. Published-reference fidelity and uncertainty about distinguishable instances remain separately reported quantities.

## 5. The sketched objective is not yet an algorithm

For n and m predicted masks define an admissible partial correspondence

\[
\Pi_{n,m}=\{\pi\in\{0,1\}^{n\times m}:
\sum_j\pi_{ij}\le1,\ \sum_i\pi_{ij}\le1\}.
\]

A fully specified finite comparison cost could take the standard form

\[
C(\pi)=\sum_{ij}\pi_{ij}w_{ij}e_{ij}
 +\sum_i\alpha_i(1-\sum_j\pi_{ij})
 +\sum_j\beta_j(1-\sum_i\pi_{ij}),
\quad
e_{ij}=1-\operatorname{IoU}(M_i,P_sM'_j).
\]

This is conventional partial assignment, not a proposed novel loss. Nonnegative weights and explicit finite penalties make the optimization defined, but do not make it useful supervision:

- With zero/uncontrolled unmatched cost, pi=0 can avoid consistency supervision.
- A learned gate can lower all w values, making wrong matched geometries cheap. Merely detaching it in one branch does not establish calibrated observability.
- Identical but wrong singleton predictions, or empty predictions in both views, can be perfectly consistent. Consistency alone cannot establish correct instance count. An independent segmentation anchor is necessary but not automatically sufficient.
- Partial one-to-one matching cannot represent every valid coarse-scale ambiguity. Two fine reference instances may become unresolved in one coarse appearance; forcing one child to match and penalizing the other may impose an unjustified count target.
- A gate estimated from current predictions risks circularly certifying the model's own omissions. Mask rasterizability is geometric feasibility, not visual observability.

Before training, one must specify gate targets/calibration, gradients, which objects may remain unmatched, how ambiguity/count alternatives are represented, and how the objective avoids newly recovered/lost-instance trade-offs. No such inference-valid observability estimator has been established in this project. Do not introduce this loss simply because its notation is well formed.

## 6. What can be inferred without new human labels

Lossless input equivalence is provable without annotator participation. It can establish representation-of-input sensitivity and quantify fidelity to unchanged published references. It cannot establish that those references are stable biological identities, that a disputed crown is visually resolvable, or that the observed sensitivity caused BAM's cross-dataset errors. Those stronger claims retain their evidence requirements.
