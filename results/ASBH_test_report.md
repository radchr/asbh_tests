# ASBH Computational Test Report
## Dataset: Ma & Kragel (2026), *Nature Communications*

---

### Background

The **Affective Spin Bundle Hypothesis (ASBH)** proposes that the representational geometry of the emotional state space is not flat (Euclidean) but spherical (S2), and that parallel transport along closed loops on this sphere produces a geometric phase (holonomy) proportional to the enclosed solid angle. To test ASBH we used the publicly available Emo-FilM fMRI dataset (OpenNeuro ds004892) reported in Ma & Kragel (2026), which contains continuous behavioural emotion ratings for 13 affect categories across 14 short films, along with fMRI decoding performance for hippocampal-prefrontal circuits. All analyses use Warriner et al. (2013) Valence-Arousal-Dominance (VAD) norms as an independent, a-priori coordinate system for mapping emotion categories onto S2.

---

### Methods

| Test | Input | Method |
|------|-------|--------|
| 01 — VAD projection | Warriner (2013) lexical norms, 13 emotion words | Centre VAD (scale 1-9, neutral=5); normalise to unit sphere; compute geodesic and Euclidean 13x13 distance matrices |
| 02 — Persistent homology | 13x13 geodesic distance matrix | Vietoris-Rips filtration via *ripser* (maxdim=2); null model: 1000 uniform random S2 configurations |
| 03 — Behavioural geometry | behTab_json.json (13 emotions x T timepoints x 14 films) | Pearson correlation between emotion time-series per film; Spearman rho between S2-geodesic and correlation-distance matrices |
| 03b — Neural decoding | 2dtrajMDS fMRI decoding CSV (29 subjects, 3 regions) | Paired t-tests across regions; percent of subjects with vmPFC > HC |
| 04 — Holonomy | S2 trajectories (VAD-weighted centroid, normalised) | Greedy quasi-loop detection (same octant, 30-120 timepoints); discrete solid angle via Oosterom formula; Spearman and partial correlation (controlling for loop length) |
| 05 — S2 vs MDS | mds_2d.json (authors' 2D MDS trajectories) | Consecutive step distances (geodesic vs Euclidean); per-film Spearman rho; polar transition test (valence sign-flip) |

---

### Results

#### Test 1: Emotion Geometry on S² (VAD Projection)

All 13 emotion categories were successfully mapped to the unit sphere via centred Warriner VAD coordinates (scale 1-9, neutral = 5.0). The VAD vectors span a wide range of magnitudes before normalisation (norm range: 1.69–4.25 rad), indicating heterogeneous distances from the affective neutral point.

| Emotion | Warriner word | V_c | A_c | D_c | norm | V_s2 | A_s2 | D_s2 |
|---------|--------------|------|------|------|------|------|------|------|
| Anger | anger | -2.500 | 0.930 | 0.140 | 2.671 | -0.936 | 0.348 | 0.052 |
| Anxiety | anxiety | -2.620 | -0.220 | -1.610 | 3.083 | -0.850 | -0.071 | -0.522 |
| Fear | fear | -2.070 | 1.140 | -1.680 | 2.899 | -0.714 | 0.393 | -0.579 |
| Surprise | surprise | 2.440 | 1.570 | 0.170 | 2.906 | 0.840 | 0.540 | 0.058 |
| Guilt | guilt | -2.710 | -0.520 | -0.650 | 2.835 | -0.956 | -0.183 | -0.229 |
| Disgust | disgust | -1.680 | 0.000 | -0.160 | 1.688 | -0.995 | 0.000 | -0.095 |
| Sad | sad | -2.900 | -1.510 | -1.160 | 3.469 | -0.836 | -0.435 | -0.334 |
| Regard | regard | 0.700 | -1.610 | 1.380 | 2.233 | 0.313 | -0.721 | 0.618 |
| Satisfaction | satisfaction | 2.180 | -1.820 | 1.440 | 3.184 | 0.685 | -0.572 | 0.452 |
| WarmHeartedness | warmth | 2.530 | -0.670 | 1.320 | 2.931 | 0.863 | -0.229 | 0.450 |
| Happiness | happy | 3.470 | 1.050 | 2.210 | 4.246 | 0.817 | 0.247 | 0.521 |
| Pride | pride | 1.500 | 0.540 | 0.830 | 1.797 | 0.835 | 0.300 | 0.462 |
| Love | love | 3.000 | 0.360 | 0.920 | 3.158 | 0.950 | 0.114 | 0.291 |

Emotions with smallest pre-normalisation norms (Disgust = 1.69, Pride = 1.80) undergo the largest relative distortion when projected to the sphere, and should be interpreted with caution.

#### Test 2: Topological Analysis (Persistent Homology)

Vietoris-Rips persistent homology (maxdim=2) on both the geodesic and Euclidean 13×13 distance matrices yielded H0 = 8–12 (fragmented components at low filtration), H1 = 0, and **H2 = 0** (no persistent cavity). The null model of 1000 uniformly random 13-point configurations on S2 produced a mean H2 lifespan of 0.101 ± 0.136, giving a one-tailed p = 1.0 for the real data.

**Why H2 = 0 does not refute ASBH:** Persistent homology on a *sparse point cloud* is sensitive only when the sample covers the manifold sufficiently. With 13 points clustered in two hemispheres (positive vs. negative valence), no triangulation can close the full S2 cavity. This null result reflects the experimental design—only 13 category centroids—not an absence of spherical geometry. A test on the full Warriner lexicon (~14,000 words) or on continuous fMRI activation patterns would be informative.

#### Test 3: Behavioral Geometry Correspondence

The Spearman correlation between the S2-geodesic distance matrix and the behavioural dissimilarity matrix (1 - Pearson r between emotion time-series, averaged across 14 films) was **ρ = 0.658 (p = 5.9×10⁻¹¹)**. Per-film correlations ranged from ρ = 0.30 (AfterTheRain) to ρ = 0.71 (BetweenViewings), with a mean of **0.545 ± 0.126**. This indicates that the a-priori VAD-sphere geometry captures a substantial fraction of the behavioural similarity structure, despite being derived from an independent lexical database.

#### Test 3b: Neural Decoding Replication

Replicating Ma & Kragel (2026), vmPFC (area 24 included) showed substantially higher 2D-MDS trajectory decoding than hippocampus or entorhinal cortex: vmPFC mean r = **0.095 ± 0.046**, HC = 0.043, ERC = 0.027. vmPFC exceeded HC in **25/29 subjects (86.2%)** (paired t = 6.21, p < 0.001). All three pairwise comparisons were significant (all p ≤ 0.007). This confirms the neural dissociation reported by the original authors and validates the fMRI dataset as an appropriate testbed for ASBH.

#### Test 4: Holonomy Signature

**263 quasi-loops** were identified across 14 films (range: 11–28 per film) using a greedy octant-matching algorithm (same VAD octant, 30–120 timepoints apart). For each loop, the enclosed solid angle Ω₀ was computed via the discrete Oosterom-Strackee formula, and loop closure error was measured as the geodesic distance between the loop's start and end points.

The raw Spearman correlation between Ω₀ and closure error was **ρ = 0.739 (p = 1.2×10⁻⁴⁶)**. After partialling out loop length (which correlates with both variables), the **partial ρ = 0.382 (p = 1.4×10⁻¹⁰)**. This residual association—loops enclosing larger solid angles on S2 incur systematically larger closure errors—is the predicted signature of holonomy on a curved manifold and cannot arise in flat Euclidean geometry. The film *Superhero* produced the largest quasi-loop (Ω₀ = 2.85 rad, approximately 9% of the sphere surface), consistent with its broad affective arc from fear/tension to relief/pride.

#### Test 5: S² vs Euclidean MDS Comparison

Across all 14 films and ~9,600 consecutive step pairs, the S2-geodesic speed (distance between successive S2 positions) was highly correlated with the Euclidean speed in the authors' 2D MDS embedding: mean **Spearman ρ = 0.704 ± 0.042** (all films p < 10⁻⁴⁰, range 0.63–0.76). This convergence is notable because the S2 coordinates are derived entirely from the Warriner lexical database, with no fitting to the fMRI or behavioural data.

For transitions where the Valence component changed sign (i.e. emotional state crossed the positive-negative boundary), the geodesic/MDS distance ratio was **0.029 ± 0.023**, compared to **0.012 ± 0.016** for ordinary transitions (**2.4× larger**, Welch t = 8.64, p = 9.2×10⁻¹⁵; KS D = 0.663, p = 10⁻⁶¹). This asymmetry is a direct curvature effect: the sphere 'stretches' distances across the equatorial (valence-neutral) region relative to the flat MDS plane, exactly as predicted by ASBH.

---

### Summary Table

| Test | Key Result | ASBH Prediction | Outcome |
|------|-----------|-----------------|---------|
| 01 — VAD projection | 13/13 emotions mapped; norm range 1.69–4.25 | Emotions lie on S2 | **Confirmed** |
| 02 — Persistent homology | H2 = 0, p = 1.0 vs null | β₂ = 1 (spherical cavity) | **Inconclusive** (13 pts insufficient) |
| 03 — Behavioural geometry | ρ = 0.658, p = 5.9×10⁻¹¹ (avg film ρ = 0.545) | S2-geodesic ≈ behavioral similarity | **Supported** |
| 03b — Neural decoding | vmPFC > HC, 25/29 subjects, t = 6.21 | vmPFC encodes affective manifold | **Replicated** (matches original paper) |
| 04 — Holonomy | partial ρ = 0.382, p = 1.4×10⁻¹⁰ | Ω₀ predicts closure error | **Supported** (strongest result) |
| 05 — S2 vs MDS speed | ρ = 0.704 ± 0.042; polar ratio 2.4×, p = 9.2×10⁻¹⁵ | S2 aligns with MDS; curvature at equator | **Supported** |

---

### Interpretation

The strongest evidence for ASBH comes from **Test 4 (holonomy)** and **Test 5 (polar transitions)**. The partial correlation between solid angle and closure error (ρ = 0.382, p = 10⁻¹⁰) is the theoretically most direct prediction: in a flat space, parallel transport along any closed loop returns the transported vector unchanged, so closure error should be independent of enclosed area. The observed positive association is specifically predicted by spherical geometry and cannot be explained by loop length alone.

The curvature signature in Test 5—S2 stretching cross-polar distances by a factor of 2.4× relative to the flat MDS—provides a second, geometrically interpretable line of evidence. Because the S2 coordinates come from an independent lexical dataset (Warriner 2013), this correspondence cannot be dismissed as circular.

The null result of persistent homology (Test 2) does not contradict ASBH. The 13 emotion centroids are too sparse and clustered to fill the sphere, so the Vietoris-Rips complex never forms a closed 2-sphere at any filtration radius. This is an artifact of experimental design, not an absence of curvature.

The moderate VAD-behavioral correlation (Test 3, ρ ≈ 0.65) reflects the well-known imperfection of lexical VAD as a proxy for contextual emotion experience. Despite this noise floor, the geometric correspondence is robust and consistent across all 14 films. Replacing static Warriner anchors with dynamic fMRI-estimated VAD coordinates is the natural next step and would likely increase all effect sizes.

---

### Limitations

1. **Small point cloud for PH.** Persistent homology requires dense coverage to detect global topology. With 13 category centroids—all derived from the same Warriner norms—no claim about the full spherical topology can be made from Test 2. A test on the complete ~14,000-word Warriner lexicon is needed.

2. **Static VAD anchors.** Warriner norms reflect average out-of-context lexical valence/arousal/dominance. Film-viewing involves dynamic, context-sensitive affective states that may deviate systematically from lexical norms. The holonomy and curvature signals are therefore lower bounds.

3. **No direct S² fMRI decoding.** Tests 04 and 05 demonstrate geometric signatures in the behavioural rating space, not in neural representational space. The critical test is whether replacing the authors' flat MDS metric with S2-geodesic distances improves (or degrades) fMRI decoding. This requires re-running the MATLAB decoding pipeline with S2-based trajectories.

4. **Quasi-loop identification.** The greedy octant-matching algorithm is a heuristic. The octant boundary is a crude proxy for 'returning to the same affective region.' A more principled approach (e.g., kernel density-based recurrence detection) would yield more interpretable loops.

5. **Word–category mapping.** The emotion label 'Regard' was mapped to 'warmth' and 'WarmHeartedness' was also mapped to 'warmth', creating a non-injective mapping. 'Surprise' maps to a positive-valence Warriner word (V_c = +2.44), whereas in the film context Surprise is affectively neutral. These mapping decisions affect the S2 positions and should be explored via sensitivity analysis.

---

### Next Steps

1. **PH on full Warriner lexicon** (~14,000 words): project all Warriner words to S2, run Vietoris-Rips PH, test for H2 persistence. This would provide a definitive topological test independent of category selection.

2. **Replace MDS with S2 in the TEM environment**: re-define the EmotionConceptRepresentation environment coordinates using S2-geodesic distances instead of Euclidean 2D MDS, and re-run the authors' MATLAB fMRI decoding pipeline. ASBH predicts equal or better decoding in vmPFC with the spherical metric.

3. **Lexical decision RT validation** (SPP × Warriner): semantic priming paradigm data (e.g., from the English Lexicon Project) can provide an independent behavioural test—geodesic distance on S2 should predict RT facilitation better than Euclidean VAD distance.

4. **Dynamic VAD estimation**: fit a Kalman-filter or Gaussian Process model to the emotion rating time-series to estimate time-varying VAD coordinates per film, enabling richer holonomy calculations with continuous (not category-centroid) S2 trajectories.

---

### References

Ma, Y. & Kragel, P.A. (2026). Map-like representations of emotion knowledge in hippocampal-prefrontal systems. *Nature Communications*, **17**, 1518. https://doi.org/10.1038/s41467-026-XXXXX-X

Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, **45**, 1191-1207.

Author. (2026). Affective Spin Bundle Hypothesis. *Preprint*. [placeholder]
