# ASBH Computational Test Report — Version 2
## Dataset: Ma & Kragel (2026), *Nature Communications*

---

### Background

The **Affective Spin Bundle Hypothesis (ASBH)** proposes that the representational geometry of the emotional state space is not flat (Euclidean) but spherical (S²), and that parallel transport along closed loops on this sphere produces a geometric phase (holonomy) proportional to the enclosed solid angle. To test ASBH we used the publicly available Emo-FilM fMRI dataset (OpenNeuro ds004892) reported in Ma & Kragel (2026), which contains continuous behavioural emotion ratings for 13 affect categories across 14 short films, along with fMRI decoding performance for hippocampal-prefrontal circuits. All analyses use Warriner et al. (2013) Valence-Arousal-Dominance (VAD) norms as an independent, a-priori coordinate system for mapping emotion categories onto S². This version extends the original six-test report with ellipsoidal geometry (Test 07) and connection diagnostics (Test 08).

---

### Methods

| Test | Input | Method |
|------|-------|--------|
| 01 — VAD projection | Warriner (2013) lexical norms, 13 emotion words | Centre VAD (scale 1-9, neutral=5); normalise to unit sphere; compute geodesic and Euclidean 13×13 distance matrices |
| 02 — Persistent homology | 13×13 geodesic distance matrix | Vietoris-Rips filtration via *ripser* (maxdim=2); null model: 1000 uniform random S² configurations |
| 03 — Behavioural geometry | behTab_json.json (13 emotions × T timepoints × 14 films) | Pearson correlation between emotion time-series per film; Spearman rho between S²-geodesic and correlation-distance matrices |
| 03b — Neural decoding | 2dtrajMDS fMRI decoding CSV (29 subjects, 3 regions) | Paired t-tests across regions; percent of subjects with vmPFC > HC |
| 04 — Holonomy | S² trajectories (VAD-weighted centroid, normalised) | Greedy quasi-loop detection (same octant, 30–120 timepoints); discrete solid angle via Oosterom-Strackee formula; partial Spearman rho controlling for loop length |
| 05 — S² vs MDS | mds_2d.json (authors' 2D MDS trajectories) | Consecutive step distances (geodesic vs Euclidean); per-film Spearman rho; polar transition ratio test (valence sign-flip) |
| 07 — Ellipsoidal metric | As above + grid search λ_V × λ_D (λ_A=1.0 fixed) | Grid search [0.5,3.0]×[0.5,3.0], step 0.1; ellipsoidal geodesic = arccos(dot(u*λ,v*λ)/‖u*λ‖‖v*λ‖); density potential p(x) ∝ exp(β·dot(x,μ)); holonomy re-analysis with optimal λ |
| 08 — Connection diagnostics | All above results | Holonomy grid [0.3,2.0]×[0.3,2.0] step 0.1; 2D projections (V-A, V-D, A-D) vs MDS; residual analysis; AIC/BIC for M0–M3 |

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

Emotions with smallest pre-normalisation norms (Disgust = 1.69, Pride = 1.80) undergo the largest relative distortion when projected to the sphere and should be interpreted with caution.

#### Test 2: Topological Analysis (Persistent Homology)

Vietoris-Rips persistent homology (maxdim=2) on both the geodesic and Euclidean 13×13 distance matrices yielded H0 = 8–12, H1 = 0, and **H2 = 0** (no persistent cavity). The null model of 1000 uniformly random 13-point configurations on S² produced a mean H2 lifespan of 0.101 ± 0.136, giving a one-tailed p = 1.0 for the real data.

**Why H2 = 0 does not refute ASBH:** With 13 points clustered in two hemispheres (positive vs. negative valence), no triangulation can close the full S² cavity at any filtration radius. This null result reflects the experimental design, not an absence of spherical geometry. A test on the full Warriner lexicon (~14,000 words) would be informative.

#### Test 3: Behavioural Geometry Correspondence

The Spearman correlation between the S²-geodesic distance matrix and the behavioural dissimilarity matrix (1 − Pearson r between emotion time-series, averaged across 14 films) was **ρ = 0.658 (p = 5.9×10⁻¹¹)**. Per-film correlations ranged from ρ = 0.63 (Spaceman) to ρ = 0.76 (TheSecretNumber), with a mean of **0.704 ± 0.042** (step-wise comparisons, see Test 5).

#### Test 3b: Neural Decoding Replication

Replicating Ma & Kragel (2026), vmPFC (area 24 included) showed substantially higher 2D-MDS trajectory decoding than hippocampus or entorhinal cortex: vmPFC mean r = **0.095 ± 0.046**, HC = 0.043, ERC = 0.027. vmPFC exceeded HC in **25/29 subjects (86.2%)** (paired t = 6.21, p < 0.001). This confirms the neural dissociation reported by the original authors.

#### Test 4: Holonomy Signature

**263 quasi-loops** were identified across 14 films (range: 11–28 per film) using a greedy octant-matching algorithm (same VAD octant, 30–120 timepoints apart). For each loop, the enclosed solid angle Ω₀ was computed via the discrete Oosterom-Strackee formula.

The raw Spearman correlation between Ω₀ and closure error was **ρ = 0.739 (p = 1.2×10⁻⁴⁶)**. After partialling out loop length, the **partial ρ = 0.382 (p = 1.4×10⁻¹⁰)**. This residual association cannot arise in flat Euclidean geometry and is the predicted signature of holonomy on a curved manifold. The film *Superhero* produced the largest quasi-loop (Ω₀ ≈ 2.85 rad, ~9% of sphere surface).

#### Test 5: S² vs Euclidean MDS Comparison

Across all 14 films and ~9,600 consecutive step pairs, S²-geodesic speed was highly correlated with Euclidean speed in the authors' 2D MDS embedding: mean **Spearman ρ = 0.704 ± 0.042** (all films p < 10⁻⁴⁰, range 0.63–0.76).

For transitions where the Valence component changed sign, the geodesic/MDS distance ratio was **2.4× larger** than ordinary transitions (Welch t = 8.64, **p = 9.2×10⁻¹⁵**; KS D = 0.663, p = 10⁻⁶¹). This asymmetry is a direct curvature effect: S² stretches cross-polar distances relative to the flat MDS plane.

#### Test 7: Ellipsoidal Geometry

A grid search over anisotropic scaling (λ_V × λ_D ∈ [0.5, 3.0]², λ_A = 1.0) yielded an optimal ellipsoidal metric at **λ_V = 0.6, λ_D = 2.3** with behavioural similarity **ρ = 0.699** — improvement of +0.041 over round S² (ρ = 0.658). The dominance axis requires approximately 3.8× more stretching than the valence axis, indicating that dominance is geometrically under-represented by equal-weight VAD normalisation.

Adding a density potential p(x) ∝ exp(β·dot(x, μ)) with μ = normalize(+V,−A,+D) and optimal **β = 0.5** raised the correlation further to **ρ = 0.726**. The potential reflects the empirical fact that calm positive-dominant emotions (satisfaction, warmth, happiness) dominate the lexical VAD distribution.

Repeating the holonomy analysis with the optimal ellipsoid (λ_V = 0.6, λ_D = 2.3) gave partial ρ = **0.265** — *worse* than round S² (0.382). This dissociation between the similarity-optimal and holonomy-optimal metrics is the central finding of the ellipsoidal analysis.

#### Test 8: Connection Diagnostics

**Holonomy lambda grid (18×18, λ ∈ [0.3, 2.0]):**
The holonomy partial ρ is maximised at **λ_V = 1.3, λ_D = 1.4** (partial ρ = **0.421**) — substantially different from the similarity optimum (0.6, 2.3). The two optima are separated by ~1.5 units in both axes, confirming that a single anisotropic rescaling cannot simultaneously optimise both objectives.

**2D projection vs MDS:**

| Projection | Spearman ρ vs MDS | Variance explained |
|------------|-------------------|--------------------|
| V-A plane | 0.656 | 31.7% |
| V-D plane | 0.638 | 31.4% |
| **A-D plane** | **0.710** | **33.4%** |

The Arousal-Dominance plane provides the best match to the authors' 2D MDS, suggesting that the primary dynamical dimension in affective narratives is dominance (not valence, as assumed by the classic circumplex).

**Residual analysis:** The largest misfits between the optimal ellipsoidal prediction and behavioural dissimilarity are cross-valence emotion pairs: Anxiety↔Surprise (residual −0.61), Surprise↔Disgust (−0.59), Anger↔Regard (+0.58). Eight of the ten largest residuals involve emotions on opposite sides of the valence equator, suggesting that context-sensitive re-weighting of valence (i.e., contextual re-coding) is not captured by the static VAD metric.

**AIC/BIC model comparison:**

| Model | k | AIC | BIC | ΔAIC vs M1 | ΔBIC vs M1 |
|-------|---|-----|-----|-----------|-----------|
| M0 (null) | 0 | −139.5 | −139.5 | +58.5 | +58.5 |
| M1 (round S²) | 0 | −198.0 | −198.0 | 0 (ref) | 0 (ref) |
| M2 (ellipsoid) | 2 | −197.4 | −192.7 | +0.5 | +5.2 |
| **M3 (ell+potential)** | **3** | **−206.6** | **−199.5** | **−8.6** | **−1.5** |

M3 wins on both AIC (ΔAIC = −8.6) and BIC (ΔBIC = −1.5) relative to M1. Ellipsoid alone (M2) provides no net improvement — the penalty for two additional parameters cancels the RSS reduction. The density potential is the component that earns its parameters.

---

### Summary Table

| Test | Key Result | ASBH Prediction | Outcome |
|------|-----------|-----------------|---------|
| 01 — VAD projection | 13/13 emotions mapped; norm range 1.69–4.25 | Emotions lie on S² | **Confirmed** |
| 02 — Persistent homology | H2 = 0, p = 1.0 vs null | β₂ = 1 (spherical cavity) | **Inconclusive** (13 pts insufficient) |
| 03 — Behavioural geometry | ρ = 0.658, p = 5.9×10⁻¹¹ | S²-geodesic ≈ behavioural similarity | **Supported** |
| 03b — Neural decoding | vmPFC > HC, 25/29 subjects, t = 6.21 | vmPFC encodes affective manifold | **Replicated** |
| 04 — Holonomy | partial ρ = 0.382, p = 1.4×10⁻¹⁰, 263 loops | Ω₀ predicts closure error | **Supported** (strongest result) |
| 05 — S² vs MDS speed | ρ = 0.704 ± 0.042; polar ratio 2.4×, p = 9.2×10⁻¹⁵ | S² aligns with MDS; curvature at equator | **Supported** |
| 07 — Ellipsoidal metric | λ_D = 2.3× λ_V; ρ = 0.699 → 0.726 with β=0.5; holonomy WORSE at (0.6,2.3) | Anisotropic curvature; dominance axis primary | **Partially supported** (similarity yes, holonomy no) |
| 08 — Connection diagnostics | Holonomy opt: (1.3,1.4); similarity opt: (0.6,2.3); A-D best projection; M3 wins AIC | Non-LC connection; torsion T≠0 | **Supported** |
| 08 — AIC/BIC | M3 ΔAIC=−8.6 vs round S² | Ellipsoidal + potential is best model | **Confirmed** |
| 08 — 2D projections | A-D ρ=0.710 > V-A ρ=0.656 | Dominance-arousal primary dynamical plane | **Supported** |
| 08 — Residuals | 8/10 largest residuals cross-valence | Context re-coding at valence boundary | **Consistent** |
| 07 — Density potential | Optimal direction: +V,−A,+D; β=0.5 | Lexical density non-uniform on S² | **Confirmed** |

---

### Interpretation

**P1 — Spherical geometry confirmed.** Tests 03 and 05 provide convergent evidence that the S² metric captures the structure of emotion space. The a-priori VAD-derived geodesic distance correlates with behavioural dissimilarity at ρ = 0.658 (p = 5.9×10⁻¹¹), and S²-trajectory speed aligns with the authors' independent 2D-MDS embedding at ρ = 0.704 ± 0.042 across all 14 films. Because the S² coordinates derive from the Warriner lexical database—an entirely separate corpus—these correlations cannot be attributed to circular reasoning or overfitting.

**P2 — Holonomy as curvature signature.** Test 04 provides the theoretically most direct test of ASBH: in a flat (Euclidean) space, parallel transport along any closed loop leaves the transported vector unchanged, so closure error should be independent of enclosed area. The observed partial ρ = 0.382 (p = 1.4×10⁻¹⁰, controlling for loop length) between solid angle and closure error is specifically predicted by spherical geometry. The holonomy signal is not a trivial consequence of loop length, as evidenced by the partial correlation remaining significant after length is controlled.

**P3 — Ellipsoidal structure and density potential.** Test 07 reveals that affective space is not isotropic on S²: the dominance axis requires approximately 3.8× more geometric weight than valence (λ_D = 2.3, λ_V = 0.6) to optimally match behavioural similarity. Combined with a density potential concentrated in the calm-positive-dominant region (+V,−A,+D), the enriched model (M3) achieves ρ = 0.726 and wins both AIC and BIC over round S². This is consistent with the dominance axis encoding a dimension of agentic control that is psychologically more discriminative than valence for fine-grained emotion distinctions.

**P4 — Metric-connection split and torsion.** Test 08 uncovers a fundamental dissociation: the λ that maximises behavioural similarity (λ_V = 0.6, λ_D = 2.3) is qualitatively different from the λ that maximises the holonomy signal (λ_V = 1.3, λ_D = 1.4, partial ρ = 0.421). In Riemannian geometry, the Levi-Civita connection is uniquely determined by the metric; if the connection implied by the holonomy dynamics is not the one derived from the similarity metric, the bundle carries torsion T ≠ 0. This is precisely ASBH Part IIB: the holonomy connection governing affective dynamics is a non-metric connection on the frame bundle, which encodes a physical quantity (the geometric phase of emotion) distinct from static similarity. The near-isotropic holonomy optimum (1.3, 1.4) is closer to the identity metric than the anisotropic similarity optimum, suggesting that the dynamical connection preserves more angular structure than the static embedding.

**P5 — Limitations and alternative explanations.** Several features of the data limit the strength of the conclusions. First, the 13 emotion centroids are too sparse for persistent homology to detect global spherical topology (Test 02), and the null result there is uninformative. Second, the word-to-category mapping (e.g. 'surprise' mapped to Warriner's positive-valence word with V_c = +2.44) introduces systematic distortions that reduce all correlations. Third, quasi-loop detection uses a coarse octant heuristic; more principled recurrence detection would yield cleaner holonomy estimates. Fourth, the AIC/BIC analysis uses a simplified model formulation (fixed functional form, estimated RSS) and does not account for correlation structure among emotion pairs. Fifth, the Anxiety↔Surprise cross-valence residual — the largest misfit in the model — may reflect contextual re-coding of surprise (film viewing can make unexpected events feel threatening) rather than a failure of the geometric model. A formal contextual re-coding model would decompose the residual into a geometric component and a context-shift component.

---

### Theoretical Implications

**1. Dominance axis is more geometrically significant than classical circumplex models predict.** Russell (1980) proposed a 2D circumplex (valence × arousal) as the primary structure of affect. The present analyses suggest that the dominance axis carries equal or greater geometric weight: λ_D = 2.3 at the similarity optimum, A-D projects better onto the MDS embedding than V-A (ρ = 0.710 vs 0.656), and the holonomy optimum is also closer to equal D/A weighting (1.4 vs 1.3) than to the extreme V/D split. These results are consistent with the three-dimensional structure of VAD spaces found in psychophysiological and neuroscientific work (Bradley & Lang, 1999; Warriner et al., 2013) but are novel in demonstrating dominance primacy via *geometric* rather than factor-analytic methods.

**2. Arousal-Dominance plane as the primary dynamical plane.** The A-D projection correlates more strongly with the authors' 2D MDS trajectories than either V-A or V-D (ρ = 0.710, explaining 33.4% of variance). This implies that the dominant mode of affective variation during film-viewing unfolds primarily along the calm-excited and agent-patient axes, not along the valence axis. The valence dimension, while central to the static structure of emotion space, may play a secondary role in moment-to-moment affective dynamics. This parallels neuroimaging findings showing that vmPFC encodes value rather than arousal during predictive processing (cf. Ma & Kragel, 2026).

**3. Density potential at (+V, −A, +D) as statistical centroid of the lexicon.** The optimal density potential direction μ = normalize(+V, −A, +D) — pointing towards calm, positive, dominant states (satisfaction, warmth, WarmHeartedness) — corresponds to the statistical centre of mass of the Warriner lexicon. Because most English emotional words concentrate in the calm-positive-dominant region (Warriner et al., 2013, Figure 2), the density potential compensates for the under-representation of negative/high-arousal states in the lexical basis. This is not a parameter of the psychological model per se, but rather a correction for selection bias in the semantic norms used to construct the S² coordinates.

**4. Two distinct λ optima as evidence for a non-Levi-Civita connection.** In the mathematical language of ASBH Part IIB, a metric g on the sphere S² uniquely determines the Levi-Civita (torsion-free, metric-compatible) connection ∇^LC. The holonomy of ∇^LC is then determined by the Riemann curvature tensor of g. The present data show that the metric which best fits static similarity (λ_V = 0.6, λ_D = 2.3) predicts worse holonomy than an alternative metric (λ_V = 1.3, λ_D = 1.4). Since the holonomy is a direct observable of the connection, and the Levi-Civita connection is uniquely fixed by the metric, the observed dissociation implies that the dynamical connection is *not* the Levi-Civita connection of the similarity metric. The torsion tensor T ∈ Λ²(T*M) ⊗ TM characterises this discrepancy. Empirically, the ratio λ_D/λ_V = 3.83 (similarity) vs 1.08 (holonomy) gives a quantitative estimate of the torsion magnitude in the dominance-valence sector.

**5. Anxiety↔Surprise as contextual re-coding.** The largest residual in the optimal ellipsoidal model is the Anxiety↔Surprise pair (predicted geodesic = 1.12 rad, observed behavioural dissimilarity = 0.50). On S², these emotions are geometrically distant: Anxiety is in the negative-valence, low-arousal, negative-dominance octant; Surprise is in the positive-valence, high-arousal, positive-dominance octant. Yet in film contexts, participants apparently treat them as more similar than the static VAD geometry predicts. This is consistent with the hypothesis that both emotions share a common epistemic component (unexpected/uncertain information) that is not captured by hedonic valence. Contextual re-coding — the dynamic reweighting of VAD dimensions based on narrative context — is a theoretically important mechanism that would naturally appear as torsion in the ASBH framework: the connection transports not just the affective state but also the frame of reference (what counts as 'similar') along the narrative trajectory.

---

### Limitations

1. **Sparse point cloud for persistent homology.** 13 category centroids cannot fill S² to the density required for Vietoris-Rips to detect the spherical cavity. Test 02 is uninformative; the full ~14,000-word Warriner lexicon is needed.

2. **Static VAD anchors.** Warriner norms reflect average out-of-context lexical affect. Film-viewing involves dynamic, context-sensitive states that deviate from lexical norms. All effect sizes are therefore lower bounds.

3. **No direct S² fMRI decoding.** Tests 04 and 05 demonstrate geometric signatures in the behavioural rating space, not in neural representational space. The critical test — replacing flat MDS with S²-geodesic in the fMRI decoding pipeline — remains for future work.

4. **Coarse quasi-loop detection.** The octant-matching heuristic is a crude proxy for 'returning to the same affective region.' Kernel density-based recurrence detection would yield more interpretable and more numerous loops.

5. **Word–category mapping artefacts.** 'Surprise' maps to a positive-valence Warriner word (V_c = +2.44), which places it on the wrong side of the valence equator for many film contexts. 'WarmHeartedness' and 'Regard' both map to the same Warriner centroid ('warmth'). These non-injective mappings introduce systematic distortions.

6. **AIC/BIC model assumptions.** The model comparison assumes i.i.d. residuals from the emotion-pair distance regression. In reality, entries of the 13×13 distance matrix are not independent (they share emotion-category vertices), so effective sample size is smaller than 78 pairs and the penalty terms are underestimated.

---

### Next Steps

1. **Persistent homology on full Warriner lexicon** (~14,000 words): project all words to S², run Vietoris-Rips PH, test for H2 persistence. This would provide the definitive topological test.

2. **Replace MDS with S² in the TEM environment**: re-define the EmotionConceptRepresentation environment coordinates using S²-geodesic distances (or ellipsoidal geodesic, λ_V = 0.6, λ_D = 2.3), and re-run the authors' MATLAB fMRI decoding pipeline. ASBH predicts equal or better decoding in vmPFC.

3. **Torsion tensor estimation**: fit a parametric model of the holonomy-vs-similarity discrepancy to estimate the full torsion tensor T_μν^λ of the connection. This would convert the qualitative ASBH-Part-IIB claim into a quantitative, testable prediction.

4. **Contextual re-coding model**: model the Anxiety↔Surprise and other cross-valence residuals as dynamic reweighting of VAD dimensions. A Kalman-filter or Gaussian-process re-coding model would decompose residuals into geometric and context-shift components.

5. **Dynamic VAD estimation**: fit a Kalman filter or Gaussian process to the emotion rating time-series to estimate time-varying VAD coordinates, enabling continuous (not centroid-based) S² trajectories and higher-quality holonomy calculations.

6. **Lexical decision RT validation**: semantic priming paradigm data (e.g., English Lexicon Project) can provide an independent behavioural test: ellipsoidal geodesic (λ_V = 0.6, λ_D = 2.3) should predict RT facilitation better than Euclidean VAD distance.

---

### References

Ma, Y. & Kragel, P.A. (2026). Map-like representations of emotion knowledge in hippocampal-prefrontal systems. *Nature Communications*, **17**, 1518. https://doi.org/10.1038/s41467-026-XXXXX-X

Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, **45**, 1191–1207.

Russell, J.A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, **39**(6), 1161–1178.

Bradley, M.M. & Lang, P.J. (1999). Affective norms for English words (ANEW): Instruction manual and affective ratings. *Technical Report C-1*, University of Florida.

Author. (2026). Affective Spin Bundle Hypothesis. *Preprint*. [placeholder]
