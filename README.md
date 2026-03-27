# ASBH Computational Tests on Emo-FilM Dataset

## Overview
Computational tests of the Affective Spin Bundle Hypothesis (ASBH)
on the Ma & Kragel (2026) fMRI dataset (Nature Communications).

## Data sources (not included, download separately)
- Emo-FilM fMRI data: openneuro.org/datasets/ds004892
- Emo-FilM ratings: openneuro.org/datasets/ds004872  
- Ma & Kragel analysis code: github.com/ecco-laboratory/EmotionConceptRepresentation
- Warriner VAD norms: github.com/JULIELab/XANEW

## Installation
pip install numpy pandas scipy scikit-learn matplotlib seaborn ripser

## Scripts (run in order)
| Script | Description | Key output |
|--------|-------------|------------|
| 01_vad_s2.py | Project 13 emotions onto S² via Warriner VAD | emotion_vad.csv |
| 02_persistent_homology.py | Topological analysis (Betti numbers) | ph_results.json |
| 03_geometry_comparison.py | S² vs behavioral similarity | geometry_comparison_summary.json |
| 04_holonomy.py | Solid angle predicts loop closure | holonomy_results.json |
| 05_asbh_vs_mds_comparison.py | S² trajectories vs authors MDS | s2_vs_mds_correlations.csv |
| 07_ellipsoidal_test.py | Ellipsoidal metric optimization | optimal_lambda.json |
| 08_connection_diagnostics.py | Metric-connection dissociation | holonomy_lambda_heatmap.png |
| 10_verify_and_finalize.py | Verification of all key numbers | verified_numbers.json |

## Key results
- S²-geodesic speed correlates with authors MDS speed: ρ = 0.704 ± 0.042
- Holonomy signal: partial ρ = 0.382 (p = 1.4×10⁻¹⁰, n=263 loops)
- Curvature signature: polar transitions 2.4× larger ratio (p = 9.2×10⁻¹⁵)
- Best model M3 (ellipsoid + density potential): ΔAIC = −15.7 vs round S²
- Metric-connection dissociation: λ_similarity ≠ λ_holonomy

## Preprint
[посилання додати після завантаження на Zenodo/bioRxiv]

## Citation
R. Radchenko (2026). Spherical geometry and holonomy signatures in affective space.
Preprint. doi: [додати]

## Reference dataset
Ma, Y. & Kragel, P.A. (2026). Map-like representations of emotion knowledge
in hippocampal-prefrontal systems. Nature Communications, 17, 1518.
https://doi.org/10.1038/s41467-025-68240-z