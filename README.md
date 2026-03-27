# \# ASBH Computational Tests on Emo-FilM Dataset

# 

# \## Overview

# Computational tests of the Affective Spin Bundle Hypothesis (ASBH)

# on the Ma \& Kragel (2026) fMRI dataset (Nature Communications).

# 

# \## Data sources (not included, download separately)

# \- Emo-FilM fMRI data: openneuro.org/datasets/ds004892

# \- Emo-FilM ratings: openneuro.org/datasets/ds004872  

# \- Ma \& Kragel analysis code: github.com/ecco-laboratory/EmotionConceptRepresentation

# \- Warriner VAD norms: github.com/JULIELab/XANEW

# 

# \## Installation

# pip install numpy pandas scipy scikit-learn matplotlib seaborn ripser

# 

# \## Scripts (run in order)

# | Script | Description | Key output |

# |--------|-------------|------------|

# | 01\_vad\_s2.py | Project 13 emotions onto S² via Warriner VAD | emotion\_vad.csv |

# | 02\_persistent\_homology.py | Topological analysis (Betti numbers) | ph\_results.json |

# | 03\_geometry\_comparison.py | S² vs behavioral similarity | geometry\_comparison\_summary.json |

# | 04\_holonomy.py | Solid angle predicts loop closure | holonomy\_results.json |

# | 05\_asbh\_vs\_mds\_comparison.py | S² trajectories vs authors MDS | s2\_vs\_mds\_correlations.csv |

# | 07\_ellipsoidal\_test.py | Ellipsoidal metric optimization | optimal\_lambda.json |

# | 08\_connection\_diagnostics.py | Metric-connection dissociation | holonomy\_lambda\_heatmap.png |

# | 10\_verify\_and\_finalize.py | Verification of all key numbers | verified\_numbers.json |

# 

# \## Key results

# \- S²-geodesic speed correlates with authors MDS speed: ρ = 0.704 ± 0.042

# \- Holonomy signal: partial ρ = 0.382 (p = 1.4×10⁻¹⁰, n=263 loops)

# \- Curvature signature: polar transitions 2.4× larger ratio (p = 9.2×10⁻¹⁵)

# \- Best model M3 (ellipsoid + density potential): ΔAIC = −15.7 vs round S²

# \- Metric-connection dissociation: λ\_similarity ≠ λ\_holonomy

# 

# \## Preprint

# \[посилання додати після завантаження на Zenodo/bioRxiv]

# 

# \## Citation

# R. Radchenko (2026). Spherical geometry and holonomy signatures in affective space.

# 

# 

# \## Reference dataset

# Ma, Y. \& Kragel, P.A. (2026). Map-like representations of emotion knowledge

# in hippocampal-prefrontal systems. Nature Communications, 17, 1518.

# https://doi.org/10.1038/s41467-025-68240-z

