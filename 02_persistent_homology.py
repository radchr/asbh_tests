"""
ASBH Test 02 — Persistent Homology on emotion distance matrices
================================================================
Tests whether the 13-emotion VAD space has the topological signature
of S² (Betti numbers beta=(1,0,1)) rather than flat R² (beta=(1,0,0)).

Theory:
  S²  → H0=1 (connected), H1=0 (no loops), H2=1 (hollow cavity)
  R²  → H0=1,             H1=0,            H2=0  (no cavity)

Outputs (asbh_tests/results/):
  ph_geodesic_barcode.png
  ph_euclidean_barcode.png
  ph_null_distribution.png
  ph_results.json
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ripser import ripser

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "asbh_tests", "results")

GEO_CSV = os.path.join(RESULTS, "geodesic_matrix.csv")
EUC_CSV = os.path.join(RESULTS, "euclidean_matrix.csv")

# ── helpers ───────────────────────────────────────────────────────────────────

def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    return df


def count_bars(diagrams, threshold, dim):
    """Return bars in H_dim with persistence > threshold."""
    dgm = diagrams[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    persistent = finite[finite[:, 1] - finite[:, 0] > threshold]
    return persistent


def longest_bar(diagrams, dim):
    """Return lifespan of the longest finite bar in H_dim (0 if none)."""
    dgm = diagrams[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) == 0:
        return 0.0
    lifespans = finite[:, 1] - finite[:, 0]
    return float(lifespans.max())


def plot_barcode(diagrams, title, save_path, max_dim=2):
    colors = {0: "#4878CF", 1: "#6ACC65", 2: "#D65F5F"}
    dim_labels = {0: "H\u2080 (components)", 1: "H\u2081 (loops)", 2: "H\u2082 (cavities)"}

    fig, axes = plt.subplots(max_dim + 1, 1, figsize=(8, 2.5 * (max_dim + 1)),
                             sharex=True)
    if max_dim == 0:
        axes = [axes]

    for dim in range(max_dim + 1):
        ax = axes[dim]
        dgm = diagrams[dim]
        finite = dgm[dgm[:, 1] < np.inf]
        # replace inf with a visible cap for plotting
        max_val = finite[:, 1].max() * 1.1 if len(finite) > 0 else 1.0
        all_bars = dgm.copy()
        all_bars[all_bars[:, 1] == np.inf, 1] = max_val * 1.1

        for k, (b, d) in enumerate(sorted(all_bars, key=lambda x: x[0])):
            ax.plot([b, d], [k, k], lw=2, color=colors.get(dim, "gray"))

        ax.set_yticks([])
        ax.set_ylabel(dim_labels.get(dim, f"H{dim}"), fontsize=9)
        ax.axvline(0, color="gray", lw=0.5, ls="--")

    axes[-1].set_xlabel("Filtration value")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def run_ripser(matrix_df, label):
    mat = matrix_df.values.astype(np.float64)
    max_dist = mat.max()
    threshold = 0.1 * max_dist
    result = ripser(mat, distance_matrix=True, maxdim=2)
    dgms = result["dgms"]

    h0 = count_bars(dgms, threshold, 0)
    h1 = count_bars(dgms, threshold, 1)
    h2 = count_bars(dgms, threshold, 2)
    h2_life = longest_bar(dgms, 2)

    print(f"\n--- {label} ---")
    print(f"  max distance      : {max_dist:.4f}")
    print(f"  threshold (10%)   : {threshold:.4f}")
    print(f"  H0 bars           : {len(h0)}  (significant components)")
    print(f"  H1 bars           : {len(h1)}  (significant loops)")
    print(f"  H2 bars           : {len(h2)}  (significant cavities)")
    print(f"  longest H2 bar    : {h2_life:.6f}")

    return dgms, h0, h1, h2, h2_life, threshold


# ── 1. load matrices ──────────────────────────────────────────────────────────
print("Loading distance matrices ...")
geo_df = load_matrix(GEO_CSV)
euc_df = load_matrix(EUC_CSV)
EMOTIONS = list(geo_df.index)
print(f"  Emotions: {EMOTIONS}")

# ── 2 & 3. ripser on real data ────────────────────────────────────────────────
dgms_geo, h0_geo, h1_geo, h2_geo, h2_life_geo, thr_geo = run_ripser(geo_df, "GEODESIC (S2 hypothesis)")
dgms_euc, h0_euc, h1_euc, h2_euc, h2_life_euc, thr_euc = run_ripser(euc_df, "EUCLIDEAN (flat hypothesis)")

# ── 4. null model: 1000 random uniform S² configurations ─────────────────────
N_NULL  = 1000
N_PTS   = 13   # same as real data

print(f"\nBuilding null model ({N_NULL} random S2 configurations, {N_PTS} points each) ...")
rng = np.random.default_rng(42)
null_h2_lifespans = []

for i in range(N_NULL):
    # uniform on S2: normalise iid Gaussian vectors
    pts = rng.standard_normal((N_PTS, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)

    # geodesic distance matrix
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    geo_null = np.arccos(dots)
    np.fill_diagonal(geo_null, 0.0)

    res = ripser(geo_null, distance_matrix=True, maxdim=2)
    null_h2_lifespans.append(longest_bar(res["dgms"], 2))

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N_NULL} done  "
              f"(mean H2 life so far: {np.mean(null_h2_lifespans):.4f})")

null_arr = np.array(null_h2_lifespans)

# ── 5. p-value ────────────────────────────────────────────────────────────────
# one-tailed: how many null samples have H2 lifespan >= real value?
p_value = float(np.mean(null_arr >= h2_life_geo))
print(f"\nReal geodesic H2 lifespan : {h2_life_geo:.6f}")
print(f"Null mean +/- std         : {null_arr.mean():.6f} +/- {null_arr.std():.6f}")
print(f"p-value (one-tailed)      : {p_value:.4f}")

# ── 6. save plots ─────────────────────────────────────────────────────────────
plot_barcode(dgms_geo, "Persistent Homology — Geodesic distances (S2 VAD)",
             os.path.join(RESULTS, "ph_geodesic_barcode.png"))

plot_barcode(dgms_euc, "Persistent Homology — Euclidean distances (flat VAD)",
             os.path.join(RESULTS, "ph_euclidean_barcode.png"))

# null distribution plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(null_arr, bins=40, color="#6ACC65", edgecolor="white", alpha=0.8,
        label=f"Null ({N_NULL} random S2, N={N_PTS})")
ax.axvline(h2_life_geo, color="#D65F5F", lw=2,
           label=f"Real geodesic H2 = {h2_life_geo:.4f}\np = {p_value:.3f}")
ax.set_xlabel("Longest H2 bar lifespan")
ax.set_ylabel("Count")
ax.set_title("H2 Lifespan: Real vs. Null (random S2 configurations)")
ax.legend(fontsize=9)
plt.tight_layout()
null_plot = os.path.join(RESULTS, "ph_null_distribution.png")
fig.savefig(null_plot, dpi=150)
plt.close(fig)
print(f"Saved: {null_plot}")

# ph_results.json
ph_results = {
    "geodesic": {
        "H0_significant": int(len(h0_geo)),
        "H1_significant": int(len(h1_geo)),
        "H2_significant": int(len(h2_geo)),
        "H2_lifespan":    round(h2_life_geo, 6),
        "p_vs_null":      round(p_value, 4),
        "null_mean":      round(float(null_arr.mean()), 6),
        "null_std":       round(float(null_arr.std()),  6),
        "threshold":      round(thr_geo, 6),
    },
    "euclidean": {
        "H0_significant": int(len(h0_euc)),
        "H1_significant": int(len(h1_euc)),
        "H2_significant": int(len(h2_euc)),
        "H2_lifespan":    round(h2_life_euc, 6),
        "threshold":      round(thr_euc, 6),
    },
}
json_path = os.path.join(RESULTS, "ph_results.json")
with open(json_path, "w") as f:
    json.dump(ph_results, f, indent=2)
print(f"Saved: {json_path}")

# ── 7. conclusion ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
ALPHA = 0.05
if h2_life_geo > 0 and p_value < ALPHA:
    verdict = (f"ASBH SUPPORTED: geodesic H2 lifespan = {h2_life_geo:.4f} "
               f"(p = {p_value:.4f} < {ALPHA})")
elif h2_life_geo > 0:
    verdict = (f"ASBH WEAK: H2 present (lifespan = {h2_life_geo:.4f}) "
               f"but NOT significant vs. null (p = {p_value:.4f})")
else:
    verdict = "ASBH NOT SUPPORTED: H2 absent (lifespan = 0)"

euc_verdict = ("Euclidean H2 present" if h2_life_euc > 0
               else "Euclidean H2 absent (consistent with flat geometry)")

print(f"  {verdict}")
print(f"  {euc_verdict}")
print("=" * 60)
