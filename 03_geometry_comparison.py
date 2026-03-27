"""
ASBH Test 03 — Geometry Comparison on Behavioral + fMRI Data
=============================================================
Part A  : Correlation-based vs. S²-geodesic similarity matrices
Part B  : fMRI decoding by region (vmPFC / HC / ERC)
Part C  : VAD peripherality vs. category-level decoding accuracy

Outputs (asbh_tests/results/):
  geometry_correlation.png
  decoding_by_region.png
  peripherality_vs_decoding.png
  geometry_comparison_summary.json
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(ROOT, "EmotionConceptRepresentation", "data")
OUTDIR  = os.path.join(ROOT, "EmotionConceptRepresentation", "outputs",
                        "rep3", "ratings_prediction_performance", "brain")
RESULTS = os.path.join(ROOT, "asbh_tests", "results")

BEH_JSON    = os.path.join(DATA, "behTab_json.json")
VAD_CSV     = os.path.join(RESULTS, "emotion_vad.csv")
GEO_CSV     = os.path.join(RESULTS, "geodesic_matrix.csv")
TRAJ_CSV    = os.path.join(OUTDIR, "2dtrajMDS",
              "2dtrajMDSRatings_prediction_performance_generalized_across_movies_hcecvmpfc.csv")
CAT_CSV     = os.path.join(OUTDIR, "category",
              "categoryRatings_prediction_performance_generalized_across_movies_hcecvmpfc.csv")

EMOTIONS = ["Anger","Anxiety","Fear","Surprise","Guilt","Disgust","Sad",
            "Regard","Satisfaction","WarmHeartedness","Happiness","Pride","Love"]

EXCLUDE_MOVIES = {"DamagedKungFu", "RidingTheRails", "LeassonLearned"}

# ═══════════════════════════════════════════════════════════════════════════════
# PART A — Correlation-based vs. S²-geodesic similarity matrices
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART A: Geometry comparison on behavioral ratings")
print("="*60)

# Load behavioral ratings
with open(BEH_JSON) as f:
    beh = json.load(f)

movies_use = [m for m in beh.keys() if m not in EXCLUDE_MOVIES]
print(f"Movies used: {len(movies_use)}  ({', '.join(movies_use)})")

# Load geodesic matrix (S²)
geo_df = pd.read_csv(GEO_CSV, index_col=0)
geo_mat = geo_df.values   # 13×13 geodesic distances

# For each movie: build 13x13 Pearson correlation matrix, convert to distance
# (dissimilarity = 1 - r, clipped to [0,2])
n_emo = len(EMOTIONS)
movie_corr_matrices = {}

for movie in movies_use:
    # use DataFrame so pairwise corr handles NaN gracefully (pairwise complete obs)
    ts_df = pd.DataFrame(
        {e: beh[movie][e] for e in EMOTIONS}, dtype=float
    )  # shape (T, 13)
    corr = ts_df.corr(method="pearson", min_periods=10)  # (13, 13) DataFrame
    dist = (1.0 - corr).values.copy()                    # writable numpy array
    # fill any remaining NaN diagonal residue with 0
    np.fill_diagonal(dist, 0.0)
    n_nan = np.sum(np.isnan(dist))
    if n_nan > 0:
        print(f"  [WARN] {movie}: {n_nan} NaN cells in correlation matrix "
              f"(filled with 0)")
        dist = np.nan_to_num(dist, nan=0.0)
    movie_corr_matrices[movie] = dist

movies_valid = movies_use
print(f"Valid movies for geometry comparison: {len(movies_valid)}")
SKIP_MOVIES = []

# Average correlation-distance matrix across valid movies
avg_corr_dist = np.mean(list(movie_corr_matrices.values()), axis=0)

# Extract upper-triangle indices (no diagonal)
triu_idx = np.triu_indices(n_emo, k=1)
geo_vals  = geo_mat[triu_idx]
corr_vals = avg_corr_dist[triu_idx]

# Spearman correlation between the two geometry matrices
rho, pval = stats.spearmanr(geo_vals, corr_vals)
print(f"\nSpearman rho (S2-geodesic vs. behavioral correlation-distance): "
      f"{rho:.4f}  (p = {pval:.4e})")

# Per-movie Spearman rho
per_movie_rho = []
for movie in movies_valid:
    dist_m = movie_corr_matrices[movie][triu_idx]
    r, _ = stats.spearmanr(geo_vals, dist_m)
    per_movie_rho.append(r)
    print(f"  {movie:25s}  rho = {r:.4f}")

valid_rho = [r for r in per_movie_rho if not np.isnan(r)]
print(f"\nMean per-movie rho: {np.mean(valid_rho):.4f} "
      f"+/- {np.std(valid_rho):.4f}  ({len(valid_rho)}/{len(per_movie_rho)} valid)")

# Plot: scatter of the two geometry matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(geo_vals, corr_vals, alpha=0.6, s=30, color="#4878CF")
m, b = np.polyfit(geo_vals, corr_vals, 1)
xs = np.linspace(geo_vals.min(), geo_vals.max(), 100)
ax.plot(xs, m*xs + b, "r--", lw=1.5)
ax.set_xlabel("S2 Geodesic distance (VAD Warriner)")
ax.set_ylabel("1 - Pearson r (behavioral, avg across movies)")
ax.set_title(f"S2-geodesic vs. behavioral similarity\n"
             f"Spearman rho = {rho:.3f}  p = {pval:.3e}", fontsize=9)

# Per-movie rho distribution
ax = axes[1]
ax.hist(valid_rho, bins=10, color="#6ACC65", edgecolor="white")
ax.axvline(np.mean(valid_rho), color="red", lw=2,
           label=f"mean = {np.mean(valid_rho):.3f}")
ax.set_xlabel("Spearman rho (per movie)")
ax.set_ylabel("Count")
ax.set_title("Distribution of per-movie geometry correlations")
ax.legend(fontsize=9)

plt.tight_layout()
geom_corr_path = os.path.join(RESULTS, "geometry_correlation.png")
fig.savefig(geom_corr_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {geom_corr_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART B — fMRI decoding by region
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART B: fMRI decoding performance by region")
print("="*60)

traj = pd.read_csv(TRAJ_CSV)
print(f"Loaded {TRAJ_CSV.split(os.sep)[-1]}: {traj.shape}")
print(f"Regions: {traj['region'].unique().tolist()}")

REGION_ORDER = ["Hippocampus", "EntorhinalCortex", "vmPFC_a24_included"]
REGION_LABELS = {"Hippocampus": "HC", "EntorhinalCortex": "ERC",
                 "vmPFC_a24_included": "vmPFC"}

# Mean decoding = mean of mds1 and mds2 per subject/region
traj["mean_dec"] = (traj["mds1"] + traj["mds2"]) / 2.0

# Summary per region
print("\nDecoding summary (mds1, mds2, mean):")
for reg in REGION_ORDER:
    sub = traj[traj["region"] == reg]
    print(f"  {REGION_LABELS[reg]:8s}  "
          f"mds1={sub['mds1'].mean():.4f}(+-{sub['mds1'].std():.4f})  "
          f"mds2={sub['mds2'].mean():.4f}(+-{sub['mds2'].std():.4f})  "
          f"mean={sub['mean_dec'].mean():.4f}(+-{sub['mean_dec'].std():.4f})")

# Pivot: subject × region for paired t-test
pivot = traj.pivot_table(index="subject", columns="region",
                         values="mean_dec", aggfunc="mean")
pivot = pivot[REGION_ORDER]

print("\nPaired t-tests (mean decoding):")
pairs = [("vmPFC_a24_included","Hippocampus"),
         ("vmPFC_a24_included","EntorhinalCortex"),
         ("Hippocampus","EntorhinalCortex")]
ttest_results = {}
for r1, r2 in pairs:
    sub_common = pivot[[r1, r2]].dropna()
    t, p = stats.ttest_rel(sub_common[r1], sub_common[r2])
    label = f"{REGION_LABELS[r1]} vs {REGION_LABELS[r2]}"
    print(f"  {label:20s}  t = {t:.4f}  p = {p:.4f}  N = {len(sub_common)}")
    ttest_results[label] = {"t": round(t,4), "p": round(p,4), "N": len(sub_common)}

# vmPFC better than HC per subject
sub_common_vhc = pivot[["vmPFC_a24_included","Hippocampus"]].dropna()
n_better = (sub_common_vhc["vmPFC_a24_included"] > sub_common_vhc["Hippocampus"]).sum()
print(f"\nvmPFC > HC: {n_better}/{len(sub_common_vhc)} subjects "
      f"({100*n_better/len(sub_common_vhc):.1f}%)")

# Violin plot
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
metrics = [("mds1", "MDS1 decoding"), ("mds2", "MDS2 decoding"),
           ("mean_dec", "Mean decoding")]

for ax, (col, label) in zip(axes, metrics):
    data_list = [traj[traj["region"]==r][col].dropna().values for r in REGION_ORDER]
    vp = ax.violinplot(data_list, showmedians=True, showmeans=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_alpha(0.7)
    for i, (d, r) in enumerate(zip(data_list, REGION_ORDER)):
        ax.scatter([i+1]*len(d), d, color="black", s=15, alpha=0.5, zorder=3)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(1, len(REGION_ORDER)+1))
    ax.set_xticklabels([REGION_LABELS[r] for r in REGION_ORDER])
    ax.set_ylabel("Pearson r (predicted vs. observed)")
    ax.set_title(label, fontsize=9)

# annotate significance
for ax in axes:
    y_max = ax.get_ylim()[1]
    # vmPFC vs HC
    r1_col, r2_col = "vmPFC_a24_included", "Hippocampus"
    sub = pivot[[r1_col, r2_col]].dropna()
    t, p = stats.ttest_rel(sub[r1_col], sub[r2_col])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    ax.annotate(f"vmPFC-HC: {sig}", xy=(0.5, 0.97), xycoords="axes fraction",
                ha="center", fontsize=7)

plt.suptitle("fMRI 2D-trajectory decoding by region", fontsize=11, fontweight="bold")
plt.tight_layout()
dec_path = os.path.join(RESULTS, "decoding_by_region.png")
fig.savefig(dec_path, dpi=150)
plt.close(fig)
print(f"Saved: {dec_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART C — VAD peripherality vs. category-level decoding
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART C: VAD peripherality vs. category decoding accuracy")
print("="*60)

vad_df = pd.read_csv(VAD_CSV)
peripherality = dict(zip(vad_df["emotion"], vad_df["norm"]))
print("VAD peripherality (Warriner centered norm):")
for e in EMOTIONS:
    print(f"  {e:20s}  norm = {peripherality[e]:.4f}")

cat = pd.read_csv(CAT_CSV)
print(f"\nLoaded category CSV: {cat.shape}, regions: {cat['region'].unique().tolist()}")

# Use vmPFC_a24_included for comparability
cat_vmpfc = cat[cat["region"] == "vmPFC_a24_included"].copy()
cat_hc    = cat[cat["region"] == "Hippocampus"].copy()

# Mean decoding per emotion across subjects
mean_dec_vmpfc = cat_vmpfc[EMOTIONS].mean()
mean_dec_hc    = cat_hc[EMOTIONS].mean()

print("\nMean vmPFC decoding per emotion:")
for e in EMOTIONS:
    print(f"  {e:20s}  vmPFC={mean_dec_vmpfc[e]:.4f}  HC={mean_dec_hc[e]:.4f}")

# Spearman: peripherality vs vmPFC decoding
periph_vals = np.array([peripherality[e] for e in EMOTIONS])
dec_vals    = mean_dec_vmpfc[EMOTIONS].values

rho_p, pval_p = stats.spearmanr(periph_vals, dec_vals)
print(f"\nSpearman rho (peripherality vs vmPFC decoding): "
      f"{rho_p:.4f}  p = {pval_p:.4f}")

# ASBH prediction: extremal emotions (Fear, Disgust, Love)
# should have large geodesic distances from each other
extremal = ["Fear", "Disgust", "Love"]
geo_mat_df = pd.read_csv(GEO_CSV, index_col=0)
print("\nGeodetic distances between extremal emotions (Fear, Disgust, Love):")
for i, e1 in enumerate(extremal):
    for e2 in extremal[i+1:]:
        d = geo_mat_df.loc[e1, e2]
        dec_e1 = mean_dec_vmpfc[e1]
        dec_e2 = mean_dec_vmpfc[e2]
        print(f"  {e1:10s} - {e2:10s}:  geodesic = {d:.4f}  "
              f"vmPFC dec({e1})={dec_e1:.4f}  vmPFC dec({e2})={dec_e2:.4f}")

# Scatter: peripherality vs decoding
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (dec_vals_plot, region_label, color) in zip(axes, [
    (mean_dec_vmpfc[EMOTIONS].values, "vmPFC", "#D65F5F"),
    (mean_dec_hc[EMOTIONS].values,    "HC",    "#4878CF"),
]):
    ax.scatter(periph_vals, dec_vals_plot, s=60, color=color, alpha=0.85, zorder=3)
    for e, x, y in zip(EMOTIONS, periph_vals, dec_vals_plot):
        ax.annotate(e, (x, y), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.5)

    rho_r, pval_r = stats.spearmanr(periph_vals, dec_vals_plot)
    m_r, b_r = np.polyfit(periph_vals, dec_vals_plot, 1)
    xs_r = np.linspace(periph_vals.min(), periph_vals.max(), 100)
    ax.plot(xs_r, m_r*xs_r + b_r, "--", color=color, lw=1.5, alpha=0.7)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("VAD peripherality (||v_centered||, Warriner)")
    ax.set_ylabel(f"Mean {region_label} decoding (Pearson r)")
    ax.set_title(f"{region_label}: peripherality vs. decoding\n"
                 f"Spearman rho = {rho_r:.3f}  p = {pval_r:.3f}", fontsize=9)

plt.suptitle("VAD Peripherality vs. Category Decoding Accuracy", fontsize=11,
             fontweight="bold")
plt.tight_layout()
periph_path = os.path.join(RESULTS, "peripherality_vs_decoding.png")
fig.savefig(periph_path, dpi=150)
plt.close(fig)
print(f"Saved: {periph_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Save summary JSON
# ═══════════════════════════════════════════════════════════════════════════════
summary = {
    "part_A_geometry_correlation": {
        "spearman_rho_averaged":      round(rho,   4),
        "spearman_p":                 round(pval,  6),
        "per_movie_rho_mean":         round(float(np.mean(valid_rho)), 4),
        "per_movie_rho_std":          round(float(np.std(valid_rho)),  4),
        "n_movies":                   len(movies_valid),
        "n_movies_skipped":           len(SKIP_MOVIES),
        "skipped_movies":             SKIP_MOVIES,
        "interpretation": (
            "S2-geodesic and behavioral correlation-distance are positively "
            f"correlated (rho={rho:.3f}), suggesting VAD-sphere distances "
            "partially capture behavioral emotion similarity."
            if rho > 0 else
            "Negative or near-zero correlation: S2-geodesic distances are "
            "not well aligned with behavioral similarity."
        ),
    },
    "part_B_decoding_by_region": {
        "vmPFC_mds1_mean":  round(traj[traj["region"]=="vmPFC_a24_included"]["mds1"].mean(), 4),
        "vmPFC_mds1_sd":    round(traj[traj["region"]=="vmPFC_a24_included"]["mds1"].std(),  4),
        "vmPFC_mds2_mean":  round(traj[traj["region"]=="vmPFC_a24_included"]["mds2"].mean(), 4),
        "vmPFC_mds2_sd":    round(traj[traj["region"]=="vmPFC_a24_included"]["mds2"].std(),  4),
        "vmPFC_mean_mean":  round(traj[traj["region"]=="vmPFC_a24_included"]["mean_dec"].mean(), 4),
        "HC_mean_mean":     round(traj[traj["region"]=="Hippocampus"]["mean_dec"].mean(), 4),
        "ERC_mean_mean":    round(traj[traj["region"]=="EntorhinalCortex"]["mean_dec"].mean(), 4),
        "paired_ttests":    ttest_results,
        "vmPFC_better_than_HC_pct": round(100*n_better/len(sub_common_vhc), 1),
    },
    "part_C_peripherality": {
        "spearman_rho_vmPFC": round(rho_p,  4),
        "spearman_p_vmPFC":   round(pval_p, 4),
        "peripherality":      {e: round(peripherality[e], 4) for e in EMOTIONS},
        "vmPFC_decoding_mean":{e: round(float(mean_dec_vmpfc[e]), 4) for e in EMOTIONS},
        "ASBH_prediction": (
            "ASBH predicts extremal emotions (Fear, Disgust, Love) are most "
            "geometrically separated on S2 and thus most decodable by vmPFC."
        ),
    },
}

json_path = os.path.join(RESULTS, "geometry_comparison_summary.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {json_path}")

# ── Final printout ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
mean_rho_str = f"{np.mean(valid_rho):.4f}" if valid_rho else "nan"
print(f"A) S2-geodesic <-> behavioral similarity: rho = {rho:.4f}  p = {pval:.4e}")
print(f"   Per-movie mean rho = {mean_rho_str}")
print(f"   {'ALIGNED' if rho > 0.2 else 'WEAKLY ALIGNED' if rho > 0 else 'NOT ALIGNED'}")
vmPFC_mean = traj[traj["region"]=="vmPFC_a24_included"]["mean_dec"].mean()
hc_mean    = traj[traj["region"]=="Hippocampus"]["mean_dec"].mean()
print(f"B) vmPFC mean decoding = {vmPFC_mean:.4f}  |  HC = {hc_mean:.4f}")
print(f"   vmPFC > HC in {n_better}/{len(sub_common_vhc)} subjects")
print(f"C) Peripherality vs vmPFC decoding: rho = {rho_p:.4f}  p = {pval_p:.4f}")
print(f"   {'SUPPORTED' if rho_p > 0 and pval_p < 0.05 else 'NOT SIGNIFICANT'}")
