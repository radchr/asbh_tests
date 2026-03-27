"""
ASBH Test 05 — S²-траєкторії vs MDS авторів
============================================
Part A : Кореляція S²-швидкостей з MDS-швидкостями по фільмах
Part B : Аналіз "полярних" переходів (зміна знаку Valence)
Part C : Heatmap тілесних кутів (фільми × часові вікна)

Outputs (asbh_tests/results/):
  s2_vs_mds_correlations.csv
  polar_transition_analysis.json
  omega_heatmap.png
  s2_trajectory_example.png
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa
from scipy import stats
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS  = os.path.join(ROOT, "asbh_tests", "results")
DATA     = os.path.join(ROOT, "EmotionConceptRepresentation", "data")

BEH_JSON = os.path.join(DATA, "behTab_json.json")
MDS_JSON = os.path.join(DATA, "mds_2d.json")
VAD_CSV  = os.path.join(RESULTS, "emotion_vad.csv")

EXCLUDE  = {"DamagedKungFu", "RidingTheRails", "LeassonLearned"}
EMOTIONS = ["Anger","Anxiety","Fear","Surprise","Guilt","Disgust","Sad",
            "Regard","Satisfaction","WarmHeartedness","Happiness","Pride","Love"]
N_WINDOWS = 10   # for heatmap

# ── load shared data ──────────────────────────────────────────────────────────
vad_df = pd.read_csv(VAD_CSV).set_index("emotion")
VAD_S2 = np.array([vad_df.loc[e, ["V_s2","A_s2","D_s2"]].values
                   for e in EMOTIONS])           # (13, 3)

with open(BEH_JSON) as f:
    beh = json.load(f)
with open(MDS_JSON) as f:
    mds_data = json.load(f)

movies = [m for m in mds_data if m not in EXCLUDE]

# ── helpers ───────────────────────────────────────────────────────────────────

def geodesic(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def make_s2_traj(movie_name):
    """
    Returns (T, 3) array of unit vectors.
    Uses probability-style weights: w[t] = ratings[t] / sum(ratings[t]).
    NaN or near-zero-sum timepoints become np.nan rows.
    """
    ratings = np.array([beh[movie_name][e] for e in EMOTIONS],
                       dtype=float)              # (13, T)
    ratings = np.nan_to_num(ratings, nan=0.0)
    ratings = np.clip(ratings, 0.0, None)        # no negative weights

    T = ratings.shape[1]
    traj = np.full((T, 3), np.nan)

    for t in range(T):
        w = ratings[:, t]
        s = w.sum()
        if s < 1e-10:
            continue
        w_norm = w / s                           # probability weights
        v = w_norm @ VAD_S2                      # (3,) weighted centroid
        n = np.linalg.norm(v)
        if n < 1e-10:
            continue
        traj[t] = v / n
    return traj


def triplet_solid_angle(a, b, c):
    """
    Solid angle of spherical triangle (a, b, c) ≈ area on unit sphere.
    Uses van Oosterom-Strackee:
      tan(Omega/2) = |a·(b×c)| / (1 + a·b + b·c + a·c)
    Returns scalar in [0, 2π).
    """
    num = abs(float(np.dot(a, np.cross(b, c))))
    den = 1.0 + float(np.dot(a, b)) + float(np.dot(b, c)) + float(np.dot(a, c))
    if abs(den) < 1e-12:
        return 0.0
    return 2.0 * np.arctan2(num, den)


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Кореляція S²-швидкостей з MDS-швидкостями
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART A: S²-trajectory speed vs MDS speed (Spearman rho)")
print("="*60)

records_corr = []

for movie in movies:
    s2_traj = make_s2_traj(movie)
    mds1 = np.array(mds_data[movie]["mds1"])
    mds2 = np.array(mds_data[movie]["mds2"])

    T = min(len(s2_traj), len(mds1))
    s2_traj = s2_traj[:T]
    mds1 = mds1[:T]; mds2 = mds2[:T]

    # consecutive distances
    s2_dists, mds_dists = [], []
    for t in range(T - 1):
        a, b = s2_traj[t], s2_traj[t+1]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        s2_dists.append(geodesic(a, b))
        mds_dists.append(np.hypot(mds1[t+1]-mds1[t], mds2[t+1]-mds2[t]))

    s2_arr  = np.array(s2_dists)
    mds_arr = np.array(mds_dists)

    rho, pval = stats.spearmanr(s2_arr, mds_arr)
    records_corr.append({
        "film":          movie,
        "spearman_rho":  round(float(rho),   4),
        "p_value":       round(float(pval),  6),
        "n_timepoints":  len(s2_arr),
    })
    print(f"  {movie:25s}  rho={rho:+.4f}  p={pval:.3e}  N={len(s2_arr)}")

corr_df = pd.DataFrame(records_corr)
csv_path = os.path.join(RESULTS, "s2_vs_mds_correlations.csv")
corr_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

mean_rho = corr_df["spearman_rho"].mean()
std_rho  = corr_df["spearman_rho"].std()
print(f"Mean rho across films: {mean_rho:.4f} +/- {std_rho:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART B: Полярні переходи (sign-flip in Valence)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART B: Polar transitions (Valence sign-flip)")
print("="*60)

polar_ratio   = []   # geodesic / MDS for polar transitions
regular_ratio = []   # same for regular transitions

for movie in movies:
    s2_traj = make_s2_traj(movie)
    mds1 = np.array(mds_data[movie]["mds1"])
    mds2 = np.array(mds_data[movie]["mds2"])
    T = min(len(s2_traj), len(mds1))

    for t in range(T - 1):
        a, b = s2_traj[t], s2_traj[t+1]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue

        geo_d = geodesic(a, b)
        mds_d = np.hypot(mds1[t+1]-mds1[t], mds2[t+1]-mds2[t])
        if mds_d < 1e-10:
            continue

        ratio = geo_d / mds_d
        # polar: sign flip in Valence component (index 0)
        is_polar = (a[0] * b[0]) < 0

        if is_polar:
            polar_ratio.append(ratio)
        else:
            regular_ratio.append(ratio)

polar_arr   = np.array(polar_ratio)
regular_arr = np.array(regular_ratio)

t_stat, p_ttest = stats.ttest_ind(polar_arr, regular_arr, equal_var=False)
rho_ks, p_ks    = stats.ks_2samp(polar_arr, regular_arr)

print(f"  Polar transitions   : N={len(polar_arr):5d}  "
      f"mean ratio = {polar_arr.mean():.4f} +/- {polar_arr.std():.4f}")
print(f"  Regular transitions : N={len(regular_arr):5d}  "
      f"mean ratio = {regular_arr.mean():.4f} +/- {regular_arr.std():.4f}")
print(f"  Welch t-test  : t = {t_stat:.4f}  p = {p_ttest:.4e}")
print(f"  KS test       : D = {rho_ks:.4f}  p = {p_ks:.4e}")
asbh_polar = bool(polar_arr.mean() > regular_arr.mean() and p_ttest < 0.05)
print(f"  ASBH prediction supported: {asbh_polar}")

polar_result = {
    "n_polar":            int(len(polar_arr)),
    "n_regular":          int(len(regular_arr)),
    "polar_ratio_mean":   round(float(polar_arr.mean()),   5),
    "polar_ratio_sd":     round(float(polar_arr.std()),    5),
    "nonpolar_ratio_mean":round(float(regular_arr.mean()), 5),
    "nonpolar_ratio_sd":  round(float(regular_arr.std()),  5),
    "t_stat":             round(float(t_stat),  4),
    "p_value":            round(float(p_ttest), 6),
    "ks_D":               round(float(rho_ks),  4),
    "ks_p":               round(float(p_ks),    6),
    "asbh_supported":     asbh_polar,
    "interpretation": (
        "ASBH SUPPORTED: geodesic/MDS ratio is significantly larger for "
        "valence-polar transitions, indicating S² stretches cross-pole distances "
        "relative to the flat MDS embedding."
        if asbh_polar else
        "ASBH NOT SUPPORTED for polar transitions: no significant difference in "
        "geodesic/MDS ratio between polar and regular transitions."
    ),
}
json_path = os.path.join(RESULTS, "polar_transition_analysis.json")
with open(json_path, "w") as f:
    json.dump(polar_result, f, indent=2)
print(f"Saved: {json_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART C: Omega heatmap (films × time windows)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART C: Omega heatmap (triplet solid angles)")
print("="*60)

omega_matrix = np.zeros((len(movies), N_WINDOWS))
movie_labels = []
movie_omega_medians = {}

for mi, movie in enumerate(movies):
    s2_traj = make_s2_traj(movie)
    T = len(s2_traj)
    movie_labels.append(movie)

    # compute triplet solid angles for all valid consecutive triplets
    triplet_omegas = []
    triplet_times  = []
    for t in range(T - 2):
        a, b, c = s2_traj[t], s2_traj[t+1], s2_traj[t+2]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
            continue
        omega = triplet_solid_angle(a, b, c)
        triplet_omegas.append(omega)
        triplet_times.append(t + 1)           # centre of triplet

    if not triplet_omegas:
        print(f"  {movie}: no valid triplets")
        continue

    omegas = np.array(triplet_omegas)
    times  = np.array(triplet_times, dtype=float)
    movie_omega_medians[movie] = float(np.median(omegas))

    # bin into N_WINDOWS equal-time windows
    bins = np.linspace(0, T, N_WINDOWS + 1)
    for w in range(N_WINDOWS):
        mask = (times >= bins[w]) & (times < bins[w+1])
        omega_matrix[mi, w] = omegas[mask].mean() if mask.any() else 0.0

    print(f"  {movie:25s}  N_triplets={len(omegas):5d}  "
          f"median_Omega={np.median(omegas):.6f}  "
          f"p95={np.percentile(omegas,95):.6f}")

# heatmap plot
fig, ax = plt.subplots(figsize=(12, max(5, len(movies) * 0.5)))

# use log scale to handle the heavy tail
omega_log = np.log1p(omega_matrix * 1000)   # log(1 + 1000*Omega) for visibility

im = ax.imshow(omega_log, aspect="auto", cmap="YlOrRd",
               interpolation="nearest")

ax.set_yticks(range(len(movie_labels)))
ax.set_yticklabels(movie_labels, fontsize=8)
ax.set_xticks(range(N_WINDOWS))
ax.set_xticklabels([f"W{i+1}" for i in range(N_WINDOWS)], fontsize=8)
ax.set_xlabel("Time window (equal segments)")
ax.set_ylabel("Film")
ax.set_title("Mean triplet solid angle $\\Omega_0$ per film × time window\n"
             "(colour = log(1 + 1000·$\\Omega$), warmer = larger affective curvature)",
             fontsize=10)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("log(1 + 1000·$\\Omega$)", fontsize=9)

plt.tight_layout()
heatmap_path = os.path.join(RESULTS, "omega_heatmap.png")
fig.savefig(heatmap_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {heatmap_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART D: Trajectory example plot (one film: S² + MDS side by side)
# ═══════════════════════════════════════════════════════════════════════════════
EXAMPLE_MOVIE = "BetweenViewings"   # long, visually rich

s2_ex   = make_s2_traj(EXAMPLE_MOVIE)
mds1_ex = np.array(mds_data[EXAMPLE_MOVIE]["mds1"])
mds2_ex = np.array(mds_data[EXAMPLE_MOVIE]["mds2"])
T_ex    = min(len(s2_ex), len(mds1_ex))
s2_ex   = s2_ex[:T_ex]

# valid timepoint mask
valid_mask = ~np.isnan(s2_ex[:, 0])
t_valid    = np.where(valid_mask)[0]

fig = plt.figure(figsize=(14, 6))

# ── left: S² trajectory ──
ax3d = fig.add_subplot(121, projection="3d")
u = np.linspace(0, 2*np.pi, 30); v = np.linspace(0, np.pi, 15)
xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax3d.plot_wireframe(xs, ys, zs, color="lightgray", alpha=0.2, lw=0.4)

pts_s2  = s2_ex[valid_mask]
n_valid = len(pts_s2)
colors  = plt.cm.plasma(np.linspace(0, 1, n_valid))
for i in range(n_valid - 1):
    ax3d.plot(pts_s2[i:i+2, 0], pts_s2[i:i+2, 1], pts_s2[i:i+2, 2],
              color=colors[i], lw=0.8, alpha=0.7)

ax3d.scatter(*pts_s2[0],  color="green", s=60, zorder=5, label="start")
ax3d.scatter(*pts_s2[-1], color="red",   s=60, zorder=5, label="end")
ax3d.set_xlabel("V", fontsize=8); ax3d.set_ylabel("A", fontsize=8)
ax3d.set_zlabel("D", fontsize=8)
ax3d.set_title(f"S² trajectory — {EXAMPLE_MOVIE}\n(colour = time: purple→yellow)",
               fontsize=9)
ax3d.legend(fontsize=7)

# ── right: MDS trajectory ──
ax2d = fig.add_subplot(122)
n_t  = T_ex
ct   = plt.cm.plasma(np.linspace(0, 1, n_t - 1))
for i in range(n_t - 1):
    ax2d.plot([mds1_ex[i], mds1_ex[i+1]], [mds2_ex[i], mds2_ex[i+1]],
              color=ct[i], lw=0.8, alpha=0.7)
ax2d.scatter(mds1_ex[0],  mds2_ex[0],  color="green", s=60, zorder=5, label="start")
ax2d.scatter(mds1_ex[-1], mds2_ex[-1], color="red",   s=60, zorder=5, label="end")
ax2d.set_xlabel("MDS1"); ax2d.set_ylabel("MDS2")
ax2d.set_title(f"Authors' 2D MDS — {EXAMPLE_MOVIE}\n(colour = time: purple→yellow)",
               fontsize=9)
ax2d.legend(fontsize=7)

plt.suptitle(f"Emotion trajectory comparison: S² (ASBH) vs. flat MDS (authors)\n"
             f"Film: {EXAMPLE_MOVIE}", fontsize=10, fontweight="bold")
plt.tight_layout()
traj_path = os.path.join(RESULTS, "s2_trajectory_example.png")
fig.savefig(traj_path, dpi=150)
plt.close(fig)
print(f"Saved: {traj_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Final console summary
# ═══════════════════════════════════════════════════════════════════════════════
best_omega_film = max(movie_omega_medians, key=movie_omega_medians.get)
best_omega_val  = movie_omega_medians[best_omega_film]

alignment = (
    "well-aligned" if mean_rho > 0.3 else
    "partially aligned" if mean_rho > 0.1 else
    "not aligned"
)

print("\n" + "="*60)
print("=== ASBH vs MDS: Geometry Comparison ===")
print("="*60)
print(f"Mean S2<->MDS trajectory correlation: "
      f"rho = {mean_rho:.4f} +/- {std_rho:.4f}")
print(f"Polar transitions: geodesic/MDS ratio = "
      f"{polar_arr.mean():.4f} (vs regular = {regular_arr.mean():.4f}), "
      f"p = {p_ttest:.4e}")
print(f"Highest Omega film: {best_omega_film} "
      f"(median Omega = {best_omega_val:.6f})")
print(f"Verdict: S2 geometry is {alignment} with authors MDS")
print("="*60)
