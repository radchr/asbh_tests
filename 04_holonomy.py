"""
ASBH Test 04 — Holonomy on Behavioural S2 Trajectories
=======================================================
Maps emotion ratings onto the unit sphere via weighted VAD centroid,
detects quasi-loops (same octant, 30-120 timepoints apart) and computes
the solid angle (holonomy) enclosed by each loop.

Core ASBH prediction:
  If the emotional manifold is S², a quasi-loop that encloses solid angle Omega
  accumulates a parallel-transport phase equal to Omega. The LARGER Omega,
  the more a "transported vector" is rotated — which should produce a
  larger effective closure error when the loop is not perfectly closed.
  Therefore we expect:  corr(Omega_0, closure_error) > 0

Outputs  (asbh_tests/results/):
  holonomy_examples.png      — 4 example loops on the sphere
  holonomy_vs_closure.png    — scatter Omega vs closure error
  holonomy_results.json
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa
from scipy import stats
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "asbh_tests", "results")
BEH_JSON = os.path.join(ROOT, "EmotionConceptRepresentation", "data", "behTab_json.json")
VAD_CSV  = os.path.join(RESULTS, "emotion_vad.csv")

EXCLUDE  = {"DamagedKungFu", "RidingTheRails", "LeassonLearned"}
EMOTIONS = ["Anger","Anxiety","Fear","Surprise","Guilt","Disgust","Sad",
            "Regard","Satisfaction","WarmHeartedness","Happiness","Pride","Love"]

LOOP_MIN, LOOP_MAX = 30, 120   # timepoint span

# ── load data ─────────────────────────────────────────────────────────────────
vad_df = pd.read_csv(VAD_CSV).set_index("emotion")
VAD_S2 = np.array([vad_df.loc[e, ["V_s2","A_s2","D_s2"]].values for e in EMOTIONS])
# VAD_S2 shape: (13, 3) — unit vectors for each emotion

with open(BEH_JSON) as f:
    beh = json.load(f)

movies = [m for m in beh if m not in EXCLUDE]

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: S² trajectory for each movie
# ═══════════════════════════════════════════════════════════════════════════════

def make_s2_trajectory(movie_data):
    """
    Returns array (T, 3) of unit vectors on S².
    position[t] = normalised weighted sum of emotion VAD_s2 vectors.
    Timepoints where ||weighted_sum|| ~ 0 are flagged as NaN rows.
    """
    ratings = np.array(
        [movie_data[e] for e in EMOTIONS], dtype=float
    )  # (13, T)

    # replace NaN ratings with 0
    ratings = np.nan_to_num(ratings, nan=0.0)

    # clip negatives to 0 (negative ratings are near-zero noise; weight can't be negative)
    ratings_pos = np.clip(ratings, 0.0, None)

    # weighted centroid: (13,T)^T @ (13,3)  →  (T,3)
    weighted = (ratings_pos.T @ VAD_S2)  # (T, 3)

    norms = np.linalg.norm(weighted, axis=1, keepdims=True)  # (T, 1)

    # mask near-zero norms
    valid = (norms[:, 0] > 1e-6)
    traj = np.zeros_like(weighted)
    traj[valid] = weighted[valid] / norms[valid]
    traj[~valid] = np.nan
    return traj


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Find quasi-loops (greedy non-overlapping)
# ═══════════════════════════════════════════════════════════════════════════════

def octant(v):
    """Return octant index 0-7 based on sign of (V, A, D)."""
    sv = int(v[0] >= 0)
    sa = int(v[1] >= 0)
    sd = int(v[2] >= 0)
    return (sv << 2) | (sa << 1) | sd


def find_quasi_loops(traj, loop_min=LOOP_MIN, loop_max=LOOP_MAX):
    """
    Greedy scan: for each t_start find the FIRST t_end in [t_start+loop_min,
    t_start+loop_max] that shares the same octant, then advance to t_end+1.
    Returns list of (t_start, t_end) tuples.
    """
    T = len(traj)
    valid = ~np.isnan(traj[:, 0])
    loops = []
    t = 0
    while t < T - loop_min:
        if not valid[t]:
            t += 1
            continue
        oct_start = octant(traj[t])
        # search window
        end_min = t + loop_min
        end_max = min(t + loop_max, T - 1)
        found = False
        for te in range(end_min, end_max + 1):
            if valid[te] and octant(traj[te]) == oct_start:
                loops.append((t, te))
                t = te + 1
                found = True
                break
        if not found:
            t += 1
    return loops


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 & 4: Solid angle and loop statistics
# ═══════════════════════════════════════════════════════════════════════════════

def geodesic_dist(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def solid_angle_loop(pts):
    """
    Discrete solid angle (holonomy) for a closed polygon on S².
    pts: (n, 3) unit vectors.  Loop closes from pts[-1] back to pts[0].

    Uses the discrete formula at each vertex i:
      dOmega_i = 2 * arctan2(
          dot(p[i], cross(p[i-1], p[i+1])),
          1 + dot(p[i-1], p[i]) + dot(p[i], p[i+1]) + dot(p[i-1], p[i+1])
      )
    Omega_0 = |sum(dOmega_i)|
    """
    n = len(pts)
    if n < 3:
        return 0.0
    omega = 0.0
    for i in range(n):
        pm = pts[(i - 1) % n]
        pc = pts[i]
        pp = pts[(i + 1) % n]

        cross_mp = np.cross(pm, pp)
        num = np.dot(pc, cross_mp)
        den = (1.0
               + np.dot(pm, pc)
               + np.dot(pc, pp)
               + np.dot(pm, pp))
        if abs(den) < 1e-10:
            continue
        omega += 2.0 * np.arctan2(num, den)

    return abs(omega)


def loop_stats(traj, t_start, t_end):
    """Return (pts, solid_angle, loop_length, closure_error)."""
    pts = traj[t_start: t_end + 1].copy()

    # close the loop for solid angle: append start point
    pts_closed = np.vstack([pts, pts[0]])

    Omega  = solid_angle_loop(pts)             # uses implicit closure
    length = sum(geodesic_dist(pts[i], pts[i+1])
                 for i in range(len(pts) - 1))
    closure = geodesic_dist(pts[0], pts[-1])

    return pts, Omega, length, closure


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop: process all movies
# ═══════════════════════════════════════════════════════════════════════════════

all_omega   = []
all_closure = []
all_length  = []
all_movie   = []

example_loops = []   # store up to 4 interesting examples for plotting

print(f"Processing {len(movies)} movies ...")
for movie in movies:
    traj = make_s2_trajectory(beh[movie])
    loops = find_quasi_loops(traj)

    if not loops:
        print(f"  {movie:25s}: 0 quasi-loops found")
        continue

    movie_omega, movie_closure = [], []
    for (ts, te) in loops:
        pts, Omega, length, closure = loop_stats(traj, ts, te)
        all_omega.append(Omega)
        all_closure.append(closure)
        all_length.append(length)
        all_movie.append(movie)
        movie_omega.append(Omega)
        movie_closure.append(closure)

        # collect diverse examples (large/small omega)
        if len(example_loops) < 4 and Omega > 0.05 and len(pts) >= 10:
            example_loops.append({
                "movie": movie, "t_start": ts, "t_end": te,
                "pts": pts, "Omega": Omega, "closure": closure,
                "length": length,
            })

    print(f"  {movie:25s}: {len(loops):3d} loops  "
          f"omega={np.mean(movie_omega):.3f}+/-{np.std(movie_omega):.3f}  "
          f"closure={np.mean(movie_closure):.3f}+/-{np.std(movie_closure):.3f}")

all_omega   = np.array(all_omega)
all_closure = np.array(all_closure)
all_length  = np.array(all_length)

print(f"\nTotal quasi-loops: {len(all_omega)}")
print(f"Omega  : mean={all_omega.mean():.4f}  std={all_omega.std():.4f}  "
      f"max={all_omega.max():.4f}")
print(f"Closure: mean={all_closure.mean():.4f}  std={all_closure.std():.4f}")
print(f"Length : mean={all_length.mean():.4f}  std={all_length.std():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Correlation Omega vs closure_error
# ═══════════════════════════════════════════════════════════════════════════════
rho, pval = stats.spearmanr(all_omega, all_closure)
print(f"\nSpearman rho (Omega vs closure_error): {rho:.4f}  p = {pval:.4e}")

rho_len, _ = stats.spearmanr(all_omega, all_length)
print(f"Spearman rho (Omega vs loop_length)  : {rho_len:.4f}")

# partial: Omega vs closure controlling for length
# residualise both against length
def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    res_x = stats.spearmanr(x, z)[0]
    res_y = stats.spearmanr(y, z)[0]
    # rank residuals
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # partial via regression on ranks
    from numpy.linalg import lstsq
    A = np.column_stack([rz, np.ones(len(rz))])
    rx_res = rx - A @ lstsq(A, rx, rcond=None)[0]
    ry_res = ry - A @ lstsq(A, ry, rcond=None)[0]
    rho_p, p_p = stats.pearsonr(rx_res, ry_res)
    return rho_p, p_p

rho_partial, pval_partial = partial_spearman(all_omega, all_closure, all_length)
print(f"Partial rho (Omega vs closure | length): {rho_partial:.4f}  p = {pval_partial:.4e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: example loops on the sphere
# ═══════════════════════════════════════════════════════════════════════════════

def sphere_wireframe(ax, alpha=0.18):
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", alpha=alpha, lw=0.4)


n_ex = min(4, len(example_loops))
fig = plt.figure(figsize=(5 * n_ex, 5))
for k, ex in enumerate(example_loops[:n_ex]):
    ax = fig.add_subplot(1, n_ex, k+1, projection="3d")
    sphere_wireframe(ax)
    pts = ex["pts"]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            color="#4878CF", lw=1.5, alpha=0.8)
    ax.scatter(*pts[0],  color="green", s=60, zorder=5, label="start")
    ax.scatter(*pts[-1], color="red",   s=60, zorder=5, label="end")
    ax.set_title(f"{ex['movie']}\nt=[{ex['t_start']},{ex['t_end']}]\n"
                 f"$\\Omega_0$={ex['Omega']:.3f} rad  "
                 f"err={ex['closure']:.3f}", fontsize=7)
    ax.set_xlabel("V", fontsize=7); ax.set_ylabel("A", fontsize=7)
    ax.set_zlabel("D", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

plt.suptitle("Example quasi-loops on S² (VAD-weighted trajectories)",
             fontsize=10, fontweight="bold")
plt.tight_layout()
ex_path = os.path.join(RESULTS, "holonomy_examples.png")
fig.savefig(ex_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {ex_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Omega vs closure_error scatter
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left: raw scatter
ax = axes[0]
sc = ax.scatter(all_omega, all_closure, c=all_length,
                cmap="viridis", s=12, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Loop length (rad)")
# regression line
m_fit, b_fit = np.polyfit(all_omega, all_closure, 1)
xs = np.linspace(all_omega.min(), all_omega.max(), 100)
ax.plot(xs, m_fit*xs + b_fit, "r--", lw=1.5)
ax.set_xlabel("Solid angle $\\Omega_0$ (rad)")
ax.set_ylabel("Closure error (rad)")
ax.set_title(f"Holonomy vs. closure error\n"
             f"Spearman $\\rho$ = {rho:.3f}  p = {pval:.3e}\n"
             f"Partial $\\rho$ (|length) = {rho_partial:.3f}  "
             f"p = {pval_partial:.3e}", fontsize=9)

# right: per-movie mean Omega vs mean closure
per_movie_data = {}
for om, cl, mv in zip(all_omega, all_closure, all_movie):
    per_movie_data.setdefault(mv, {"omega":[], "closure":[]})
    per_movie_data[mv]["omega"].append(om)
    per_movie_data[mv]["closure"].append(cl)

mvs = list(per_movie_data.keys())
mv_omega   = [np.mean(per_movie_data[m]["omega"])   for m in mvs]
mv_closure = [np.mean(per_movie_data[m]["closure"]) for m in mvs]
ax2 = axes[1]
ax2.scatter(mv_omega, mv_closure, s=60, color="#D65F5F", zorder=3)
for mv, ox, cy in zip(mvs, mv_omega, mv_closure):
    ax2.annotate(mv[:8], (ox, cy), textcoords="offset points",
                 xytext=(3, 3), fontsize=6)
m2, b2 = np.polyfit(mv_omega, mv_closure, 1)
xs2 = np.linspace(min(mv_omega), max(mv_omega), 50)
ax2.plot(xs2, m2*xs2 + b2, "k--", lw=1)
rho_mv, pval_mv = stats.spearmanr(mv_omega, mv_closure)
ax2.set_xlabel("Mean $\\Omega_0$ per movie (rad)")
ax2.set_ylabel("Mean closure error per movie (rad)")
ax2.set_title(f"Per-movie means\nSpearman $\\rho$ = {rho_mv:.3f}  p = {pval_mv:.3f}",
              fontsize=9)

plt.suptitle("Holonomy (Solid Angle) vs. Loop Closure Error",
             fontsize=11, fontweight="bold")
plt.tight_layout()
sc_path = os.path.join(RESULTS, "holonomy_vs_closure.png")
fig.savefig(sc_path, dpi=150)
plt.close(fig)
print(f"Saved: {sc_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Save JSON
# ═══════════════════════════════════════════════════════════════════════════════
per_movie_summary = {
    mv: {
        "n_loops":        len(per_movie_data[mv]["omega"]),
        "mean_omega":     round(float(np.mean(per_movie_data[mv]["omega"])),  5),
        "mean_closure":   round(float(np.mean(per_movie_data[mv]["closure"])), 5),
    }
    for mv in mvs
}

results = {
    "n_loops":              int(len(all_omega)),
    "mean_omega":           round(float(all_omega.mean()),   5),
    "std_omega":            round(float(all_omega.std()),    5),
    "mean_closure":         round(float(all_closure.mean()), 5),
    "corr_omega_closure": {
        "spearman_rho":     round(float(rho),          4),
        "p_value":          round(float(pval),          6),
        "partial_rho":      round(float(rho_partial),   4),
        "partial_p":        round(float(pval_partial),  6),
    },
    "corr_omega_length": {
        "spearman_rho":     round(float(rho_len),       4),
    },
    "per_movie":            per_movie_summary,
    "interpretation": (
        f"Spearman rho(Omega, closure) = {rho:.3f} (p={pval:.3e}). "
        + ("ASBH SUPPORTED: larger solid angle correlates with larger closure error, "
           "consistent with non-trivial holonomy on S2."
           if rho > 0.1 and pval < 0.05
           else
           "ASBH WEAK / NOT SUPPORTED: no significant positive correlation between "
           "enclosed solid angle and loop closure error.")
    ),
}

json_path = os.path.join(RESULTS, "holonomy_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {json_path}")

# ─── Final verdict ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("HOLONOMY RESULT")
print("="*60)
print(f"  Total quasi-loops   : {len(all_omega)}")
print(f"  Mean solid angle    : {all_omega.mean():.4f} rad  "
      f"(range {all_omega.min():.4f}-{all_omega.max():.4f})")
print(f"  Omega vs closure    : rho = {rho:.4f}  p = {pval:.4e}")
print(f"  Partial rho (|len)  : {rho_partial:.4f}  p = {pval_partial:.4e}")
if rho > 0.1 and pval < 0.05:
    print("  => ASBH SUPPORTED: holonomy signal detected")
else:
    print("  => ASBH NOT SUPPORTED / INCONCLUSIVE")
print("="*60)
