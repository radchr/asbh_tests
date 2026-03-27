"""Generate ASBH_summary_figure_v2.png — 6-panel 2x3 figure."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

ROOT = Path(__file__).parent
RES  = ROOT / "results"
DATA = ROOT.parent / "EmotionConceptRepresentation" / "data"
WAR  = ROOT.parent / "data" / "warriner" / "Ratings_Warriner_et_al.csv"

# ── colours ────────────────────────────────────────────────────────────────────
C_BLUE   = "#2166AC"
C_ORANGE = "#F4A322"
C_RED    = "#D6604D"
C_GREEN  = "#4DAC26"
C_GREY   = "#999999"
FILM_CMAP = plt.cm.tab20

# ==============================================================================
# Load cached data
# ==============================================================================

# --- VAD S2 positions ---------------------------------------------------------
vad_s2 = {}
with open(RES / "emotion_vad.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vad_s2[row["emotion"]] = np.array([float(row["V_s2"]),
                                            float(row["A_s2"]),
                                            float(row["D_s2"])])

emotions = list(vad_s2.keys())
pts = np.array([vad_s2[e] for e in emotions])

# --- holonomy -----------------------------------------------------------------
with open(RES / "holonomy_results.json") as f:
    hol = json.load(f)

# Load full loop data from behTab to reconstruct omega/closure arrays
with open(DATA / "behTab_json.json") as f:
    beh_raw = json.load(f)

EXCLUDE = {"DamagedKungFu", "RidingTheRails", "LeassonLearned"}
movies  = [m for m in beh_raw if m not in EXCLUDE]

def make_s2_traj(movie):
    emo_data = beh_raw[movie]
    emo_list = sorted(emo_data.keys())
    T = len(next(iter(emo_data.values())))
    mat = np.zeros((T, len(emo_list)))
    for j, e in enumerate(emo_list):
        vals = np.array(emo_data[e], dtype=float)
        mat[:, j] = np.where(np.isnan(vals), 0.0, np.clip(vals, 0, None))
    vs2 = np.array([vad_s2[e] for e in emo_list])
    col_sums = mat.sum(axis=0)
    valid = col_sums > 1e-10
    w = np.zeros_like(mat)
    w[:, valid] = mat[:, valid] / col_sums[None, valid]
    # w is (T, J), vs2 is (J, 3) -> result is (T, 3)
    weighted = w @ vs2
    norms = np.linalg.norm(weighted, axis=1)
    traj = np.full(weighted.shape, np.nan)
    ok = norms > 1e-10
    traj[ok] = weighted[ok] / norms[ok, None]
    return traj

def solid_angle_loop(pts_loop):
    n = len(pts_loop)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        a = pts_loop[(i-1) % n]
        b = pts_loop[i]
        c = pts_loop[(i+1) % n]
        num = np.dot(b, np.cross(a, c))
        den = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(a, c)
        total += 2.0 * np.arctan2(num, den)
    return abs(total)

def find_loops(traj):
    T = len(traj)
    omegas, closures, lengths = [], [], []
    used = set()
    for i in range(T):
        if np.any(np.isnan(traj[i])):
            continue
        oct_i = tuple(np.sign(traj[i]).astype(int))
        for j in range(i + 30, min(i + 121, T)):
            if j in used or np.any(np.isnan(traj[j])):
                continue
            oct_j = tuple(np.sign(traj[j]).astype(int))
            if oct_i == oct_j:
                loop = traj[i:j+1]
                loop = loop[~np.any(np.isnan(loop), axis=1)]
                if len(loop) < 3:
                    continue
                om = solid_angle_loop(loop)
                cl = float(np.arccos(np.clip(np.dot(traj[i], traj[j]), -1, 1)))
                omegas.append(om)
                closures.append(cl)
                lengths.append(j - i)
                used.add(j)
                break
    return np.array(omegas), np.array(closures), np.array(lengths)

print("Recomputing holonomy loops for scatter...")
all_om, all_cl, all_ln, all_mv = [], [], [], []
for movie in movies:
    traj = make_s2_traj(movie)
    om, cl, ln = find_loops(traj)
    all_om.extend(om)
    all_cl.extend(cl)
    all_ln.extend(ln)
    all_mv.extend([movie] * len(om))
all_om = np.array(all_om)
all_cl = np.array(all_cl)
all_ln = np.array(all_ln)
print(f"  {len(all_om)} loops")

# --- S2 vs MDS per-film correlations ------------------------------------------
rho_per_film = {}
with open(RES / "s2_vs_mds_correlations.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["film"].strip():
            rho_per_film[row["film"]] = float(row["spearman_rho"])

film_names = list(rho_per_film.keys())
rho_vals   = list(rho_per_film.values())

# --- polar transitions --------------------------------------------------------
with open(RES / "polar_transition_analysis.json") as f:
    polar = json.load(f)

# --- ellipsoidal comparison ---------------------------------------------------
with open(RES / "ellipsoidal_comparison.csv") as f:
    reader = csv.DictReader(f)
    ell_rows = list(reader)

# Build polar ratio table: V, A, D
axis_ratios = {}
with open(RES / "axis_polar_ratios.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row["type"]   # e.g. "V-polar", "A-polar", "D-polar", "non-polar"
        axis_ratios[label] = {
            "mean_ratio": float(row["mean_ratio"]),
            "p": row["p_vs_nonpolar"] or "n/a"
        }

# --- model comparison ---------------------------------------------------------
with open(RES / "model_comparison.json") as f:
    mc = json.load(f)

model_names = list(mc.keys())
aic_vals = [mc[m]["AIC"] for m in model_names]
bic_vals = [mc[m]["BIC"] for m in model_names]

# --- holonomy heatmap (load from diagnostics) --------------------------------
# We'll rebuild a small grid for illustration using stored data
# The actual grid is 18x18 — we'll try to load from a saved numpy array or skip
# Since we didn't save the raw grid, we'll use the stored partial rho values
# and annotate the heatmap manually

# Load full grid from scratch (fast vectorized version)
def vad_s2_ell(lam):
    scaled = pts * np.array(lam)
    norms  = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.clip(norms, 1e-10, None)

def partial_rho_fast(x, y, z):
    from scipy import stats
    def rank_residual(a, b):
        ra = stats.rankdata(a).astype(float)
        rb = stats.rankdata(b).astype(float)
        rb -= rb.mean()
        beta = np.dot(ra - ra.mean(), rb) / (np.dot(rb, rb) + 1e-20)
        return ra - ra.mean() - beta * rb
    rx = rank_residual(x, z)
    ry = rank_residual(y, z)
    if rx.std() < 1e-10 or ry.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])

def holonomy_rho_for_lam(lv, ld):
    vs2 = vad_s2_ell([lv, 1.0, ld])
    oms, cls, lns = [], [], []
    for movie in movies:
        emo_data = beh_raw[movie]
        emo_list = sorted(emo_data.keys())
        T = len(next(iter(emo_data.values())))
        mat = np.zeros((T, len(emo_list)))
        for j, e in enumerate(emo_list):
            vals = np.array(emo_data[e], dtype=float)
            mat[:, j] = np.where(np.isnan(vals), 0.0, np.clip(vals, 0, None))
        col_sums = mat.sum(axis=0)
        valid = col_sums > 1e-10
        w = np.zeros_like(mat)
        w[:, valid] = mat[:, valid] / col_sums[None, valid]
        weighted = w @ vs2
        norms = np.linalg.norm(weighted, axis=1)
        traj = np.full(weighted.shape, np.nan)
        ok = norms > 1e-10
        traj[ok] = weighted[ok] / norms[ok, None]
        T2 = len(traj)
        used = set()
        for i in range(T2):
            if np.any(np.isnan(traj[i])):
                continue
            oct_i = tuple(np.sign(traj[i]).astype(int))
            for jj in range(i + 30, min(i + 121, T2)):
                if jj in used or np.any(np.isnan(traj[jj])):
                    continue
                oct_j = tuple(np.sign(traj[jj]).astype(int))
                if oct_i == oct_j:
                    loop = traj[i:jj+1]
                    loop = loop[~np.any(np.isnan(loop), axis=1)]
                    if len(loop) < 3:
                        continue
                    om = solid_angle_loop(loop)
                    cl = float(np.arccos(np.clip(np.dot(traj[i], traj[jj]), -1, 1)))
                    oms.append(om); cls.append(cl); lns.append(jj - i)
                    used.add(jj)
                    break
    if len(oms) < 10:
        return np.nan
    return partial_rho_fast(np.array(oms), np.array(cls), np.array(lns))

# Use a coarser grid for the figure (8×8) to keep runtime manageable
print("Computing holonomy heatmap (coarse 8x8 grid)...")
lv_g = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0])
ld_g = np.array([0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.3])
hol_grid = np.full((len(lv_g), len(ld_g)), np.nan)
for i, lv in enumerate(lv_g):
    for j, ld in enumerate(ld_g):
        hol_grid[i, j] = holonomy_rho_for_lam(lv, ld)
    print(f"  lv={lv:.1f} done")

# behavioral grid (just a placeholder interpolation from known points)
# We'll annotate positions on the holonomy heatmap
print("Building behavioral similarity heatmap (coarse)...")
from scipy import stats as sp_stats

def beh_rho_for_lam(lv, ld):
    vs2 = vad_s2_ell([lv, 1.0, ld])
    # compute geodesic matrix
    n = len(vs2)
    geo = np.zeros((n, n))
    for ii in range(n):
        for jj in range(ii+1, n):
            d = float(np.arccos(np.clip(np.dot(vs2[ii], vs2[jj]), -1, 1)))
            geo[ii, jj] = geo[jj, ii] = d
    # load behavioral dissimilarity (precomputed for round S2; reuse same)
    # We use the precomputed upper-triangle from the global beh_dis
    return geo

# Load behavioral dissimilarity from geometry comparison results
# Use the s2_vs_mds file for round S2 rho, and optimal_lambda for ellipsoid rho
# For the figure we'll show just the holonomy heatmap and annotate both optima

# ==============================================================================
# Figure
# ==============================================================================
print("Drawing figure...")
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                       left=0.07, right=0.97, top=0.88, bottom=0.08)

ax_A = fig.add_subplot(gs[0, 0], projection="3d")
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[0, 2])
ax_D = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[1, 1])
ax_F = fig.add_subplot(gs[1, 2])

# ── PANEL A: 13 emotions on S2 ─────────────────────────────────────────────────
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax_A.plot_wireframe(xs, ys, zs, color="lightgrey", alpha=0.3, linewidth=0.4, rstride=2, cstride=2)

valence = pts[:, 0]
norm_v  = Normalize(vmin=-1, vmax=1)
cmap_v  = plt.cm.RdBu

# colour by V_s2
for i, e in enumerate(emotions):
    c = cmap_v(norm_v(valence[i]))
    ax_A.scatter(*pts[i], color=c, s=60, zorder=5, depthshade=False)
    ax_A.text(pts[i, 0]*1.12, pts[i, 1]*1.12, pts[i, 2]*1.12,
              e[:6], fontsize=6.5, ha="center", va="center")

ax_A.set_title("A. Emotion geometry on S²\n(VAD projection, colour=valence)",
               fontsize=9, fontweight="bold")
ax_A.set_xlabel("V", fontsize=8); ax_A.set_ylabel("A", fontsize=8); ax_A.set_zlabel("D", fontsize=8)
ax_A.tick_params(labelsize=6)
sm = ScalarMappable(norm=norm_v, cmap=cmap_v)
sm.set_array([])
fig.colorbar(sm, ax=ax_A, pad=0.02, shrink=0.6, label="Valence", orientation="vertical")

# ── PANEL B: Holonomy scatter Omega vs closure ──────────────────────────────────
# subsample for clarity
np.random.seed(42)
idx_s = np.random.choice(len(all_om), size=min(200, len(all_om)), replace=False)
om_sub = all_om[idx_s]
cl_sub = all_cl[idx_s]
ln_sub = all_ln[idx_s]

sc = ax_B.scatter(np.log1p(om_sub * 100), cl_sub,
                  c=ln_sub, cmap="viridis_r", s=20, alpha=0.6, linewidths=0)
ax_B.set_xlabel("log(1 + 100*solid angle)", fontsize=9)
ax_B.set_ylabel("Closure error (rad)", fontsize=9)
ax_B.set_title("B. Holonomy signature\npartial rho=0.382 (p=1.4e-10)",
               fontsize=9, fontweight="bold")
fig.colorbar(sc, ax=ax_B, label="Loop length (tp)", shrink=0.7, pad=0.02)
ax_B.annotate("partial rho=0.382\np=1.4e-10", xy=(0.05, 0.93),
              xycoords="axes fraction", fontsize=8,
              bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"))

# ── PANEL C: per-film rho violin ───────────────────────────────────────────────
ax_C.violinplot([rho_vals], positions=[0], widths=0.6,
                showmedians=True, showextrema=True)
jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(rho_vals))
cmap_f = FILM_CMAP
for i, (r, j) in enumerate(zip(rho_vals, jitter)):
    ax_C.scatter(j, r, color=cmap_f(i / len(rho_vals)), s=40, zorder=5, alpha=0.85)
    ax_C.annotate(film_names[i][:8], (j, r), fontsize=5.5, ha="left",
                  xytext=(3, 0), textcoords="offset points", color="dimgrey")
ax_C.axhline(np.median(rho_vals), color=C_RED, lw=1.5, ls="--", label=f"median={np.median(rho_vals):.3f}")
ax_C.set_xticks([0]); ax_C.set_xticklabels(["S2 vs MDS step-rho"])
ax_C.set_ylabel("Spearman rho", fontsize=9)
ax_C.set_title("C. S2 trajectory vs MDS\nper-film correlation",
               fontsize=9, fontweight="bold")
ax_C.legend(fontsize=8)

# ── PANEL D: Polar ratio bars (V, A, D, non-polar) ─────────────────────────────
bar_labels  = ["D-polar", "A-polar", "V-polar", "non-polar"]
bar_vals    = [axis_ratios[k]["mean_ratio"] for k in bar_labels]
bar_colors  = [C_BLUE, C_GREEN, C_ORANGE, C_GREY]
bars = ax_D.bar(bar_labels, bar_vals, color=bar_colors, width=0.55, edgecolor="white", linewidth=0.5)
ax_D.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
for b, v in zip(bars, bar_vals):
    ax_D.text(b.get_x() + b.get_width()/2, v + 0.04,
              f"{v:.2f}x", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax_D.set_ylabel("Median ratio (ellipsoid / MDS)", fontsize=9)
ax_D.set_title("D. Polar transition asymmetry\n(ellipsoid geodesic / MDS Euclidean)",
               fontsize=9, fontweight="bold")
ax_D.set_ylim(0, max(bar_vals) * 1.25)
ax_D.tick_params(axis="x", labelsize=8)

# ── PANEL E: Holonomy lambda heatmap ─────────────────────────────────────────────
im = ax_E.imshow(hol_grid.T, origin="lower", aspect="auto",
                 extent=[lv_g[0]-0.05, lv_g[-1]+0.05, ld_g[0]-0.05, ld_g[-1]+0.05],
                 cmap="RdYlGn", vmin=0.2, vmax=0.45, interpolation="bilinear")
fig.colorbar(im, ax=ax_E, label="partial rho (holonomy)", shrink=0.8, pad=0.02)
# mark optima
ax_E.plot(1.3, 1.4, "k*", ms=14, label="holonomy opt (1.3,1.4)")
ax_E.plot(0.6, 2.3, "ws", ms=10, markeredgecolor="k", label="similarity opt (0.6,2.3)")
ax_E.set_xlabel("lambda_V", fontsize=9)
ax_E.set_ylabel("lambda_D", fontsize=9)
ax_E.set_title("E. Holonomy partial-rho heatmap\n(* holonomy opt | sq similarity opt)",
               fontsize=9, fontweight="bold")
ax_E.legend(fontsize=7, loc="upper right")

# ── PANEL F: AIC/BIC grouped bars ──────────────────────────────────────────────
x  = np.arange(len(model_names))
w  = 0.35
short_names = ["M0\n(null)", "M1\n(round S2)", "M2\n(ellipsoid)", "M3\nell+pot"]
b1 = ax_F.bar(x - w/2, aic_vals, w, color=C_BLUE,   label="AIC", alpha=0.85)
b2 = ax_F.bar(x + w/2, bic_vals, w, color=C_ORANGE, label="BIC", alpha=0.85)
ax_F.set_xticks(x); ax_F.set_xticklabels(short_names, fontsize=8)
ax_F.set_ylabel("AIC / BIC (lower = better)", fontsize=9)
ax_F.set_title("F. AIC/BIC model comparison\nM3 wins: ΔAIC=−8.6 vs M1",
               fontsize=9, fontweight="bold")
ax_F.legend(fontsize=9)
# annotate M3
ax_F.annotate("Best", xy=(3, min(aic_vals[3], bic_vals[3])),
              xytext=(2.4, min(aic_vals[3], bic_vals[3]) - 6),
              fontsize=8, color="darkgreen", fontweight="bold",
              arrowprops=dict(arrowstyle="->", color="darkgreen"))

# ── suptitle & footnote ────────────────────────────────────────────────────────
fig.suptitle(
    "ASBH Computational Evidence: Spherical Geometry with\n"
    "Non-Levi-Civita Connection in Affective Space",
    fontsize=13, fontweight="bold", y=0.965)

fig.text(0.5, 0.005,
         "Data: Emo-FilM (Ma & Kragel, 2026, Nat Commun).  "
         "VAD: Warriner et al. (2013).  N=29 fMRI participants.",
         ha="center", fontsize=8, color="dimgrey", style="italic")

out_path = RES / "ASBH_summary_figure_v2.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close(fig)
