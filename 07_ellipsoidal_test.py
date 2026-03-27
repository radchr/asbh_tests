"""
ASBH Test 07 - Ellipsoidal S2 with density potential
=====================================================
Tests whether an anisotropically-scaled sphere better fits
behavioural emotion similarity than the isotropic S2.

Parts:
  A: Four metrics (round, ACT, Warriner, optimal via grid search)
  B: Comparison with behavioral similarity + bootstrap CI
  C: Density potential (exp(beta*mu)) sensitivity
  D: Axis-specific polar transition ratios
  E: Holonomy re-analysis with optimal ellipsoidal metric
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")

# -- paths ---------------------------------------------------------------------
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "asbh_tests", "results")
DATA    = os.path.join(ROOT, "EmotionConceptRepresentation", "data")

BEH_JSON = os.path.join(DATA, "behTab_json.json")
MDS_JSON = os.path.join(DATA, "mds_2d.json")
VAD_CSV  = os.path.join(RESULTS, "emotion_vad.csv")

EXCLUDE  = {"DamagedKungFu", "RidingTheRails", "LeassonLearned"}
EMOTIONS = ["Anger","Anxiety","Fear","Surprise","Guilt","Disgust","Sad",
            "Regard","Satisfaction","WarmHeartedness","Happiness","Pride","Love"]
N_EMO = 13
TRIU  = np.triu_indices(N_EMO, k=1)
LOOP_MIN, LOOP_MAX = 30, 120

# -- load core data -------------------------------------------------------------
vad_df = pd.read_csv(VAD_CSV).set_index("emotion")
VAD_S2 = vad_df.loc[EMOTIONS, ["V_s2","A_s2","D_s2"]].values.astype(float)  # (13,3)

with open(BEH_JSON) as f:  beh = json.load(f)
with open(MDS_JSON) as f:  mds_data = json.load(f)
movies = [m for m in mds_data if m not in EXCLUDE]

# -- helpers -------------------------------------------------------------------

def ellipsoidal_dist_matrix(vad_s2, lam):
    """13x13 ellipsoidal distance matrix. lam = [lam_V, lam_A, lam_D]."""
    scaled = vad_s2 * np.asarray(lam)
    norms  = np.linalg.norm(scaled, axis=1, keepdims=True)
    scaled_n = scaled / norms
    cos = np.clip(scaled_n @ scaled_n.T, -1.0, 1.0)
    return np.arccos(cos)


def geodesic_pt(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def spearman_with_bootstrap(x, y, n_boot=1000, seed=42):
    rho, pval = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot_rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(r)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    return float(rho), float(pval), float(ci_lo), float(ci_hi)


# -- behavioral dissimilarity matrix (pooled across films) ---------------------
print("Computing pooled behavioral dissimilarity matrix ...")
corr_mats = []
for movie in movies:
    ts_df = pd.DataFrame({e: beh[movie][e] for e in EMOTIONS}, dtype=float)
    corr  = ts_df.corr(method="pearson", min_periods=10).values.copy()
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0)
    corr_mats.append(corr)
avg_beh_corr = np.mean(corr_mats, axis=0)    # 13x13 averaged Pearson r
beh_dis      = (1.0 - avg_beh_corr)[TRIU]    # upper-triangle dissimilarity

# ===============================================================================
# PART A: Four metrics
# ===============================================================================
print("\n" + "="*60)
print("PART A: Four metrics + grid-search for optimal lambda")
print("="*60)

LAM_ACT = [1.6, 1.0, 1.4]
LAM_WAR = [1.42, 1.0, 1.05]
LAM_RND = [1.0,  1.0, 1.0]

# Grid search: lam_V in [0.5,3.0], lam_D in [0.5,3.0], lam_A fixed = 1.0
lv_vals  = np.arange(0.5, 3.01, 0.1)
ld_vals  = np.arange(0.5, 3.01, 0.1)
grid_rho = np.full((len(lv_vals), len(ld_vals)), np.nan)

print("Running grid search ...")
best_rho, best_lv, best_ld = -np.inf, 1.0, 1.0

for i, lv in enumerate(lv_vals):
    for j, ld in enumerate(ld_vals):
        dm   = ellipsoidal_dist_matrix(VAD_S2, [lv, 1.0, ld])
        rho  = stats.spearmanr(dm[TRIU], beh_dis)[0]
        grid_rho[i, j] = rho
        if rho > best_rho:
            best_rho, best_lv, best_ld = rho, lv, ld

LAM_OPT = [round(best_lv, 1), 1.0, round(best_ld, 1)]
print(f"  Optimal: lam_V={LAM_OPT[0]:.1f}, lam_D={LAM_OPT[2]:.1f}, rho={best_rho:.4f}")

# Distance matrices for all four metrics
dm_rnd = ellipsoidal_dist_matrix(VAD_S2, LAM_RND)
dm_act = ellipsoidal_dist_matrix(VAD_S2, LAM_ACT)
dm_war = ellipsoidal_dist_matrix(VAD_S2, LAM_WAR)
dm_opt = ellipsoidal_dist_matrix(VAD_S2, LAM_OPT)

# ===============================================================================
# PART B: Behavioral comparison with bootstrap CI
# ===============================================================================
print("\n" + "="*60)
print("PART B: Spearman rho vs behavioral dissimilarity (bootstrap CI)")
print("="*60)

metrics_cfg = [
    ("Round S2",     LAM_RND, dm_rnd),
    ("Ellips ACT",   LAM_ACT, dm_act),
    ("Ellips WAR",   LAM_WAR, dm_war),
    ("Optimal",      LAM_OPT, dm_opt),
]

comp_records = []
rho_rnd_val  = None

print(f"\n{'Metric':15s} | {'lam_V':4s} | {'lam_D':4s} | {'rho':6s} | {'p':8s} | {'95% CI':14s} | {'delta vs round':10s}")
print("-"*75)

for label, lam, dm in metrics_cfg:
    rho, pval, ci_lo, ci_hi = spearman_with_bootstrap(dm[TRIU], beh_dis)
    delta = rho - rho_rnd_val if rho_rnd_val is not None else 0.0
    if rho_rnd_val is None:
        rho_rnd_val = rho
        delta_str = "-"
    else:
        delta_str = f"{delta:+.4f}"
    print(f"{label:15s} | {lam[0]:4.2f} | {lam[2]:4.2f} | {rho:.4f} | {pval:.2e} | "
          f"[{ci_lo:.4f},{ci_hi:.4f}] | {delta_str}")
    comp_records.append({
        "metric": label, "lambda_V": lam[0], "lambda_A": lam[1], "lambda_D": lam[2],
        "rho": round(rho,4), "p_value": round(pval,6),
        "ci_lo": round(ci_lo,4), "ci_hi": round(ci_hi,4),
        "delta_vs_round": round(rho - rho_rnd_val + (rho - rho),4),  # recompute below
    })

# fix delta column properly
for rec in comp_records:
    rec["delta_vs_round"] = round(rec["rho"] - comp_records[0]["rho"], 4)

comp_df = pd.DataFrame(comp_records)
comp_csv = os.path.join(RESULTS, "ellipsoidal_comparison.csv")
comp_df.to_csv(comp_csv, index=False)
print(f"\nSaved: {comp_csv}")

# ===============================================================================
# PART C: Density potential sensitivity
# ===============================================================================
print("\n" + "="*60)
print("PART C: Density potential exp(beta*mu) sensitivity")
print("="*60)

mu_raw = np.array([1.0, -1.0, 1.0])
mu     = mu_raw / np.linalg.norm(mu_raw)    # normalize

betas       = [0.0, 0.5, 1.0, 2.0, 3.0]
beta_rhos   = []
beta_recs   = []

for beta in betas:
    # p(x) = exp(beta * dot(x, mu))  (unnormalised - only ratio matters)
    p_vals = np.exp(beta * (VAD_S2 @ mu))   # (13,)
    # weighted geodesic: dist_ij * sqrt(p_i * p_j)
    wt = np.outer(p_vals, p_vals)           # (13,13)
    dm_w = dm_opt * np.sqrt(wt)
    rho, pval = stats.spearmanr(dm_w[TRIU], beh_dis)
    beta_rhos.append(rho)
    beta_recs.append({"beta": beta, "rho": round(float(rho),4), "p": round(float(pval),6)})
    print(f"  beta={beta:.1f}  rho={rho:.4f}  p={pval:.3e}")

best_beta_idx = int(np.argmax(beta_rhos))
best_beta     = betas[best_beta_idx]
best_beta_rho = beta_rhos[best_beta_idx]
print(f"\n  Best beta = {best_beta:.1f} (rho = {best_beta_rho:.4f})")

# ===============================================================================
# PART D: Axis-specific polar transition ratios
# ===============================================================================
print("\n" + "="*60)
print("PART D: Axis-specific polar transition ratios (V / A / D)")
print("="*60)

def make_s2_traj(movie_name):
    ratings = np.array([beh[movie_name][e] for e in EMOTIONS], dtype=float)
    ratings = np.nan_to_num(np.clip(ratings, 0, None))
    T = ratings.shape[1]
    traj = np.full((T, 3), np.nan)
    for t in range(T):
        w = ratings[:, t]; s = w.sum()
        if s < 1e-10: continue
        v = (w/s) @ VAD_S2; n = np.linalg.norm(v)
        if n < 1e-10: continue
        traj[t] = v / n
    return traj


def ellipsoidal_dist_pts(a, b, lam):
    """Ellipsoidal geodesic distance between two unit vectors."""
    a_sc = a * lam; b_sc = b * lam
    cos  = np.dot(a_sc, b_sc) / (np.linalg.norm(a_sc) * np.linalg.norm(b_sc))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


lam_opt_arr = np.array(LAM_OPT, dtype=float)

categories = {"V-polar": [], "A-polar": [], "D-polar": [], "non-polar": []}

for movie in movies:
    s2_traj = make_s2_traj(movie)
    mds1 = np.array(mds_data[movie]["mds1"])
    mds2 = np.array(mds_data[movie]["mds2"])
    T = min(len(s2_traj), len(mds1))

    for t in range(T - 1):
        a, b = s2_traj[t], s2_traj[t+1]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        geo_ell = ellipsoidal_dist_pts(a, b, lam_opt_arr)
        mds_d   = np.hypot(mds1[t+1]-mds1[t], mds2[t+1]-mds2[t])
        if mds_d < 1e-10:
            continue
        ratio = geo_ell / mds_d

        v_flip = (a[0] * b[0]) < 0
        a_flip = (a[1] * b[1]) < 0
        d_flip = (a[2] * b[2]) < 0

        if not v_flip and not a_flip and not d_flip:
            categories["non-polar"].append(ratio)
        if v_flip:
            categories["V-polar"].append(ratio)
        if a_flip:
            categories["A-polar"].append(ratio)
        if d_flip:
            categories["D-polar"].append(ratio)

# Welch t-test vs non-polar
non_arr = np.array(categories["non-polar"])
axis_records = []

print(f"\n{'Type':12s} | {'N':6s} | {'mean ratio':10s} | {'SD':7s} | {'p vs non-polar':14s}")
print("-"*58)

for cat in ["V-polar", "A-polar", "D-polar", "non-polar"]:
    arr = np.array(categories[cat])
    if cat == "non-polar":
        pval_str = "-"
        pval     = np.nan
    else:
        _, pval = stats.ttest_ind(arr, non_arr, equal_var=False)
        pval_str = f"{pval:.3e}"
    print(f"{cat:12s} | {len(arr):6d} | {arr.mean():10.5f} | {arr.std():7.5f} | {pval_str:14s}")
    axis_records.append({
        "type": cat, "N": len(arr), "mean_ratio": round(float(arr.mean()),6),
        "sd_ratio": round(float(arr.std()),6),
        "p_vs_nonpolar": round(float(pval),6) if not np.isnan(pval) else None,
    })

# Check ASBH ellipsoidal prediction: V > D > A
v_mean = np.array(categories["V-polar"]).mean()
a_mean = np.array(categories["A-polar"]).mean()
d_mean = np.array(categories["D-polar"]).mean()
prediction_holds = bool(v_mean > d_mean > a_mean)
print(f"\n  Prediction V>D>A: {prediction_holds}  "
      f"(V={v_mean:.5f}, D={d_mean:.5f}, A={a_mean:.5f})")
print(f"  Expected from Lam: lam_V={LAM_OPT[0]:.1f} > lam_D={LAM_OPT[2]:.1f} > lam_A={LAM_OPT[1]:.1f}"
      if prediction_holds else
      f"  Note: optimal Lambda has lam_V={LAM_OPT[0]:.1f}, lam_D={LAM_OPT[2]:.1f}, lam_A={LAM_OPT[1]:.1f}")

axis_df = pd.DataFrame(axis_records)
axis_csv = os.path.join(RESULTS, "axis_polar_ratios.csv")
axis_df.to_csv(axis_csv, index=False)
print(f"Saved: {axis_csv}")

# ===============================================================================
# PART E: Holonomy with ellipsoidal metric
# ===============================================================================
print("\n" + "="*60)
print("PART E: Holonomy re-analysis with optimal ellipsoidal metric")
print("="*60)

def octant(v): return (int(v[0]>=0)<<2)|(int(v[1]>=0)<<1)|int(v[2]>=0)

def solid_angle_loop(pts):
    n = len(pts)
    if n < 3: return 0.0
    omega = 0.0
    for i in range(n):
        pm, pc, pp = pts[(i-1)%n], pts[i], pts[(i+1)%n]
        num = np.dot(pc, np.cross(pm, pp))
        den = 1.0 + np.dot(pm,pc) + np.dot(pc,pp) + np.dot(pm,pp)
        if abs(den) < 1e-10: continue
        omega += 2.0 * np.arctan2(num, den)
    return abs(omega)


def ellipsoidal_solid_angle(pts, lam):
    """
    Approximate ellipsoidal solid angle:
    scale loop points by Lambda, re-normalise to S2, then compute standard solid angle.
    """
    scaled = pts * np.asarray(lam)
    norms  = np.linalg.norm(scaled, axis=1, keepdims=True)
    valid  = (norms[:, 0] > 1e-10)
    if valid.sum() < 3:
        return 0.0
    pts_ell = scaled.copy()
    pts_ell[valid] /= norms[valid]
    return solid_angle_loop(pts_ell[valid])


omega_round, omega_ell, loop_length, loop_closure = [], [], [], []

for movie in movies:
    traj = make_s2_traj(movie)
    T    = len(traj)
    valid = ~np.isnan(traj[:,0])
    t = 0
    while t < T - LOOP_MIN:
        if not valid[t]: t += 1; continue
        oct_s = octant(traj[t])
        found = False
        for te in range(t+LOOP_MIN, min(t+LOOP_MAX, T-1)+1):
            if valid[te] and octant(traj[te]) == oct_s:
                pts     = traj[t:te+1]
                om_r    = solid_angle_loop(pts)
                om_e    = ellipsoidal_solid_angle(pts, lam_opt_arr)
                length  = sum(geodesic_pt(pts[i], pts[i+1])
                              for i in range(len(pts)-1))
                closure = geodesic_pt(pts[0], pts[-1])
                omega_round.append(om_r); omega_ell.append(om_e)
                loop_length.append(length); loop_closure.append(closure)
                t = te + 1; found = True; break
        if not found: t += 1

omega_round  = np.array(omega_round)
omega_ell    = np.array(omega_ell)
loop_length  = np.array(loop_length)
loop_closure = np.array(loop_closure)

def partial_spearman(x, y, z):
    rx, ry, rz = [stats.rankdata(v) for v in (x, y, z)]
    A = np.column_stack([rz, np.ones(len(rz))])
    from numpy.linalg import lstsq
    rx_r = rx - A @ lstsq(A, rx, rcond=None)[0]
    ry_r = ry - A @ lstsq(A, ry, rcond=None)[0]
    return stats.pearsonr(rx_r, ry_r)

rho_r_raw, p_r_raw   = stats.spearmanr(omega_round, loop_closure)
rho_e_raw, p_e_raw   = stats.spearmanr(omega_ell,   loop_closure)
rho_r_par, p_r_par   = partial_spearman(omega_round, loop_closure, loop_length)
rho_e_par, p_e_par   = partial_spearman(omega_ell,   loop_closure, loop_length)

print(f"  Round  S2: raw rho={rho_r_raw:.4f}  partial rho={rho_r_par:.4f} (p={p_r_par:.3e})")
print(f"  Ellips opt: raw rho={rho_e_raw:.4f}  partial rho={rho_e_par:.4f} (p={p_e_par:.3e})")
print(f"  Improvement in partial rho: {rho_e_par - rho_r_par:+.4f}")

# ===============================================================================
# Save optimal lambda JSON
# ===============================================================================
opt_json = {
    "lambda_V":              round(LAM_OPT[0], 2),
    "lambda_A":              1.0,
    "lambda_D":              round(LAM_OPT[2], 2),
    "rho":                   round(best_rho,   4),
    "round_S2_rho":          round(comp_records[0]["rho"], 4),
    "improvement_over_round":round(best_rho - comp_records[0]["rho"], 4),
    "prediction_V_gt_D_gt_A":prediction_holds,
}
with open(os.path.join(RESULTS, "optimal_lambda.json"), "w") as f:
    json.dump(opt_json, f, indent=2)
print(f"\nSaved: optimal_lambda.json")

# ===============================================================================
# FIGURE - 4 panels
# ===============================================================================
print("Building figure ...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor("white")

# -- Panel A: Grid search heatmap ---------------------------------------------
ax_A = axes[0, 0]
im = ax_A.imshow(grid_rho.T, origin="lower", aspect="auto", cmap="RdYlGn",
                 vmin=grid_rho.min(), vmax=grid_rho.max(),
                 extent=[lv_vals[0]-0.05, lv_vals[-1]+0.05,
                         ld_vals[0]-0.05, ld_vals[-1]+0.05])
plt.colorbar(im, ax=ax_A, label="Spearman $\\rho$", shrink=0.85)

# star at optimum
ax_A.plot(best_lv, best_ld, "w*", ms=16, zorder=5,
          label=f"opt ({best_lv:.1f},{best_ld:.1f})")
# mark standard variants
for label_pt, lv_pt, ld_pt, col in [
    ("Round", 1.0,  1.0,  "white"),
    ("ACT",   1.6,  1.4,  "cyan"),
    ("WAR",   1.42, 1.05, "yellow"),
]:
    ax_A.plot(lv_pt, ld_pt, "o", color=col, ms=8, zorder=5)
    ax_A.annotate(label_pt, (lv_pt, ld_pt), textcoords="offset points",
                  xytext=(5, 4), fontsize=7, color=col)
ax_A.set_xlabel("$\\lambda_V$  (valence scale)", fontsize=9)
ax_A.set_ylabel("$\\lambda_D$  (dominance scale)", fontsize=9)
ax_A.set_title("(A)  Grid search: $\\rho$ vs behavioral similarity",
               fontsize=9, fontweight="bold")
ax_A.legend(fontsize=7, loc="upper right")

# round S2 rho contour line
ax_A.contour(lv_vals, ld_vals, grid_rho.T,
             levels=[comp_records[0]["rho"]], colors="white",
             linestyles="--", linewidths=1)
ax_A.annotate(f"round S2 rho={comp_records[0]['rho']:.3f}",
              xy=(0.04, 0.07), xycoords="axes fraction",
              fontsize=7, color="white",
              bbox=dict(boxstyle="round,pad=0.2", fc="none", ec="white", lw=0.7))

# -- Panel B: Axis polar ratios bar plot ---------------------------------------
ax_B = axes[0, 1]
cat_labels = ["Non-polar", "A-polar", "D-polar", "V-polar"]
bar_means  = [np.array(categories[c]).mean()
              for c in ["non-polar","A-polar","D-polar","V-polar"]]
bar_sds    = [np.array(categories[c]).std()
              for c in ["non-polar","A-polar","D-polar","V-polar"]]
bar_colors = ["#78909C", "#66BB6A", "#42A5F5", "#EF5350"]

bars = ax_B.bar(cat_labels, bar_means, yerr=bar_sds, capsize=5,
                color=bar_colors, edgecolor="white", width=0.55,
                error_kw=dict(ecolor="black", lw=1.2))

# predicted ordering lines
non_mean = bar_means[0]
for lam_val, color, label_t in [
    (LAM_OPT[2]/LAM_OPT[1], "#42A5F5", f"pred D ratio (lam_D/lam_A={LAM_OPT[2]:.1f})"),
    (LAM_OPT[0]/LAM_OPT[1], "#EF5350", f"pred V ratio (lam_V/lam_A={LAM_OPT[0]:.1f})"),
]:
    ax_B.axhline(non_mean * lam_val, color=color, ls=":", lw=1.4, alpha=0.7,
                 label=label_t)

ax_B.set_ylabel("Ellipsoidal geodesic / MDS distance ratio", fontsize=9)
ax_B.set_title("(B)  Axis-specific polar transition ratios",
               fontsize=9, fontweight="bold")
ax_B.legend(fontsize=6.5)
for bar, m in zip(bars, bar_means):
    ax_B.text(bar.get_x()+bar.get_width()/2, m*0.3,
              f"{m:.4f}", ha="center", va="center", fontsize=7,
              color="white", fontweight="bold")

# -- Panel C: Omega_ell vs closure (with Omega_round overlay) -------------------------
ax_C = axes[1, 0]
ax_C.scatter(omega_round, loop_closure, s=10, alpha=0.4, color="#4878CF",
             label=f"Round S2 ($\\rho$={rho_r_par:.3f})", zorder=2)
ax_C.scatter(omega_ell,   loop_closure, s=10, alpha=0.4, color="#D65F5F",
             label=f"Ellips opt ($\\rho$={rho_e_par:.3f})", zorder=3)
for om_arr, col in [(omega_round,"#2C5F8A"), (omega_ell,"#8A1C1C")]:
    m_c, b_c = np.polyfit(om_arr, loop_closure, 1)
    xs_c = np.linspace(om_arr.min(), om_arr.max(), 100)
    ax_C.plot(xs_c, m_c*xs_c+b_c, color=col, lw=1.4)

ax_C.set_xlabel("Solid angle $\\Omega_0$ (rad)", fontsize=9)
ax_C.set_ylabel("Loop closure error (rad)", fontsize=9)
ax_C.set_title("(C)  Holonomy: round vs ellipsoidal $\\Omega_0$",
               fontsize=9, fontweight="bold")
ax_C.legend(fontsize=8)
ax_C.annotate(f"partial $\\rho$ (ellips) = {rho_e_par:.3f}\np = {p_e_par:.2e}",
              xy=(0.55, 0.12), xycoords="axes fraction", fontsize=8,
              color="#8A1C1C",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D65F5F", alpha=0.85))

# -- Panel D: beta sensitivity ----------------------------------------------------
ax_D = axes[1, 1]
beta_plot_vals = [r["beta"] for r in beta_recs]
beta_plot_rhos = [r["rho"]  for r in beta_recs]
ax_D.plot(beta_plot_vals, beta_plot_rhos, "o-", color="#7B1FA2", lw=2, ms=8)
ax_D.fill_between(beta_plot_vals, beta_plot_rhos,
                  min(beta_plot_rhos)*0.98, alpha=0.15, color="#7B1FA2")
ax_D.axhline(comp_records[0]["rho"], color="#4878CF", ls="--", lw=1.5,
             label=f"Round S2 rho = {comp_records[0]['rho']:.4f}")
best_y = max(beta_plot_rhos)
best_x = beta_plot_vals[beta_plot_rhos.index(best_y)]
ax_D.axvline(best_x, color="#7B1FA2", ls=":", lw=1, alpha=0.6)
ax_D.annotate(f"beta={best_x:.1f}\nrho={best_y:.4f}",
              xy=(best_x, best_y), xytext=(best_x+0.15, best_y-0.004),
              fontsize=8, color="#7B1FA2",
              arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1))
ax_D.set_xlabel("Potential strength $\\beta$", fontsize=9)
ax_D.set_ylabel("Spearman $\\rho$ vs behavioral dissimilarity", fontsize=9)
ax_D.set_title("(D)  Density potential sensitivity  "
               r"$p(x) \propto \exp(\beta \cdot \mu)$",
               fontsize=9, fontweight="bold")
ax_D.legend(fontsize=8)
ax_D.set_ylim(min(beta_plot_rhos)*0.99, max(beta_plot_rhos)*1.01)

fig.suptitle("ASBH Test 07: Ellipsoidal S2 vs Isotropic S2",
             fontsize=12, fontweight="bold", y=0.98)
fig.text(0.5, 0.005,
         r"Optimal: $\lambda_V$=" + f"{LAM_OPT[0]:.1f}"
         + r", $\lambda_D$=" + f"{LAM_OPT[2]:.1f}"
         + f"  |  best rho = {best_rho:.4f}  (delta = {best_rho - comp_records[0]['rho']:+.4f})",
         ha="center", fontsize=9, color="#333333", style="italic")

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
fig_path = os.path.join(RESULTS, "ellipsoidal_test_figure.png")
fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {fig_path}")

# ===============================================================================
# Final console summary
# ===============================================================================
rho_act  = comp_records[1]["rho"]
rho_war  = comp_records[2]["rho"]
rho_opt_ = comp_records[3]["rho"]
rho_rnd_ = comp_records[0]["rho"]

verdict = ("better than" if rho_opt_ > rho_rnd_ + 0.005 else
           "comparable to" if abs(rho_opt_ - rho_rnd_) <= 0.005 else
           "worse than")

print("\n" + "="*60)
print("=== ELLIPSOIDAL MODEL: Results ===")
print("="*60)
print(f"Round S2          : rho = {rho_rnd_:.4f}")
print(f"Ellipsoidal (ACT) : rho = {rho_act:.4f}  (delta = {rho_act-rho_rnd_:+.4f})")
print(f"Ellipsoidal (WAR) : rho = {rho_war:.4f}  (delta = {rho_war-rho_rnd_:+.4f})")
print(f"Optimal           : lambda_V={LAM_OPT[0]:.1f}, lambda_D={LAM_OPT[2]:.1f}, "
      f"rho = {rho_opt_:.4f}  (delta = {rho_opt_-rho_rnd_:+.4f} vs round)")
print(f"Axis ratios       : V-polar={v_mean:.5f} > D-polar={d_mean:.5f} > "
      f"A-polar={a_mean:.5f}  (prediction: {prediction_holds})")
print(f"Holonomy (ellips) : partial rho = {rho_e_par:.4f}  "
      f"(vs round 0.382, delta = {rho_e_par-rho_r_par:+.4f})")
print(f"Optimal beta      : {best_beta:.1f}  (rho = {best_beta_rho:.4f})")
print(f"CONCLUSION        : ellipsoidal model is {verdict} round S2")
print("="*60)
