"""
ASBH Test 08 - Connection diagnostics
======================================
Tests whether one metric governs both behavioural similarity and holonomy,
or whether the two structures require different lambda parameters
(evidence for a 'twisted connection' beyond Levi-Civita).

Parts:
  A: Grid search lambda -> holonomy partial-rho (heatmap)
  B: 2D axis-pair projections vs authors MDS
  C: Residual analysis of optimal ellipsoid
  D: AIC/BIC model comparison (4 models)
"""

import os, json, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from numpy.linalg import lstsq
warnings.filterwarnings("ignore")

# stdout UTF-8 safe
sys.stdout = __import__("io").TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace"
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "asbh_tests", "results")
DATA    = os.path.join(ROOT, "EmotionConceptRepresentation", "data")

BEH_JSON  = os.path.join(DATA, "behTab_json.json")
MDS_JSON  = os.path.join(DATA, "mds_2d.json")
VAD_CSV   = os.path.join(RESULTS, "emotion_vad.csv")
OPT_JSON  = os.path.join(RESULTS, "optimal_lambda.json")

EXCLUDE  = {"DamagedKungFu", "RidingTheRails", "LeassonLearned"}
EMOTIONS = ["Anger","Anxiety","Fear","Surprise","Guilt","Disgust","Sad",
            "Regard","Satisfaction","WarmHeartedness","Happiness","Pride","Love"]
N_EMO    = 13
TRIU     = np.triu_indices(N_EMO, k=1)
LOOP_MIN, LOOP_MAX = 30, 120

# ── load ──────────────────────────────────────────────────────────────────────
vad_df   = pd.read_csv(VAD_CSV).set_index("emotion")
VAD_c    = vad_df.loc[EMOTIONS, ["V_c","A_c","D_c"]].values.astype(float)
VAD_S2   = vad_df.loc[EMOTIONS, ["V_s2","A_s2","D_s2"]].values.astype(float)

with open(BEH_JSON) as f:  beh = json.load(f)
with open(MDS_JSON) as f:  mds = json.load(f)
with open(OPT_JSON) as f:  opt = json.load(f)
movies = [m for m in mds if m not in EXCLUDE]

LAM_OPT_BEH = [opt["lambda_V"], opt["lambda_A"], opt["lambda_D"]]
print(f"Behavioural-optimal lambda: V={LAM_OPT_BEH[0]}, D={LAM_OPT_BEH[2]}")

# ── precompute ratings (clip negative → 0) ────────────────────────────────────
ratings_all = {}
for movie in movies:
    r = np.array([beh[movie][e] for e in EMOTIONS], dtype=float)
    ratings_all[movie] = np.nan_to_num(np.clip(r, 0, None))  # (13, T)

# ── helpers ───────────────────────────────────────────────────────────────────

def vad_s2_ell(lam):
    """Emotion unit vectors for given lambda = [lV, lA, lD]."""
    sc = VAD_c * np.asarray(lam)
    return sc / np.linalg.norm(sc, axis=1, keepdims=True)


def make_traj(movie, vad_s2):
    """Vectorised S2 trajectory for given emotion unit vectors."""
    r = ratings_all[movie]               # (13, T)
    col_sums = r.sum(axis=0)             # (T,)
    valid = col_sums > 1e-10
    w = np.zeros_like(r)
    w[:, valid] = r[:, valid] / col_sums[None, valid]
    weighted = w.T @ vad_s2              # (T, 3)
    norms = np.linalg.norm(weighted, axis=1)
    traj = np.full(weighted.shape, np.nan)
    v2 = valid & (norms > 1e-10)
    traj[v2] = weighted[v2] / norms[v2, None]
    return traj                           # (T, 3)


def octant(v):
    return (int(v[0] >= 0) << 2) | (int(v[1] >= 0) << 1) | int(v[2] >= 0)


def solid_angle(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    om = 0.0
    for i in range(n):
        pm, pc, pp = pts[(i-1) % n], pts[i], pts[(i+1) % n]
        num = np.dot(pc, np.cross(pm, pp))
        den = 1.0 + np.dot(pm, pc) + np.dot(pc, pp) + np.dot(pm, pp)
        if abs(den) < 1e-10:
            continue
        om += 2.0 * np.arctan2(num, den)
    return abs(om)


def find_loops_and_stats(traj):
    """Return (omega_arr, closure_arr, length_arr)."""
    T = len(traj)
    valid = ~np.isnan(traj[:, 0])
    omegas, closures, lengths = [], [], []
    t = 0
    while t < T - LOOP_MIN:
        if not valid[t]:
            t += 1
            continue
        oct_s = octant(traj[t])
        found = False
        for te in range(t + LOOP_MIN, min(t + LOOP_MAX, T - 1) + 1):
            if valid[te] and octant(traj[te]) == oct_s:
                pts     = traj[t:te + 1]
                om      = solid_angle(pts)
                cl      = float(np.arccos(np.clip(np.dot(pts[0], pts[-1]), -1, 1)))
                ln      = sum(
                    float(np.arccos(np.clip(np.dot(pts[i], pts[i+1]), -1, 1)))
                    for i in range(len(pts) - 1)
                )
                omegas.append(om)
                closures.append(cl)
                lengths.append(ln)
                t = te + 1
                found = True
                break
        if not found:
            t += 1
    return np.array(omegas), np.array(closures), np.array(lengths)


def partial_rho(x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    A = np.column_stack([rz, np.ones(len(rz))])
    rx_r = rx - A @ lstsq(A, rx, rcond=None)[0]
    ry_r = ry - A @ lstsq(A, ry, rcond=None)[0]
    r, p  = stats.pearsonr(rx_r, ry_r)
    return float(r), float(p)


def holonomy_partial_rho_for_lambda(lv, ld):
    """Full pipeline: lambda -> partial rho (holonomy)."""
    vs2 = vad_s2_ell([lv, 1.0, ld])
    oms, cls, lns = [], [], []
    for movie in movies:
        traj = make_traj(movie, vs2)
        om, cl, ln = find_loops_and_stats(traj)
        oms.extend(om.tolist()); cls.extend(cl.tolist()); lns.extend(ln.tolist())
    om_arr, cl_arr, ln_arr = np.array(oms), np.array(cls), np.array(lns)
    if len(om_arr) < 10:
        return np.nan
    rho, _ = partial_rho(om_arr, cl_arr, ln_arr)
    return rho


# ── behavioral dissimilarity matrix ──────────────────────────────────────────
print("Computing behavioural dissimilarity matrix ...")
corr_mats = []
for movie in movies:
    ts_df = pd.DataFrame({e: beh[movie][e] for e in EMOTIONS}, dtype=float)
    cm    = ts_df.corr(method="pearson", min_periods=10).values.copy()
    np.fill_diagonal(cm, 0.0)
    corr_mats.append(np.nan_to_num(cm, nan=0.0))
avg_corr  = np.mean(corr_mats, axis=0)
beh_dis   = (1.0 - avg_corr)[TRIU]

# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Holonomy lambda grid search
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART A: Holonomy grid search (lambda_V x lambda_D)")
print("="*60)

lv_vals = np.arange(0.3, 2.01, 0.1)
ld_vals = np.arange(0.3, 2.01, 0.1)
grid_holo = np.full((len(lv_vals), len(ld_vals)), np.nan)

t0 = time.time()
for i, lv in enumerate(lv_vals):
    for j, ld in enumerate(ld_vals):
        grid_holo[i, j] = holonomy_partial_rho_for_lambda(lv, ld)
    print(f"  lv={lv:.1f} done ({time.time()-t0:.1f}s)", flush=True)

# Find holonomy optimum
best_idx  = np.unravel_index(np.nanargmax(grid_holo), grid_holo.shape)
best_lv_h = float(lv_vals[best_idx[0]])
best_ld_h = float(ld_vals[best_idx[1]])
best_rho_h = float(grid_holo[best_idx])

# Round S2 value (lambda=1,1,1)
idx_rnd_v = int(np.argmin(np.abs(lv_vals - 1.0)))
idx_rnd_d = int(np.argmin(np.abs(ld_vals - 1.0)))
rho_rnd_h = float(grid_holo[idx_rnd_v, idx_rnd_d])

same_lambda = (abs(best_lv_h - LAM_OPT_BEH[0]) < 0.15 and
               abs(best_ld_h - LAM_OPT_BEH[2]) < 0.15)

print(f"\nHolonomy optimum: lV={best_lv_h:.1f}, lD={best_ld_h:.1f}, "
      f"partial-rho={best_rho_h:.4f}")
print(f"Round S2 holonomy: partial-rho={rho_rnd_h:.4f}")
print(f"Behavioural optimum: lV={LAM_OPT_BEH[0]}, lD={LAM_OPT_BEH[2]}")
print(f"Same lambda for both? {same_lambda}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART B: 2D axis-pair projections vs MDS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART B: 2D projections vs authors MDS")
print("="*60)

# Use round S2 trajectories for projection analysis
vs2_rnd = vad_s2_ell([1.0, 1.0, 1.0])

PROJECTIONS = {
    "V-A": (0, 1),
    "V-D": (0, 2),
    "A-D": (1, 2),
}

proj_records = []
# per-projection step-distance arrays across all films
proj_step_dists = {k: [] for k in PROJECTIONS}
mds_step_dists  = []

for movie in movies:
    traj   = make_traj(movie, vs2_rnd)
    m1     = np.array(mds[movie]["mds1"])
    m2     = np.array(mds[movie]["mds2"])
    T      = min(len(traj), len(m1))
    traj   = traj[:T]

    for t in range(T - 1):
        a, b = traj[t], traj[t+1]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        mds_d = np.hypot(m1[t+1] - m1[t], m2[t+1] - m2[t])
        mds_step_dists.append(mds_d)
        for k, (i0, i1) in PROJECTIONS.items():
            d2 = np.hypot(b[i0] - a[i0], b[i1] - a[i1])
            proj_step_dists[k].append(d2)

mds_arr = np.array(mds_step_dists)
print(f"\n{'Projection':8s} | {'rho vs MDS':10s} | {'p':8s} | {'var_explained':13s}")
print("-"*50)

proj_corrs = {}
for k, sd in proj_step_dists.items():
    sd_arr = np.array(sd)
    rho, pval = stats.spearmanr(sd_arr, mds_arr)
    # variance explained: R^2 of linear fit
    slope, intercept, r_val, _, _ = stats.linregress(sd_arr, mds_arr)
    r2 = r_val ** 2
    print(f"  {k:6s}   | {rho:10.4f} | {pval:.3e} | {r2:.4f}")
    proj_corrs[k] = {"rho": round(float(rho), 4), "p": round(float(pval), 6),
                     "var_explained": round(r2, 4)}
    proj_records.append({"projection": k, "spearman_rho": round(float(rho),4),
                         "p_value": round(float(pval),6), "var_explained": round(r2,4)})

best_proj = max(proj_corrs, key=lambda k: proj_corrs[k]["rho"])
print(f"\nBest projection for MDS: {best_proj} (rho={proj_corrs[best_proj]['rho']:.4f})")

pd.DataFrame(proj_records).to_csv(
    os.path.join(RESULTS, "projection_comparison.csv"), index=False)
print(f"Saved: projection_comparison.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PART C: Residual analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART C: Residual analysis (optimal ellipsoid)")
print("="*60)

# Optimal ellipsoidal distance matrix
sc_opt = VAD_c * np.array(LAM_OPT_BEH)
sc_opt /= np.linalg.norm(sc_opt, axis=1, keepdims=True)
cos_opt = np.clip(sc_opt @ sc_opt.T, -1.0, 1.0)
dm_opt  = np.arccos(cos_opt)
pred_raw = dm_opt[TRIU]

# Fit linear regression: beh_dis ~ a*pred + b
slope_c, intercept_c, _, _, _ = stats.linregress(pred_raw, beh_dis)
pred_fit = slope_c * pred_raw + intercept_c
residuals = beh_dis - pred_fit

# Build residual dataframe
res_rows = []
triu_i, triu_j = TRIU
for k, (i, j) in enumerate(zip(triu_i, triu_j)):
    res_rows.append({
        "emotion_i":           EMOTIONS[i],
        "emotion_j":           EMOTIONS[j],
        "behavioral_dissim":   round(float(beh_dis[k]),   5),
        "predicted_ell":       round(float(pred_fit[k]),   5),
        "residual":            round(float(residuals[k]),  5),
    })
res_df = pd.DataFrame(res_rows).sort_values("residual", key=np.abs, ascending=False)
res_df.to_csv(os.path.join(RESULTS, "residual_analysis.csv"), index=False)

print("\nTop 10 largest |residual| pairs:")
print(f"{'Emotion i':22s} | {'Emotion j':22s} | {'beh_dis':8s} | {'pred':7s} | {'resid':7s}")
print("-"*80)
for _, row in res_df.head(10).iterrows():
    print(f"  {row['emotion_i']:20s} | {row['emotion_j']:20s} | "
          f"{row['behavioral_dissim']:8.4f} | {row['predicted_ell']:7.4f} | "
          f"{row['residual']:+7.4f}")

# Systematic pattern check
top10 = res_df.head(10)
pos_emo  = {"Satisfaction","WarmHeartedness","Happiness","Pride","Love","Regard","Surprise"}
neg_emo  = {"Anger","Anxiety","Fear","Guilt","Disgust","Sad"}
n_cross  = sum(1 for _, r in top10.iterrows()
               if (r["emotion_i"] in pos_emo) != (r["emotion_j"] in pos_emo))
print(f"\nCross-valence pairs in top-10 residuals: {n_cross}/10")
print(f"Saved: residual_analysis.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PART D: AIC / BIC model comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART D: AIC/BIC model comparison")
print("="*60)

n_pairs = len(beh_dis)

def model_rss_after_regression(predicted, actual):
    """RSS after fitting optimal linear rescaling."""
    s, ic, _, _, _ = stats.linregress(predicted, actual)
    return float(np.sum((actual - (s * predicted + ic)) ** 2))

# M0: null — predict mean
rss_M0 = float(np.sum((beh_dis - beh_dis.mean()) ** 2))

# M1: round S2 — 0 free geometry params
dm_rnd  = np.arccos(np.clip(VAD_S2 @ VAD_S2.T, -1, 1))
rss_M1  = model_rss_after_regression(dm_rnd[TRIU], beh_dis)

# M2: optimal ellipsoid — 2 free params (lV, lD)
rss_M2  = model_rss_after_regression(dm_opt[TRIU], beh_dis)

# M3: optimal ellipsoid + potential (beta=0.5) — 3 free params
BETA_OPT = 0.5
mu_raw   = np.array([1.0, -1.0, 1.0])
mu       = mu_raw / np.linalg.norm(mu_raw)
p_vals   = np.exp(BETA_OPT * (sc_opt @ mu))
wt_m3    = np.outer(p_vals, p_vals)
dm_M3    = dm_opt * np.sqrt(wt_m3)
rss_M3   = model_rss_after_regression(dm_M3[TRIU], beh_dis)

models = {
    "M0 (null)":        {"k": 0, "rss": rss_M0, "desc": "constant prediction"},
    "M1 (round S2)":    {"k": 0, "rss": rss_M1, "desc": "lambda=1,1,1"},
    "M2 (ellipsoid)":   {"k": 2, "rss": rss_M2, "desc": f"lV={LAM_OPT_BEH[0]},lD={LAM_OPT_BEH[2]}"},
    "M3 (ell+pot)":     {"k": 3, "rss": rss_M3, "desc": f"above + beta={BETA_OPT}"},
}

print(f"\n{'Model':20s} | {'k':2s} | {'RSS':8s} | {'AIC':8s} | {'BIC':8s} | "
      f"{'ΔAIC':8s} | {'ΔBIC':8s}")
print("-"*80)

aic_M1 = n_pairs * np.log(rss_M1 / n_pairs) + 2 * 0
bic_M1 = n_pairs * np.log(rss_M1 / n_pairs) + 0 * np.log(n_pairs)
model_results = {}

for name, m in models.items():
    aic = n_pairs * np.log(m["rss"] / n_pairs) + 2 * m["k"]
    bic = n_pairs * np.log(m["rss"] / n_pairs) + m["k"] * np.log(n_pairs)
    d_aic = aic - aic_M1
    d_bic = bic - bic_M1
    print(f"  {name:18s} | {m['k']:2d} | {m['rss']:8.5f} | {aic:8.3f} | {bic:8.3f} | "
          f"{d_aic:+8.3f} | {d_bic:+8.3f}")
    model_results[name] = {
        "k": m["k"], "rss": round(m["rss"],5), "desc": m["desc"],
        "AIC": round(aic,3), "BIC": round(bic,3),
        "delta_AIC": round(d_aic,3), "delta_BIC": round(d_bic,3),
    }

best_aic_name = min(model_results, key=lambda x: model_results[x]["AIC"])
best_bic_name = min(model_results, key=lambda x: model_results[x]["BIC"])
print(f"\nBest AIC: {best_aic_name}")
print(f"Best BIC: {best_bic_name}")

with open(os.path.join(RESULTS, "model_comparison.json"), "w") as f:
    json.dump(model_results, f, indent=2)
print("Saved: model_comparison.json")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding figure ...")

fig = plt.figure(figsize=(15, 11))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── Panel A: Holonomy heatmap ─────────────────────────────────────────────────
ax_A = fig.add_subplot(gs[0, 0])

vmin_h = np.nanpercentile(grid_holo, 5)
vmax_h = np.nanpercentile(grid_holo, 95)
im_A = ax_A.imshow(
    grid_holo.T, origin="lower", aspect="auto", cmap="RdYlGn",
    vmin=vmin_h, vmax=vmax_h,
    extent=[lv_vals[0]-0.05, lv_vals[-1]+0.05,
            ld_vals[0]-0.05, ld_vals[-1]+0.05]
)
plt.colorbar(im_A, ax=ax_A, label="Holonomy partial rho", shrink=0.85)

ax_A.plot(best_lv_h, best_ld_h, "w*", ms=16, zorder=6,
          label=f"Holonomy opt ({best_lv_h:.1f},{best_ld_h:.1f})")
ax_A.plot(LAM_OPT_BEH[0], LAM_OPT_BEH[2], "cs", ms=10, zorder=6,
          label=f"Beh.sim opt ({LAM_OPT_BEH[0]},{LAM_OPT_BEH[2]})")
ax_A.plot(1.0, 1.0, "wo", ms=8, zorder=5)
ax_A.annotate("Round S2", (1.0, 1.0), textcoords="offset points",
              xytext=(5, 4), fontsize=7, color="white")

ax_A.set_xlabel("lambda_V", fontsize=9)
ax_A.set_ylabel("lambda_D", fontsize=9)
ax_A.set_title("(A)  Holonomy partial-rho heatmap\n"
               f"opt ({best_lv_h:.1f},{best_ld_h:.1f}) vs beh.sim opt "
               f"({LAM_OPT_BEH[0]},{LAM_OPT_BEH[2]})",
               fontsize=9, fontweight="bold")
ax_A.legend(fontsize=7, loc="upper right")
# contour at round S2 value
try:
    ax_A.contour(lv_vals, ld_vals, grid_holo.T,
                 levels=[rho_rnd_h], colors="white", linestyles=":", linewidths=1)
except Exception:
    pass

# ── Panel B: 3 scatter plots (V-A, V-D, A-D vs MDS) ─────────────────────────
ax_B = fig.add_subplot(gs[0, 1])
ax_B.axis("off")
gs_B = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], wspace=0.35)

colors_B = {"V-A": "#4878CF", "V-D": "#D65F5F", "A-D": "#6ACC65"}
for pi, proj_k in enumerate(["V-A", "V-D", "A-D"]):
    ax_sub = fig.add_subplot(gs_B[0, pi])
    sd = np.array(proj_step_dists[proj_k])
    # subsample for speed
    idx_ss = np.random.default_rng(42).choice(len(sd), min(3000, len(sd)), replace=False)
    ax_sub.scatter(sd[idx_ss], mds_arr[idx_ss], s=3, alpha=0.25,
                   color=colors_B[proj_k])
    m_b, b_b = np.polyfit(sd, mds_arr, 1)
    xs_b = np.linspace(sd.min(), sd.max(), 50)
    ax_sub.plot(xs_b, m_b*xs_b+b_b, "k-", lw=1.2)
    r = proj_corrs[proj_k]["rho"]
    ax_sub.set_title(f"{proj_k}\nrho={r:.3f}", fontsize=8, fontweight="bold",
                     color=colors_B[proj_k])
    ax_sub.set_xlabel("2D proj step", fontsize=7)
    if pi == 0:
        ax_sub.set_ylabel("MDS step", fontsize=7)
    ax_sub.tick_params(labelsize=6)

fig.text(0.77, 0.955, "(B)  Axis projections vs MDS (step distances)",
         ha="center", fontsize=9, fontweight="bold")

# ── Panel C: Residuals bar chart ──────────────────────────────────────────────
ax_C = fig.add_subplot(gs[1, 0])
top_res = res_df.head(12)
pair_labels = [f"{r['emotion_i'][:6]}-{r['emotion_j'][:6]}"
               for _, r in top_res.iterrows()]
res_vals    = top_res["residual"].values
bar_c       = ["#EF5350" if v > 0 else "#42A5F5" for v in res_vals]
bars = ax_C.barh(range(len(pair_labels)), res_vals[::-1], color=bar_c[::-1],
                 edgecolor="white", height=0.65)
ax_C.set_yticks(range(len(pair_labels)))
ax_C.set_yticklabels(pair_labels[::-1], fontsize=7)
ax_C.axvline(0, color="black", lw=0.8)
ax_C.set_xlabel("Residual (actual - predicted)", fontsize=9)
ax_C.set_title(f"(C)  Top-12 residuals (optimal ellipsoid)\n"
               f"cross-valence in top-10: {n_cross}/10",
               fontsize=9, fontweight="bold")

# ── Panel D: AIC/BIC bar ──────────────────────────────────────────────────────
ax_D = fig.add_subplot(gs[1, 1])
m_names = list(model_results.keys())
d_aic_vals = [model_results[m]["delta_AIC"] for m in m_names]
d_bic_vals = [model_results[m]["delta_BIC"] for m in m_names]
x = np.arange(len(m_names))
w = 0.35
ax_D.bar(x - w/2, d_aic_vals, w, label="ΔAIC vs M1", color="#7B1FA2", alpha=0.8)
ax_D.bar(x + w/2, d_bic_vals, w, label="ΔBIC vs M1", color="#0288D1", alpha=0.8)
ax_D.axhline(0, color="black", lw=0.8, ls="--")
ax_D.set_xticks(x)
ax_D.set_xticklabels([m.split("(")[0].strip() for m in m_names], fontsize=8)
ax_D.set_ylabel("Delta AIC / BIC vs M1 (round S2)", fontsize=9)
ax_D.set_title("(D)  Model comparison\n(negative = better than round S2)",
               fontsize=9, fontweight="bold")
ax_D.legend(fontsize=8)
for xi, (da, db) in enumerate(zip(d_aic_vals, d_bic_vals)):
    ax_D.text(xi - w/2, da + (0.3 if da >= 0 else -0.8),
              f"{da:+.1f}", ha="center", fontsize=7, color="#7B1FA2")
    ax_D.text(xi + w/2, db + (0.3 if db >= 0 else -0.8),
              f"{db:+.1f}", ha="center", fontsize=7, color="#0288D1")

fig.suptitle("ASBH Test 08: Connection Diagnostics",
             fontsize=12, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig_path = os.path.join(RESULTS, "diagnostics_figure.png")
fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {fig_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════════

# Compute rho for behavioural similarity with optimal behavioural lambda
dm_beh_opt  = np.arccos(np.clip(sc_opt @ sc_opt.T, -1, 1))
rho_beh_opt = stats.spearmanr(dm_beh_opt[TRIU], beh_dis)[0]

metric_verdict = ("different" if not same_lambda else "same")
metric_implication = ("suggests separate metrics for similarity and dynamics"
                      if not same_lambda else "consistent with a single metric")

print("\n" + "="*60)
print("=== CONNECTION DIAGNOSTICS ===")
print("="*60)
print(f"Optimal lambda (behavioural similarity):  "
      f"V={LAM_OPT_BEH[0]}, D={LAM_OPT_BEH[2]} (rho={rho_beh_opt:.4f})")
print(f"Optimal lambda (holonomy):                "
      f"V={best_lv_h:.1f}, D={best_ld_h:.1f} (partial-rho={best_rho_h:.4f})")
print(f"Lambda difference: {metric_verdict} -- {metric_implication}")
print(f"Best 2D projection for MDS authors: "
      f"{best_proj} (rho={proj_corrs[best_proj]['rho']:.4f})")
print(f"Largest residual: "
      f"{res_df.iloc[0]['emotion_i']} <-> {res_df.iloc[0]['emotion_j']} "
      f"(delta={res_df.iloc[0]['residual']:+.4f})")
print(f"Best AIC model: {best_aic_name}")
print(f"Best BIC model: {best_bic_name}")
print(f"CONCLUSION: space has {metric_verdict} metric for similarity and dynamics")
print("="*60)
