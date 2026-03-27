"""
ASBH Test 01 — VAD coordinates on S²
=====================================
Loads Warriner VAD norms, maps 13 emotion categories to words,
centers and normalises to unit sphere, then computes geodesic and
Euclidean distance matrices.

Outputs (all in asbh_tests/results/):
  emotion_vad.csv          — per-emotion VAD + S² coords
  geodesic_matrix.csv      — 13×13 great-circle distances (radians)
  euclidean_matrix.csv     — 13×13 Euclidean distances (centered VAD)
  emotion_sphere_3d.png    — 3-D scatter on S²
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3-d projection)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAR_CSV   = os.path.join(ROOT, "data", "warriner", "Ratings_Warriner_et_al.csv")
RESULTS   = os.path.join(ROOT, "asbh_tests", "results")
os.makedirs(RESULTS, exist_ok=True)

# ── emotion → Warriner word mapping ──────────────────────────────────────────
EMOTION_WORD = {
    "Anger":           "anger",
    "Anxiety":         "anxiety",
    "Fear":            "fear",
    "Surprise":        "surprise",
    "Guilt":           "guilt",
    "Disgust":         "disgust",
    "Sad":             "sad",
    "Regard":          "regard",
    "Satisfaction":    "satisfaction",
    "WarmHeartedness": "warmth",
    "Happiness":       "happy",
    "Pride":           "pride",
    "Love":            "love",
}

EMOTIONS = list(EMOTION_WORD.keys())

# ── 1. load Warriner and build lookup ─────────────────────────────────────────
print("Loading Warriner norms …")
war = pd.read_csv(WAR_CSV)

# Build lowercase → row index for fast lookup
war["Word_lower"] = war["Word"].str.lower().str.strip()
word_lookup = war.set_index("Word_lower")[["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]]

# ── 2. extract VAD for each emotion ──────────────────────────────────────────
records = []
for emotion in EMOTIONS:
    target = EMOTION_WORD[emotion].lower().strip()

    if target in word_lookup.index:
        row = word_lookup.loc[target]
        # handle duplicates — keep first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        matched_word = row["Word"]
        V_raw = float(row["V.Mean.Sum"])
        A_raw = float(row["A.Mean.Sum"])
        D_raw = float(row["D.Mean.Sum"])
        found = True
    else:
        print(f"  [WARN] '{target}' not found in Warriner for emotion '{emotion}'")
        matched_word = target
        V_raw = A_raw = D_raw = 5.0   # neutral fallback
        found = False

    # center: scale is 1–9, neutral = 5
    V_c = V_raw - 5.0
    A_c = A_raw - 5.0
    D_c = D_raw - 5.0
    norm = np.sqrt(V_c**2 + A_c**2 + D_c**2)

    records.append({
        "emotion":       emotion,
        "word_warriner": matched_word,
        "found":         found,
        "V_raw": V_raw, "A_raw": A_raw, "D_raw": D_raw,
        "V_c":   V_c,   "A_c":   A_c,   "D_c":   D_c,
        "norm":  norm,
    })

    status = "OK" if found else "MISSING"
    print(f"  [{status}] {emotion:20s} -> {matched_word:15s}  "
          f"V_c={V_c:+.3f}  A_c={A_c:+.3f}  D_c={D_c:+.3f}  ||v||={norm:.4f}")

# ── 3. normalise to S² ────────────────────────────────────────────────────────
print("\nNormalising to S² …")
for r in records:
    if r["norm"] > 0:
        r["V_s2"] = r["V_c"] / r["norm"]
        r["A_s2"] = r["A_c"] / r["norm"]
        r["D_s2"] = r["D_c"] / r["norm"]
    else:
        # degenerate: map to north pole
        r["V_s2"], r["A_s2"], r["D_s2"] = 0.0, 0.0, 1.0
        print(f"  [WARN] zero-norm vector for '{r['emotion']}', mapped to north pole")

# ── 4 & 5. distance matrices ──────────────────────────────────────────────────
n = len(EMOTIONS)
s2_coords = np.array([[r["V_s2"], r["A_s2"], r["D_s2"]] for r in records])
vc_coords  = np.array([[r["V_c"],  r["A_c"],  r["D_c"]]  for r in records])

geo_mat  = np.zeros((n, n))
euc_mat  = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        # geodesic: arccos of dot product, clipped for numerical safety
        dot_ij = np.dot(s2_coords[i], s2_coords[j])
        geo_mat[i, j] = np.arccos(np.clip(dot_ij, -1.0, 1.0))
        # Euclidean in centered VAD space
        euc_mat[i, j] = np.linalg.norm(vc_coords[i] - vc_coords[j])

# ── 6. save outputs ───────────────────────────────────────────────────────────

# emotion_vad.csv
vad_df = pd.DataFrame([{
    "emotion":       r["emotion"],
    "word_warriner": r["word_warriner"],
    "V_c":           r["V_c"],
    "A_c":           r["A_c"],
    "D_c":           r["D_c"],
    "norm":          r["norm"],
    "V_s2":          r["V_s2"],
    "A_s2":          r["A_s2"],
    "D_s2":          r["D_s2"],
} for r in records])
vad_path = os.path.join(RESULTS, "emotion_vad.csv")
vad_df.to_csv(vad_path, index=False)
print(f"\nSaved: {vad_path}")

# geodesic_matrix.csv
geo_df = pd.DataFrame(geo_mat, index=EMOTIONS, columns=EMOTIONS)
geo_path = os.path.join(RESULTS, "geodesic_matrix.csv")
geo_df.to_csv(geo_path)
print(f"Saved: {geo_path}")

# euclidean_matrix.csv
euc_df = pd.DataFrame(euc_mat, index=EMOTIONS, columns=EMOTIONS)
euc_path = os.path.join(RESULTS, "euclidean_matrix.csv")
euc_df.to_csv(euc_path)
print(f"Saved: {euc_path}")

# ── 7. 3-D sphere scatter ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

# wireframe sphere for reference
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(xs, ys, zs, color="lightgray", alpha=0.25, linewidth=0.5)

# emotion points — colour by V_s2 (valence)
V_s2_vals = s2_coords[:, 0]
sc = ax.scatter(
    s2_coords[:, 0], s2_coords[:, 1], s2_coords[:, 2],
    c=V_s2_vals, cmap="RdYlGn", vmin=-1, vmax=1,
    s=80, zorder=5,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Valence (S²)", fontsize=9)

# labels
for r, (x, y, z) in zip(records, s2_coords):
    ax.text(x * 1.05, y * 1.05, z * 1.05,
            r["emotion"], fontsize=7, ha="center", va="bottom")

ax.set_xlabel("Valence (V_s2)")
ax.set_ylabel("Arousal (A_s2)")
ax.set_zlabel("Dominance (D_s2)")
ax.set_title("Emotion Categories on S²\n(Warriner VAD, centered & normalised)")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
plt.tight_layout()

png_path = os.path.join(RESULTS, "emotion_sphere_3d.png")
fig.savefig(png_path, dpi=150)
plt.close(fig)
print(f"Saved: {png_path}")

print("\nDone.")
