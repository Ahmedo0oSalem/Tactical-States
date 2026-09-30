"""
cluster_pipeline_v3.py   (v4 logic + BALL information)
======================================================
Same as cluster_pipeline_noball.py (no auto-k, group-balanced weighting, held-out validation
groups, split-half stability, PCA-retention diagnostic, optional territory residualisation)
PLUS the ball block from ball_features.py:

  - ball_features.build_ball_features() -> 15 path scalars + 8 ball-grid PCs, attack-normalised.
  - The ball block is z-scored / PCA'd separately, then SCALED to carry a fixed share of the total
    clustering variance (BALL_FRACTION, default 15 %), split between path scalars and grid PCs by
    BALL_GRID_SHARE. The base block is group-weighted first, so BALL_FRACTION is measured against
    the weighted base.
  - Existing phase-table columns with "ball" in the name stay excluded from the base block.
  - Feature groups "ball_path" / "ball_grid" appear in the eta^2 importance (orange).
  - Per-cluster plots have a 3rd column: mean ball-occupancy map.
  - Baseline check for every k in FINAL_KS: same weighting WITHOUT ball, and the ARI between the two
    partitions (how much the ball actually changed the clustering).

Config via environment variables (all optional)
  UPLOAD_DIR, OUT_DIR
  FINAL_KS (2,4,6) K_MAX (15) GROUP_ALPHA (0.5) HOLDOUT_GROUPS PCA_VAR (0.90)
  STAB_REPEATS (5, 0 = off) TERRITORY_RESIDUAL (0)
  BALL_FRACTION (0.15; 0 disables ball) BALL_GRID_SHARE (0.5) N_GRID_PCS (8)
Match direction overrides: edit BALL_DIRECTION_OVERRIDES below.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from ball_features import (build_ball_features, scale_block_to_fraction,
                           PATH_FEATURES)

UPLOAD = Path(os.environ.get("UPLOAD_DIR", "."))
OUT = Path(os.environ.get("OUT_DIR", "./cluster_output_v3"))
OUT.mkdir(parents=True, exist_ok=True)

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
N_ROWS, N_COLS = 12, 16

FINAL_KS = [int(x) for x in os.environ.get("FINAL_KS", "2,4,6").split(",") if x.strip()]
K_MAX = int(os.environ.get("K_MAX", 15))
GROUP_ALPHA = float(os.environ.get("GROUP_ALPHA", 0.5))
HOLDOUT_GROUPS = {g.strip() for g in os.environ.get(
    "HOLDOUT_GROUPS", "match_events,passing,pressure,carrying,touch_movement,other").split(",")
    if g.strip()}
PCA_VAR = float(os.environ.get("PCA_VAR", 0.90))
STAB_REPEATS = int(os.environ.get("STAB_REPEATS", 5))
TERRITORY_RESIDUAL = os.environ.get("TERRITORY_RESIDUAL", "0") == "1"
TERRITORY_ANCHORS = ("pitch_control_mean", "attacking_centroid_x_mean")
PALETTE = ["#4c78a8", "#e45756", "#54a24b", "#b279a2", "#72b7b2", "#eeca3b"]

# ---- ball configuration ------------------------------------------------
BALL_FRACTION = float(os.environ.get("BALL_FRACTION", 0.15))     # share of total variance
BALL_GRID_SHARE = float(os.environ.get("BALL_GRID_SHARE", 0.5))  # part of it given to grid PCs
N_GRID_PCS = int(os.environ.get("N_GRID_PCS", 8))
# {match_id: True/False}; True = home attacks towards +x in period 2.
# Match 10504's automatic vote was split 2/3 (both directions give mean x ~52) -> verify it.
BALL_DIRECTION_OVERRIDES = {}
USE_BALL = BALL_FRACTION > 0


def find_file(name):
    alts = [name]
    pre = "Outputs_4th_Run_"
    if name.startswith(pre):
        alts.append("Outputs\\4th_Run\\" + name[len(pre):])
    for d in (UPLOAD, UPLOAD / "pca_output"):
        for a in alts:
            if (d / a).exists():
                return d / a
    raise FileNotFoundError(f"Could not find {name} (tried {alts}) in {UPLOAD} or {UPLOAD / 'pca_output'}")


# ------------------------------------------------------------------
# 1. Load + merge
# ------------------------------------------------------------------
phases = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases.parquet"))
pass_mv = pd.read_parquet(find_file("Outputs_4th_Run_pass_movement_scores.parquet"))
form_win = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases_formation_windows.parquet"))
pca_scores = pd.read_parquet(find_file("pca_scores.parquet"))
pca_model = np.load(find_file("pca_model.npz"), allow_pickle=True)

for d in (phases, pass_mv, form_win, pca_scores):
    d["match_id"] = d["match_id"].astype(str)
for d in (phases, pca_scores):
    d["window_id"] = d["window_id"].astype(str)

num_cols = pass_mv.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ("sequence_id", "game_event_id", "possession_event_id", "period")]
pm_agg = (pass_mv.groupby(["match_id", "sequence_id", "team"])[num_cols]
          .mean().add_prefix("touch_").reset_index())

fw_num = ["avgCostPerPlayer", "confidence", "mean_compactness", "mean_width", "mean_depth"]
fw_agg = (form_win.groupby(["match_id", "team", "period"])[fw_num]
          .mean().add_prefix("formwin_").reset_index())

pc_cols = [c for c in pca_scores.columns if c.startswith("PC")]
pca_sub = pca_scores[["match_id", "window_id", "team", "period"] + pc_cols]

df = phases.merge(pm_agg, on=["match_id", "sequence_id", "team"], how="left")
df = df.merge(fw_agg, on=["match_id", "team", "period"], how="left")
df = df.merge(pca_sub, on=["match_id", "window_id", "team", "period"], how="left")
assert len(df) == len(phases)
print(f"merged: {df.shape}  |  matches: {df['match_id'].nunique()}")

# ------------------------------------------------------------------
# 1b. Ball features (aligned row-for-row to df)
# ------------------------------------------------------------------
ball_path_cols, ball_grid_cols = [], []
if USE_BALL:
    ball_path = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases_ball_path.parquet"))
    ball_traj = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases_ball_trajectory.parquet"))
    ball_feats, ball_extra = build_ball_features(
        ball_path, ball_traj, n_grid_pcs=N_GRID_PCS,
        direction_overrides=BALL_DIRECTION_OVERRIDES)
    ball_path_cols = list(PATH_FEATURES)
    ball_grid_cols = [c for c in ball_feats.columns if c.startswith("ballgrid_PC")]

    bf = df[["match_id", "window_id"]].merge(ball_feats, on=["match_id", "window_id"], how="left")
    assert len(bf) == len(df), "duplicate (match_id, window_id) keys in ball features"
    has_ball = bf["ball_start_x"].notna().to_numpy()
    print(f"[ball] {has_ball.sum()}/{len(df)} phases matched to ball data ({has_ball.mean():.1%})")
    if has_ball.mean() < 0.5:
        raise RuntimeError("Less than half the phases matched ball data - check that "
                           "window_id / match_id formats agree between the tables.")
    bf.drop(columns=["match_id", "window_id"]).to_parquet(OUT / "ball_features_per_phase.parquet",
                                                          index=False)
    # row index into ball_extra["grids"] for each df row (-1 = no ball data)
    gkey = {(m, w): i for i, (m, w) in enumerate(zip(ball_extra["keys"]["match_id"],
                                                     ball_extra["keys"]["window_id"]))}
    grid_row = np.array([gkey.get((m, w), -1) for m, w in zip(df["match_id"], df["window_id"])])
    ball_grids = ball_extra["grids"]

# ------------------------------------------------------------------
# 2. Feature selection (ball excluded) + group tagging
# ------------------------------------------------------------------
id_like = {"match_id", "sequence_id", "window_id", "windowIndex", "start_frame", "end_frame",
           "game_event_id", "possession_event_id"}
numeric_df = df.select_dtypes(include=[np.number]).copy()

drop_cols = set()
for c in numeric_df.columns:
    lc = c.lower()
    if "ball" in lc:
        drop_cols.add(c)
    if c in id_like or lc.endswith("_id") or "frame" in lc:
        drop_cols.add(c)
    if lc in ("start_sec", "end_sec", "start_sec_period", "end_sec_period",
              "touch_event_start_time", "touch_event_end_time", "touch_event_time",
              "touch_next_event_time"):
        drop_cols.add(c)

feature_cols = [c for c in numeric_df.columns if c not in drop_cols]
X_raw = numeric_df[feature_cols]
keep_cols = [c for c in feature_cols if X_raw[c].isna().mean() < 0.6]
X_raw = X_raw[keep_cols]
nunique = X_raw.nunique()
keep_cols2 = nunique[nunique > 1].index.tolist()
X_raw = X_raw[keep_cols2]
print(f"features available: {len(keep_cols2)}")


def group_of(col):
    lc = col.lower()
    if lc.startswith("pc") and lc[2:].isdigit():
        return "positional_pca"
    if "depth" in lc:
        return "depth"
    if "width" in lc:
        return "width"
    if "compactness" in lc:
        return "compactness"
    if "elongation" in lc:
        return "elongation"
    if "_area" in lc or lc.startswith("area"):
        return "area"
    if "centroid" in lc and "distance" not in lc:
        return "centroid_position"
    if "centroid_distance" in lc or "longitudinal_centroid" in lc or "lateral_centroid" in lc:
        return "team_vs_team_distance"
    if "speed" in lc:
        return "player_speed"
    if "pitch_control" in lc:
        return "pitch_control"
    if "pressure" in lc or "pressured" in lc:
        return "pressure"
    if "pass" in lc or "n_passs" in lc or "lines_broken" in lc:
        return "passing"
    if "progression" in lc:
        return "progression"
    if "carry" in lc or "carrys" in lc:
        return "carrying"
    if "touch_" in lc or "disp" in lc:
        return "touch_movement"
    if "formation" in lc or "formwin" in lc:
        return "formation"
    if "n_events" in lc or "n_shots" in lc or "goals" in lc or "n_clearances" in lc \
       or "n_challenges" in lc or "n_crosss" in lc or "n_rebounds" in lc \
       or "chances_created" in lc or "dangerous_positions" in lc or "events_per_second" in lc \
       or "shots_on_target" in lc:
        return "match_events"
    if "nearest_" in lc or "pairwise" in lc:
        return "player_spacing"
    if "duration" in lc:
        return "phase_duration"
    return "other"


feature_groups = {c: group_of(c) for c in keep_cols2}
for c in ball_path_cols:
    feature_groups[c] = "ball_path"
for c in ball_grid_cols:
    feature_groups[c] = "ball_grid"
with open(OUT / "feature_groups.txt", "w") as f:
    for grp in sorted(set(feature_groups.values())):
        cols = sorted([c for c, g in feature_groups.items() if g == grp])
        f.write(f"\n== {grp} ({len(cols)} features) ==\n")
        for c in cols:
            f.write(f"  {c}\n")

# ------------------------------------------------------------------
# 3. Scale, (optionally residualise territory), group-weight, PCA-denoise
# ------------------------------------------------------------------
X_imp = SimpleImputer(strategy="median").fit_transform(X_raw)
M = StandardScaler().fit_transform(X_imp)          # every feature, unit variance (eta^2 uses this)
col_idx = {c: i for i, c in enumerate(keep_cols2)}
groups_arr = np.array([feature_groups[c] for c in keep_cols2])

# positional PCs: keep their relative variances (PC1 largest) instead of z-scoring all 20
pc_idx = np.where(groups_arr == "positional_pca")[0]
if len(pc_idx) and "PC1" in col_idx:
    sd_pc1 = X_imp[:, col_idx["PC1"]].std()
    for i in pc_idx:
        M[:, i] = (X_imp[:, i] - X_imp[:, i].mean()) / sd_pc1

base = M
if TERRITORY_RESIDUAL:
    anchors = [c for c in TERRITORY_ANCHORS if c in col_idx]
    A = np.column_stack([np.ones(len(M))] + [M[:, col_idx[c]] for c in anchors])
    beta = np.linalg.lstsq(A, M, rcond=None)[0]
    base = M - A @ beta            # NOT re-standardised: columns mostly explained by territory stay small
    print(f"[territory] residualised on {anchors}")

use_mask = ~np.isin(groups_arr, list(HOLDOUT_GROUPS))
cluster_cols = [c for c, u in zip(keep_cols2, use_mask) if u]
holdout_cols = [c for c, u in zip(keep_cols2, use_mask) if not u]
g_used = groups_arr[use_mask]
Mu = M[:, use_mask]
col_scale = np.ones(Mu.shape[1])
for g in np.unique(g_used):
    idx = np.where(g_used == g)[0]
    nominal = Mu[:, idx].var(axis=0).sum()
    # group variance -> nominal ** (1 - alpha); nominal == n_features for z-scored groups,
    # and ~effective dimensionality for the positional-PC group (so 20 PCs don't count as 20 features)
    col_scale[idx] = np.sqrt(nominal ** (1.0 - GROUP_ALPHA) / nominal)
Xw = base[:, use_mask] * col_scale

base_variance = float(Xw.var(axis=0).sum())      # group-weighted base block

if USE_BALL:
    path_imp = SimpleImputer(strategy="median").fit_transform(bf[ball_path_cols])
    path_z = StandardScaler().fit_transform(path_imp)
    # grid PCs are centred already; phases without ball data -> 0 (= average phase)
    grid_pcs = bf[ball_grid_cols].fillna(0.0).to_numpy()

    base_fraction = 1.0 - BALL_FRACTION
    path_w = scale_block_to_fraction(path_z, base_variance,
                                     BALL_FRACTION * (1 - BALL_GRID_SHARE), base_fraction)
    grid_w = scale_block_to_fraction(grid_pcs, base_variance,
                                     BALL_FRACTION * BALL_GRID_SHARE, base_fraction)
    X_all = np.hstack([Xw, path_w, grid_w])
    names_all = cluster_cols + ball_path_cols + ball_grid_cols
    tot = float(np.var(X_all, axis=0).sum())
    print(f"[ball] variance share -> base {base_variance / tot:.1%} | "
          f"path {np.var(path_w, axis=0).sum() / tot:.1%} | "
          f"grid {np.var(grid_w, axis=0).sum() / tot:.1%}")
    # un-weighted matrix with real names, for eta^2 (scale-invariant per column)
    imp_names = keep_cols2 + ball_path_cols + ball_grid_cols
    imp_matrix = np.hstack([M, path_z, grid_pcs])
else:
    X_all, names_all = Xw, cluster_cols
    imp_names, imp_matrix = keep_cols2, M

g_all = np.array([feature_groups[c] for c in names_all])
tot_var = X_all.var(axis=0).sum()
share = (pd.Series(X_all.var(axis=0), index=g_all).groupby(level=0).sum() / tot_var).sort_values(ascending=False)
print(f"\nclustering on {len(names_all)} features; base held out for validation: "
      f"{len(holdout_cols)} ({sorted(HOLDOUT_GROUPS)})")
print(f"GROUP_ALPHA={GROUP_ALPHA} -> share of clustering variance per group:")
print((share * 100).round(1).to_string())

pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=0)
X_pca = pca.fit_transform(X_all)
print(f"\nPCA denoise: {X_pca.shape[1]} components ({PCA_VAR:.0%} variance)")

Xrec = pca.inverse_transform(X_pca)
ss_tot = ((X_all - X_all.mean(0)) ** 2).sum(0)
ss_res = ((X_all - Xrec) ** 2).sum(0)
ret = pd.DataFrame({"tot": ss_tot, "res": ss_res}, index=g_all).groupby(level=0).sum()
ret["variance_retained"] = 1 - ret["res"] / ret["tot"]
ret = ret[["variance_retained"]].sort_values("variance_retained")
ret.to_csv(OUT / "pca_retention_by_group.csv")
print("variance retained per group after PCA denoise:")
print(ret.round(3).to_string())

# ------------------------------------------------------------------
# 4. Diagnostic sweep: silhouette, inertia, split-half stability
# ------------------------------------------------------------------
matches_arr = df["match_id"].to_numpy()


def split_half_ari(X, k, repeats):
    """ARI between two k-means fitted on disjoint halves of the matches, both applied to ALL rows."""
    rng = np.random.default_rng(0)
    um = np.unique(matches_arr)
    out = []
    for r in range(repeats):
        perm = rng.permutation(um)
        half = len(um) // 2
        a = np.isin(matches_arr, perm[:half])
        b = np.isin(matches_arr, perm[half:])
        ka = KMeans(n_clusters=k, n_init=3, random_state=r).fit(X[a])
        kb = KMeans(n_clusters=k, n_init=3, random_state=r + 100).fit(X[b])
        out.append(adjusted_rand_score(ka.predict(X), kb.predict(X)))
    return float(np.mean(out)), float(np.std(out))


ks = list(range(2, K_MAX + 1))
assert all(k in ks for k in FINAL_KS), f"FINAL_KS {FINAL_KS} must lie within 2..{K_MAX}"
sil_scores, inertias, stab_mean, stab_std, all_labels = [], [], [], [], {}
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X_pca)
    sil_scores.append(silhouette_score(X_pca, labels))
    inertias.append(km.inertia_)
    all_labels[k] = labels
    if STAB_REPEATS > 0:
        m_, s_ = split_half_ari(X_pca, k, STAB_REPEATS)
    else:
        m_, s_ = np.nan, np.nan
    stab_mean.append(m_); stab_std.append(s_)
    print(f"k={k:2d}  silhouette={sil_scores[-1]:.3f}  split-half ARI={m_:.3f}")

sweep_df = pd.DataFrame({"k": ks, "silhouette": sil_scores, "inertia": inertias,
                         "stability_ari_mean": stab_mean, "stability_ari_std": stab_std})
sweep_df.to_csv(OUT / "silhouette_by_k.csv", index=False)
print("\nNo automatic k selection: full outputs for k in", FINAL_KS)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
axes[0].plot(ks, sil_scores, marker="o")
axes[0].set_xlabel("k"); axes[0].set_ylabel("silhouette score"); axes[0].set_title("Silhouette vs k")
axes[1].plot(ks, inertias, marker="o", color="#e45756")
axes[1].set_xlabel("k"); axes[1].set_ylabel("inertia"); axes[1].set_title("Elbow plot")
if STAB_REPEATS > 0:
    axes[2].errorbar(ks, stab_mean, yerr=stab_std, marker="o", color="#54a24b", capsize=3)
    axes[2].set_ylim(0, 1.02)
axes[2].set_xlabel("k"); axes[2].set_ylabel("ARI (half A vs half B of matches)")
axes[2].set_title("Split-half stability (higher = reproducible)")
for ax in axes:
    for fk in FINAL_KS:
        ax.axvline(fk, color="grey", ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT / "silhouette_sweep_2_15.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------
# 4b. Baseline without ball: how much did the ball change the clustering?
# ------------------------------------------------------------------
base_labels = {}
if USE_BALL:
    Xb = PCA(n_components=PCA_VAR, svd_solver="full", random_state=0).fit_transform(Xw)
    base_rows = []
    for k in FINAL_KS:
        lb = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xb)
        base_labels[k] = lb
        base_rows.append(dict(k=k,
                              silhouette_no_ball=silhouette_score(Xb, lb),
                              silhouette_with_ball=sil_scores[ks.index(k)],
                              ARI_ball_vs_no_ball=adjusted_rand_score(all_labels[k], lb)))
    base_cmp = pd.DataFrame(base_rows)
    base_cmp.to_csv(OUT / "ball_vs_no_ball_comparison.csv", index=False)
    print("\nBall vs no-ball clustering (ARI 1.0 = identical, ~0 = unrelated):")
    print(base_cmp.round(3).to_string(index=False))

# ------------------------------------------------------------------
# 5. Assignments for every k
# ------------------------------------------------------------------
assign = df[["match_id", "sequence_id", "window_id", "team", "period"] + pc_cols].copy()
for k in ks:
    assign[f"cluster_k{k}"] = all_labels[k]
assign.to_parquet(OUT / "cluster_assignments_all_k.parquet", index=False)

# ------------------------------------------------------------------
# 6. Grouped feature importance (eta^2) - ALL features, held-out groups shown in grey
# ------------------------------------------------------------------

def group_importance(labels, k_label, color):
    Z = pd.DataFrame(imp_matrix, columns=imp_names)
    mu = Z.mean()
    gm = Z.groupby(labels).mean()
    sizes = pd.Series(labels).value_counts().reindex(gm.index).to_numpy()
    ss_b = ((gm - mu) ** 2).mul(sizes, axis=0).sum()
    ss_t = ((Z - mu) ** 2).sum()
    eta = (ss_b / ss_t.where(ss_t > 0)).fillna(0)
    imp = pd.DataFrame({"feature": imp_names, "group": [feature_groups[c] for c in imp_names],
                        "eta_squared": eta.to_numpy()})
    imp.to_csv(OUT / f"cluster_feature_importance_{k_label}.csv", index=False)
    g = imp.groupby("group")["eta_squared"].mean().sort_values(ascending=False)
    g.to_csv(OUT / f"cluster_group_importance_{k_label}.csv")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    g.plot(kind="barh", ax=ax, color=["#bbbbbb" if i in HOLDOUT_GROUPS
                                      else ("#f58518" if i.startswith("ball") else color)
                                      for i in g.index])
    ax.invert_yaxis()
    ax.set_xlabel("mean eta-squared (higher = more separates clusters)")
    ax.set_title(f"Which feature GROUPS separate the clusters ({k_label})\n[grey = held out, orange = ball]",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / f"group_importance_{k_label}.png", dpi=130)
    plt.close(fig)
    return g


# ------------------------------------------------------------------
# 7. Per-cluster player recurrence maps (reconstructed from the PCA model)
# ------------------------------------------------------------------
mean_vec = pca_model["mean"]
components = pca_model["components"]
size = N_ROWS * N_COLS
pc_scores_matrix = df[pc_cols].to_numpy()
valid = ~np.isnan(pc_scores_matrix).any(axis=1)


def _pitch(ax):
    L, W = PITCH_LENGTH, PITCH_WIDTH
    ax.set_facecolor("#1c1c1c")
    kw = dict(fill=False, color="white", lw=0.9, alpha=0.8)
    ax.add_patch(patches.Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2, L / 2], [0, W], color="white", lw=0.8, alpha=0.8)
    ax.add_patch(patches.Circle((L / 2, W / 2), 9.15, **kw))
    for xg, d in ((0, 1), (L, -1)):
        ax.add_patch(patches.Rectangle((xg, W / 2 - 20.15), d * 16.5, 40.3, **kw))
        ax.add_patch(patches.Rectangle((xg, W / 2 - 9.15), d * 5.5, 18.3, **kw))
    ax.set_xlim(-2, L + 2); ax.set_ylim(-2, W + 2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def make_recurrence_maps(labels, k_label):
    maps = {}
    for c in sorted(np.unique(labels)):
        mask = (labels == c) & valid
        mean_scores = pc_scores_matrix[mask].mean(axis=0)
        recon = np.clip(mean_vec + mean_scores @ components, 0, None) ** 2
        ball_map = None
        if USE_BALL:
            rows = grid_row[(labels == c) & (grid_row >= 0)]
            ball_map = ball_grids[rows].mean(axis=0) if len(rows) else np.zeros((N_ROWS, N_COLS))
        maps[c] = (recon[:size].reshape(N_ROWS, N_COLS),
                   recon[size:].reshape(N_ROWS, N_COLS), ball_map, mask.sum())
    n = len(maps)
    ncols = 3 if USE_BALL else 2
    fig, axes = plt.subplots(n, ncols, figsize=(5.2 * ncols, 3.1 * n), squeeze=False)
    vmax = max(max(m[0].max(), m[1].max()) for m in maps.values())
    bmax = max(m[2].max() for m in maps.values()) if USE_BALL else None
    for i, c in enumerate(sorted(maps)):
        inp, outp, bm, cnt = maps[c]
        panels = [("IN POSSESSION", inp, vmax), ("OUT OF POSSESSION", outp, vmax)]
        if USE_BALL:
            panels.append(("BALL", bm, bmax))
        for j, (name, grid, vm) in enumerate(panels):
            ax = axes[i, j]
            _pitch(ax)
            im = ax.imshow(grid, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH], origin="lower",
                           cmap="hot", vmin=0, vmax=vm, alpha=0.9, aspect="auto",
                           interpolation="nearest")
            ax.set_title(f"Cluster {c} (n={cnt})  ·  {name}", fontsize=9)
            if j >= 1:
                plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f"Average recurrence maps per cluster ({k_label})\n"
                 "team in possession attacks left → right  ·  brighter = more time spent there",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(OUT / f"cluster_recurrence_maps_{k_label}.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------
# 8. Per-k outputs: importance, maps, profile, held-out validation profile
# ------------------------------------------------------------------
profile_feats = [c for c in [
    "duration", "attacking_width_mean", "attacking_depth_mean", "attacking_compactness_mean",
    "defending_width_mean", "defending_depth_mean", "defending_compactness_mean",
    "centroid_distance_mean", "mean_player_speed", "n_events", "n_passs",
    "pass_completion_rate", "n_shots", "goals", "pressure_rate", "pitch_control_mean",
    "PC1", "PC2", "PC3",
] if c in df.columns]

for i_k, k_ in enumerate(FINAL_KS):
    lab = all_labels[k_]
    kl = f"k{k_}"
    g = group_importance(lab, kl, PALETTE[i_k % len(PALETTE)])
    print(f"\nGroup importance (mean eta^2) at {kl}:\n{g.round(3).to_string()}")
    make_recurrence_maps(lab, kl)

    p_ = df[profile_feats].copy()
    if USE_BALL:
        for c in ("ball_start_x", "ball_end_x", "ball_dx", "ball_max_x", "ball_lat_shift",
                  "ball_path_length", "ball_straightness", "ball_speed_mps"):
            p_[c] = bf[c].to_numpy()
    p_["cluster"] = lab
    table = p_.groupby("cluster").mean(numeric_only=True).round(2)
    table["n_phases"] = p_.groupby("cluster").size()
    table.to_csv(OUT / f"cluster_profile_{kl}.csv")
    print(f"\nCluster profile ({kl}):\n{table}")

    if holdout_cols:
        hv = X_raw[holdout_cols].copy()
        hv["cluster"] = lab
        hv.groupby("cluster").mean().round(3).to_csv(OUT / f"cluster_holdout_profile_{kl}.csv")

print(f"\nSaved everything to {OUT}/")