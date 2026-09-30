"""
cluster_pipeline_noball.py   (v4 - no ball features)
====================================================
Clustering of tactical phases WITHOUT any ball features. Standalone.

What changed vs the previous version
  1. NO auto-selection of k. The sweep (k=2..K_MAX) is still run as a diagnostic, but the
     detailed outputs (maps, importance, profiles) are produced for every k in FINAL_KS.
  2. Group-balanced feature weighting (GROUP_ALPHA). Previously ~72% of the input variance came
     from six families with 34-52 near-duplicate columns (start/end/mean/min/max...). Now each
     group's total variance is  n_features ** (1 - GROUP_ALPHA)
     (effective dimensionality for the positional-PC group):
         0.0 = old behaviour (proportional to size), 1.0 = every group counts equally.
  3. Positional-PC columns keep their relative variances (PC1 no longer weighs the same as PC20).
  4. Outcome / behaviour groups are HELD OUT of the clustering (HOLDOUT_GROUPS) so that shots,
     goals, passing, pressure can be used as non-circular validation. They still get eta^2 and
     a per-cluster profile table.
  5. Split-half stability (ARI between k-means fitted on two disjoint halves of the MATCHES,
     both applied to all phases) is computed for every k and plotted next to silhouette/elbow.
  6. Diagnostic: how much variance of each feature group survives the PCA denoise step.
  7. Optional TERRITORY_RESIDUAL=1: regress attacking_centroid_x_mean out of every feature
     first, so clusters describe shape/tempo instead of "which half".
  8. Ball features are excluded entirely (dropped before clustering, not just held out) --
     this pipeline only sees positional/formation/event/touch/pitch-control data.
  9. Per-cluster PLAYER recurrence maps (the plotted images) are the RAW maps from
     Outputs\4th_Run\tactical_phases_recurrence.parquet (a straight average of each cluster's
     member phases' actual occupancy grids, attack-direction normalised), not a PCA
     reconstruction.
 10. RECURRENCE_FEATURES=1 (NEW, default on): the CLUSTERING features themselves now include
     the full flattened per-phase in-possession + out-of-possession occupancy grid AT NATIVE
     RESOLUTION (e.g. 68x105 = ~1m/cell, whatever the recurrence parquet was built at -- see
     RECURRENCE_GRID_ROWS/COLS to override), instead of relying on the offline positional_pca
     PC1..PCn summary from pca_scores.parquet. The old PCs are still merged in and profiled,
     but held out of clustering by default (see HOLDOUT_GROUPS). Unlike a normal feature group,
     recurrence_map is NOT independently z-scored per cell (that forces a center-circle cell
     and a never-visited corner cell to the same variance=1, which blows up the group's
     effective dimensionality and drowns out everything else) and is exempted from the pooled
     PCA-denoise step (so it reaches k-means as the genuine raw map, not a reconstruction) --
     see Section 3. Set RECURRENCE_FEATURES=0 to go back to the old PC-based behaviour.
 11. PITCH_CONTROL_FEATURES=1 (NEW, default on): the full per-cell pitch-control map
     (Outputs\4th_Run\tactical_phases_pitch_control.parquet, ONE grid per phase -- already
     collapsed to the possession team's control fraction, not home/away, so no in/out-poss
     pair the way recurrence_map has) is added as its own "pitch_control_map" clustering
     group, at native resolution, with the SAME peak-scaling + PCA-passthrough treatment as
     recurrence_map. The 8 pitch_control_mean/std/min/max/range/start/end/change SCALAR
     columns (a plain aggregate summary, not a map) are a separate, ordinary group
     "pitch_control_scalar" -- previously dropped entirely, now clustered normally like
     depth/width/etc. Set PITCH_CONTROL_FEATURES=0 to exclude the map (the scalar group is
     unaffected by this flag; move it to HOLDOUT_GROUPS if you want it validation-only).

     PATCH (memory fix): _load_raw_pitch_control() used to run unconditionally at import
     time regardless of PITCH_CONTROL_FEATURES, building a full (n_phases, n_rows, n_cols)
     float64 array just to be ignored. It's now only called when PITCH_CONTROL_FEATURES=1.
     Both raw grid builders (_load_raw_recurrence's home/away grids and
     _load_raw_pitch_control's grid) now allocate float32 instead of float64, roughly
     halving peak memory for these arrays. _load_raw_recurrence() itself is still
     unconditional -- Section 7/8's per-cluster recurrence PLOTS use raw_in_poss/raw_out_poss
     even when RECURRENCE_FEATURES=0, so it can't be skipped, only shrunk.

Config via environment variables (all optional)
  UPLOAD_DIR, OUT_DIR
  FINAL_KS         default "2,4,6"      k values that get full outputs
  K_MAX            default 15           sweep upper bound
  GROUP_ALPHA      default 0.5
  HOLDOUT_GROUPS   default "match_events,passing,pressure,carrying,touch_movement,other"
  PCA_VAR          default 0.90
  STAB_REPEATS     default 5            0 disables the stability computation
  TERRITORY_RESIDUAL  default 0
  RECURRENCE_FEATURES default 1         cluster on raw 12x16 recurrence grids, not PCA PCs
  PITCH_CONTROL_FEATURES default 1      cluster on the raw pitch-control map too (see item 11)
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

UPLOAD = Path(os.environ.get("UPLOAD_DIR", "."))
OUT = Path(os.environ.get("OUT_DIR", "./cluster_output_noball"))
OUT.mkdir(parents=True, exist_ok=True)

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

FINAL_KS = [int(x) for x in os.environ.get("FINAL_KS", "2,4,6").split(",") if x.strip()]
K_MAX = int(os.environ.get("K_MAX", 15))
GROUP_ALPHA = float(os.environ.get("GROUP_ALPHA", 0.5))
RECURRENCE_FEATURES = os.environ.get("RECURRENCE_FEATURES", "1") == "1"
# Optional downsample of the fine grid before it becomes clustering features. Default 0 =
# use the NATIVE resolution as-loaded (e.g. 68x105 / ~1m cells) untouched. Set >0 only if you
# want to compare against a coarser grid -- the dimensionality problem this used to cause is
# now fixed via pooled peak-scaling in Section 3, not by throwing away resolution.
RECURRENCE_GRID_ROWS = int(os.environ.get("RECURRENCE_GRID_ROWS", 0))
RECURRENCE_GRID_COLS = int(os.environ.get("RECURRENCE_GRID_COLS", 0))
# Optional Gaussian smoothing (in grid cells) applied AT NATIVE RESOLUTION before flattening.
# Default 0 = off, raw map unchanged. This does NOT reduce resolution or cell count -- it makes
# adjacent cells correlated so two phases with the same shape shifted by 1-2 cells don't look
# maximally far apart under Euclidean distance (which flattened-pixel k-means uses and which
# is not translation-invariant). Try e.g. 1.0-2.0 if recurrence_map's own eta^2 stays near-zero
# regardless of its variance share -- that pattern means it's a distance-metric problem, not a
# weighting problem, and GROUP_ALPHA can't fix it.
RECURRENCE_SMOOTH_SIGMA = float(os.environ.get("RECURRENCE_SMOOTH_SIGMA", 0))
# Same idea, for the separate pitch-control grid (tactical_phases_pitch_control.parquet).
# Default 1 = included in clustering (this is the "put it back" toggle) as its own
# "pitch_control_map" group, same peak-scaling + PCA-passthrough treatment as recurrence_map.
PITCH_CONTROL_FEATURES = os.environ.get("PITCH_CONTROL_FEATURES", "1") == "1"
PITCH_CONTROL_SMOOTH_SIGMA = float(os.environ.get("PITCH_CONTROL_SMOOTH_SIGMA", 0))
# When RECURRENCE_FEATURES=0, the raw recurrence grids aren't needed for clustering -- but
# Section 8 still unconditionally calls make_recurrence_maps() to produce the diagnostic PNGs,
# which DOES need them, so _load_raw_recurrence() (a large parquet read + array build) was
# still running every time regardless of RECURRENCE_FEATURES. Set SKIP_RECURRENCE_MAPS=1 for
# sweep runs (e.g. scanning GROUP_ALPHA/PCA_VAR) where you don't need those PNGs each time --
# this skips the load entirely AND skips generating the PNGs. Only takes effect when
# RECURRENCE_FEATURES=0 (if it's on, the grids are needed for clustering itself and are always
# loaded regardless of this flag).
SKIP_RECURRENCE_MAPS = os.environ.get("SKIP_RECURRENCE_MAPS", "0") == "1"
_default_holdout = "match_events,passing,pressure,carrying,touch_movement,other"
if RECURRENCE_FEATURES:
    _default_holdout += ",positional_pca"
HOLDOUT_GROUPS = {g.strip() for g in os.environ.get("HOLDOUT_GROUPS", _default_holdout).split(",")
    if g.strip()}
PCA_VAR = float(os.environ.get("PCA_VAR", 0.90))
STAB_REPEATS = int(os.environ.get("STAB_REPEATS", 5))
TERRITORY_RESIDUAL = os.environ.get("TERRITORY_RESIDUAL", "0") == "1"
TERRITORY_ANCHORS = ("attacking_centroid_x_mean",)
PALETTE = ["#4c78a8", "#e45756", "#54a24b", "#b279a2", "#72b7b2", "#eeca3b"]


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
# 1a. Raw per-phase recurrence maps (fine grid -- auto-detected, was 68x105 = ~1m/cell in
#     practice, NOT the 12x16 the build script's comments describe). Loaded BEFORE feature
#     selection because, when RECURRENCE_FEATURES=1, a DOWNSAMPLED version of these grids
#     becomes the clustering features -- not just plotting inputs. Kept at full resolution
#     here for the per-cluster heatmap plots (Section 7/8); downsampled separately below
#     (Section 1b) for clustering, because 68x105x2=14,280 raw cells is mostly per-phase
#     sampling noise (a phase only touches a few dozen cells per side) and swamped every
#     other feature group even after GROUP_ALPHA weighting.
#
#     NOTE (memory): this loader is unconditional -- Section 7/8's per-cluster recurrence
#     PLOTS need raw_in_poss/raw_out_poss even when RECURRENCE_FEATURES=0, so it can't be
#     skipped based on that flag. The home/away grids are built as float32 (not float64)
#     to roughly halve the memory this allocates.
# ------------------------------------------------------------------
def _load_raw_recurrence():
    rec = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases_recurrence.parquet"))
    if "side" not in rec.columns:
        raise SystemExit(
            "No `side` column in the recurrence parquet: this is the old combined-map file. "
            "Re-run build_tactical_phases.py (per-team Step 6) first.")
    rec = rec.assign(match_id=rec["match_id"].astype(str), window_id=rec["window_id"].astype(str))

    n_rows_raw = int(rec["row"].max()) + 1
    n_cols_raw = int(rec["col"].max()) + 1
    per_match_extent = rec.groupby("match_id").agg(max_row=("row", "max"), max_col=("col", "max"))
    if per_match_extent["max_row"].nunique() > 1 or per_match_extent["max_col"].nunique() > 1:
        print("WARNING: raw recurrence grid extent varies by match_id (different pitch "
              "dimensions per Step 6) -- cells will not be spatially aligned across matches, "
              "so averaging within a cluster will blur across pitch sizes:")
        print(per_match_extent)

    key = rec["match_id"] + "|" + rec["window_id"]
    codes, uniques = pd.factorize(key)
    rec = rec.assign(_k=codes)
    n = len(uniques)
    rphases = (rec.drop_duplicates("_k")[["_k", "match_id", "window_id", "team", "period"]]
               .sort_values("_k").reset_index(drop=True))

    grids = {}
    for side in ("home", "away"):
        d = rec[rec["side"] == side]
        # PATCH: float32 instead of float64 -- this is the largest allocation in the whole
        # script (n_phases x n_rows x n_cols, x2 for home/away); halving it materially
        # reduces peak memory without changing any downstream math (recurrence_frac values
        # are fractions in [0,1], well within float32 precision for this purpose).
        g = np.zeros((n, n_rows_raw, n_cols_raw), dtype=np.float32)
        g[d["_k"].to_numpy(), d["row"].to_numpy(), d["col"].to_numpy()] = d["recurrence_frac"].to_numpy()
        grids[side] = g

    # attacking-direction normalisation (same logic as pca_recurrence.py): rotate 180 degrees
    # so the team in possession always attacks toward +x.
    def centroid_x(g):
        xs = (np.arange(n_cols_raw) + 0.5) * PITCH_LENGTH / n_cols_raw
        mass = g.sum(axis=(1, 2))
        return (g.sum(axis=1) * xs).sum(axis=1) / np.where(mass > 0, mass, np.nan)

    mid = 0.5 * (centroid_x(grids["home"]) + centroid_x(grids["away"]))
    ddf = rphases[["match_id", "period", "team"]].copy()
    ddf["diff"] = mid
    home_dir = {}
    for (m, p), grp in ddf.groupby(["match_id", "period"]):
        h = grp.loc[grp["team"] == "home", "diff"].mean()
        a = grp.loc[grp["team"] == "away", "diff"].mean()
        margin = h - a
        home_dir[(m, p)] = (1 if margin > 0 else -1) if (np.isfinite(margin) and margin != 0) \
            else (1 if int(p) == 1 else -1)
    hd = np.array([home_dir[(m, p)] for m, p in zip(rphases["match_id"], rphases["period"])])
    attack_dir = np.where(rphases["team"].to_numpy() == "home", hd, -hd)
    flip = attack_dir == -1
    for s in grids:
        grids[s] = grids[s].copy()
        grids[s][flip] = grids[s][flip][:, ::-1, ::-1]

    is_home = (rphases["team"].to_numpy() == "home")[:, None, None]
    in_poss = np.where(is_home, grids["home"], grids["away"])
    out_poss = np.where(is_home, grids["away"], grids["home"])

    key_to_idx = {(m, w): i for i, (m, w) in enumerate(zip(rphases["match_id"], rphases["window_id"]))}
    # exposed so _load_raw_pitch_control can apply the IDENTICAL flip to the same phases,
    # instead of re-deriving a fresh (and potentially inconsistent) direction heuristic from a
    # file that has no separate home/away grids of its own to infer direction from.
    flip_by_key = {(m, w): bool(f) for m, w, f in
                   zip(rphases["match_id"], rphases["window_id"], flip)}
    return key_to_idx, in_poss, out_poss, n_rows_raw, n_cols_raw, flip_by_key


if SKIP_RECURRENCE_MAPS and not RECURRENCE_FEATURES:
    raw_key_to_idx, raw_in_poss, raw_out_poss, N_ROWS, N_COLS, raw_flip_by_key = {}, None, None, 0, 0, {}
    print("[recurrence] SKIP_RECURRENCE_MAPS=1 and RECURRENCE_FEATURES=0 -- skipped loading "
          "raw recurrence grids entirely (per-cluster recurrence PNGs will be skipped too).")
else:
    raw_key_to_idx, raw_in_poss, raw_out_poss, N_ROWS, N_COLS, raw_flip_by_key = _load_raw_recurrence()


# ------------------------------------------------------------------
# 1a-pc. Raw per-phase PITCH CONTROL map (Outputs\4th_Run\tactical_phases_pitch_control.parquet).
#     Distinct from the 8 pitch_control_mean/std/min/max/range/start/end/change SCALAR columns
#     already in phases.parquet (those get their own ordinary "pitch_control_scalar" group in
#     Section 2, unlike this one). This is a genuine per-cell grid: ONE map per phase (already
#     collapsed to the possession team's control fraction vs the opponent, not home/away, so
#     unlike recurrence_map there's no separate in/out-possession pair -- the complement is
#     just 1-grid), on the SAME 1x1m x_edges/y_edges as the recurrence maps.
#
#     PATCH (memory fix): this used to be called unconditionally at import time -- building a
#     full (n_phases, n_rows, n_cols) float64 array even when PITCH_CONTROL_FEATURES=0, i.e.
#     even when nothing downstream ever reads it. It's now only loaded when
#     PITCH_CONTROL_FEATURES=1 (the only thing that consumes raw_pitch_control is the
#     `if PITCH_CONTROL_FEATURES:` block in Section 1b). When skipped, the raw_pitch_control
#     variable is None and PC_N_ROWS/PC_N_COLS are 0 -- nothing else in the script touches
#     them unless PITCH_CONTROL_FEATURES is on. Also switched to float32 like the recurrence
#     grids above.
# ------------------------------------------------------------------
def _load_raw_pitch_control(flip_by_key):
    pc = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases_pitch_control.parquet"))
    pc = pc.assign(match_id=pc["match_id"].astype(str), window_id=pc["window_id"].astype(str))

    n_rows_pc = int(pc["row"].max()) + 1
    n_cols_pc = int(pc["col"].max()) + 1
    if (n_rows_pc, n_cols_pc) != (N_ROWS, N_COLS):
        print(f"WARNING: pitch-control grid is {n_rows_pc}x{n_cols_pc}, recurrence grid is "
              f"{N_ROWS}x{N_COLS} -- expected them to match (same x_edges/y_edges upstream).")

    key = pc["match_id"] + "|" + pc["window_id"]
    codes, uniques = pd.factorize(key)
    pc = pc.assign(_k=codes)
    n = len(uniques)
    pphases = (pc.drop_duplicates("_k")[["_k", "match_id", "window_id"]]
               .sort_values("_k").reset_index(drop=True))

    # PATCH: float32 instead of float64.
    grid = np.zeros((n, n_rows_pc, n_cols_pc), dtype=np.float32)
    grid[pc["_k"].to_numpy(), pc["row"].to_numpy(), pc["col"].to_numpy()] = pc["pitch_control_frac"].to_numpy()

    flip = np.array([flip_by_key.get((m, w), False)
                      for m, w in zip(pphases["match_id"], pphases["window_id"])])
    grid = grid.copy()
    grid[flip] = grid[flip][:, ::-1, ::-1]

    key_to_idx = {(m, w): i for i, (m, w) in enumerate(zip(pphases["match_id"], pphases["window_id"]))}
    return key_to_idx, grid, n_rows_pc, n_cols_pc


# PATCH: only pay for this (a full n_phases x n_rows x n_cols array) when it will actually be
# used. Previously this ran unconditionally regardless of PITCH_CONTROL_FEATURES.
if PITCH_CONTROL_FEATURES:
    pc_key_to_idx, raw_pitch_control, PC_N_ROWS, PC_N_COLS = _load_raw_pitch_control(raw_flip_by_key)
else:
    pc_key_to_idx, raw_pitch_control, PC_N_ROWS, PC_N_COLS = {}, None, 0, 0
    print("[pitch_control] PITCH_CONTROL_FEATURES=0 -- skipped loading the raw pitch-control "
          "grid entirely (this used to load unconditionally and was a major memory cost).")


def _area_weight_matrix(n_in, n_out):
    """(n_out, n_in) matrix redistributing n_in unit-width fine bins into n_out coarse bins
    by exact overlap area. Each column sums to 1, so `W @ grid` conserves total mass -- a
    phase's recurrence map still sums to 1 after downsampling, it's just spread over fewer,
    denser cells instead of thousands of mostly-empty ones."""
    edges_out = np.linspace(0, n_in, n_out + 1)
    W = np.zeros((n_out, n_in))
    for i in range(n_out):
        lo, hi = edges_out[i], edges_out[i + 1]
        j0, j1 = int(np.floor(lo)), int(np.ceil(hi))
        for j in range(max(0, j0), min(n_in, j1)):
            overlap = max(0.0, min(hi, j + 1) - max(lo, j))
            if overlap > 0:
                W[i, j] = overlap
    return W


def _downsample_grids(grids, n_rows_out, n_cols_out):
    """grids: (N, R, C) -> (N, n_rows_out, n_cols_out), mass-conserving block downsample."""
    n, r, c = grids.shape
    if (r, c) == (n_rows_out, n_cols_out):
        return grids
    Wr = _area_weight_matrix(r, n_rows_out)   # (n_rows_out, r)
    Wc = _area_weight_matrix(c, n_cols_out)   # (n_cols_out, c)
    step1 = np.einsum("ir,nrc->nic", Wr, grids)      # (N, n_rows_out, c)
    return np.einsum("nic,jc->nij", step1, Wc)        # (N, n_rows_out, n_cols_out)


# ------------------------------------------------------------------
# 1. Load + merge
# ------------------------------------------------------------------
phases = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases.parquet"))
pass_mv = pd.read_parquet(find_file("Outputs_4th_Run_pass_movement_scores.parquet"))
form_win = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases_formation_windows.parquet"))
pca_scores = pd.read_parquet(find_file("pca_scores.parquet"))

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
# 1b. Align raw recurrence grids to df's row order, and (if RECURRENCE_FEATURES=1)
#     flatten them into per-phase feature columns: rec_in_r{r}_c{c} / rec_out_r{r}_c{c},
#     384 columns total (192 cells x {in-possession, out-of-possession}). These are the
#     FULL 2D occupancy map, not a PCA summary of it -- clustering sees actual shape.
# ------------------------------------------------------------------
raw_row = np.array([raw_key_to_idx.get((m, w), -1) for m, w in zip(df["match_id"], df["window_id"])])
print(f"[recurrence] {(raw_row >= 0).sum()}/{len(df)} phases matched to raw recurrence maps "
      f"(grid {N_ROWS}x{N_COLS})")

if RECURRENCE_FEATURES:
    matched = raw_row >= 0
    safe_idx = np.where(matched, raw_row, 0)
    rows_out = RECURRENCE_GRID_ROWS or N_ROWS
    cols_out = RECURRENCE_GRID_COLS or N_COLS
    ds_in = _downsample_grids(raw_in_poss, rows_out, cols_out)   # no-op at native resolution
    ds_out = _downsample_grids(raw_out_poss, rows_out, cols_out)
    if RECURRENCE_SMOOTH_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        # sigma=0 on the phase axis (no smoothing across phases, only within each phase's grid)
        ds_in = gaussian_filter(ds_in, sigma=(0, RECURRENCE_SMOOTH_SIGMA, RECURRENCE_SMOOTH_SIGMA))
        ds_out = gaussian_filter(ds_out, sigma=(0, RECURRENCE_SMOOTH_SIGMA, RECURRENCE_SMOOTH_SIGMA))
        print(f"[recurrence] Gaussian-smoothed at native {rows_out}x{cols_out} resolution "
              f"(sigma={RECURRENCE_SMOOTH_SIGMA} cells) -- shape/cell-count unchanged")
    rec_in = ds_in[safe_idx].reshape(len(df), -1).astype(float)
    rec_out = ds_out[safe_idx].reshape(len(df), -1).astype(float)
    rec_in[~matched] = np.nan
    rec_out[~matched] = np.nan
    rec_cols_in = [f"rec_in_r{r:02d}_c{c:02d}" for r in range(rows_out) for c in range(cols_out)]
    rec_cols_out = [f"rec_out_r{r:02d}_c{c:02d}" for r in range(rows_out) for c in range(cols_out)]
    rec_df = pd.DataFrame(np.hstack([rec_in, rec_out]), columns=rec_cols_in + rec_cols_out, index=df.index)
    df = pd.concat([df, rec_df], axis=1)
    print(f"[recurrence] grid {N_ROWS}x{N_COLS} -> features at {rows_out}x{cols_out} "
          f"and added {rec_df.shape[1]} recurrence-cell features "
          f"({matched.sum()}/{len(df)} phases have real values, rest NaN -> median-imputed below)")

if PITCH_CONTROL_FEATURES:
    pc_row = np.array([pc_key_to_idx.get((m, w), -1) for m, w in zip(df["match_id"], df["window_id"])])
    pc_matched = pc_row >= 0
    pc_safe_idx = np.where(pc_matched, pc_row, 0)
    pc_rows_out = RECURRENCE_GRID_ROWS or PC_N_ROWS
    pc_cols_out = RECURRENCE_GRID_COLS or PC_N_COLS
    ds_pc = _downsample_grids(raw_pitch_control, pc_rows_out, pc_cols_out)
    if PITCH_CONTROL_SMOOTH_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        ds_pc = gaussian_filter(ds_pc, sigma=(0, PITCH_CONTROL_SMOOTH_SIGMA, PITCH_CONTROL_SMOOTH_SIGMA))
        print(f"[pitch_control] Gaussian-smoothed at {pc_rows_out}x{pc_cols_out} "
              f"(sigma={PITCH_CONTROL_SMOOTH_SIGMA} cells)")
    pc_feat = ds_pc[pc_safe_idx].reshape(len(df), -1).astype(float)
    pc_feat[~pc_matched] = np.nan
    pc_cols_names = [f"pcmap_r{r:02d}_c{c:02d}" for r in range(pc_rows_out) for c in range(pc_cols_out)]
    pc_feat_df = pd.DataFrame(pc_feat, columns=pc_cols_names, index=df.index)
    df = pd.concat([df, pc_feat_df], axis=1)
    print(f"[pitch_control] grid {PC_N_ROWS}x{PC_N_COLS} -> features at {pc_rows_out}x{pc_cols_out} "
          f"and added {pc_feat_df.shape[1]} pitch-control-cell features "
          f"({pc_matched.sum()}/{len(df)} phases have real values, rest NaN -> median-imputed below)")

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
    if "pitch_control" in lc:
        return "pitch_control_scalar"
    if lc.startswith("pcmap_"):
        return "pitch_control_map"
    if lc.startswith("rec_in_") or lc.startswith("rec_out_"):
        return "recurrence_map"
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

# recurrence_map / pitch_control_map: same trick, and for the same reason -- but the divisor
# must be the PEAK per-cell std (the busiest cell on the pitch), not the pooled/average std.
# Dividing by the AVERAGE column variance is a no-op: nominal = sum(var_i) / mean(var_i) ==
# n_features by simple algebra, REGARDLESS of how concentrated the real signal is (this is
# exactly what happened with recurrence_map's first attempt -- effective_dim came back ~= raw
# column count, unchanged from naive z-scoring). Dividing by the MAX per-cell std instead
# (mirroring how PC1 specifically, not an average PC, scales positional_pca) makes nominal
# reflect how many cells actually carry real signal relative to the busiest one -- a handful of
# frequently-occupied/controlled cells stay near variance~1, the thousands of near-empty cells
# collapse toward ~0, and nominal comes out close to the TRUE effective number of informative
# cells. Both raw spatial-grid groups get this treatment; every other group keeps normal
# per-column z-scoring above.
RAW_MAP_GROUPS = ("recurrence_map", "pitch_control_map")
for grp in RAW_MAP_GROUPS:
    grp_idx = np.where(groups_arr == grp)[0]
    if len(grp_idx):
        peak_sd = X_imp[:, grp_idx].std(axis=0).max()
        for i in grp_idx:
            M[:, i] = (X_imp[:, i] - X_imp[:, i].mean()) / peak_sd

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
nominal_per_group = {}
for g in np.unique(g_used):
    idx = np.where(g_used == g)[0]
    nominal = Mu[:, idx].var(axis=0).sum()
    nominal_per_group[g] = (nominal, len(idx))
    # group variance -> nominal ** (1 - alpha); nominal == n_features for z-scored groups,
    # and ~effective dimensionality for pooled-scaled groups (positional_pca, recurrence_map)
    # so their raw column COUNT doesn't equal their apparent size the way it does elsewhere
    col_scale[idx] = np.sqrt(nominal ** (1.0 - GROUP_ALPHA) / nominal)
Xw = base[:, use_mask] * col_scale

tot_var = Xw.var(axis=0).sum()
share = (pd.Series(Xw.var(axis=0), index=g_used).groupby(level=0).sum() / tot_var).sort_values(ascending=False)
print(f"\nclustering on {len(cluster_cols)} features; held out for validation: "
      f"{len(holdout_cols)} ({sorted(HOLDOUT_GROUPS)})")
print("effective dimensionality (pre-reweight variance sum) per group -- for pooled-scaled "
      "groups (positional_pca, recurrence_map) this is NOT the same as raw column count:")
for g, (nom, nf) in sorted(nominal_per_group.items(), key=lambda kv: -kv[1][0]):
    print(f"  {g:22s} n_features={nf:5d}  effective_dim~{nom:8.1f}")
print(f"GROUP_ALPHA={GROUP_ALPHA} -> share of clustering variance per group:")
print((share * 100).round(1).to_string())
print("If recurrence_map's share is still dominating, raise GROUP_ALPHA (e.g. 0.7-0.9) rather "
      "than reducing resolution.")

# Exempt positional_pca and recurrence_map from the pooled denoise -- both have diffuse
# variance spectra with no dominant components (a lossy offline PCA summary for the former;
# a full-resolution spatial map for the latter). Pooling either into one global PCA with
# everything else double-compresses it and starves it of its variance budget. Denoise the
# OTHER groups together, then concatenate these two back untouched (they keep the
# group_alpha weighting already applied above via col_scale) -- recurrence_map reaches
# k-means as the genuine raw 1x1m map, just correctly scaled, not PCA-reconstructed.
PASSTHROUGH_GROUPS = {"positional_pca", "recurrence_map", "pitch_control_map"}
pospca_mask = np.isin(g_used, list(PASSTHROUGH_GROUPS))
Xw_other = Xw[:, ~pospca_mask]
Xw_pospca = Xw[:, pospca_mask]
g_other = g_used[~pospca_mask]

pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=0)
X_pca_other = pca.fit_transform(Xw_other)
print(f"\nPCA denoise (excl. {sorted(PASSTHROUGH_GROUPS)}): {X_pca_other.shape[1]} components ({PCA_VAR:.0%} variance)")

X_pca = np.hstack([X_pca_other, Xw_pospca])

np.save(OUT / "X_pca_clustering_space.npy", X_pca)

Xrec_other = pca.inverse_transform(X_pca_other)
ss_tot = ((Xw_other - Xw_other.mean(0)) ** 2).sum(0)
ss_res = ((Xw_other - Xrec_other) ** 2).sum(0)
ret = pd.DataFrame({"tot": ss_tot, "res": ss_res}, index=g_other).groupby(level=0).sum()
ret["variance_retained"] = 1 - ret["res"] / ret["tot"]
# passthrough groups aren't in g_other (removed pre-groupby) -> add them back manually as
# 100% retained at THIS stage (any lossy compression for them happened earlier upstream, if any)
for grp in PASSTHROUGH_GROUPS:
    if grp in g_used:
        ret.loc[grp, "variance_retained"] = 1.0
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
# PATCH (memory fix): silhouette_score's default computes the FULL pairwise distance matrix
# over every sample -- O(n^2) memory, not O(n). At n ~= 42,000 phases that's
# 42000**2 * 8 bytes =~ 14 GB for ONE call, on the very first k=2 iteration, on top of
# everything else already in memory -- this was the actual OOM-kill, not the parquet loads
# or the recurrence/pitch-control grids (those are now gated off and this still died).
# sample_size subsamples SAMPLE_SIZE rows (with a fixed random_state for reproducibility
# across k) and computes silhouette on that subsample only -- O(sample_size^2) memory, an
# unbiased-ish estimate of the true score, standard practice at this n.
SILHOUETTE_SAMPLE_SIZE = int(os.environ.get("SILHOUETTE_SAMPLE_SIZE", 5000))
sil_scores, inertias, stab_mean, stab_std, all_labels = [], [], [], [], {}
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X_pca)
    sil_scores.append(silhouette_score(
        X_pca, labels,
        sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(labels)),
        random_state=0,
    ))
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
# 5. Assignments for every k
# ------------------------------------------------------------------
assign = df[["match_id", "sequence_id", "window_id", "team", "period"] + pc_cols].copy()
for k in ks:
    assign[f"cluster_k{k}"] = all_labels[k]
assign.to_parquet(OUT / "cluster_assignments_all_k.parquet", index=False)

# ------------------------------------------------------------------
# 6. Grouped feature importance (eta^2) - ALL features, held-out groups shown in grey
# ------------------------------------------------------------------
imp_names, imp_matrix = keep_cols2, M


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
    g.plot(kind="barh", ax=ax, color=["#bbbbbb" if i in HOLDOUT_GROUPS else color for i in g.index])
    ax.invert_yaxis()
    ax.set_xlabel("mean eta-squared (higher = more separates clusters)")
    ax.set_title(f"Which feature GROUPS separate the clusters ({k_label})\n[grey = held out, not used to cluster]",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / f"group_importance_{k_label}.png", dpi=130)
    plt.close(fig)
    return g


# ------------------------------------------------------------------
# 7. Per-cluster player recurrence maps for plotting. raw_in_poss / raw_out_poss /
#    raw_row / N_ROWS / N_COLS were already loaded in Section 1b and (if
#    RECURRENCE_FEATURES=1) are also the CLUSTERING features themselves.
# ------------------------------------------------------------------

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
        idx = raw_row[(labels == c) & (raw_row >= 0)]
        if len(idx):
            inp = raw_in_poss[idx].mean(axis=0)
            outp = raw_out_poss[idx].mean(axis=0)
        else:
            inp = np.zeros((N_ROWS, N_COLS)); outp = np.zeros((N_ROWS, N_COLS))
        maps[c] = (inp, outp, (labels == c).sum())
    n = len(maps)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.1 * n), squeeze=False)
    vmax = max(max(m[0].max(), m[1].max()) for m in maps.values())
    for i, c in enumerate(sorted(maps)):
        inp, outp, cnt = maps[c]
        for j, (name, grid) in enumerate((("IN POSSESSION", inp), ("OUT OF POSSESSION", outp))):
            ax = axes[i, j]
            _pitch(ax)
            im = ax.imshow(grid, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH], origin="lower",
                           cmap="hot", vmin=0, vmax=vmax, alpha=0.9, aspect="auto",
                           interpolation="nearest")
            ax.set_title(f"Cluster {c} (n={cnt})  ·  {name}", fontsize=9)
            if j == 1:
                plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f"Average recurrence map per cluster ({k_label}, raw phase averages)\n"
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
    "pass_completion_rate", "n_shots", "goals", "pressure_rate",
    "PC1", "PC2", "PC3",
] if c in df.columns]

for i_k, k_ in enumerate(FINAL_KS):
    lab = all_labels[k_]
    kl = f"k{k_}"
    g = group_importance(lab, kl, PALETTE[i_k % len(PALETTE)])
    print(f"\nGroup importance (mean eta^2) at {kl}:\n{g.round(3).to_string()}")
    if raw_in_poss is not None:
        make_recurrence_maps(lab, kl)
    else:
        print(f"[recurrence] Skipped cluster_recurrence_maps_{kl}.png "
              f"(SKIP_RECURRENCE_MAPS=1, raw grids not loaded)")

    p_ = df[profile_feats].copy()
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