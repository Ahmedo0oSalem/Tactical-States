"""
formation_recurrence_maps.py
=============================
Sums per-phase recurrence maps GROUPED BY FORMATION instead of by k-means cluster.

For every tactical phase that has a fitted `dominant_formation` (from
build_tactical_phases.py's fit_phase_formation), this pulls that phase's raw
recurrence map (in-possession + out-of-possession, attack-direction normalised --
same convention as cluster_pipeline_noball.py's _load_raw_recurrence) and SUMS
the per-cell recurrence_frac across every phase sharing that formation.

IMPORTANT -- per-match grid alignment:
    tactical_phases_recurrence.parquet is built with each match's grid sized to
    that match's OWN pitch_length/pitch_width (see build_grid_edges() /
    n_cols_1m/n_rows_1m in build_tactical_phases.py). Two matches on slightly
    different real pitch dimensions do NOT share the same cell grid. Summing
    raw cells across matches without accounting for this would silently
    misalign geometry. This script reprojects every phase's native-resolution
    grid onto one common TARGET grid (mass-conserving area-weighted resample,
    the same method already used for clustering-feature downsampling in
    cluster_pipeline_noball.py) before summing anything across matches.

Output:
    - one PNG per formation (in-possession + out-of-possession side by side)
    - a single parquet with the summed grids for further use
    - a formation x formation Pearson correlation matrix (in-possession,
      out-of-possession, and combined), as CSVs + a heatmap PNG. High pairwise
      correlation between two formations means their spatial occupancy
      patterns are effectively indistinguishable -- i.e. the formation label
      is not adding information beyond what a shared "shape" already implies.

Run:
    python formation_recurrence_maps.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

UPLOAD = Path(os.environ.get("UPLOAD_DIR", "."))
OUT = Path(os.environ.get("OUT_DIR", "./formation_recurrence_maps"))
OUT.mkdir(parents=True, exist_ok=True)

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
# Common grid every match's native-resolution map gets reprojected onto before
# summing across matches (see module docstring). ~1m cells at the nominal
# 105x68 pitch, same default as build_tactical_phases.py's RECURRENCE_N_COLS/ROWS.
TARGET_N_COLS = int(os.environ.get("TARGET_N_COLS", 105))
TARGET_N_ROWS = int(os.environ.get("TARGET_N_ROWS", 68))

# Which column to group formations by: "dominant_formation" (most specific,
# e.g. "433", "4231"), "formation_variant", or "formation_family" (coarsest,
# e.g. "back-4"). Set via env var if you want a coarser grouping.
FORMATION_COL = os.environ.get("FORMATION_COL", "dominant_formation")
# Skip formations with fewer than this many phases -- a formation seen in only
# a handful of phases produces a noisy, not-meaningfully-summed map.
MIN_PHASES_PER_FORMATION = int(os.environ.get("MIN_PHASES_PER_FORMATION", 20))


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
# Mass-conserving area-weighted regrid (ported from cluster_pipeline_noball.py) --
# used here to bring every match's native-resolution grid onto one common
# TARGET_N_ROWS x TARGET_N_COLS grid before summing across matches.
# ------------------------------------------------------------------
def _area_weight_matrix(n_in, n_out):
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


def _regrid(grid, n_rows_out, n_cols_out):
    """grid: (n_rows_in, n_cols_in) -> (n_rows_out, n_cols_out), mass-conserving."""
    r, c = grid.shape
    if (r, c) == (n_rows_out, n_cols_out):
        return grid
    Wr = _area_weight_matrix(r, n_rows_out)
    Wc = _area_weight_matrix(c, n_cols_out)
    return Wr @ grid @ Wc.T


# ------------------------------------------------------------------
# 1. Load phases (formation labels) + recurrence (raw per-cell maps)
# ------------------------------------------------------------------
print("Loading tactical_phases.parquet (formation labels) ...")
phases = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases.parquet"))
phases = phases.assign(match_id=phases["match_id"].astype(str), window_id=phases["window_id"].astype(str))
if FORMATION_COL not in phases.columns:
    raise SystemExit(f"'{FORMATION_COL}' not in tactical_phases.parquet columns: {list(phases.columns)}")
n_with_formation = phases[FORMATION_COL].notna().sum()
print(f"  {len(phases):,} phases total, {n_with_formation:,} ({100*n_with_formation/len(phases):.1f}%) "
      f"have a fitted '{FORMATION_COL}'")

print("Loading tactical_phases_recurrence.parquet (raw per-cell maps) ...")
rec = pd.read_parquet(find_file("Outputs_4th_Run_tactical_phases_recurrence.parquet"))
if "side" not in rec.columns:
    raise SystemExit("No `side` column in the recurrence parquet -- this is the old combined-map "
                      "file, not the per-team Step 6 output. Re-run build_tactical_phases.py.")
rec = rec.assign(match_id=rec["match_id"].astype(str), window_id=rec["window_id"].astype(str))

# native per-match grid extent (rows/cols can differ by match -- see module docstring)
per_match_extent = rec.groupby("match_id").agg(max_row=("row", "max"), max_col=("col", "max"))
n_distinct_extents = per_match_extent.drop_duplicates().shape[0]
print(f"  recurrence grid: {n_distinct_extents} distinct (rows,cols) extent(s) across "
      f"{rec['match_id'].nunique()} matches -- reprojecting every phase onto a common "
      f"{TARGET_N_ROWS}x{TARGET_N_COLS} grid before summing anything across matches.")

# ------------------------------------------------------------------
# 2. Build in-possession / out-of-possession grids per phase, attack-direction
#    normalised (identical logic/convention to cluster_pipeline_noball.py's
#    _load_raw_recurrence), REPROJECTED onto the common target grid per match.
# ------------------------------------------------------------------
key = rec["match_id"] + "|" + rec["window_id"]
codes, uniques = pd.factorize(key)
rec = rec.assign(_k=codes)
n_phases_rec = len(uniques)
rphases = (rec.drop_duplicates("_k")[["_k", "match_id", "window_id", "team", "period"]]
           .sort_values("_k").reset_index(drop=True))

in_poss = np.zeros((n_phases_rec, TARGET_N_ROWS, TARGET_N_COLS), dtype=np.float32)
out_poss = np.zeros((n_phases_rec, TARGET_N_ROWS, TARGET_N_COLS), dtype=np.float32)

print("Reprojecting each match's native-resolution grids onto the common grid ...")
for match_id, grp in rec.groupby("match_id"):
    n_rows_native = int(grp["row"].max()) + 1
    n_cols_native = int(grp["col"].max()) + 1
    m_phase_idx = grp["_k"].unique()
    m_k_to_local = {k: i for i, k in enumerate(m_phase_idx)}
    n_local = len(m_phase_idx)

    home_native = np.zeros((n_local, n_rows_native, n_cols_native), dtype=np.float32)
    away_native = np.zeros((n_local, n_rows_native, n_cols_native), dtype=np.float32)
    for side, arr in (("home", home_native), ("away", away_native)):
        d = grp[grp["side"] == side]
        li = d["_k"].map(m_k_to_local).to_numpy()
        arr[li, d["row"].to_numpy(), d["col"].to_numpy()] = d["recurrence_frac"].to_numpy()

    # regrid every phase's native grid onto the common target resolution
    home_common = np.stack([_regrid(home_native[i], TARGET_N_ROWS, TARGET_N_COLS) for i in range(n_local)])
    away_common = np.stack([_regrid(away_native[i], TARGET_N_ROWS, TARGET_N_COLS) for i in range(n_local)])

    local_rphases = rphases[rphases["match_id"] == match_id].set_index("_k").loc[m_phase_idx]
    is_home = (local_rphases["team"].to_numpy() == "home")
    global_idx = m_phase_idx

    ip = np.where(is_home[:, None, None], home_common, away_common)
    op = np.where(is_home[:, None, None], away_common, home_common)
    in_poss[global_idx] = ip
    out_poss[global_idx] = op

# attack-direction normalisation (same heuristic as cluster_pipeline_noball.py):
# rotate 180deg so the team in possession always attacks toward +x
def centroid_x(g):
    xs = (np.arange(TARGET_N_COLS) + 0.5) * PITCH_LENGTH / TARGET_N_COLS
    mass = g.sum(axis=(1, 2))
    return (g.sum(axis=1) * xs).sum(axis=1) / np.where(mass > 0, mass, np.nan)

# recompute home/away combined-grid centroids per phase for direction inference
# (reuse in_poss/out_poss: whichever is "home" side's actual grid depends on team,
#  so infer directly from is_home per phase using a fresh pass)
ddf = rphases[["match_id", "period", "team"]].copy()
# proxy: use in_poss as "this phase's own team" grid and out_poss as opponent's,
# midpoint of the two is a good direction proxy regardless of home/away identity
ddf["diff"] = 0.5 * (centroid_x(in_poss) + centroid_x(out_poss))
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
in_poss[flip] = in_poss[flip][:, ::-1, ::-1]
out_poss[flip] = out_poss[flip][:, ::-1, ::-1]

print(f"  built {n_phases_rec:,} attack-normalised, common-grid phase maps")

# ------------------------------------------------------------------
# 3. Attach formation labels, sum per formation
# ------------------------------------------------------------------
rphases = rphases.merge(
    phases[["match_id", "window_id", FORMATION_COL]],
    on=["match_id", "window_id"], how="left",
)
counts = rphases[FORMATION_COL].value_counts()
keep_formations = counts[counts >= MIN_PHASES_PER_FORMATION].index.tolist()
print(f"\nFormations with >= {MIN_PHASES_PER_FORMATION} phases (grouping by '{FORMATION_COL}'):")
for f in keep_formations:
    print(f"  {f!s:>10}  n={counts[f]:,}")
skipped = counts[counts < MIN_PHASES_PER_FORMATION]
if len(skipped):
    print(f"  (skipped {len(skipped)} formation(s) with < {MIN_PHASES_PER_FORMATION} phases: "
          f"{dict(skipped)})")

summed_rows = []
# keep the raw (rows, cols) grids around too -- the flat per-cell dataframe above is
# handy for saving/plotting, but the correlation step below wants each formation's
# map as one flat vector, so build that alongside instead of re-pivoting the dataframe.
mean_grids = {}   # formation -> (ip_mean, op_mean), each (TARGET_N_ROWS, TARGET_N_COLS)
for f in keep_formations:
    idx = rphases.index[rphases[FORMATION_COL] == f].to_numpy()
    ip_sum = in_poss[idx].sum(axis=0)
    op_sum = out_poss[idx].sum(axis=0)
    # per-phase MEAN (sum / n_phases) -- used for the correlation matrix below so that
    # formations with very different phase counts are compared on the same footing.
    # (Pearson correlation is scale-invariant per vector anyway, so this doesn't change
    # the correlation numbers vs. using the sums directly -- it just keeps the
    # intermediate values on a comparable, interpretable scale.)
    mean_grids[f] = (ip_sum / max(len(idx), 1), op_sum / max(len(idx), 1))
    for r in range(TARGET_N_ROWS):
        for c in range(TARGET_N_COLS):
            summed_rows.append({"formation": f, "n_phases": int(len(idx)),
                                 "row": r, "col": c,
                                 "in_possession_sum": float(ip_sum[r, c]),
                                 "out_of_possession_sum": float(op_sum[r, c])})

summed_df = pd.DataFrame(summed_rows)
summed_df.to_parquet(OUT / f"formation_recurrence_sums_{FORMATION_COL}.parquet", index=False)
print(f"\nSaved summed grids: {OUT / f'formation_recurrence_sums_{FORMATION_COL}.parquet'} "
      f"({len(summed_df):,} rows)")

# ------------------------------------------------------------------
# 3b. Formation x formation Pearson correlation matrices
# ------------------------------------------------------------------
# Idea: flatten each formation's (mean) recurrence map into one long vector of
# per-cell occupancy. Two formations whose maps are spatially near-identical
# will have a near-1.0 correlation between their flattened vectors, regardless
# of overall scale (phase count) -- i.e. despite having different formation
# labels, they occupy/control the pitch in the same way. That's the "formation
# label isn't adding information" attack: a high off-diagonal correlation
# means the two labels are close to redundant, spatially.
print("\nComputing formation x formation correlation matrices ...")


def _corr_matrix(vectors_by_formation):
    """vectors_by_formation: {formation: 1D np.array} -> (formations, corr_df)."""
    formations = list(vectors_by_formation.keys())
    mat = np.stack([vectors_by_formation[f].ravel() for f in formations])  # (n_formations, n_cells)
    corr_df = pd.DataFrame(np.corrcoef(mat), index=formations, columns=formations)
    return formations, corr_df


ip_vectors = {f: mean_grids[f][0] for f in keep_formations}
op_vectors = {f: mean_grids[f][1] for f in keep_formations}
# "combined": in-possession and out-of-possession maps concatenated into one vector
# per formation, so the correlation reflects the FULL phase signature (both what a
# team does on the ball and what the opponent does off it), not just one side of it.
combined_vectors = {f: np.concatenate([mean_grids[f][0].ravel(), mean_grids[f][1].ravel()])
                     for f in keep_formations}

_, corr_ip = _corr_matrix(ip_vectors)
_, corr_op = _corr_matrix(op_vectors)
formations, corr_combined = _corr_matrix(combined_vectors)

corr_ip.to_csv(OUT / f"formation_correlation_in_possession_{FORMATION_COL}.csv")
corr_op.to_csv(OUT / f"formation_correlation_out_of_possession_{FORMATION_COL}.csv")
corr_combined.to_csv(OUT / f"formation_correlation_combined_{FORMATION_COL}.csv")
print(f"  saved 3 correlation CSVs (in-possession / out-of-possession / combined) to {OUT}")

# quick console summary: highest off-diagonal pairs, i.e. the formations that look most
# like near-duplicates of each other spatially
off_diag = corr_combined.where(~np.eye(len(formations), dtype=bool))
pairs = (off_diag.stack()
         .rename("corr").reset_index()
         .rename(columns={"level_0": "formation_a", "level_1": "formation_b"}))
pairs = pairs[pairs["formation_a"] < pairs["formation_b"]]  # de-dup symmetric pairs
pairs = pairs.sort_values("corr", ascending=False)
print("\nTop 10 most spatially-similar formation pairs (combined in+out-of-possession corr):")
print(pairs.head(10).to_string(index=False))
print(f"\n  median off-diagonal correlation: {pairs['corr'].median():.3f}")
print(f"  mean off-diagonal correlation:   {pairs['corr'].mean():.3f}")

# ------------------------------------------------------------------
# 3c. Correlation matrix heatmaps (annotated, like the group-importance plots)
#     The self-correlation diagonal (always 1.00) is REMOVED from the plot:
#     cells are blanked (white), unlabeled, and the color scale is fit to the
#     off-diagonal range. The saved CSVs above still contain the full matrix.
# ------------------------------------------------------------------
def _plot_corr_heatmap(corr_df, title, out_path):
    n = len(corr_df)
    vals = corr_df.values.astype(float).copy()
    np.fill_diagonal(vals, np.nan)                 # remove self-correlations

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")                          # diagonal cells render blank

    vmax = np.nanmax(vals)
    fig, ax = plt.subplots(figsize=(0.55 * n + 2.5, 0.55 * n + 2.0))
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr_df.index, fontsize=8)
    # annotate every off-diagonal cell -- fine up to ~20-25 formations, like the ones in this dataset
    for i in range(n):
        for j in range(n):
            if i == j:
                continue                           # no "1.00" label on the diagonal
            val = vals[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.6 * vmax else "black", fontsize=6)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


_plot_corr_heatmap(corr_ip, f"Formation x formation correlation -- IN POSSESSION maps ({FORMATION_COL})",
                    OUT / f"formation_correlation_in_possession_{FORMATION_COL}.png")
_plot_corr_heatmap(corr_op, f"Formation x formation correlation -- OUT OF POSSESSION maps ({FORMATION_COL})",
                    OUT / f"formation_correlation_out_of_possession_{FORMATION_COL}.png")
_plot_corr_heatmap(corr_combined, f"Formation x formation correlation -- COMBINED maps ({FORMATION_COL})",
                    OUT / f"formation_correlation_combined_{FORMATION_COL}.png")
print(f"  saved 3 correlation heatmap PNGs to {OUT}")

# ------------------------------------------------------------------
# 3d. Restrict the PLOT to a hand-picked, de-duplicated set of formations.
#     (correlation matrices / summed parquet above still cover everything --
#     this only trims what gets drawn in the per-formation grid figure below,
#     dropping the formations whose spatial pattern was near-identical to one
#     already kept -- see the r > 0.8 clusters in the correlation heatmap.)
# ------------------------------------------------------------------
KEEP_FOR_PLOT = [
    "1423",       # kept over 2233 (r=0.86)
    "1234",       # kept over 2431, 1324 (r=0.81/0.81/0.86 cluster)
    "3511flat",   # kept over 343flat, metodo (r=0.94 vs metodo)
    "3511",       # kept over 451, 3421, 3331, 4141 (r=0.79-0.93 cluster)
    "1432", "pyramid", "wm", "32221", "3421flat", "541", "42211",  # ungrouped, unchanged
]
missing = [f for f in KEEP_FOR_PLOT if f not in keep_formations]
if missing:
    print(f"  WARNING: requested formation(s) not in keep_formations, skipping: {missing}")
keep_formations = [f for f in KEEP_FOR_PLOT if f in keep_formations]
print(f"\nPlotting reduced, de-duplicated set ({len(keep_formations)} formations): {keep_formations}")

# ------------------------------------------------------------------
# 4. Plot (per-formation recurrence maps, as before)
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


n_f = len(keep_formations)
fig, axes = plt.subplots(n_f, 2, figsize=(11, 3.1 * n_f), squeeze=False)
# PATCH: vmax used to be computed globally across every formation, so a formation with a
# handful of phases (peak sum ~1-2) got crushed to near-black next to a formation with
# thousands of phases (peak sum ~12+) on the same color scale -- it wasn't empty, just
# invisible. Each formation's row now gets its OWN vmax (shared between its two panels so
# in-possession vs out-of-possession stays comparable), so every formation's shape is
# visible regardless of its phase count. The raw summed values in the saved parquet are
# unaffected by this -- it's a plotting-only fix.
for i, f in enumerate(keep_formations):
    sub = summed_df[summed_df["formation"] == f]
    ip_grid = np.zeros((TARGET_N_ROWS, TARGET_N_COLS))
    op_grid = np.zeros((TARGET_N_ROWS, TARGET_N_COLS))
    ip_grid[sub["row"].to_numpy(), sub["col"].to_numpy()] = sub["in_possession_sum"].to_numpy()
    op_grid[sub["row"].to_numpy(), sub["col"].to_numpy()] = sub["out_of_possession_sum"].to_numpy()
    n_ph = int(sub["n_phases"].iloc[0])
    vmax_f = max(ip_grid.max(), op_grid.max())
    if vmax_f <= 0:
        vmax_f = 1.0
    for j, (name, grid) in enumerate((("IN POSSESSION", ip_grid), ("OUT OF POSSESSION", op_grid))):
        ax = axes[i, j]
        _pitch(ax)
        im = ax.imshow(grid, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH], origin="lower",
                       cmap="hot", vmin=0, vmax=vmax_f, alpha=0.9, aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"Formation {f} (n={n_ph:,} phases)  ·  {name}", fontsize=9)
        if j == 1:
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

fig.suptitle(f"Summed recurrence map per formation (grouped by '{FORMATION_COL}')\n"
             "team in possession attacks left -> right  ·  brighter = more summed occupancy",
             fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.97))
out_png = OUT / f"formation_recurrence_maps_{FORMATION_COL}.png"
plt.savefig(out_png, dpi=120)
plt.close(fig)
print(f"Saved plot: {out_png}")
# ------------------------------------------------------------------
# 5. Normalized plots (mean per phase, i.e. sum / n_phases) -- so total
#    mass is comparable across formations regardless of phase count.
#    In-possession and out-of-possession each saved as their own PNG.
#    (it reuses mean_grids, keep_formations, _pitch, PITCH_LENGTH/WIDTH,
#    TARGET_N_ROWS/COLS, summed_df's n_phases, FORMATION_COL, OUT).
# ------------------------------------------------------------------
def _plot_normalized(which, title_suffix, out_name):
    """which: 0 for in-possession, 1 for out-of-possession (index into mean_grids[f] tuple)."""
    n_f = len(keep_formations)
    fig, axes = plt.subplots(n_f, 1, figsize=(6, 3.1 * n_f), squeeze=False)
    # shared vmax across ALL formations here (not per-formation like section 4) --
    # that's the whole point: with mean-per-phase values, formations are now on
    # the same scale, so a single shared vmax is what actually makes them comparable.
    vmax_shared = max(mean_grids[f][which].max() for f in keep_formations)
    if vmax_shared <= 0:
        vmax_shared = 1.0
    for i, f in enumerate(keep_formations):
        grid = mean_grids[f][which]
        n_ph = int(summed_df.loc[summed_df["formation"] == f, "n_phases"].iloc[0])
        ax = axes[i, 0]
        _pitch(ax)
        im = ax.imshow(grid, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH], origin="lower",
                        cmap="hot", vmin=0, vmax=vmax_shared, alpha=0.9, aspect="auto",
                        interpolation="nearest")
        ax.set_title(f"Formation {f} (n={n_ph:,} phases)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.suptitle(f"Normalized (mean-per-phase) recurrence map per formation -- {title_suffix}\n"
                 "team in possession attacks left -> right  ·  same color scale across all formations",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = OUT / out_name
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


_plot_normalized(0, "IN POSSESSION", f"formation_recurrence_maps_normalized_in_possession_{FORMATION_COL}.png")
_plot_normalized(1, "OUT OF POSSESSION", f"formation_recurrence_maps_normalized_out_of_possession_{FORMATION_COL}.png")
# ------------------------------------------------------------------
# 6. Subsampling test -- is "patchiness" purely a sample-size artifact,
#    or does formation SHAPE itself matter beyond sample size?
#
# Logic: take a high-n formation (e.g. 1234, n=4,658) and repeatedly draw
# random subsamples of size n_low (matching a real low-n formation, e.g.
# pyramid, n=61) from its phases, rebuild the mean map from just that
# subsample, and see whether it now looks as patchy as the real low-n
# formation.
#
#   - If the subsampled high-n formation looks JUST AS patchy as the real
#     low-n formation at matched n -> patchiness is a pure sample-size
#     artifact, not a property of formation shape.
#   - If the subsampled high-n formation still looks noticeably SMOOTHER
#     than the real low-n formation at the same n -> shape itself
#     contributes something beyond sample size (Dr. Tamer's stronger claim).
#
# To make "how patchy" objective rather than just visual, this also computes
# a simple roughness/patchiness score per map: the fraction of a map's total
# mass that sits in "isolated" nonzero cells (cells with no nonzero neighbor).
# A smooth/filled map has almost no isolated cells; a speckled map has many.
# ------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: F811 (already imported above; kept for standalone use)

N_SUBSAMPLE_DRAWS = 20          # average over several random draws for a stable estimate
RNG_SEED = 0
# (high_n_formation, low_n_formation) pairs to test -- edit to match whichever
# pairs you want to check; these are the ones from the figure.
SUBSAMPLE_TEST_PAIRS = [
    ("1234", "pyramid"),
    ("3511flat", "wm"),
    ("1432", "32221"),
    ("3511", "3421flat"),
    ("541", "42211"),
]


def _patchiness_score(grid):
    """Fraction of the grid's total mass sitting in cells with no nonzero
    4-neighbor. 0 = perfectly smooth/filled, higher = more speckled/patchy."""
    nz = grid > 0
    if not nz.any() or grid.sum() <= 0:
        return np.nan
    has_nonzero_neighbor = np.zeros_like(nz)
    has_nonzero_neighbor[1:, :]  |= nz[:-1, :]
    has_nonzero_neighbor[:-1, :] |= nz[1:, :]
    has_nonzero_neighbor[:, 1:]  |= nz[:, :-1]
    has_nonzero_neighbor[:, :-1] |= nz[:, 1:]
    isolated = nz & ~has_nonzero_neighbor
    return float(grid[isolated].sum() / grid.sum())


def _mean_grid_from_indices(idx, which_poss):
    """which_poss: in_poss or out_poss array; idx: phase indices to average over."""
    if len(idx) == 0:
        return np.zeros((TARGET_N_ROWS, TARGET_N_COLS))
    return which_poss[idx].mean(axis=0)


def run_subsampling_test(pairs=SUBSAMPLE_TEST_PAIRS, n_draws=N_SUBSAMPLE_DRAWS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    results = []

    fig, axes = plt.subplots(len(pairs), 4, figsize=(16, 3.4 * len(pairs)))
    col_titles = ["A: real HIGH-n\n(full sample)", "B: real LOW-n\n(actual formation)",
                  "C: HIGH-n subsampled\nto match B's n (mean of draws)",
                  "Patchiness score\n(lower = smoother)"]

    for row_i, (hi_f, lo_f) in enumerate(pairs):
        hi_idx = rphases.index[rphases[FORMATION_COL] == hi_f].to_numpy()
        lo_idx = rphases.index[rphases[FORMATION_COL] == lo_f].to_numpy()
        n_hi, n_lo = len(hi_idx), len(lo_idx)
        if n_hi == 0 or n_lo == 0:
            print(f"  skipping {hi_f}/{lo_f}: missing from data (n_hi={n_hi}, n_lo={n_lo})")
            continue

        # A: full high-n map
        grid_hi_full = _mean_grid_from_indices(hi_idx, in_poss)
        score_hi_full = _patchiness_score(grid_hi_full)

        # B: real low-n map (as-is, this is literally what's in the original figure)
        grid_lo_real = _mean_grid_from_indices(lo_idx, in_poss)
        score_lo_real = _patchiness_score(grid_lo_real)

        # C: high-n formation, repeatedly subsampled down to n_lo phases,
        # averaged over n_draws random draws (both the grid itself, for display,
        # and the patchiness score, since the score is noisy per single draw).
        sub_grids = []
        sub_scores = []
        n_draw = min(n_lo, n_hi)
        for _ in range(n_draws):
            draw_idx = rng.choice(hi_idx, size=n_draw, replace=False)
            g = _mean_grid_from_indices(draw_idx, in_poss)
            sub_grids.append(g)
            sub_scores.append(_patchiness_score(g))
        grid_hi_sub = np.mean(sub_grids, axis=0)   # representative subsampled map for display
        score_hi_sub_mean = float(np.nanmean(sub_scores))
        score_hi_sub_std = float(np.nanstd(sub_scores))

        results.append({
            "high_n_formation": hi_f, "n_high": n_hi, "score_high_full": score_hi_full,
            "low_n_formation": lo_f, "n_low": n_lo, "score_low_real": score_lo_real,
            "score_high_subsampled_mean": score_hi_sub_mean,
            "score_high_subsampled_std": score_hi_sub_std,
            # How much of the real low-n formation's patchiness is "explained away"
            # by sample size alone. ~1.0 -> fully explained by sample size (Claim A).
            # Much less than 1.0 -> low-n formation is patchier than sample size alone
            # would predict -> some shape-specific effect on top of sample size (Claim B).
            "pct_patchiness_explained_by_n": (
                float(score_hi_sub_mean / score_lo_real) if score_lo_real else np.nan
            ),
        })

        grids_to_plot = [grid_hi_full, grid_lo_real, grid_hi_sub]
        vmax_row = max(g.max() for g in grids_to_plot if g.max() > 0) or 1.0
        for col_i, g in enumerate(grids_to_plot):
            ax = axes[row_i, col_i]
            _pitch(ax)
            ax.imshow(g, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH], origin="lower",
                      cmap="hot", vmin=0, vmax=vmax_row, alpha=0.9, aspect="auto",
                      interpolation="nearest")
            if col_i == 0:
                ax.set_title(f"{hi_f} (n={n_hi:,}) full\npatchiness={score_hi_full:.3f}", fontsize=8)
            elif col_i == 1:
                ax.set_title(f"{lo_f} (n={n_lo}) real\npatchiness={score_lo_real:.3f}", fontsize=8)
            else:
                ax.set_title(f"{hi_f} subsampled to n={n_draw}\n"
                             f"patchiness={score_hi_sub_mean:.3f}±{score_hi_sub_std:.3f}", fontsize=8)

        ax_text = axes[row_i, 3]
        ax_text.axis("off")
        pct = results[-1]["pct_patchiness_explained_by_n"]
        verdict = ("mostly SAMPLE SIZE" if 0.8 <= pct <= 1.25 else
                   ("real is PATCHIER than n alone predicts -> shape matters"
                    if pct < 0.8 else
                    "real is SMOOTHER than n alone predicts (unexpected)"))
        ax_text.text(0.02, 0.6,
                     f"{hi_f} full:        {score_hi_full:.3f}\n"
                     f"{hi_f}\u2192n={n_draw} (mean): {score_hi_sub_mean:.3f}\n"
                     f"{lo_f} real:       {score_lo_real:.3f}\n\n"
                     f"ratio (subsampled/real) = {pct:.2f}\n\n"
                     f"verdict: {verdict}",
                     fontsize=9, va="top", family="monospace")

    for j, t in enumerate(col_titles):
        axes[0, j].annotate(t, xy=(0.5, 1.28), xycoords="axes fraction",
                             ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Subsampling test: is patchiness explained by sample size alone?\n"
                 "(in-possession maps; ratio near 1.0 = yes, ratio << 1.0 = formation shape matters too)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = OUT / f"subsampling_patchiness_test_{FORMATION_COL}.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nSaved: {out_png}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / f"subsampling_patchiness_test_{FORMATION_COL}.csv", index=False)
    print(f"Saved: {OUT / f'subsampling_patchiness_test_{FORMATION_COL}.csv'}")
    print("\n" + results_df.to_string(index=False))
    return results_df


# ------------------------------------------------------------------
# 6 (FIXED). Subsampling test -- v2 patchiness metric.
#
# WHY THE OLD METRIC BROKE:
#   The old _patchiness_score() only counted a cell as "patchy" if it had
#   ZERO nonzero neighbors at all. A speckled map often has its hot pixels
#   sitting in small 2-4 cell clumps (not perfectly isolated single pixels),
#   so that metric scored genuinely patchy-looking maps (wm, 3421flat) as
#   0.000 -- perfectly smooth -- which is clearly wrong and produced the
#   nan / 21.29 ratios.
#
# THE FIX: use TOTAL VARIATION (mean absolute difference between every
# horizontally/vertically adjacent cell pair), normalized by the map's total
# mass. This measures how much a map's density changes from one cell to the
# next, on average -- a smooth/filled map has small, gradual differences
# between neighbors; a speckled map has large jumps (bright cell next to a
# dark one) regardless of whether the bright pixels are perfectly isolated
# or sit in small clumps. This is scale-invariant (normalized by total mass,
# same reasoning as the correlation normalization discussed earlier) and
# does not break on small clusters the way the isolated-cell count did.
#
# It only redefines _patchiness_score and reruns run_subsampling_test, so
# having both definitions is harmless; the later definition wins.
# ------------------------------------------------------------------

def _patchiness_score(grid):
    """Total-variation roughness, normalized by total mass.
    0 = perfectly smooth (constant-density), higher = more jagged/speckled.
    Robust to small clusters of hot pixels, unlike the old isolated-cell
    count -- it looks at EVERY adjacent-cell pair's difference, not just
    whether a cell has zero neighbors."""
    total = grid.sum()
    if total <= 0:
        return np.nan
    tv_h = np.abs(np.diff(grid, axis=1)).sum()   # horizontal neighbor differences
    tv_v = np.abs(np.diff(grid, axis=0)).sum()   # vertical neighbor differences
    return float((tv_h + tv_v) / total)


print("\n" + "=" * 70)
print("Rerunning subsampling test with fixed (total-variation) patchiness metric")
print("=" * 70)
run_subsampling_test()