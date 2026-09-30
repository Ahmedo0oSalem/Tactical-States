"""
pca_recurrence.py
=================
PCA over tactical phases, using each phase's per-team recurrence maps
(output of build_tactical_phases.py Step 6, the per-side version).

One phase = one feature vector:
    poss_opp (default): [ in-possession team's cells | out-of-possession team's cells ]
    home_away         : [ home cells | away cells ]
    combined          : [ both teams blended, cells ]

Grid shape
----------
The recurrence grid is sized per-match at ~1m/cell from that match's actual
pitch dimensions (see build_tactical_phases.py Step 6 / build_match_phases),
NOT a fixed 12x16 zone grid. This script infers (n_rows, n_cols) from the
data's own row/col columns rather than hardcoding it, and warns if different
matches in the same file have different grid extents (which would mean the
same cell (r, c) is a different physical spot in different matches, making a
shared PCA feature space spatially meaningless until resampled onto a common
grid).

Direction normalisation (important)
-----------------------------------
Teams swap ends at half-time, so raw maps are mirrored between periods and
PCA would spend its top components on "which half is this". The script
detects each match/period's attacking direction from the data (play sits
further toward a team's attacking goal while that team has the ball) and rotates the
maps 180 degrees where needed, so the team in possession always attacks
toward +x (drawn as left -> right). The detected directions are printed so
you can sanity-check them.

Usage:
    python3 pca_recurrence.py 'Outputs\\4th_Run\\tactical_phases_recurrence.parquet'
    python3 pca_recurrence.py <parquet> --features home_away --n-components 15
    python3 pca_recurrence.py <parquet> --match 10502            # one match only
    python3 pca_recurrence.py <parquet> --transform none --orient none

Outputs (in --outdir, default pca_output/):
    pca_scores.parquet          one row per phase: keys + PC1..PCk
    pca_loadings.parquet        one row per (block, row, col): PC1..PCk weights
    pca_explained_variance.csv  variance ratio per component
    pca_model.npz               mean + components (+ grid shape), to project new phases later
    pca_scree.png, pca_loadings.png, pca_scatter.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
TEAM_COLORS = {"home": "#1f77ff", "away": "#ff3b3b"}


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------
def load_grids(rec, n_rows=None, n_cols=None):
    """-> phases DataFrame (one row per phase), {'home','away'} -> (n, R, C), n_rows, n_cols.

    Grid shape is NOT fixed (see module docstring): it's inferred from the
    data's own row/col columns unless explicitly overridden. If different
    matches have different grid extents, that's a sign the underlying maps
    aren't spatially aligned across matches -- warn loudly rather than
    silently mixing incompatible cells into one PCA feature space.
    """
    if "side" not in rec.columns:
        raise SystemExit(
            "No `side` column: this is the old combined recurrence parquet. "
            "Re-run build_tactical_phases.py (per-team Step 6) first.")

    data_rows = int(rec["row"].max()) + 1
    data_cols = int(rec["col"].max()) + 1
    if n_rows is None:
        n_rows = data_rows
    if n_cols is None:
        n_cols = data_cols

    if int(rec["row"].max()) >= n_rows or int(rec["col"].max()) >= n_cols:
        raise SystemExit(
            f"Data contains row/col values that don't fit in a "
            f"{n_rows}x{n_cols} grid (row max={rec['row'].max()}, "
            f"col max={rec['col'].max()}). Pass the correct --n-rows/--n-cols "
            f"or omit them to auto-detect.")

    per_match_extent = rec.groupby("match_id").agg(
        max_row=("row", "max"), max_col=("col", "max"))
    if per_match_extent["max_row"].nunique() > 1 or per_match_extent["max_col"].nunique() > 1:
        print("WARNING: grid extent varies by match_id (different pitch dimensions "
              "produce different ~1m grids per Step 6):")
        print(per_match_extent)
        print(f"Proceeding with a shared {n_rows}x{n_cols} grid (zero-padded for "
              f"smaller matches), but cell (r, c) will NOT correspond to the same "
              f"physical pitch location across matches with different extents. "
              f"Results will be spatially meaningless unless you resample onto a "
              f"common grid first.\n")

    key = rec["match_id"].astype(str) + "|" + rec["window_id"].astype(str)
    codes, uniques = pd.factorize(key)
    rec = rec.assign(_k=codes)
    n = len(uniques)

    phases = (rec.drop_duplicates("_k")
                 [["_k", "match_id", "window_id", "sequence_id", "team",
                   "period", "frame_count"]]
                 .sort_values("_k").reset_index(drop=True))
    assert len(phases) == n and (phases["_k"].to_numpy() == np.arange(n)).all()

    grids = {}
    for side in ("home", "away"):
        d = rec[rec["side"] == side]
        g = np.zeros((n, n_rows, n_cols))
        g[d["_k"].to_numpy(), d["row"].to_numpy(), d["col"].to_numpy()] = \
            d["recurrence_frac"].to_numpy()
        grids[side] = g
    return phases.drop(columns="_k"), grids, n_rows, n_cols


# ------------------------------------------------------------------------------
# Attacking-direction detection + normalisation
# ------------------------------------------------------------------------------
def _centroid_x(g):
    n_cols = g.shape[2]
    xs = (np.arange(n_cols) + 0.5) * PITCH_LENGTH / n_cols
    mass = g.sum(axis=(1, 2))
    return (g.sum(axis=1) * xs).sum(axis=1) / np.where(mass > 0, mass, np.nan)


def detect_attack_direction(phases, grids):
    """+1 if the team in possession attacks toward +x in that phase, else -1.

    For each (match, period): compare where the play sits (mean x of all 20
    outfielders) in phases where HOME has the ball against phases where AWAY
    has it. Play drifts toward the goal the possessing team attacks, so if
    home attacks +x the margin is > 0.
    """
    mid = 0.5 * (_centroid_x(grids["home"]) + _centroid_x(grids["away"]))
    df = phases[["match_id", "period", "team"]].copy()
    df["diff"] = mid

    home_dir = {}
    rows = []
    for (m, p), grp in df.groupby(["match_id", "period"]):
        h = grp.loc[grp["team"] == "home", "diff"].mean()
        a = grp.loc[grp["team"] == "away", "diff"].mean()
        margin = h - a
        if np.isfinite(margin) and margin != 0:
            d, how = (1 if margin > 0 else -1), "detected"
        else:
            d, how = (1 if int(p) == 1 else -1), "ASSUMED (not enough data)"
        home_dir[(m, p)] = d
        rows.append((m, int(p), "right (+x)" if d == 1 else "left (-x)",
                     margin, how))

    print("\nDetected attacking direction of HOME team:")
    print(f"  {'match':<8}{'period':<8}{'home attacks':<14}{'margin (m)':<16}basis")
    for m, p, dirn, margin, how in rows:
        print(f"  {m:<8}{p:<8}{dirn:<14}{margin:<16.3f}{how}")
    print("  (margin near 0 = weak evidence -- check that row by eye; expect roughly +/- several metres)\n")

    hd = np.array([home_dir[(m, p)] for m, p in zip(phases["match_id"], phases["period"])])
    return np.where(phases["team"].to_numpy() == "home", hd, -hd)


def rotate_to_attack_right(grids, attack_dir):
    flip = attack_dir == -1
    out = {}
    for s, g in grids.items():
        g = g.copy()
        g[flip] = g[flip][:, ::-1, ::-1]     # 180-degree rotation of the pitch
        out[s] = g
    return out


# ------------------------------------------------------------------------------
# Feature matrix
# ------------------------------------------------------------------------------
def build_blocks(phases, grids, features):
    is_home = (phases["team"].to_numpy() == "home")[:, None, None]
    if features == "poss_opp":
        return [("IN POSSESSION", np.where(is_home, grids["home"], grids["away"])),
                ("OUT OF POSSESSION", np.where(is_home, grids["away"], grids["home"]))]
    if features == "home_away":
        return [("HOME", grids["home"]), ("AWAY", grids["away"])]
    if features == "combined":
        return [("BOTH TEAMS", 0.5 * (grids["home"] + grids["away"]))]
    raise ValueError(features)


def to_matrix(blocks, transform):
    parts = []
    for _, b in blocks:
        b = np.sqrt(b) if transform == "sqrt" else b
        parts.append(b.reshape(len(b), -1))
    return np.concatenate(parts, axis=1)


# ------------------------------------------------------------------------------
# PCA (plain SVD, no sklearn needed)
# ------------------------------------------------------------------------------
def run_pca(X):
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    ratio = var / var.sum()

    # Fix arbitrary SVD signs: largest |loading| in each component is positive.
    sign = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    sign[sign == 0] = 1
    Vt = Vt * sign[:, None]
    U = U * sign[None, :]
    scores = U * S
    return mean, Vt, var, ratio, scores


# ------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------
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


def plot_scree(ratio, out, n_show=20):
    k = min(n_show, len(ratio))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(1, k + 1), ratio[:k] * 100, color="#4c78a8")
    ax.set_xlabel("principal component"); ax.set_ylabel("variance explained (%)")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, k + 1), np.cumsum(ratio)[:k] * 100, color="#e45756", marker="o", ms=3)
    ax2.set_ylabel("cumulative (%)"); ax2.set_ylim(0, 100)
    ax.set_title("Scree plot")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(fig)


def plot_loadings(components, blocks, ratio, out, n_show, n_rows, n_cols):
    n_blocks = len(blocks)
    size = n_rows * n_cols
    n_show = min(n_show, len(components))
    fig, axes = plt.subplots(n_show, n_blocks, figsize=(5.6 * n_blocks, 3.1 * n_show),
                             squeeze=False)
    for i in range(n_show):
        comp = components[i]
        vmax = np.abs(comp).max()
        for b, (name, _) in enumerate(blocks):
            ax = axes[i, b]
            _pitch(ax)
            w = comp[b * size:(b + 1) * size].reshape(n_rows, n_cols)
            im = ax.imshow(w, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH], origin="lower",
                           cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.9,
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"PC{i + 1} ({ratio[i] * 100:.1f}%)  ·  {name}", fontsize=9)
            if b == n_blocks - 1:
                plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Component loadings  (red = more occupancy than average, blue = less)\n"
                 "team in possession attacks left → right", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out, dpi=110); plt.close(fig)


def plot_scatter(scores, phases, ratio, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c = phases["team"].map(TEAM_COLORS)
    axes[0].scatter(scores[:, 0], scores[:, 1], c=c, s=8, alpha=0.4, linewidths=0)
    axes[0].set_title("coloured by team in possession (blue home, red away)")
    for per, col in ((1, "#2ca02c"), (2, "#9467bd")):
        m = phases["period"].to_numpy() == per
        axes[1].scatter(scores[m, 0], scores[m, 1], c=col, s=8, alpha=0.4,
                        linewidths=0, label=f"period {per}")
    axes[1].legend(); axes[1].set_title("coloured by period")
    for ax in axes:
        ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
        ax.axhline(0, color="#999", lw=0.5); ax.axvline(0, color="#999", lw=0.5)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--features", choices=["poss_opp", "home_away", "combined"],
                    default="poss_opp")
    ap.add_argument("--transform", choices=["sqrt", "none"], default="sqrt",
                    help="sqrt (Hellinger) tames the heavy skew of occupancy shares")
    ap.add_argument("--orient", choices=["auto", "none"], default="auto")
    ap.add_argument("--match", help="restrict to one match_id")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="drop phases shorter than this many tracking frames")
    ap.add_argument("--n-components", type=int, default=20, help="components to save")
    ap.add_argument("--show", type=int, default=6, help="components to draw as maps")
    ap.add_argument("--n-rows", type=int, default=None,
                    help="override auto-detected grid rows (default: inferred from data)")
    ap.add_argument("--n-cols", type=int, default=None,
                    help="override auto-detected grid cols (default: inferred from data)")
    ap.add_argument("--outdir", default="pca_output")
    a = ap.parse_args()

    rec = pd.read_parquet(a.parquet)
    if a.match:
        rec = rec[rec["match_id"].astype(str) == str(a.match)]
    phases, grids, n_rows, n_cols = load_grids(rec, n_rows=a.n_rows, n_cols=a.n_cols)
    print(f"{len(phases):,} phases from {phases['match_id'].nunique()} match(es)")
    print(f"grid shape: {n_rows} x {n_cols} (auto-detected from row/col columns)"
          if a.n_rows is None and a.n_cols is None else
          f"grid shape: {n_rows} x {n_cols} (overridden via --n-rows/--n-cols)")

    # drop phases where a side has no valid data or that are too short
    ok = (grids["home"].sum(axis=(1, 2)) > 0.5) & (grids["away"].sum(axis=(1, 2)) > 0.5)
    ok &= phases["frame_count"].to_numpy() >= a.min_frames
    if (~ok).any():
        print(f"dropping {(~ok).sum():,} phases (missing a side or < {a.min_frames} frames)")
    phases = phases[ok].reset_index(drop=True)
    grids = {s: g[ok] for s, g in grids.items()}

    if a.orient == "auto":
        attack_dir = detect_attack_direction(phases, grids)
        grids = rotate_to_attack_right(grids, attack_dir)
        phases["attack_dir_original"] = attack_dir
        print(f"rotated {(attack_dir == -1).sum():,} phases so possession attacks →")

    blocks = build_blocks(phases, grids, a.features)
    X = to_matrix(blocks, a.transform)
    print(f"feature matrix: {X.shape[0]:,} phases x {X.shape[1]} features "
          f"({a.features}, transform={a.transform})")

    mean, comps, var, ratio, scores = run_pca(X)
    cum = np.cumsum(ratio)
    for t in (0.5, 0.8, 0.9, 0.95):
        print(f"  components for {int(t * 100)}% variance: {int(np.searchsorted(cum, t) + 1)}")
    print("  top components: " +
          "  ".join(f"PC{i + 1}={ratio[i] * 100:.1f}%" for i in range(min(8, len(ratio)))))

    k = min(a.n_components, len(ratio))
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    sc = phases.copy()
    for i in range(k):
        sc[f"PC{i + 1}"] = scores[:, i]
    sc.to_parquet(out / "pca_scores.parquet", index=False)

    size = n_rows * n_cols
    lo = []
    for b, (name, _) in enumerate(blocks):
        r, c = np.divmod(np.arange(size), n_cols)
        df = pd.DataFrame({"block": name, "row": r, "col": c})
        for i in range(k):
            df[f"PC{i + 1}"] = comps[i, b * size:(b + 1) * size]
        lo.append(df)
    pd.concat(lo, ignore_index=True).to_parquet(out / "pca_loadings.parquet", index=False)

    pd.DataFrame({"pc": np.arange(1, len(ratio) + 1), "variance": var,
                  "ratio": ratio, "cumulative": cum}).to_csv(
        out / "pca_explained_variance.csv", index=False)
    np.savez(out / "pca_model.npz", mean=mean, components=comps[:k],
             features=a.features, transform=a.transform,
             n_rows=n_rows, n_cols=n_cols)

    plot_scree(ratio, out / "pca_scree.png")
    plot_loadings(comps, blocks, ratio, out / "pca_loadings.png", a.show, n_rows, n_cols)
    plot_scatter(scores, phases, ratio, out / "pca_scatter.png")
    print(f"\nSaved everything to {out}/")


if __name__ == "__main__":
    main()