"""
animate_ball_trajectory.py
==========================
Step through a match's tactical phases in chronological order, one frame per
phase, showing where the BALL went. Same idea as animate_recurrence.py, but
for the ball: it combines the two ball parquets --

    *_ball_trajectory.parquet   12 x 16 ball-occupancy grid per phase
                                (match_id, window_id, row, col, recurrence_frac ...)
    *_ball_path.parquet         the actual (x, y) ball path per phase
                                (match_id, window_id, path_x, path_y, ...)

Modes
-----
    combined (default)  one pitch: occupancy heatmap + the ball path drawn on
                        top (time-coloured line, start dot, end star), plus
                        optional fading ghosts of the previous phases (--trail)
    split               left = occupancy heatmap, right = ball path
    heatmap             heatmap only (closest to animate_recurrence.py)
    path                path only

Usage:
    python animate_ball_trajectory.py traj.parquet path.parquet --match 10502
    python animate_ball_trajectory.py traj.parquet path.parquet --match 10502 --mode split
    python animate_ball_trajectory.py traj.parquet path.parquet --match 10502 --team home --trail 5
    python animate_ball_trajectory.py traj.parquet path.parquet --match 10502 --max-phases 60 --out test.gif

Or from Python:
    from animate_ball_trajectory import animate_ball
    animate_ball(traj_df, path_df, match_id="10502", out_path="10502_ball.mp4")

Notes
-----
* Phases are ordered by (period, running counter in window_id), same as
  animate_recurrence.py, so both animations line up phase-for-phase.
* Heatmap = share of the phase's valid ball frames spent in each cell
  (recurrence_frac, sums to 1 per phase). `upsample` only smooths the display.
* Row = y bin, col = x bin (0,0 = bottom-left of the pitch) -- verified against
  the raw path coordinates.
"""

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
N_ROWS = 12
N_COLS = 16

TEAM_COLORS = {"home": "#4da3ff", "away": "#ff6b6b"}
# Dark -> team colour -> near-white, so low cells fade into the pitch.
TEAM_CMAPS = {
    t: LinearSegmentedColormap.from_list(t, ["#0a1a0a", c, "#ffffff"])
    for t, c in {"home": "#1f77ff", "away": "#ff2d2d"}.items()
}
# Path colouring: start of the phase = dim, end = bright (shows direction).
PATH_CMAPS = {
    t: LinearSegmentedColormap.from_list(t + "_path", [dark, bright])
    for t, (dark, bright) in {"home": ("#0b3d91", "#e8f4ff"),
                              "away": ("#7a0c0c", "#fff0e8")}.items()
}


def draw_pitch(ax, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
    ax.add_patch(patches.Rectangle((0, 0), pitch_length, pitch_width,
                                   fill=False, color="white", lw=1.5))
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width],
            color="white", lw=1.2)
    ax.add_patch(patches.Circle((pitch_length / 2, pitch_width / 2), 9.15,
                                fill=False, color="white", lw=1.2))
    for x_goal in (0, pitch_length):
        direction = 1 if x_goal == 0 else -1
        ax.add_patch(patches.Rectangle(
            (x_goal, pitch_width / 2 - 20.15),
            direction * 16.5, 40.3, fill=False, color="white", lw=1.2))
        ax.add_patch(patches.Rectangle(
            (x_goal, pitch_width / 2 - 9.15),
            direction * 5.5, 18.3, fill=False, color="white", lw=1.2))
    ax.set_xlim(-2, pitch_length + 2)
    ax.set_ylim(-2, pitch_width + 2)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d6a2d")
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------
# data prep
# --------------------------------------------------------------------------
def _as_array(v):
    """Path columns are stored as JSON strings; accept lists/arrays too."""
    if isinstance(v, str):
        v = json.loads(v)
    return np.asarray(v, dtype=float)


def _phase_table(path_df, match_id, team=None):
    """One row per phase, chronological."""
    sub = path_df[path_df["match_id"].astype(str) == str(match_id)]
    if team is not None:
        sub = sub[sub["team"] == team]
    if sub.empty:
        raise ValueError(f"No rows for match_id={match_id!r}, team={team!r}")
    phases = sub.drop_duplicates(["match_id", "window_id"]).copy()
    phases["_ctr"] = phases["window_id"].str.split("_").str[-1].astype(int)
    phases = phases.sort_values(["period", "_ctr"]).reset_index(drop=True)
    return phases


def _build_grids(traj_df, phases, match_id, value_col, n_rows, n_cols):
    """(n_phases, n_rows, n_cols) occupancy grids aligned with `phases`."""
    sub = traj_df[traj_df["match_id"].astype(str) == str(match_id)]
    idx = {w: i for i, w in enumerate(phases["window_id"])}
    sub = sub[sub["window_id"].isin(idx)]
    g = np.zeros((len(phases), n_rows, n_cols))
    i = sub["window_id"].map(idx).to_numpy()
    g[i, sub["row"].to_numpy(), sub["col"].to_numpy()] = sub[value_col].to_numpy()
    return g


def _segments(x, y):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def animate_ball(traj_df, path_df, match_id, out_path=None, team=None,
                 mode="combined", value_col="recurrence_frac", fps=6, step=1,
                 max_phases=None, upsample=3, fixed_scale=True, trail=3,
                 dpi=110, fps_tracking=25.0, pitch_length=PITCH_LENGTH,
                 pitch_width=PITCH_WIDTH, n_rows=N_ROWS, n_cols=N_COLS):
    """One frame per phase.
       trail = number of previous phases kept as fading ghost paths (0 = off)."""
    if mode not in ("combined", "split", "heatmap", "path"):
        raise ValueError(f"unknown mode {mode!r}")

    phases = _phase_table(path_df, match_id, team)
    grids = _build_grids(traj_df, phases, match_id, value_col, n_rows, n_cols)

    if step > 1:
        keep = np.arange(0, len(phases), step)
        phases = phases.iloc[keep].reset_index(drop=True)
        grids = grids[keep]
    if max_phases:
        phases = phases.iloc[:max_phases].reset_index(drop=True)
        grids = grids[:max_phases]
    n = len(phases)

    xs = [_as_array(v) for v in phases["path_x"]]
    ys = [_as_array(v) for v in phases["path_y"]]

    def up(g):
        return zoom(g, upsample, order=1) if upsample > 1 else g

    pos = grids[grids > 0]
    vmax = np.percentile(pos, 99.5) if (fixed_scale and pos.size) else max(grids.max(), 1e-9)

    show_heat = mode in ("combined", "split", "heatmap")
    show_path = mode in ("combined", "split", "path")
    ext = [0, pitch_length, 0, pitch_width]

    if mode == "split":
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
        ax_heat, ax_path = axes
        fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.08, wspace=0.03)
        ax_heat.set_title("ball occupancy", color="#ccc", fontsize=10)
        ax_path.set_title("ball path", color="#ccc", fontsize=10)
    else:
        fig, ax0 = plt.subplots(figsize=(9.5, 6.2))
        ax_heat = ax_path = ax0
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08)
    fig.patch.set_facecolor("#111")

    for ax in {id(ax_heat): ax_heat, id(ax_path): ax_path}.values():
        draw_pitch(ax, pitch_length, pitch_width)

    # first-phase team decides initial colormap; update() swaps it per phase
    t0 = phases.loc[0, "team"]
    im = None
    if show_heat:
        im = ax_heat.imshow(up(grids[0]), extent=ext, origin="lower",
                            cmap=TEAM_CMAPS.get(t0, "viridis"), alpha=0.85,
                            aspect="auto", interpolation="nearest",
                            vmin=0, vmax=vmax, zorder=2)

    ghosts, main_lc, start_pt, end_pt = [], None, None, None
    if show_path:
        for _ in range(max(trail, 0)):
            lc = LineCollection([], linewidths=1.4, zorder=3)
            ax_path.add_collection(lc)
            ghosts.append(lc)
        main_lc = LineCollection([], linewidths=2.6, zorder=5)
        ax_path.add_collection(main_lc)
        start_pt = ax_path.scatter([], [], s=55, c="white", edgecolors="black",
                                   linewidths=1.0, zorder=6)
        end_pt = ax_path.scatter([], [], s=190, c="#ffd21f", marker="*",
                                 edgecolors="black", linewidths=0.9, zorder=7)

    title = fig.suptitle("", fontsize=12, color="white")
    progress = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="#aaa")

    def update(k):
        r = phases.iloc[k]
        team_k = r["team"]

        if show_heat:
            im.set_data(up(grids[k]))
            im.set_cmap(TEAM_CMAPS.get(team_k, "viridis"))

        if show_path:
            x, y = xs[k], ys[k]
            cmap = PATH_CMAPS.get(team_k, PATH_CMAPS["home"])
            if len(x) >= 2:
                segs = _segments(x, y)
                main_lc.set_segments(segs)
                main_lc.set_colors(cmap(np.linspace(0.15, 1, len(segs))))
            else:
                main_lc.set_segments([])
            start_pt.set_offsets([[x[0], y[0]]])
            end_pt.set_offsets([[x[-1], y[-1]]])

            for j, lc in enumerate(ghosts):          # j=0 -> most recent past phase
                kk = k - 1 - j
                if kk < 0 or len(xs[kk]) < 2:
                    lc.set_segments([])
                    continue
                lc.set_segments(_segments(xs[kk], ys[kk]))
                alpha = 0.55 * (1 - j / (len(ghosts) + 1))
                lc.set_color(TEAM_COLORS.get(phases.loc[kk, "team"], "white"))
                lc.set_alpha(alpha)

        dur = r["frame_count"] / fps_tracking if "frame_count" in r else len(xs[k]) / fps_tracking
        extra = ""
        if "path_length_m" in r and pd.notna(r["path_length_m"]):
            extra = f"  ·  {r['path_length_m']:.0f} m travelled"
        title.set_text(f"match {match_id}  ·  period {r['period']}  ·  "
                       f"{team_k.upper()} in possession  ·  {dur:.1f}s{extra}")
        title.set_color(TEAM_COLORS.get(team_k, "white"))
        progress.set_text(f"phase {k + 1}/{n}   ({r['window_id']})")

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)

    if out_path is None:
        out_path = f"ball_{match_id}_{mode}{'_' + team if team else ''}.mp4"
    writer = (PillowWriter(fps=fps) if out_path.lower().endswith(".gif")
              else FFMpegWriter(fps=fps, bitrate=2500))
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}  ({n} phases @ {fps} fps = {n / fps:.1f}s)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", help="*_ball_trajectory.parquet (12x16 occupancy grid)")
    ap.add_argument("path", help="*_ball_path.parquet (raw x/y path per phase)")
    ap.add_argument("--match", required=True)
    ap.add_argument("--team", choices=["home", "away"])
    ap.add_argument("--out")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--step", type=int, default=1, help="use every Nth phase")
    ap.add_argument("--max-phases", type=int)
    ap.add_argument("--upsample", type=int, default=3,
                    help="display upsampling factor; 1 = original 12x16 blocks")
    ap.add_argument("--trail", type=int, default=3,
                    help="previous phases kept as fading ghost paths (0 = off)")
    ap.add_argument("--mode", choices=["combined", "split", "heatmap", "path"],
                    default="combined")
    ap.add_argument("--per-phase-scale", action="store_true",
                    help="normalise heatmap brightness per phase instead of globally")
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args()

    traj = pd.read_parquet(a.trajectory)
    paths = pd.read_parquet(a.path)
    animate_ball(traj, paths, a.match, out_path=a.out, team=a.team, fps=a.fps,
                 step=a.step, max_phases=a.max_phases, upsample=a.upsample,
                 trail=a.trail, mode=a.mode, dpi=a.dpi,
                 fixed_scale=not a.per_phase_scale)
