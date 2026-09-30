"""
build_tactical_phases.py
========================

Combined pipeline:

  STEP 1 — Per-pass movement scores.
      For every completed pass/cross in every match, compute the mean and
      max Euclidean displacement of the attacking team's 10 outfielders
      between:
          x1 = this pass's snapshot
          x2 = the snapshot of the IMMEDIATELY NEXT event in the same
               sequence (any type: pass, carry, challenge, foul, shot ...)
      Writes pass_movement_scores.parquet + a 4-panel histogram.

  STEP 2 — Tactical phase segmentation.
      Within each PFF sequence, cut events into phases. A phase closes as
      soon as any of the following occurs (whichever fires first):

          1. Stoppage break — a null-sequence stoppage event (OUT, SUB,
             FOUL, kickoffs, END, ...) falls between the previous open-play
             event and this one. The phase closes at the last event
             BEFORE the stoppage; play resumes in a new phase when the
             next non-null-sequence event appears.
          2. threshold_mean  — mean_disp > THRESHOLD_MEAN
          3. threshold_max   — max_disp  > THRESHOLD_MAX
          4. threshold_both  — both of the above
          5. failed_pass     — any PA/CR with outcome != 'C'
          6. shot            — any SH
          7. max_passes      — MAX_WINDOW_PASSES completed passes accumulated
          8. max_time        — MAX_WINDOW_SECONDS elapsed since phase start
          9. sequence_end    — the sequence ended without any close trigger

      The closing event belongs to the closing phase (Reading 1). Segment
      bounds use the actual EVENT moment (eventTime), not the enclosing
      game event's start/end times (which can span tens of seconds).
      Aggregate the same features as the possession pipeline, but over
      each phase's frames. Formation is fit using the phase's own frames.

  STEP 6 — Per-team recurrence maps (home and away, 10 outfielders each).

Dependencies:
    possession_pipeline.py   (the corrected, game-clock-aware pipeline)

Run:
    python build_tactical_phases.py
"""

import bisect
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------------------------------------------------------------
# Import everything we reuse from the corrected possession pipeline.
# ------------------------------------------------------------------------------
from possession_pipeline import (
    # Config
    PROCESSED_DIR, EPV_GRID_PATH, XT_GRID_PATH,
    FORMATION_MIN_OUTFIELD_PLAYERS,
    # Grid loaders
    load_epv_grid, load_grid,
    # Per-frame series
    compute_pc_for_match, compute_pitch_control_grid_frame, compute_obso_for_match,
    compute_signed_epv_series, bucket_epv_by_second,
    compute_das_and_offball_xt_for_match, compute_signed_xt_series_vectorized,
    # Spatial + aggregation
    compute_all_spatial_features, agg_series,
    # Formation
    build_templates, resolve_goalkeepers,
    compute_frame_weights, build_formation_windows, build_formation_segments,
    derive_hierarchy, match_formation, get_orientation, attach_formation,
    # Frame lookup
    load_tracking, load_flat_events, frame_range_mask,
    _possession_frame_bounds, _game_clock_bounds, frame_at_time,
    # Event keys
    _get,
    GE_SEQUENCE_KEYS, GE_START_TIME_KEYS, GE_END_TIME_KEYS,
    GE_SETPIECE_KEYS, GE_VIDEO_MISSING_KEYS, GE_GAME_EVENT_ID_KEYS,
    GE_HOME_TEAM_KEYS,
    PE_TYPE_KEYS, PE_OUTCOME_KEYS, PE_ID_KEYS,
    # Possession-row builder (reused as-is)
    build_possession_row,
)

logger = logging.getLogger(__name__)
_T0 = time.time()


# ==============================================================================
# LOCAL EVENT-KEY EXTENSIONS (not exported by possession_pipeline)
# ==============================================================================
GE_EVENT_TIME_KEYS = ["eventTime", "event_time"]
GE_EVENT_TYPE_KEYS = ["gameEventType", "game_event_type"]


# ==============================================================================
# CONFIG
# ==============================================================================
NUM_MATCHES = None
# 1 = sequential (one match's tracking + grid data in memory at a time). This was 3, running
# up to 3 matches' full tracking arrays + per-cell recurrence/pitch-control grids in memory
# concurrently -- combined with the accumulate-everything-then-concat pattern in main() (see
# the streaming-writer note near RECURRENCE_PARQUET/PITCH_CONTROL_PARQUET below), that's what
# was driving memory usage. Raise this back up only once the streaming writes below are
# confirmed to keep steady-state memory flat across a full run.
NUM_WORKERS = 1

# ---- Step 1 outputs ---------------------------------------------------------
MOVEMENT_PARQUET = r"Outputs\4th_Run\pass_movement_scores.parquet"
MOVEMENT_FIGURE  = r"Outputs\4th_Run\pass_movement_scores_histogram.png"

RECOMPUTE_MOVEMENT_SCORES = True   # True: rerun extraction. False: reuse parquet.
MIN_VALID_OUTFIELDERS = 7
EXCLUDE_GK = True
PASS_EVENT_TYPES = ("PA", "CR")    # only passes and crosses are scored / counted

MAX_REASONABLE_PASS_DURATION = 10.0  # seconds; anything above is an artifact

# ---- Step 2 outputs ---------------------------------------------------------
OUTPUT_PATH = r"Outputs\4th_Run\tactical_phases.parquet"
FORMATION_WINDOWS_OUTPUT = r"Outputs\4th_Run\tactical_phases_formation_windows.parquet"

# ---- Step 6 outputs — recurrence maps ---------------------------------------
RECURRENCE_PARQUET = r"Outputs\4th_Run\tactical_phases_recurrence.parquet"
# Grid resolution the pitch is divided into (rows x cols).
# x in [0, pitch_length], y in [0, pitch_width] (non-centered convention).
# Cells are sized to be ~1m x 1m: n_cols/n_rows are set per-match from the
# actual pitch_length/pitch_width (see build_grid_edges call in
# build_match_phases), so these two constants are only the DEFAULTS used by
# build_grid_edges() and by the later get_phase_*_grid() lookup/plot helpers
# when no explicit n_cols/n_rows is passed. They assume a standard 105m x 68m
# pitch; if a match's actual pitch_length/pitch_width differs, that match's
# real map will still be ~1m cells (computed dynamically), just with a
# slightly different cell count than these defaults imply.
RECURRENCE_N_COLS = 105
RECURRENCE_N_ROWS = 68
# Each phase produces one map per side (10 outfielders each, GK excluded).
SIDES = ("home", "away")

# ---- Step 6b outputs — ball trajectory maps ----------------------------------
# Same grid as the player recurrence maps, but built from the ball's own
# tracking trace instead of player positions. Two artefacts per phase:
#   1. a grid occupancy map (which cells the ball passed through, and how
#      often), same shape/convention as compute_phase_recurrence's output,
#      so it can be plotted with the exact same pitch-heatmap code.
#   2. the raw ordered path (frame-by-frame x,y actually visited, in time
#      order) so the true trajectory (a line, with direction) can also be
#      drawn, not just a blurred heatmap.
BALL_TRAJECTORY_PARQUET = r"Outputs\4th_Run\tactical_phases_ball_trajectory.parquet"
BALL_PATH_PARQUET       = r"Outputs\4th_Run\tactical_phases_ball_path.parquet"

# ---- Step 6c outputs — per-phase pitch control maps -------------------------
# Same ~1m grid as the player recurrence maps (x_edges/y_edges from build_grid_edges), but each
# cell holds the PHASE-AVERAGED fraction of frames in which the team in possession's nearest
# player controlled that cell (a Voronoi-style pitch-control map, from possession_pipeline's
# compute_pitch_control_grid_frame), not an occupancy count. So unlike the recurrence/ball maps,
# a phase's pitch-control map does NOT sum to 1 -- each cell is independently a [0, 1] fraction.
PITCH_CONTROL_PARQUET = r"Outputs\4th_Run\tactical_phases_pitch_control.parquet"
# Evaluating a full grid_res x grid_res nearest-neighbour assignment on every single tracking
# frame in every phase is unnecessary for a phase-level average -- this subsamples frames
# WITHIN each phase (independent of possession_pipeline's own PC_DOWNSAMPLE, which only
# controls its match-level pc_df/home_control scalar series). 1 = every frame.
PHASE_PITCH_CONTROL_DOWNSAMPLE = 2
# Key(s) to look for the ball's tracking array under `track`. load_tracking()
# (in possession_pipeline) is assumed to expose the ball the same way it
# exposes players -- an (n_frames, 2) array of x,y in the same non-centered
# pitch convention used for {side}_xy. If your build uses a different key
# (e.g. track["ball"]["xy"], or a 3rd z-column), adjust BALL_XY_KEY /
# _get_ball_xy() below; everything downstream just needs an (n_frames, 2) array.
BALL_XY_KEY = "ball_xy"

# ---- Segmentation params ----------------------------------------------------
THRESHOLD_MEAN     = 2.0           # metres; mean displacement of 10 outfielders
THRESHOLD_MAX      = 4.0           # metres; single most-moved outfielder
MAX_WINDOW_PASSES  = 12
MAX_WINDOW_SECONDS = 30.0
MIN_PHASE_FRAMES   = 3             # skip phases with fewer tracking frames

# PFF tags open-play events with a `sequence` number and tags stoppages with
# `sequence = null`. We treat the null-sequence events as temporal break points:
# if one falls between two open-play events of the same sequence, the phase is
# closed at the last open-play event before the stoppage.
STOPPAGE_EVENT_TYPES = {
    "OUT", "SUB", "OFF", "ON",
    "FIRSTKICKOFF", "SECONDKICKOFF", "THIRD_KICKOFF", "FOURTHKICKOFF",
    "FOUL", "END", "VID",
    # "G" (post/bar rebound) is deliberately EXCLUDED — the ball stays in play.
}


# ==============================================================================
# LOGGING
# ==============================================================================
def log(msg, level="INFO"):
    print(f"[{time.time() - _T0:7.1f}s] [{level}] {msg}", flush=True)


# ==============================================================================
# GENERIC EVENT HELPERS
# ==============================================================================
_SUBDICT_KEYS = ("gameEvents", "initialTouch", "possessionEvents",
                 "fouls", "game_event", "possession_event")


def _flatten_event(rec):
    if not isinstance(rec, dict):
        return {}
    flat = dict(rec)
    for key in _SUBDICT_KEYS:
        sub = rec.get(key)
        if isinstance(sub, dict):
            flat.update(sub)
    return flat


def _load_events_raw(match_id, processed_dir):
    p = Path(processed_dir) / str(match_id) / "events.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("events", [raw]))
    if not isinstance(raw, list):
        raw = [raw]
    return [_flatten_event(r) for r in raw if isinstance(r, dict)]


def _event_moment(r):
    """Actual moment of a possession event. `eventTime` is per-event; the
    `startTime`/`endTime` fields refer to the ENCLOSING GAME EVENT, which can
    span tens of seconds (e.g. a keeper holding the ball before distributing).
    Use `eventTime` whenever you want the real chronological position."""
    t = _get(r, GE_EVENT_TIME_KEYS)
    if t is None:
        return None
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def _snapshot_map(players):
    """{player_id: (x, y)} for outfielders with finite positions."""
    m = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        if EXCLUDE_GK and p.get("positionGroupType") == "GK":
            continue
        pid = p.get("playerId")
        x, y = p.get("x"), p.get("y")
        if pid is None or x is None or y is None:
            continue
        try:
            m[pid] = (float(x), float(y))
        except (TypeError, ValueError):
            continue
    return m


def _side_players(ev, home_team):
    return ev.get("homePlayers" if home_team else "awayPlayers") or []


def _has_team_snapshot(ev):
    return bool(ev.get("homePlayers") or ev.get("awayPlayers"))


def _collect_break_times(events):
    """Sorted eventTime values of null-sequence stoppage events.

    These are PFF's own "play stopped" markers — the ball went out, a
    substitution occurred, a foul happened, a kickoff was taken, etc. They
    carry `sequence = null`, so they sit temporally between two open-play
    events of the surrounding sequence. We use them to split phases.
    """
    times = []
    for r in events:
        if _get(r, GE_SEQUENCE_KEYS) is not None:
            continue
        gtype = _get(r, GE_EVENT_TYPE_KEYS)
        if gtype not in STOPPAGE_EVENT_TYPES:
            continue
        t = _event_moment(r)
        if t is not None:
            times.append(t)
    return sorted(times)


def _has_break_between(break_times, t_prev, t_now):
    """True if any break time falls in the half-open interval (t_prev, t_now]."""
    if not break_times or t_prev is None or t_now is None:
        return False
    if t_now <= t_prev:
        return False
    i = bisect.bisect_right(break_times, t_prev)
    return i < len(break_times) and break_times[i] <= t_now


# ==============================================================================
# ============ STEP 1: PER-PASS MOVEMENT SCORES ===============================
# ==============================================================================
def _extract_movement_scores_for_match(match_id, processed_dir):
    events = _load_events_raw(match_id, processed_dir)
    if not events:
        return []

    groups = defaultdict(list)
    for r in events:
        seq = _get(r, GE_SEQUENCE_KEYS)
        if seq is not None:
            groups[seq].append(r)

    records = []
    for seq, seq_rows in groups.items():
        # Order events by their TRUE chronological moment (eventTime), not by
        # the enclosing game event's start_time.
        seq_rows = sorted(seq_rows, key=lambda r: _event_moment(r) or 0.0)

        for i, ev in enumerate(seq_rows):
            if _get(ev, PE_TYPE_KEYS) not in PASS_EVENT_TYPES:
                continue
            if _get(ev, PE_OUTCOME_KEYS) != "C":
                continue
            home_team = _get(ev, GE_HOME_TEAM_KEYS)
            if home_team is None:
                continue
            team = "home" if home_team else "away"

            t_start = _get(ev, GE_START_TIME_KEYS)
            t_end   = _get(ev, GE_END_TIME_KEYS)
            t_event = _event_moment(ev)

            base = {
                "match_id":            str(match_id),
                "sequence_id":         seq,
                "game_event_id":       _get(ev, GE_GAME_EVENT_ID_KEYS),
                "possession_event_id": _get(ev, PE_ID_KEYS),
                "period":              _get(ev, ["period"]),
                "team":                team,
                "setpiece_type":       _get(ev, GE_SETPIECE_KEYS),
                "event_type":          _get(ev, PE_TYPE_KEYS),
                "event_start_time":    float(t_start) if t_start is not None else None,
                "event_end_time":      float(t_end)   if t_end   is not None else None,
                "event_time":          t_event,
            }

            if i + 1 >= len(seq_rows):
                rec = dict(base); rec["status"] = "no_next_event"
                records.append(rec); continue
            next_ev = seq_rows[i + 1]

            if not _has_team_snapshot(ev) or not _has_team_snapshot(next_ev):
                rec = dict(base); rec["status"] = "missing_snapshot"
                records.append(rec); continue

            before = _snapshot_map(_side_players(ev, home_team))
            after  = _snapshot_map(_side_players(next_ev, home_team))
            if not before or not after:
                rec = dict(base); rec["status"] = "empty_snapshot"
                records.append(rec); continue

            common = set(before) & set(after)
            n_valid = len(common)
            if n_valid < MIN_VALID_OUTFIELDERS:
                rec = dict(base); rec["status"] = "few_valid"
                rec["n_valid"] = int(n_valid); records.append(rec); continue

            disps = np.empty(n_valid, dtype=np.float64)
            for j, pid in enumerate(common):
                x1, y1 = before[pid]
                x2, y2 = after[pid]
                disps[j] = float(np.hypot(x2 - x1, y2 - y1))

            # Enclosing-game-event span = the "touch duration" of this pass.
            touch_duration = None
            if t_start is not None and t_end is not None:
                d = float(t_end) - float(t_start)
                if 0.0 <= d <= MAX_REASONABLE_PASS_DURATION:
                    touch_duration = d

            # Time to the next event = eventTime(next) - eventTime(this).
            t_next = _event_moment(next_ev)
            time_to_next = None
            if t_event is not None and t_next is not None:
                time_to_next = float(t_next) - float(t_event)

            rec = dict(base)
            rec.update({
                "status":          "ok",
                "next_event_type": _get(next_ev, PE_TYPE_KEYS) or _get(next_ev, GE_EVENT_TYPE_KEYS),
                "next_event_time": t_next,
                "n_valid":         int(n_valid),
                "touch_duration":  touch_duration,
                "time_to_next":    time_to_next,
                "mean_disp":       float(disps.mean()),
                "median_disp":     float(np.median(disps)),
                "max_disp":        float(disps.max()),
                "min_disp":        float(disps.min()),
                "std_disp":        float(disps.std()),
                "n_above_1m":      int((disps > 1.0).sum()),
                "n_above_2m":      int((disps > 2.0).sum()),
                "n_above_5m":      int((disps > 5.0).sum()),
            })
            records.append(rec)
    return records


def _plot_movement_histogram(ok, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    _hist_panel(axes[0, 0], ok["mean_disp"],
                "mean player displacement per pass (m)",
                "Mean displacement per pass", bins=100)
    _hist_panel(axes[0, 1], ok["mean_disp"],
                "mean player displacement per pass (m)",
                "Mean displacement per pass", bins=100, log_y=True)

    _hist_panel(axes[1, 0], ok["max_disp"],
                "max player displacement per pass (m)",
                "Max displacement (single most-moved player)", bins=100)
    _hist_panel(axes[1, 1], ok["max_disp"],
                "max player displacement per pass (m)",
                "Max displacement (single most-moved player)",
                bins=100, log_y=True)

    _hist_panel(axes[2, 0], ok["time_to_next"],
                "seconds from this pass to next event",
                "Time to next event", bins=100)
    _hist_panel(axes[2, 1], ok["touch_duration"],
                "touch duration (s)",
                "Touch duration (enclosing game event, capped 10 s)", bins=80)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _hist_panel(ax, values, xlabel, title, bins=80, log_y=False):
    v = pd.Series(values).dropna()
    ax.hist(v, bins=bins, edgecolor="none")
    if log_y:
        ax.set_yscale("log")
        title = title + "  (log y)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=10)


def run_movement_extraction(match_ids):
    """Step 1: scan all matches, save parquet + histogram, return the lookup.

    Returns {possession_event_id: {"mean_disp": float, "max_disp": float}}.
    """
    log("=" * 70)
    log(f"STEP 1 — Per-pass movement scores ({len(match_ids)} matches)")
    log("=" * 70)

    parquet_path = Path(MOVEMENT_PARQUET)
    if parquet_path.exists() and not RECOMPUTE_MOVEMENT_SCORES:
        log(f"Reusing existing {MOVEMENT_PARQUET}")
        df = pd.read_parquet(parquet_path)
        ok = df[df["status"] == "ok"]
        log(f"  {len(ok):,} scored passes loaded")
        return {
            int(r["possession_event_id"]): {
                "mean_disp": float(r["mean_disp"]),
                "max_disp":  float(r["max_disp"]),
            }
            for _, r in ok.iterrows()
            if pd.notna(r["possession_event_id"])
        }

    all_records = []
    for i, mid in enumerate(match_ids, 1):
        t0 = time.time()
        recs = _extract_movement_scores_for_match(mid, PROCESSED_DIR)
        n_ok = sum(1 for r in recs if r.get("status") == "ok")
        all_records.extend(recs)
        log(f"  [{i:>2}/{len(match_ids)}] {mid}  scored={n_ok:>5}  "
            f"total={len(recs):>5}  ({time.time()-t0:.2f}s)")

    df = pd.DataFrame(all_records)
    log(f"\nTotal rows: {len(df):,}")
    for s, n in df["status"].value_counts().items():
        log(f"  {s:<18} {n:>7,}")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        log("No scorable passes; aborting.", level="WARN")
        return {}

    ok.to_parquet(MOVEMENT_PARQUET, index=False)
    log(f"Saved: {MOVEMENT_PARQUET}  ({len(ok):,} scored passes)")

    log("=== mean_disp distribution ===")
    for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        log(f"  p{int(q*100):>2}: {ok['mean_disp'].quantile(q):>7.3f} m")
    log(f"  mean:  {ok['mean_disp'].mean():>7.3f} m")
    log(f"  max:   {ok['mean_disp'].max():>7.3f} m")

    log("=== max_disp distribution ===")
    for q in (0.50, 0.75, 0.90, 0.95, 0.99):
        log(f"  p{int(q*100):>2}: {ok['max_disp'].quantile(q):>7.3f} m")

    log("=== time_to_next distribution ===")
    for q in (0.05, 0.50, 0.95, 0.99):
        log(f"  p{int(q*100):>2}: {ok['time_to_next'].quantile(q):>7.3f} s")

    _plot_movement_histogram(ok, MOVEMENT_FIGURE)
    log(f"Saved: {MOVEMENT_FIGURE}")

    return {
        int(r["possession_event_id"]): {
            "mean_disp": float(r["mean_disp"]),
            "max_disp":  float(r["max_disp"]),
        }
        for _, r in ok.iterrows()
        if pd.notna(r["possession_event_id"])
    }


# ==============================================================================
# ============ STEP 2: TACTICAL PHASE SEGMENTATION ============================
# ==============================================================================
def build_tactical_phases_from_events(events, track, score_lookup,
                                      threshold_mean, threshold_max):
    """Walk each sequence and produce phase dicts.

    Phase closure priority (first match wins):
        stoppage_break > failed_pass > shot
                       > threshold_both / threshold_mean / threshold_max
                       > max_passes > max_time > sequence_end
    """
    break_times = _collect_break_times(events)

    groups = defaultdict(list)
    for r in events:
        seq = _get(r, GE_SEQUENCE_KEYS)
        if seq is not None:
            groups[seq].append(r)

    phases = []
    phase_counter = [0]

    for seq, seq_rows in groups.items():
        # Order by the TRUE event moment, not the enclosing game event start.
        seq_rows = sorted(seq_rows, key=lambda r: _event_moment(r) or 0.0)
        if not seq_rows:
            continue

        home_team = _get(seq_rows[0], GE_HOME_TEAM_KEYS)
        if home_team is None:
            continue
        team = "home" if home_team else "away"

        current_window = []
        window_start_time = None
        n_completed_passes = 0
        prev_t = None

        for ev in seq_rows:
            t_now = _event_moment(ev)
            if t_now is None:
                continue

            # ---- 1. Stoppage break --------------------------------------
            # Checked BEFORE appending this event, so the closing phase does
            # not include the post-stoppage event.
            if current_window and _has_break_between(break_times, prev_t, t_now):
                ph = _finalize_phase(seq, team, current_window, track,
                                     "stoppage_break", phase_counter,
                                     threshold_mean, threshold_max)
                if ph is not None:
                    phases.append(ph)
                current_window = []
                window_start_time = None
                n_completed_passes = 0

            if not current_window:
                window_start_time = t_now
                n_completed_passes = 0

            current_window.append(ev)
            prev_t = t_now

            pe_type = _get(ev, PE_TYPE_KEYS)
            outcome = _get(ev, PE_OUTCOME_KEYS)
            reason = None

            # ---- 2/3/4. Threshold closures on completed passes ----------
            if pe_type in PASS_EVENT_TYPES and outcome == "C":
                n_completed_passes += 1
                score = score_lookup.get(_get(ev, PE_ID_KEYS))
                if score is not None:
                    mean_d = score["mean_disp"]
                    max_d  = score["max_disp"]
                    hit_mean = mean_d > threshold_mean
                    hit_max  = max_d  > threshold_max
                    if hit_mean and hit_max:
                        reason = "threshold_both"
                    elif hit_mean:
                        reason = "threshold_mean"
                    elif hit_max:
                        reason = "threshold_max"

            # ---- 5. Failed pass -----------------------------------------
            if reason is None and pe_type in PASS_EVENT_TYPES and outcome != "C":
                reason = "failed_pass"

            # ---- 6. Shot -------------------------------------------------
            if reason is None and pe_type == "SH":
                reason = "shot"

            # ---- 7. Max passes ------------------------------------------
            if reason is None and n_completed_passes >= MAX_WINDOW_PASSES:
                reason = "max_passes"

            # ---- 8. Max time (backstop for any event type) --------------
            if (reason is None and window_start_time is not None and
                    (t_now - window_start_time) > MAX_WINDOW_SECONDS):
                reason = "max_time"

            if reason is not None:
                ph = _finalize_phase(seq, team, current_window, track,
                                     reason, phase_counter,
                                     threshold_mean, threshold_max)
                if ph is not None:
                    phases.append(ph)
                current_window = []
                window_start_time = None
                n_completed_passes = 0

        # ---- 9. Sequence end -------------------------------------------
        if current_window:
            ph = _finalize_phase(seq, team, current_window, track,
                                 "sequence_end", phase_counter,
                                 threshold_mean, threshold_max)
            if ph is not None:
                phases.append(ph)

    return phases


def _finalize_phase(seq, team, window_rows, track, reason, counter,
                    threshold_mean, threshold_max):
    """Package a phase. Bounds use each event's TRUE moment (eventTime),
    not the enclosing game event's start/end times."""
    if not window_rows:
        return None

    # ----- Frame resolution (ID -> game clock -> legacy time) -----
    start_idx, end_idx = _possession_frame_bounds(track, window_rows)
    if start_idx is None or end_idx is None:
        gc_start, gc_end = _game_clock_bounds(track, window_rows)
        if gc_start is not None and gc_end is not None:
            start_idx, end_idx = gc_start, gc_end

    # ----- eventTime-based bounds (the fix) -----
    moments = [_event_moment(r) for r in window_rows]
    moments = [m for m in moments if m is not None]
    if not moments:
        return None
    start_sec = float(min(moments))
    end_sec   = float(max(moments))

    period_val = _get(window_rows[0], ["period"])
    period = int(period_val) if period_val is not None else 1

    if start_idx is None or end_idx is None:
        start_idx = frame_at_time(track, start_sec, period)
        end_idx   = frame_at_time(track, end_sec,   period)
    if start_idx is None or end_idx is None:
        return None

    start_frame = int(track["frame_num"][start_idx])
    end_frame   = int(track["frame_num"][end_idx])
    start_sec_period = float(track["elapsed"][start_idx])
    end_sec_period   = float(track["elapsed"][end_idx])

    # ----- game event dedup -----
    seen_ge, game_events = set(), []
    for r in window_rows:
        geid = _get(r, GE_GAME_EVENT_ID_KEYS)
        if geid not in seen_ge:
            seen_ge.add(geid)
            game_events.append(r)
    game_events.sort(key=lambda r: _event_moment(r) or 0.0)

    duration = end_sec - start_sec if (np.isfinite(start_sec) and np.isfinite(end_sec)) else np.nan

    counter[0] += 1
    return {
        "sequence_id": seq,
        "window_id": f"{seq}_{counter[0]}",
        "team": team,
        "period": period,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration": duration,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "start_sec_period": start_sec_period,
        "end_sec_period": end_sec_period,
        "setpiece_type": _get(game_events[0], GE_SETPIECE_KEYS) if game_events else None,
        "video_missing": any(bool(_get(r, GE_VIDEO_MISSING_KEYS, False)) for r in game_events),
        "possession_events": window_rows,
        "game_events": game_events,
        "_reason_closed": reason,
        "_threshold_mean": threshold_mean,
        "_threshold_max": threshold_max,
    }


# ==============================================================================
# STEP 6 — PHASE-LEVEL PER-TEAM RECURRENCE MAPS
# ==============================================================================
# For every tactical phase:
#   1. Use EVERY tracking frame from start_frame through end_frame.
#   2. Select the 10 active outfield players from each team, excluding GKs.
#   3. Bin each team's player positions into the SAME 12 x 16 pitch grid,
#      SEPARATELY for home and away.
#   4. Average the occupancy over all valid player-frame observations.
#
# Result:
#       ONE tactical phase -> TWO 12 x 16 recurrence maps (home + away),
#                             2 x 192 = 384 rows
#
# A player does NOT need to make a pass/event to contribute. Their tracking
# position is included at every valid tracking frame in the phase.
#
# Columns:
#   team                      team IN POSSESSION for the phase
#   side                      whose players this map describes ("home"/"away")
#   recurrence_count          that side's player-frame observations in the cell
#   recurrence_frac           count / that side's total  -> sums to 1 per
#                             (phase, side)
#   recurrence_frac_combined  count / both sides' total  -> sums to 1 per
#                             phase; summing it over both sides reproduces
#                             the old blended 20-player map
# ==============================================================================

def build_grid_edges(pitch_length, pitch_width,
                     n_cols=RECURRENCE_N_COLS, n_rows=RECURRENCE_N_ROWS):
    """Non-centered pitch convention: x in [0, pitch_length], y in [0, pitch_width].

    This matches possession_pipeline's own tracking coordinate system (it
    shifts raw centered coordinates by +pitch_length/2 / +pitch_width/2 at
    load time -- see its "Applying coordinate shift" log line), NOT a
    centered [-L/2, L/2] convention. Using the wrong convention here doesn't
    error, it silently clips almost every real coordinate into the last
    row/column bin -- verified against a real run: rows 0-4 and columns 0-7
    were entirely empty, with row 11 and column 15 alone absorbing 72% and
    62% of all recurrence respectively. Confirm this against your own
    possession_pipeline if its coordinate convention ever changes.
    """
    x_edges = np.linspace(0.0, pitch_length, n_cols + 1)
    y_edges = np.linspace(0.0, pitch_width, n_rows + 1)
    return x_edges, y_edges


def _select_phase_outfield_players(track, frame_idx, side, goalkeepers,
                                    n_players=10):
    """Select the 10 outfield players with the greatest valid tracking
    coverage inside THIS phase. Goalkeepers are excluded first.

    `load_tracking()` contains the player axes for the whole match, so this
    phase-local selection prevents substitutes from being treated as active
    players merely because they appear somewhere in the match.
    """
    xy = track[f"{side}_xy"][frame_idx]
    ids = track[f"{side}_ids"]
    gk_ids = {str(x) for x in (goalkeepers.get(side) or set())}

    candidates = []
    for j, pid in enumerate(ids):
        if str(pid) in gk_ids:
            continue

        pos = xy[:, j, :]
        valid = np.all(np.isfinite(pos), axis=1)
        n_valid = int(valid.sum())
        if n_valid > 0:
            candidates.append((n_valid, j, pid))

    candidates.sort(key=lambda x: (-x[0], str(x[2])))
    return [j for _, j, _ in candidates[:n_players]]


def compute_phase_recurrence(match_id, ph, track, frame_idx,
                             x_edges, y_edges, goalkeepers):
    """Two recurrence maps per phase: one for the 10 home outfielders and one
    for the 10 away outfielders (goalkeepers excluded), on the same grid.

    Returns row dicts for cells that were actually occupied (zero-count cells are omitted --
    get_phase_recurrence_grid() reconstructs them as 0.0 by default, so this is lossless and
    keeps memory bounded on matches with many phases).
    """
    n_rows_grid = len(y_edges) - 1
    n_cols_grid = len(x_edges) - 1
    n_cells = n_rows_grid * n_cols_grid

    if len(frame_idx) == 0:
        return []

    counts = {}
    n_players = {}
    for side in SIDES:
        idx = _select_phase_outfield_players(
            track, frame_idx, side, goalkeepers, n_players=10)
        n_players[side] = len(idx)

        if not idx:
            counts[side] = np.zeros(n_cells, dtype=np.int64)
            continue

        # (n_frames, n_players, 2) -> (n_frames * n_players, 2)
        pos = track[f"{side}_xy"][frame_idx][:, idx, :].reshape(-1, 2)
        pos = pos[np.all(np.isfinite(pos), axis=1)]

        if len(pos) == 0:
            counts[side] = np.zeros(n_cells, dtype=np.int64)
            continue

        c = np.searchsorted(x_edges, pos[:, 0], side="right") - 1
        r = np.searchsorted(y_edges, pos[:, 1], side="right") - 1
        c = np.clip(c, 0, n_cols_grid - 1)
        r = np.clip(r, 0, n_rows_grid - 1)
        counts[side] = np.bincount(r * n_cols_grid + c,
                                   minlength=n_cells).astype(np.int64)

    side_total = {s: int(counts[s].sum()) for s in SIDES}
    total_all = side_total["home"] + side_total["away"]
    if total_all == 0:
        return []

    if n_players["home"] < 10 or n_players["away"] < 10:
        logger.warning(
            f"[{match_id}] phase {ph['window_id']} has only "
            f"{n_players['home']} home + {n_players['away']} away tracked outfielders"
        )

    # MEMORY FIX: only emit cells the players actually occupied (cnt > 0). A ~1m grid over a
    # full pitch has ~7,000 cells, but a single short phase only ever touches a small fraction
    # of them -- the old code built a Python dict for EVERY cell of EVERY phase (all-zero cells
    # included), which is what was blowing up memory across a match with many phases.
    # get_phase_recurrence_grid() already reconstructs the full grid by starting from zeros and
    # only placing the rows that exist, so dropping true-zero cells here is lossless.
    rows = []
    for side in SIDES:
        cnt = counts[side]
        st = side_total[side]
        frac_side = cnt / st if st > 0 else np.zeros(n_cells)
        frac_comb = cnt / total_all

        nz = np.nonzero(cnt)[0]
        if len(nz) == 0:
            continue
        row_idx, col_idx = np.divmod(nz, n_cols_grid)
        side_df = pd.DataFrame({
            "match_id": str(match_id),
            "window_id": ph["window_id"],
            "sequence_id": ph["sequence_id"],
            "team": ph["team"],                       # team in possession
            "side": side,                             # whose players
            "period": ph["period"],
            "row": row_idx.astype(int),
            "col": col_idx.astype(int),
            "frame_count": int(len(frame_idx)),
            "n_home_outfielders": int(n_players["home"]),
            "n_away_outfielders": int(n_players["away"]),
            "n_outfielders": int(n_players["home"] + n_players["away"]),
            "n_side_outfielders": int(n_players[side]),
            "valid_player_frame_count": int(total_all),
            "side_valid_player_frame_count": int(st),
            "recurrence_count": cnt[nz].astype(int),
            "recurrence_frac": frac_side[nz].astype(float),
            "recurrence_frac_combined": frac_comb[nz].astype(float),
        })
        rows.extend(side_df.to_dict("records"))
    return rows


def validate_phase_recurrence(recurrence_df):
    """Each (phase, side) map should sum to 1; each phase's combined map too."""
    if recurrence_df.empty:
        logger.warning("Recurrence dataframe is empty.")
        return

    per_side = (recurrence_df
                .groupby(["match_id", "window_id", "side"], as_index=False)
                ["recurrence_frac"].sum())
    empty_sides = int((per_side["recurrence_frac"] == 0).sum())
    ok_sides = per_side[per_side["recurrence_frac"] > 0]
    bad_side = ~np.isclose(ok_sides["recurrence_frac"], 1.0, atol=1e-6)

    combined = (recurrence_df
                .groupby(["match_id", "window_id"], as_index=False)
                ["recurrence_frac_combined"].sum())
    bad_comb = ~np.isclose(combined["recurrence_frac_combined"], 1.0, atol=1e-6)

    logger.info("=== RECURRENCE MAP SANITY CHECK (per side) ===")
    logger.info(f"  Phases: {len(combined):,}  |  side maps: {len(per_side):,}")
    logger.info(f"  Side maps with no valid observations: {empty_sides:,}")
    logger.info(f"  Side maps with sum != 1: {int(bad_side.sum()):,}")
    logger.info(f"  Combined maps with sum != 1: {int(bad_comb.sum()):,}")
    for s in SIDES:
        n = recurrence_df[recurrence_df["side"] == s]["n_side_outfielders"]
        logger.info(f"  {s}: mean outfielders/phase = {n.mean():.2f}, min = {n.min()}")


# ==============================================================================
# STEP 6b — PER-PHASE BALL TRAJECTORY MAPS
# ==============================================================================
def _get_ball_xy(track):
    """(n_frames, 2) array of ball x,y in the same pitch convention as
    track[f"{side}_xy"]. Tries a couple of common shapes defensively; adjust
    here if your possession_pipeline.load_tracking() exposes it differently.
    """
    if BALL_XY_KEY in track:
        arr = np.asarray(track[BALL_XY_KEY])
        return arr[:, :2] if arr.ndim == 2 and arr.shape[1] >= 2 else None
    ball = track.get("ball")
    if isinstance(ball, dict) and "xy" in ball:
        arr = np.asarray(ball["xy"])
        return arr[:, :2] if arr.ndim == 2 and arr.shape[1] >= 2 else None
    return None


def compute_phase_ball_trajectory(match_id, ph, track, frame_idx, x_edges, y_edges):
    """Ball-only counterpart of compute_phase_recurrence.

    Returns:
      grid_rows -> list of dicts, one per OCCUPIED grid cell (zero-count cells omitted --
                   get_phase_ball_grid() reconstructs them as 0.0 by default), same schema
                   style as the player maps: recurrence_count / recurrence_frac (sums to 1
                   per phase across the emitted nonzero cells).
      path_row  -> a single dict for this phase holding the ordered list of
                   (x, y) the ball actually visited, in time order, plus
                   simple trajectory summary stats (path length, net
                   displacement, straightness). None if no valid ball frames.
    """
    n_rows_grid = len(y_edges) - 1
    n_cols_grid = len(x_edges) - 1
    n_cells = n_rows_grid * n_cols_grid

    if len(frame_idx) == 0:
        return [], None

    ball_xy_all = _get_ball_xy(track)
    if ball_xy_all is None:
        logger.warning(f"[{match_id}] phase {ph['window_id']}: no ball tracking array found "
                       f"(looked for track['{BALL_XY_KEY}']) -- skipping ball trajectory")
        return [], None

    pos = ball_xy_all[frame_idx]
    valid = np.all(np.isfinite(pos), axis=1)
    pos_valid = pos[valid]

    if len(pos_valid) == 0:
        return [], None

    c = np.searchsorted(x_edges, pos_valid[:, 0], side="right") - 1
    r = np.searchsorted(y_edges, pos_valid[:, 1], side="right") - 1
    c = np.clip(c, 0, n_cols_grid - 1)
    r = np.clip(r, 0, n_rows_grid - 1)
    counts = np.bincount(r * n_cols_grid + c, minlength=n_cells).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return [], None
    frac = counts / total

    # MEMORY FIX: same sparsification as compute_phase_recurrence -- only emit cells the ball
    # actually passed through (counts > 0) instead of a dict for all ~7,000 grid cells per
    # phase. get_phase_ball_grid() already defaults missing cells to 0, so this is lossless.
    nz = np.nonzero(counts)[0]
    if len(nz) == 0:
        grid_rows = []
    else:
        row_idx, col_idx = np.divmod(nz, n_cols_grid)
        grid_rows = pd.DataFrame({
            "match_id": str(match_id),
            "window_id": ph["window_id"],
            "sequence_id": ph["sequence_id"],
            "team": ph["team"],              # team in possession during this phase
            "period": ph["period"],
            "row": row_idx.astype(int),
            "col": col_idx.astype(int),
            "frame_count": int(len(frame_idx)),
            "valid_ball_frame_count": total,
            "recurrence_count": counts[nz].astype(int),
            "recurrence_frac": frac[nz].astype(float),
        }).to_dict("records")

    # Ordered path + simple trajectory descriptors, for drawing an actual
    # line (with direction) rather than only a blurred occupancy heatmap.
    step_vecs = np.diff(pos_valid, axis=0)
    path_length = float(np.linalg.norm(step_vecs, axis=1).sum()) if len(pos_valid) > 1 else 0.0
    net_disp = float(np.linalg.norm(pos_valid[-1] - pos_valid[0])) if len(pos_valid) > 1 else 0.0
    straightness = float(net_disp / path_length) if path_length > 0 else np.nan

    path_row = {
        "match_id": str(match_id),
        "window_id": ph["window_id"],
        "sequence_id": ph["sequence_id"],
        "team": ph["team"],
        "period": ph["period"],
        "n_points": int(len(pos_valid)),
        "path_x": pos_valid[:, 0].astype(float).tolist(),
        "path_y": pos_valid[:, 1].astype(float).tolist(),
        "start_x": float(pos_valid[0, 0]), "start_y": float(pos_valid[0, 1]),
        "end_x": float(pos_valid[-1, 0]), "end_y": float(pos_valid[-1, 1]),
        "path_length_m": path_length,
        "net_displacement_m": net_disp,
        "straightness": straightness,   # 1.0 = dead straight, lower = more winding
    }
    return grid_rows, path_row


def validate_ball_trajectory(traj_df):
    """Each phase's ball occupancy map should sum to 1."""
    if traj_df.empty:
        logger.warning("Ball trajectory dataframe is empty.")
        return
    per_phase = (traj_df.groupby(["match_id", "window_id"], as_index=False)
                 ["recurrence_frac"].sum())
    bad = ~np.isclose(per_phase["recurrence_frac"], 1.0, atol=1e-6)
    logger.info("=== BALL TRAJECTORY MAP SANITY CHECK ===")
    logger.info(f"  Phases with a ball map: {len(per_phase):,}")
    logger.info(f"  Maps with sum != 1: {int(bad.sum()):,}")
    n = traj_df.groupby(["match_id", "window_id"])["valid_ball_frame_count"].first()
    logger.info(f"  mean valid ball frames/phase = {n.mean():.2f}, min = {n.min()}")


def compute_phase_pitch_control(match_id, ph, track, frame_idx, x_edges, y_edges,
                                downsample=PHASE_PITCH_CONTROL_DOWNSAMPLE):
    """Pitch-control counterpart of compute_phase_recurrence / compute_phase_ball_trajectory:
    one map per phase (not per side -- home/away is collapsed to "possession team" vs
    "opponent" so it's directly comparable across phases regardless of which literal side was
    on the ball), on the same x_edges/y_edges grid.

    Each cell's value is the MEAN, over this phase's (downsampled) frames, of whether the team
    in possession (ph["team"]) had the nearest player to that cell. That makes it a genuine
    per-cell control fraction in [0, 1], not an occupancy count -- it does not sum to 1 across
    a map the way the recurrence/ball maps do (see PITCH_CONTROL_PARQUET note above).

    Returns row dicts for cells with a nonzero control fraction (exact-0.0 cells are omitted --
    get_phase_pitch_control_grid() reconstructs them as 0.0 by default, so this is lossless and
    keeps memory bounded on matches with many phases).
    """
    n_rows_grid = len(y_edges) - 1
    n_cols_grid = len(x_edges) - 1
    n_cells = n_rows_grid * n_cols_grid

    if len(frame_idx) == 0:
        return []

    sample_idx = frame_idx[::downsample] if downsample > 1 else frame_idx
    if len(sample_idx) == 0:
        sample_idx = frame_idx[:1]

    team = ph["team"]
    total = np.zeros((n_rows_grid, n_cols_grid), dtype=float)
    n_valid_frames = 0
    for i in sample_idx:
        home_xy = track["home_xy"][i]
        away_xy = track["away_xy"][i]
        grid = compute_pitch_control_grid_frame(home_xy, away_xy, x_edges, y_edges)
        if grid is None:
            continue
        own_grid = grid if team == "home" else (1.0 - grid)
        total += own_grid
        n_valid_frames += 1

    if n_valid_frames == 0:
        return []

    mean_grid = total / n_valid_frames

    # MEMORY FIX: same sparsification idea as the recurrence/ball maps. Cells with an exact 0.0
    # control fraction reconstruct correctly with no row at all, since get_phase_pitch_control_
    # grid() already initializes the grid to zeros -- so we only need a row for cells with a
    # nonzero value. This is what was generating a dict for every one of ~7,000 cells x every
    # phase in the match and blowing up memory; most cells in a short phase are untouched.
    flat = mean_grid.reshape(-1)
    nz = np.nonzero(flat)[0]
    if len(nz) == 0:
        return []
    row_idx, col_idx = np.divmod(nz, n_cols_grid)
    rows = pd.DataFrame({
        "match_id": str(match_id),
        "window_id": ph["window_id"],
        "sequence_id": ph["sequence_id"],
        "team": team,                                  # team in possession
        "period": ph["period"],
        "row": row_idx.astype(int),
        "col": col_idx.astype(int),
        "frame_count": int(len(frame_idx)),
        "valid_pitch_control_frame_count": int(n_valid_frames),
        "pitch_control_frac": flat[nz].astype(float),
    }).to_dict("records")
    return rows


def validate_phase_pitch_control(pc_df):
    """Pitch-control cells are independent [0, 1] fractions, not a distribution over cells, so
    there's no sum-to-1 check (unlike validate_phase_recurrence / validate_ball_trajectory).
    Instead sanity-check the value range and overall coverage."""
    if pc_df.empty:
        logger.warning("Pitch control dataframe is empty.")
        return
    vals = pc_df["pitch_control_frac"]
    out_of_range = int(((vals < -1e-6) | (vals > 1 + 1e-6)).sum())
    per_phase = pc_df.groupby(["match_id", "window_id"])["valid_pitch_control_frame_count"].first()
    logger.info("=== PITCH CONTROL MAP SANITY CHECK ===")
    logger.info(f"  Phases with a pitch-control map: {len(per_phase):,}")
    logger.info(f"  Cells outside [0, 1]: {out_of_range:,} / {len(pc_df):,}")
    logger.info(f"  mean valid (downsampled) frames/phase = {per_phase.mean():.2f}, min = {per_phase.min()}")
    logger.info(f"  mean control fraction across all cells = {vals.mean():.3f} (~0.5 expected pooled "
               f"across possession/opposition phases)")


def get_phase_pitch_control_grid(pc_df, match_id, window_id,
                                 n_rows=RECURRENCE_N_ROWS, n_cols=RECURRENCE_N_COLS):
    """One phase's pitch-control map as a (n_rows, n_cols) array of [0, 1] control fractions --
    mirrors get_phase_ball_grid / get_phase_recurrence_grid so the same plotting code works on
    all three. Cells with no data default to 0 (rather than the recurrence maps' implicit 0),
    which is a real value here (never controlled), not a missing-data placeholder.
    """
    sub = pc_df[
        (pc_df["match_id"].astype(str) == str(match_id)) &
        (pc_df["window_id"].astype(str) == str(window_id))
    ]
    grid = np.zeros((n_rows, n_cols), dtype=float)
    np.add.at(grid, (sub["row"].to_numpy(int), sub["col"].to_numpy(int)),
              sub["pitch_control_frac"].to_numpy(float))
    return grid


def get_phase_ball_grid(traj_df, match_id, window_id,
                        n_rows=RECURRENCE_N_ROWS, n_cols=RECURRENCE_N_COLS):
    """One phase's ball occupancy map as a (n_rows, n_cols) array -- mirrors
    get_phase_recurrence_grid() so the same plotting code works on either.

    n_rows/n_cols default to a standard 105x68m pitch (1m cells). Grid size
    is now set per-match from that match's real pitch_length/pitch_width
    (see build_match_phases), so if a specific match's pitch differs from
    105x68, pass that match's actual n_rows/n_cols here explicitly -- using
    the defaults for a non-standard pitch will misalign row/col indices.
    """
    sub = traj_df[
        (traj_df["match_id"].astype(str) == str(match_id)) &
        (traj_df["window_id"].astype(str) == str(window_id))
    ]
    grid = np.zeros((n_rows, n_cols), dtype=float)
    np.add.at(grid, (sub["row"].to_numpy(int), sub["col"].to_numpy(int)),
              sub["recurrence_frac"].to_numpy(float))
    return grid


def get_phase_recurrence_grid(recurrence_df, match_id, window_id,
                              side=None, value_col=None,
                              n_rows=RECURRENCE_N_ROWS,
                              n_cols=RECURRENCE_N_COLS):
    """One phase's map as a (n_rows, n_cols) array.

    side="home" / "away" -> that side only (sums to 1).
    side=None            -> both sides together (sums to 1), same as the old map.

    n_rows/n_cols default to a standard 105x68m pitch (1m cells); pass a
    match's actual grid shape explicitly if its pitch differs (see note on
    get_phase_ball_grid above).
    """
    sub = recurrence_df[
        (recurrence_df["match_id"].astype(str) == str(match_id)) &
        (recurrence_df["window_id"].astype(str) == str(window_id))
    ]
    if side is not None:
        sub = sub[sub["side"] == side]
        col = value_col or "recurrence_frac"
    else:
        col = value_col or "recurrence_frac_combined"

    grid = np.zeros((n_rows, n_cols), dtype=float)
    np.add.at(grid, (sub["row"].to_numpy(int), sub["col"].to_numpy(int)),
              sub[col].to_numpy(float))
    return grid


# ==============================================================================
# PER-PHASE FORMATION FIT
# ==============================================================================
def fit_phase_formation(track, phase_frames, team, period, templates, goalkeepers):
    side = team
    xy_arr = track[f"{side}_xy"][phase_frames]
    ids = track[f"{side}_ids"]
    gk = goalkeepers.get(side) or set()

    avg_positions = []
    for j, pid in enumerate(ids):
        if str(pid) in gk:
            continue
        pos = xy_arr[:, j, :]
        valid = np.all(np.isfinite(pos), axis=1)
        if valid.sum() < 3:
            continue
        avg_positions.append(pos[valid].mean(axis=0))

    if len(avg_positions) < FORMATION_MIN_OUTFIELD_PLAYERS:
        return None

    avg_positions = np.array(avg_positions)
    orientation = get_orientation(side, period, track["meta"])
    formation, cost, _ = match_formation(avg_positions, templates, orientation)
    fit_quality = 1.0 / (1.0 + float(cost))
    hier = derive_hierarchy(formation)

    return {
        "dominant_formation":   formation,
        "formation_variant":    hier["variant"],
        "formation_family":     hier["family"],
        "formation_confidence": fit_quality,
    }


# ==============================================================================
# PER-MATCH DRIVER
# ==============================================================================
def build_match_phases(match_id, processed_dir, epv_grid, xt_grid, score_lookup):
    logger.info(f"[{match_id}] Step 1/5: Loading tracking ...")
    track = load_tracking(match_id, processed_dir)
    if track is None:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame())

    logger.info(f"[{match_id}] Step 2/5: Loading events, building phases ...")
    events = load_flat_events(match_id, processed_dir)
    phases = build_tactical_phases_from_events(
        events, track, score_lookup, THRESHOLD_MEAN, THRESHOLD_MAX
    )
    if not phases:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame())

    logger.info(f"[{match_id}] Step 3/5: Computing per-frame series ...")
    owner_arr = np.full(len(track["period"]), None, dtype=object)
    for ph in phases:
        owner_arr[frame_range_mask(track, ph["start_frame"], ph["end_frame"])] = ph["team"]
    owner_lookup = lambda period, i: owner_arr[i]

    pc_df = compute_pc_for_match(track)
    obso_df = compute_obso_for_match(track, epv_grid, owner_lookup) if epv_grid is not None else pd.DataFrame()
    signed_epv_df = compute_signed_epv_series(track, epv_grid, owner_lookup) if epv_grid is not None else pd.DataFrame()
    epv_bucket_df = bucket_epv_by_second(signed_epv_df)

    effective_xt_grid = xt_grid if xt_grid is not None else epv_grid
    das_offball_df = compute_das_and_offball_xt_for_match(track, epv_grid, owner_lookup, xt_grid=xt_grid) \
        if epv_grid is not None else pd.DataFrame()
    signed_xt_df = compute_signed_xt_series_vectorized(track, effective_xt_grid, owner_arr) \
        if effective_xt_grid is not None else pd.DataFrame()

    logger.info(f"[{match_id}] Step 3.5/5: Spatial features ...")
    spatial_features = compute_all_spatial_features(track)

    logger.info(f"[{match_id}] Step 4/6: Formation templates + GKs ...")
    templates = build_templates(track["pitch_length"], track["pitch_width"])
    goalkeepers = resolve_goalkeepers(track)

    # ~1m x 1m cells: size the grid from THIS match's actual pitch dimensions
    # rather than the fixed 16x12 grid, so resolution is real-world 1m per
    # square regardless of small pitch-length/width differences across venues.
    n_cols_1m = max(1, round(track["pitch_length"]))
    n_rows_1m = max(1, round(track["pitch_width"]))
    x_edges, y_edges = build_grid_edges(track["pitch_length"], track["pitch_width"],
                                         n_cols=n_cols_1m, n_rows=n_rows_1m)

    logger.info(f"[{match_id}] Step 5/6: Aggregating phase-level features ...")
    rows = []
    rec_rows = []
    ball_grid_rows = []
    ball_path_rows = []
    pc_grid_rows = []
    failed = 0
    for ph in phases:
        try:
            frame_idx = frame_range_mask(track, ph["start_frame"], ph["end_frame"])
            if len(frame_idx) < MIN_PHASE_FRAMES:
                continue

            row = build_possession_row(
                match_id, ph, track, pc_df, obso_df,
                pd.DataFrame(),          # no precomputed formation segments
                spatial_features,
                das_offball_df=das_offball_df,
                signed_xt_df=signed_xt_df,
            )
            if row is None:
                failed += 1
                continue

            row["window_id"]        = ph["window_id"]
            row["reason_closed"]    = ph["_reason_closed"]
            row["threshold_mean"]   = ph["_threshold_mean"]
            row["threshold_max"]    = ph["_threshold_max"]
            row["n_passes_in_window"] = sum(
                1 for r in ph["possession_events"]
                if _get(r, PE_TYPE_KEYS) in PASS_EVENT_TYPES
                and _get(r, PE_OUTCOME_KEYS) == "C"
            )

            ff = fit_phase_formation(track, frame_idx, ph["team"], ph["period"],
                                     templates, goalkeepers)
            if ff is not None:
                row.update(ff)
            else:
                row["dominant_formation"]   = None
                row["formation_variant"]    = None
                row["formation_family"]     = None
                row["formation_confidence"] = np.nan

            rows.append(row)

            rec_rows.extend(compute_phase_recurrence(
                match_id, ph, track, frame_idx, x_edges, y_edges, goalkeepers))

            b_grid, b_path = compute_phase_ball_trajectory(
                match_id, ph, track, frame_idx, x_edges, y_edges)
            ball_grid_rows.extend(b_grid)
            if b_path is not None:
                ball_path_rows.append(b_path)

            pc_grid_rows.extend(compute_phase_pitch_control(
                match_id, ph, track, frame_idx, x_edges, y_edges))
        except Exception as ex:
            logger.error(f"[{match_id}] phase {ph.get('window_id')} failed: {ex}")
            failed += 1

    logger.info(f"[{match_id}] Step 6/6: Per-team recurrence maps ... {len(rec_rows):,} rows "
               f"(grid={n_rows_1m}x{n_cols_1m}, ~1m cells, 2 sides per phase)")
    logger.info(f"[{match_id}] Step 6b/6: Ball trajectory maps ... {len(ball_grid_rows):,} grid rows, "
               f"{len(ball_path_rows):,} phase paths (ball excluded from player recurrence maps above)")
    logger.info(f"[{match_id}] Step 6c/6: Pitch control maps ... {len(pc_grid_rows):,} grid rows "
               f"(grid={n_rows_1m}x{n_cols_1m}, ~1m cells, 1 possession-relative map per phase)")

    logger.info(f"[{match_id}] Phases: {len(phases)}  |  ok: {len(rows)}  |  failed: {failed}")

    poss_df = pd.DataFrame(rows)
    rec_df = pd.DataFrame(rec_rows)
    ball_grid_df = pd.DataFrame(ball_grid_rows)
    ball_path_df = pd.DataFrame(ball_path_rows)
    pc_grid_df = pd.DataFrame(pc_grid_rows)

    # Sliding-window formation segments (for optional joins downstream)
    try:
        w_home, w_away = compute_frame_weights(track, events)
        windows_df = build_formation_windows(track, templates, goalkeepers, w_home, w_away)
        windows_df = windows_df.assign(match_id=str(match_id)) if not windows_df.empty else windows_df
    except Exception as ex:
        logger.warning(f"[{match_id}] formation windows failed: {ex}")
        windows_df = pd.DataFrame()

    return poss_df, windows_df, rec_df, ball_grid_df, ball_path_df, pc_grid_df


# ==============================================================================
# MULTICORE WORKER
# ==============================================================================
_WORKER = {}

def _init_worker(processed_dir, epv_grid_path, xt_grid_path, score_lookup):
    _WORKER["processed_dir"] = processed_dir
    _WORKER["epv_grid"] = load_epv_grid(epv_grid_path)
    _WORKER["xt_grid"]  = load_grid(xt_grid_path)
    _WORKER["score_lookup"] = score_lookup

def _process_one_match_worker(match_id):
    t0 = time.time()
    try:
        poss_df, windows_df, rec_df, ball_grid_df, ball_path_df, pc_grid_df = build_match_phases(
            match_id,
            _WORKER["processed_dir"],
            _WORKER["epv_grid"],
            _WORKER["xt_grid"],
            _WORKER["score_lookup"],
        )
        return (match_id, poss_df, windows_df, rec_df, ball_grid_df, ball_path_df, pc_grid_df,
                None, time.time() - t0)
    except Exception:
        import traceback
        return (match_id, None, None, None, None, None, None,
                traceback.format_exc(), time.time() - t0)


# ==============================================================================
# STREAMING PARQUET WRITER — memory fix for the recurrence / pitch-control tables
# ==============================================================================
# recurrence_df and pc_grid_df are one row per (phase, grid cell) -- at a ~1m grid over a
# full pitch that's ~7,000 cells x 2 sides x hundreds of phases PER MATCH. The original code
# appended every match's DataFrame to a Python list (all_rec / all_pc_grid) and only called
# pd.concat + to_parquet once at the very end of main(), so the ENTIRE run's worth of grid
# rows sat in memory simultaneously regardless of NUM_WORKERS. This writes each match's rows
# to disk as soon as that match finishes, so steady-state memory is bounded by ~1 match's grid
# output, not all 10. poss_df/windows_df/ball_path_df are left as plain in-memory
# accumulate-then-concat -- they're one row per PHASE (not per cell), orders of magnitude
# smaller, and not what was reported as the memory problem.
class StreamingParquetWriter:
    def __init__(self, path):
        self.path = str(path)
        self._writer = None
        self.n_rows = 0

    def write(self, df):
        if df is None or df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        elif not table.schema.equals(self._writer.schema):
            # column order/dtype drift between matches (e.g. an all-NaN column inferred as a
            # different type) -- reconcile to the writer's original schema rather than crash
            # partway through a run.
            table = table.cast(self._writer.schema)
        self._writer.write_table(table)
        self.n_rows += len(df)

    def close(self):
        if self._writer is not None:
            self._writer.close()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    root = Path(PROCESSED_DIR)
    match_ids = sorted(
        d.name for d in root.iterdir()
        if d.is_dir()
        and (d / "metadata.json").exists()
        and (d / "tracking.jsonl.bz2").exists()
        and (d / "events.json").exists()
    ) if root.exists() else []
    if NUM_MATCHES is not None:
        match_ids = match_ids[:NUM_MATCHES]

    if not match_ids:
        print(f"No matches found under {PROCESSED_DIR}.")
        return

    # ---------- STEP 1 ----------
    score_lookup = run_movement_extraction(match_ids)
    if not score_lookup:
        print("No scored passes; aborting.")
        return

    # ---------- STEP 2 ----------
    log("")
    log("=" * 70)
    log(f"STEP 2 — Tactical phase segmentation")
    log(f"  threshold_mean = {THRESHOLD_MEAN} m  |  threshold_max = {THRESHOLD_MAX} m")
    log(f"  max_passes = {MAX_WINDOW_PASSES}  |  max_time = {MAX_WINDOW_SECONDS}s")
    log(f"  also closes on: failed_pass | shot | stoppage_break | sequence_end")
    log(f"  workers = {NUM_WORKERS}")
    log("=" * 70)

    all_poss, all_windows = [], []
    all_ball_path = []
    t_start = time.time()
    done = [0]

    # Stream these three straight to disk per-match instead of accumulating in memory (see
    # StreamingParquetWriter docstring above) -- this is the actual memory fix.
    rec_writer = StreamingParquetWriter(RECURRENCE_PARQUET)
    ball_grid_writer = StreamingParquetWriter(BALL_TRAJECTORY_PARQUET)
    pc_writer = StreamingParquetWriter(PITCH_CONTROL_PARQUET)

    def _absorb(mid, poss_df, windows_df, rec_df, ball_grid_df, ball_path_df, pc_grid_df, err, dt):
        done[0] += 1
        if err is not None:
            print(f"  [{done[0]}/{len(match_ids)}] {mid}  FAILED ({dt:.1f}s)")
            print(err)
            return
        n = 0 if poss_df is None or poss_df.empty else len(poss_df)
        n_rec = 0 if rec_df is None or rec_df.empty else len(rec_df)
        n_ball = 0 if ball_grid_df is None or ball_grid_df.empty else len(ball_grid_df)
        n_pc = 0 if pc_grid_df is None or pc_grid_df.empty else len(pc_grid_df)
        if poss_df is not None and not poss_df.empty:
            all_poss.append(poss_df)
        if windows_df is not None and not windows_df.empty:
            all_windows.append(windows_df)
        rec_writer.write(rec_df)
        ball_grid_writer.write(ball_grid_df)
        if ball_path_df is not None and not ball_path_df.empty:
            all_ball_path.append(ball_path_df)
        pc_writer.write(pc_grid_df)
        print(f"  [{done[0]}/{len(match_ids)}] {mid}  phases={n:>5}  recurrence_rows={n_rec:>6}  "
             f"ball_traj_rows={n_ball:>6}  pitch_control_rows={n_pc:>6}  ({dt:.1f}s)")

    if NUM_WORKERS == 1:
        _init_worker(PROCESSED_DIR, EPV_GRID_PATH, XT_GRID_PATH, score_lookup)
        for mid in match_ids:
            _absorb(*_process_one_match_worker(mid))
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=NUM_WORKERS,
                initializer=_init_worker,
                initargs=(PROCESSED_DIR, EPV_GRID_PATH, XT_GRID_PATH, score_lookup),
            ) as ex:
                futs = [ex.submit(_process_one_match_worker, mid) for mid in match_ids]
                for fut in as_completed(futs):
                    _absorb(*fut.result())
        except Exception as ex:
            print(f"  [WARN] parallel path failed ({ex!r}); falling back to single")
            _init_worker(PROCESSED_DIR, EPV_GRID_PATH, XT_GRID_PATH, score_lookup)
            for mid in match_ids:
                _absorb(*_process_one_match_worker(mid))

    print(f"\nAll matches done in {time.time() - t_start:.1f}s.")

    if all_poss:
        result = pd.concat(all_poss, ignore_index=True)
        result.to_parquet(OUTPUT_PATH, index=False)
        print(f"SUCCESS: {OUTPUT_PATH}  --  {len(result):,} phases  x  {result.shape[1]} columns")
        print("  reason_closed breakdown:")
        for r, n in result["reason_closed"].value_counts().items():
            print(f"    {r:<18} {n:>7,} ({100*n/len(result):.1f}%)")

        print("\n  phase duration percentiles (s):")
        d = result["duration"].dropna()
        for q in (0.50, 0.75, 0.90, 0.95, 0.99, 0.999):
            print(f"    p{int(q*1000)/10:>5.1f}: {d.quantile(q):>8.2f}")
        print(f"    max:   {d.max():>8.2f}")

        print("\n  top-5 longest phases:")
        top5 = result.nlargest(5, "duration")[
            ["match_id", "sequence_id", "window_id", "team", "period",
             "duration", "reason_closed"]
        ]
        for _, r in top5.iterrows():
            print(f"    {r['match_id']:<6} seq={r['sequence_id']:<5.0f} "
                  f"{r['window_id']:<12} {r['team']:<5} p{r['period']}  "
                  f"dur={r['duration']:>7.2f}s  closed={r['reason_closed']}")

    if all_windows:
        pd.concat(all_windows, ignore_index=True).to_parquet(FORMATION_WINDOWS_OUTPUT, index=False)

    # rec_writer / ball_grid_writer / pc_writer were already streaming rows to disk match-by-
    # match during the loop above -- just close them (flushes + finalizes the parquet footer).
    # Sanity-check validation still needs the full table, so we read it back ONCE here (a
    # single read, after every per-match object has already been freed) rather than holding
    # it in memory for the whole run the way the old accumulate-then-concat did.
    rec_writer.close()
    if rec_writer.n_rows:
        print(f"SUCCESS: {RECURRENCE_PARQUET}  --  {rec_writer.n_rows:,} grid-cell rows (2 sides per phase)")
        validate_phase_recurrence(pd.read_parquet(RECURRENCE_PARQUET))
    else:
        print(f"NOTE: no recurrence grid rows produced.")

    ball_grid_writer.close()
    if ball_grid_writer.n_rows:
        print(f"SUCCESS: {BALL_TRAJECTORY_PARQUET}  --  {ball_grid_writer.n_rows:,} grid-cell rows (1 ball map per phase)")
        validate_ball_trajectory(pd.read_parquet(BALL_TRAJECTORY_PARQUET))
    else:
        print(f"NOTE: no ball trajectory grid rows produced -- check BALL_XY_KEY / "
             f"_get_ball_xy() against your load_tracking() output.")

    if all_ball_path:
        ball_path_result = pd.concat(all_ball_path, ignore_index=True)
        ball_path_result.to_parquet(BALL_PATH_PARQUET, index=False)
        print(
            f"SUCCESS: {BALL_PATH_PARQUET}  --  "
            f"{len(ball_path_result):,} phase-level ball paths"
        )
        print("  ball path length percentiles (m):")
        pl = ball_path_result["path_length_m"].dropna()
        for q in (0.50, 0.75, 0.90, 0.95, 0.99):
            print(f"    p{int(q*100):>3}: {pl.quantile(q):>8.2f}")

    pc_writer.close()
    if pc_writer.n_rows:
        print(f"SUCCESS: {PITCH_CONTROL_PARQUET}  --  "
              f"{pc_writer.n_rows:,} grid-cell rows (1 possession-relative pitch-control map per phase)")
        validate_phase_pitch_control(pd.read_parquet(PITCH_CONTROL_PARQUET))
    else:
        print(f"NOTE: no pitch control grid rows produced -- check that possession_pipeline's "
             f"compute_pitch_control_grid_frame is importable and track['home_xy']/['away_xy'] "
             f"have valid frames.")

    print(f"\n{'='*70}\nPIPELINE COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()