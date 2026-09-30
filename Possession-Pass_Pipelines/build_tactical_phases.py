"""
build_tactical_phases.py
========================

Combined pipeline:

  STEP 1 — Per-pass movement scores.
      For every completed pass/cross in every match, compute the mean
      Euclidean displacement of the attacking team's 10 outfielders between:
          x1 = this pass's snapshot
          x2 = the snapshot of the immediately NEXT event in the same
               sequence (any type: pass, carry, challenge, foul, shot...)
      Writes pass_movement_scores.parquet + a 4-panel histogram.

  STEP 2 — Tactical phase segmentation.
      Within each PFF sequence, cut events into phases. A phase closes as
      soon as any of the following occurs (whichever fires first):

          1. Stoppage break — a null-sequence stoppage event (OUT, SUB,
             FOUL, kickoffs, END, ...) falls between the previous open-play
             event and this one.
          2. threshold_mean / threshold_max / threshold_both
          3. failed_pass  — any PA/CR with outcome != 'C'
          4. shot         — any SH
          5. max_passes   — MAX_WINDOW_PASSES completed passes accumulated
          6. max_time     — MAX_WINDOW_SECONDS elapsed since phase start
          7. sequence_end — the sequence ended without any close trigger

      Segment bounds use the actual EVENT moment (eventTime), not the
      enclosing game event's start/end times.

  PITCH CONTROL:
      The `pc_map` column (flat 75*45 = 3375 float16 values) is written as
      a fixed-size pyarrow list so the parquet stays compact and every row
      has the same shape.

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
    PC_GRID_RES_X, PC_GRID_RES_Y, PC_MAP_SIZE, PC_MAP_DTYPE,
    # Grid loaders
    load_epv_grid, load_grid,
    # Per-frame series
    compute_pc_for_match, compute_obso_for_match,
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
    # Flatten helper (deduplicated: identical to the pipeline's own copy)
    flatten_event, _SUBDICT_KEYS,
)

logger = logging.getLogger(__name__)
_T0 = time.time()

# ==============================================================================
# LOCAL EVENT-KEY EXTENSIONS
# ==============================================================================
GE_EVENT_TIME_KEYS = ["eventTime", "event_time"]
GE_EVENT_TYPE_KEYS = ["gameEventType", "game_event_type"]

# ==============================================================================
# CONFIG
# ==============================================================================
NUM_MATCHES = 64
NUM_WORKERS = 3

MOVEMENT_PARQUET = "output/pass_movement_scores.parquet"
MOVEMENT_FIGURE  = "output/pass_movement_scores_histogram.png"
RECOMPUTE_MOVEMENT_SCORES = True
MIN_VALID_OUTFIELDERS = 7
EXCLUDE_GK = True
PASS_EVENT_TYPES = ("PA", "CR")
MAX_REASONABLE_PASS_DURATION = 10.0

OUTPUT_PATH = "output/tactical_phases.parquet"
FORMATION_WINDOWS_OUTPUT = "output/tactical_phases_formation_windows.parquet"

THRESHOLD_MEAN     = 2.0
THRESHOLD_MAX      = 4.0
MAX_WINDOW_PASSES  = 8
MAX_WINDOW_SECONDS = 30.0
MIN_PHASE_FRAMES   = 3

STOPPAGE_EVENT_TYPES = {
    "OUT", "SUB", "OFF", "ON",
    "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF",
    "FOUL", "END", "VID",
}
# PFF's authoritative dead-ball signal (event spec section 2): `outType` on `OUT` events.
#   W = Whistle (refereed stoppage), H = Home Score, A = Away Score (ball in the net).
# Both halt play and end a phase. T = Out of Touch only means the ball crossed the
# touchline and the same sequence usually continues through the throw-in, so it does not
# close a phase. Preferred over the hardcoded type list -- see VERIFICATION_FINDINGS.md F6.
DEAD_BALL_OUT_TYPES = {"W", "H", "A"}

def log(msg, level="INFO"):
    print(f"[{time.time() - _T0:7.1f}s] [{level}] {msg}", flush=True)

# ==============================================================================
# GENERIC EVENT HELPERS
# ==============================================================================
# `flatten_event`, `_SUBDICT_KEYS`, and `load_flat_events` are imported from
# possession_pipeline (B8: previously duplicated here, and the two copies had already
# diverged -- the pipeline's read `grades`, this copy did not).

def _load_events_raw(match_id, processed_dir):
    return load_flat_events(match_id, processed_dir)

def _event_moment(r):
    t = _get(r, GE_EVENT_TIME_KEYS)
    if t is None:
        return None
    try:
        return float(t)
    except (TypeError, ValueError):
        return None

def _snapshot_map(players):
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
    times = []
    for r in events:
        if _get(r, GE_SEQUENCE_KEYS) is not None:
            continue
        # Authoritative dead-ball signal: a whistle or a score on an OUT event.
        out_type = _get(r, ["outType"])
        if out_type in DEAD_BALL_OUT_TYPES:
            t = _event_moment(r)
            if t is not None:
                times.append(t)
            continue
        # Fallback: stoppages that carry no OUT record (substitutions, fouls, cards,
        # kickoffs, END, VID). OUT itself is handled by the whistle/score signal above.
        gtype = _get(r, GE_EVENT_TYPE_KEYS)
        if gtype not in STOPPAGE_EVENT_TYPES or gtype == "OUT":
            continue
        t = _event_moment(r)
        if t is not None:
            times.append(t)
    return sorted(times)

def _has_break_between(break_times, t_prev, t_now):
    if not break_times or t_prev is None or t_now is None:
        return False
    if t_now <= t_prev:
        return False
    i = bisect.bisect_right(break_times, t_prev)
    return i < len(break_times) and break_times[i] <= t_now

# ==============================================================================
# STEP 1: PER-PASS MOVEMENT SCORES
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

            touch_duration = None
            if t_start is not None and t_end is not None:
                d = float(t_end) - float(t_start)
                if 0.0 <= d <= MAX_REASONABLE_PASS_DURATION:
                    touch_duration = d

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

def _hist_panel(ax, values, xlabel, title, bins=80, log_y=False):
    v = pd.Series(values).dropna()
    ax.hist(v, bins=bins, edgecolor="none")
    if log_y:
        ax.set_yscale("log")
        title = title + "  (log y)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=10)

def _plot_movement_histogram(ok, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    _hist_panel(axes[0, 0], ok["mean_disp"], "mean player displacement per pass (m)",
                "Mean displacement per pass", bins=100)
    _hist_panel(axes[0, 1], ok["mean_disp"], "mean player displacement per pass (m)",
                "Mean displacement per pass", bins=100, log_y=True)
    _hist_panel(axes[1, 0], ok["max_disp"], "max player displacement per pass (m)",
                "Max displacement (single most-moved player)", bins=100)
    _hist_panel(axes[1, 1], ok["max_disp"], "max player displacement per pass (m)",
                "Max displacement (single most-moved player)", bins=100, log_y=True)
    _hist_panel(axes[2, 0], ok["time_to_next"], "seconds from this pass to next event",
                "Time to next event", bins=100)
    _hist_panel(axes[2, 1], ok["touch_duration"], "touch duration (s)",
                "Touch duration (enclosing game event, capped 10 s)", bins=80)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def run_movement_extraction(match_ids):
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

    Path(MOVEMENT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
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
# STEP 2: TACTICAL PHASE SEGMENTATION
# ==============================================================================
def build_tactical_phases_from_events(events, track, score_lookup,
                                      threshold_mean, threshold_max):
    break_times = _collect_break_times(events)

    groups = defaultdict(list)
    for r in events:
        seq = _get(r, GE_SEQUENCE_KEYS)
        if seq is not None:
            groups[seq].append(r)

    phases = []
    phase_counter = [0]

    for seq, seq_rows in groups.items():
        seq_rows = sorted(seq_rows, key=lambda r: _event_moment(r) or 0.0)
        if not seq_rows:
            continue

        home_team = _get(seq_rows[0], GE_HOME_TEAM_KEYS)
        if home_team is None:
            continue
        # ~1/3 of PFF sequences contain both teams' events (the minority side contributes
        # defensive actions: clearances, challenges, rebounds). The first event is not a
        # reliable owner -- a sequence can legitimately open with the defending team -- so
        # attribute the phase to whichever team supplies the majority of its events.
        n_home = sum(1 for r in seq_rows if _get(r, GE_HOME_TEAM_KEYS) is True)
        n_away = sum(1 for r in seq_rows if _get(r, GE_HOME_TEAM_KEYS) is False)
        team = "home" if n_home >= n_away else "away"

        current_window = []
        window_start_time = None
        n_completed_passes = 0
        prev_t = None

        for ev in seq_rows:
            t_now = _event_moment(ev)
            if t_now is None:
                continue

            # 1. Stoppage break (before append)
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

            if reason is None and pe_type in PASS_EVENT_TYPES and outcome != "C":
                reason = "failed_pass"

            if reason is None and pe_type == "SH":
                reason = "shot"

            if reason is None and n_completed_passes >= MAX_WINDOW_PASSES:
                reason = "max_passes"

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

        if current_window:
            ph = _finalize_phase(seq, team, current_window, track,
                                 "sequence_end", phase_counter,
                                 threshold_mean, threshold_max)
            if ph is not None:
                phases.append(ph)

    return phases

def _finalize_phase(seq, team, window_rows, track, reason, counter,
                    threshold_mean, threshold_max):
    if not window_rows:
        return None

    period_val = _get(window_rows[0], ["period"])
    phase_period = int(period_val) if period_val is not None else None

    start_idx, end_idx = _possession_frame_bounds(track, window_rows)
    if start_idx is None or end_idx is None:
        gc_start, gc_end = _game_clock_bounds(track, window_rows, phase_period)
        if gc_start is not None and gc_end is not None:
            start_idx, end_idx = gc_start, gc_end

    moments = [_event_moment(r) for r in window_rows]
    moments = [m for m in moments if m is not None]
    if not moments:
        return None
    start_sec = float(min(moments))
    end_sec   = float(max(moments))
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
# PER-PHASE FORMATION FIT
# ==============================================================================
def fit_phase_formation(track, phase_frames, team, period, templates, goalkeepers):
    side = team
    xy_arr = track[f"{side}_xy"][phase_frames]
    ids = track[f"{side}_ids"]
    # goalkeepers are resolved per (side, period) so a substituted keeper is excluded
    # in the period where the replacement actually plays.
    gk = goalkeepers.get((side, int(period))) or set()

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
    orientation = get_orientation(side, period, track)
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
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f"[{match_id}] Step 2/5: Loading events, building phases ...")
    events = load_flat_events(match_id, processed_dir)
    phases = build_tactical_phases_from_events(
        events, track, score_lookup, THRESHOLD_MEAN, THRESHOLD_MAX
    )
    if not phases:
        return pd.DataFrame(), pd.DataFrame()

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
    # Only frames inside a phase are ever read downstream, so restrict the expensive
    # per-frame convex-hull loop to that union (mirrors build_match in the pipeline).
    needed_frames = np.zeros(len(track["period"]), dtype=bool)
    for ph in phases:
        needed_frames[frame_range_mask(track, ph["start_frame"], ph["end_frame"])] = True
    spatial_features = compute_all_spatial_features(track, needed_frames)

    logger.info(f"[{match_id}] Step 4/5: Formation templates + GKs ...")
    templates = build_templates(track["pitch_length"], track["pitch_width"])
    goalkeepers = resolve_goalkeepers(track)

    logger.info(f"[{match_id}] Step 5/5: Aggregating phase-level features ...")
    rows = []
    failed = 0
    dropped_by_reason = defaultdict(int)
    dropped_frame_counts = defaultdict(list)
    for ph in phases:
        try:
            frame_idx = frame_range_mask(track, ph["start_frame"], ph["end_frame"])
            if len(frame_idx) < MIN_PHASE_FRAMES:
                reason = ph["_reason_closed"]
                dropped_by_reason[reason] += 1
                dropped_frame_counts[reason].append(len(frame_idx))
                continue

            row = build_possession_row(
                match_id, ph, track, pc_df, obso_df,
                pd.DataFrame(),
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
        except Exception as ex:
            logger.error(f"[{match_id}] phase {ph.get('window_id')} failed: {ex}")
            failed += 1

    logger.info(f"[{match_id}] Phases: {len(phases)}  |  ok: {len(rows)}  |  failed: {failed}")
    if dropped_by_reason:
        # ~1/3 of built phases resolve to fewer than MIN_PHASE_FRAMES tracking frames (a
        # near-instant closedown resolves to one or two frames); report them so the
        # kept/total ratio is auditable instead of silently shrinking the output.
        summary = ", ".join(f"{k}={v}" for k, v in sorted(
            dropped_by_reason.items(), key=lambda kv: -kv[1]))
        logger.info(f"[{match_id}] Dropped (<{MIN_PHASE_FRAMES} frames): "
                    f"{sum(dropped_by_reason.values())}  [{summary}]")

    poss_df = pd.DataFrame(rows)

    try:
        w_home, w_away = compute_frame_weights(track, events)
        windows_df = build_formation_windows(track, templates, goalkeepers, w_home, w_away)
        windows_df = windows_df.assign(match_id=str(match_id)) if not windows_df.empty else windows_df
    except Exception as ex:
        logger.warning(f"[{match_id}] formation windows failed: {ex}")
        windows_df = pd.DataFrame()

    return poss_df, windows_df

# ==============================================================================
# SAVE HELPER (pc_map as fixed-size list)
# ==============================================================================
def _save_with_pc_map(df, path, grid_x, grid_y):
    """Write df to parquet, encoding `pc_map` as a fixed-size float16 list
    of length grid_x * grid_y. Missing / wrong-size maps become all-NaN."""
    flat_size = grid_x * grid_y

    maps_list = []
    for m in df["pc_map"].values:
        if m is None:
            maps_list.append(np.full(flat_size, np.nan, dtype=np.float16))
            continue
        arr = np.asarray(m, dtype=np.float16).ravel()
        if arr.shape[0] != flat_size:
            arr = np.full(flat_size, np.nan, dtype=np.float16)
        maps_list.append(arr)

    flat = np.concatenate(maps_list)
    pc_map_col = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.float16()),
        list_size=flat_size,
    )

    df_no_pc = df.drop(columns=["pc_map"])
    table = pa.Table.from_pandas(df_no_pc, preserve_index=False)
    table = table.append_column("pc_map", pc_map_col)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")

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
        poss_df, windows_df = build_match_phases(
            match_id,
            _WORKER["processed_dir"],
            _WORKER["epv_grid"],
            _WORKER["xt_grid"],
            _WORKER["score_lookup"],
        )
        return match_id, poss_df, windows_df, None, time.time() - t0
    except Exception:
        import traceback
        return match_id, None, None, traceback.format_exc(), time.time() - t0

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

    score_lookup = run_movement_extraction(match_ids)
    if not score_lookup:
        print("No scored passes; aborting.")
        return

    log("")
    log("=" * 70)
    log(f"STEP 2 — Tactical phase segmentation")
    log(f"  threshold_mean = {THRESHOLD_MEAN} m  |  threshold_max = {THRESHOLD_MAX} m")
    log(f"  max_passes = {MAX_WINDOW_PASSES}  |  max_time = {MAX_WINDOW_SECONDS}s")
    log(f"  also closes on: failed_pass | shot | stoppage_break | sequence_end")
    log(f"  pc_map grid = {PC_GRID_RES_X}x{PC_GRID_RES_Y} = {PC_MAP_SIZE} cells")
    log(f"  workers = {NUM_WORKERS}")
    log("=" * 70)

    all_poss, all_windows = [], []
    t_start = time.time()
    done = [0]

    def _absorb(mid, poss_df, windows_df, err, dt):
        done[0] += 1
        if err is not None:
            print(f"  [{done[0]}/{len(match_ids)}] {mid}  FAILED ({dt:.1f}s)")
            print(err)
            return
        n = 0 if poss_df is None or poss_df.empty else len(poss_df)
        if poss_df is not None and not poss_df.empty:
            all_poss.append(poss_df)
        if windows_df is not None and not windows_df.empty:
            all_windows.append(windows_df)
        print(f"  [{done[0]}/{len(match_ids)}] {mid}  phases={n:>5}  ({dt:.1f}s)")

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
        _save_with_pc_map(result, OUTPUT_PATH, PC_GRID_RES_X, PC_GRID_RES_Y)
        print(f"SUCCESS: {OUTPUT_PATH}  --  {len(result):,} phases  x  {result.shape[1]} columns  "
              f"(pc_map = fixed-size list, {PC_GRID_RES_X}x{PC_GRID_RES_Y} = {PC_MAP_SIZE} cells)")
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
        Path(FORMATION_WINDOWS_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_windows, ignore_index=True).to_parquet(FORMATION_WINDOWS_OUTPUT, index=False)

    print(f"\n{'='*70}\nPIPELINE COMPLETE\n{'='*70}")

if __name__ == "__main__":
    main()