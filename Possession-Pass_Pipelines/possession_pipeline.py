# ==============================================================================
# POSSESSION-LEVEL TACTICAL STATE PIPELINE (VECTORIZED + COORD FIX + MULTICORE)
#
# GAME-CLOCK FALLBACK ADDED:
#   - load_tracking now records periodGameClockTime per frame.
#   - frame_at_game_clock() does a match-relative lookup that works for ALL
#     periods including extra time.
#   - build_possessions_from_events uses three-tier resolution:
#       1. possession_event_id / game_event_id -> tracking frame
#       2. periodGameClockTime vs startGameClock (match-relative, no reset)
#       3. (legacy) video-time vs period-elapsed -- only if the first two fail
#   - event_period_sec() also uses the game-clock fallback.
#   - Phantom post-END possessions (shootout kicks tagged P3/P4 after the
#     tracking feed stops) are skipped.
#
# PITCH-CONTROL MAP:
#   - compute_pitch_control_frame returns a binary (PC_GRID_RES_X, PC_GRID_RES_Y)
#     ownership map, not a scalar.
#   - compute_pc_for_match stores one map per sampled frame.
#   - build_possession_row averages the frame maps over the phase, flips
#     coordinates so the attacking team always attacks left->right, and
#     stores the flat map under key "pc_map" (shape = PC_MAP_SIZE).
# ==============================================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import bz2
import json
import logging
import math
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from mplsoccer import Pitch

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
import faulthandler
faulthandler.enable()

# ==============================================================================
# 1. CONFIG
# ==============================================================================
PROCESSED_DIR  = "processed_tracking"
NUM_MATCHES = 64
NUM_WORKERS = 3
OUTPUT_PATH_BASE    = "possession_level_dataset_ET_Only.parquet"
FORMATION_WINDOWS_BASE = "formations_windows_ET_Only.parquet"
FORMATION_SEGMENTS_BASE = "formation_segments_ET_Only.parquet"
OUTPUT_PATH    = OUTPUT_PATH_BASE
FORMATION_WINDOWS_OUTPUT = FORMATION_WINDOWS_BASE
FORMATION_SEGMENTS_OUTPUT = FORMATION_SEGMENTS_BASE
EPV_GRID_PATH  = "EPV_grid.csv"

USE_SMOOTHED_TRACKING = True
RELIABILITY_MODE = "all"
RELIABILITY_THRESHOLD = 0.5

OBSO_RADIUS_M = 5.0
OBSO_DOWNSAMPLE = 5
PC_DOWNSAMPLE = 3

# ---- Pitch control grid -----------------------------------------------------
PC_GRID_RES_X = 75              # resolution along pitch length (x axis)
PC_GRID_RES_Y = 45              # resolution along pitch width  (y axis)
PC_MAP_DTYPE  = np.float16      # storage dtype for the flat map
PC_MAP_SIZE   = PC_GRID_RES_X * PC_GRID_RES_Y   # = 3375

FORMATION_WINDOW_SECONDS = 180
FORMATION_STRIDE_SECONDS = 60
FORMATION_MIN_FRAMES_PER_WINDOW = 100
FORMATION_MIN_OUTFIELD_PLAYERS = 8
FORMATION_GK_MIN_FRAMES = 200
MIN_WINDOW_CONFIDENCE = 0.0

FOUL_DECAY_SECONDS = 15.0
SETPIECE_RECOVERY_SECONDS = 10.0
SUB_DECAY_SECONDS = 30.0
TURNOVER_DECAY_SECONDS = 5.0
RELIABILITY_PERCENTILE = 90.0

DAS_EPV_THRESHOLD = 0.05
DAS_MIN_DURATION_SECONDS = 3.0

XT_GRID_PATH = "xT_grid.csv"
DAS_GRID_RES = 30
DAS_DOWNSAMPLE = 5
OFFBALL_CARRIER_RADIUS_M = 3.0

POST_END_GRACE_SECONDS = 5.0

# ==============================================================================
# 2. FIELD KEYS
# ==============================================================================
PLAYER_ID_KEYS   = ["jerseyNum", "jersey_num"]
PLAYER_X_KEYS    = ["x", "X"]
PLAYER_Y_KEYS    = ["y", "Y"]
PLAYER_Z_KEYS    = ["z", "Z"]
PLAYER_VIS_KEYS  = ["visibility"]
PLAYER_CONF_KEYS = ["confidence"]
VISIBLE_VALUES = {"VISIBLE", "visible", "ESTIMATED", "estimated"}
REJECT_CONFIDENCE_VALUES = {}

def _get(d, keys, default=None):
    if not isinstance(d, dict): return default
    for k in keys:
        if k in d and d[k] is not None: return d[k]
    return default

# ==============================================================================
# 3. EPV GRID + PITCH CONTROL + OBSO
# ==============================================================================
def load_epv_grid(path):
    return np.loadtxt(path, delimiter=",") if Path(path).exists() else None

def load_grid(path):
    return np.loadtxt(path, delimiter=",") if Path(path).exists() else None

def epv_value(epv_grid, x, y, pitch_length, pitch_width, direction):
    n_rows, n_cols = epv_grid.shape
    gx = x if direction == 1 else (pitch_length - x)
    gy = y
    col = int(np.clip(gx / pitch_length * n_cols, 0, n_cols - 1))
    row = int(np.clip(gy / pitch_width * n_rows, 0, n_rows - 1))
    return float(epv_grid[row, col])

def get_base_directions(meta, extra_time=False):
    key = "homeTeamStartLeftExtraTime" if extra_time else "homeTeamStartLeft"
    hsl = meta.get(key, meta.get("homeTeamStartLeft", True))
    return (1 if hsl else -1), (-1 if hsl else 1)

def attack_direction(team, period, meta):
    h_dir, a_dir = get_base_directions(meta, extra_time=period in (3, 4))
    base = h_dir if team == "home" else a_dir
    return base * (1 if int(period) % 2 == 1 else -1)

def compute_pitch_control_frame(home_xy, away_xy, pitch_length, pitch_width,
                                grid_res_x=PC_GRID_RES_X,
                                grid_res_y=PC_GRID_RES_Y):
    """Binary ownership map of the pitch.

    Returns
    -------
    ownership : np.ndarray of shape (grid_res_x, grid_res_y), dtype uint8
        1 where home's nearest player is closer than any away player's
        nearest player, 0 elsewhere. Returns None if fewer than 4 players
        are tracked in this frame.
    """
    home_xy = home_xy[np.all(np.isfinite(home_xy), axis=1)]
    away_xy = away_xy[np.all(np.isfinite(away_xy), axis=1)]
    if len(home_xy) + len(away_xy) < 4:
        return None

    pts    = np.vstack([home_xy, away_xy])
    labels = np.array([1] * len(home_xy) + [0] * len(away_xy), dtype=np.uint8)

    xs = np.linspace(0, pitch_length, grid_res_x)
    ys = np.linspace(0, pitch_width,  grid_res_y)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")            # both (grid_res_x, grid_res_y)
    gp = np.column_stack((gx.ravel(), gy.ravel()))          # (grid_res_x*grid_res_y, 2)

    nearest   = np.argmin(cdist(gp, pts), axis=1)
    ownership = labels[nearest].reshape(grid_res_x, grid_res_y)
    return ownership

def compute_pc_for_match(track, downsample=PC_DOWNSAMPLE):
    """One row per sampled frame with a (grid_res_x, grid_res_y) uint8 map."""
    records = []
    n = len(track["period"])
    for i in range(0, n, downsample):
        ownership = compute_pitch_control_frame(
            track["home_xy"][i], track["away_xy"][i],
            track["pitch_length"], track["pitch_width"],
        )
        if ownership is None:
            continue
        records.append({
            "period":    int(track["period"][i]),
            "elapsed":   float(track["elapsed"][i]),
            "frame_idx": i,
            "pc_map":    ownership,
        })
    return pd.DataFrame(records)

def compute_obso_for_match(track, epv_grid, owner_lookup, radius=OBSO_RADIUS_M, downsample=OBSO_DOWNSAMPLE):
    if epv_grid is None: return pd.DataFrame()
    pl, pw = track["pitch_length"], track["pitch_width"]
    n = len(track["period"])
    step = 1.0
    offsets = [(dx, dy) for dx in np.arange(-radius, radius + step, step)
               for dy in np.arange(-radius, radius + step, step) if dx * dx + dy * dy <= radius * radius]
    records = []
    for i in range(0, n, downsample):
        bx, by = track["ball_xy"][i]
        if not np.isfinite(bx) or not np.isfinite(by): continue
        period = int(track["period"][i])
        team = owner_lookup(period, i)
        if team is None: continue
        direction = attack_direction(team, period, track["meta"])
        max_epv = epv_value(epv_grid, bx, by, pl, pw, direction)
        for dx, dy in offsets:
            xt, yt = bx + dx, by + dy
            if xt < 0 or xt > pl or yt < 0 or yt > pw: continue
            v = epv_value(epv_grid, xt, yt, pl, pw, direction)
            if v > max_epv: max_epv = v
        records.append({"period": period, "elapsed": float(track["elapsed"][i]), "team": team, "obso": max_epv})
    return pd.DataFrame(records)

def compute_signed_epv_series(track, epv_grid, owner_lookup):
    if epv_grid is None: return pd.DataFrame()
    pl, pw = track["pitch_length"], track["pitch_width"]
    n = len(track["period"])
    records = []
    for i in range(n):
        bx, by = track["ball_xy"][i]
        if not np.isfinite(bx) or not np.isfinite(by): continue
        period = int(track["period"][i])
        team = owner_lookup(period, i)
        if team is None: continue
        direction = attack_direction(team, period, track["meta"])
        v = epv_value(epv_grid, bx, by, pl, pw, direction)
        records.append({"period": period, "elapsed": float(track["elapsed"][i]),
                         "team": team, "signed_epv": v if team == "home" else -v})
    return pd.DataFrame(records)

def bucket_epv_by_second(signed_epv_df):
    if signed_epv_df.empty: return pd.DataFrame(columns=["period", "secondIntoPeriod", "meanSignedEPV"])
    rows = []
    for p, grp in signed_epv_df.groupby("period"):
        bucket = np.floor(grp["elapsed"].to_numpy()).astype(int)
        s = grp["signed_epv"].to_numpy()
        for b in np.unique(bucket):
            m = bucket == b
            rows.append({"period": int(p), "secondIntoPeriod": int(b), "meanSignedEPV": float(np.mean(s[m]))})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------
# 3b. DAS + xT + off-ball xT
# ------------------------------------------------------------------------
def _grid_points(pitch_length, pitch_width, grid_res):
    xs = np.linspace(0, pitch_length, grid_res)
    ys = np.linspace(0, pitch_width, grid_res)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack((gx.ravel(), gy.ravel()))

def _grid_value_lookup(value_grid, grid_points, pitch_length, pitch_width, direction):
    n_rows, n_cols = value_grid.shape
    gx = grid_points[:, 0] if direction == 1 else (pitch_length - grid_points[:, 0])
    gy = grid_points[:, 1]
    col = np.clip((gx / pitch_length * n_cols).astype(int), 0, n_cols - 1)
    row = np.clip((gy / pitch_width * n_rows).astype(int), 0, n_rows - 1)
    return value_grid[row, col]

def compute_das_and_offball_xt_for_match(track, epv_grid, owner_lookup, xt_grid=None,
                                          grid_res=DAS_GRID_RES, downsample=DAS_DOWNSAMPLE,
                                          carrier_radius=OFFBALL_CARRIER_RADIUS_M):
    if epv_grid is None:
        return pd.DataFrame()
    if xt_grid is None:
        xt_grid = epv_grid

    pl, pw = track["pitch_length"], track["pitch_width"]
    grid_points = _grid_points(pl, pw, grid_res)
    cell_area = (pl / grid_res) * (pw / grid_res)

    epv_dir = {1: _grid_value_lookup(epv_grid, grid_points, pl, pw, 1),
               -1: _grid_value_lookup(epv_grid, grid_points, pl, pw, -1)}
    xt_dir = {1: _grid_value_lookup(xt_grid, grid_points, pl, pw, 1),
              -1: _grid_value_lookup(xt_grid, grid_points, pl, pw, -1)}

    n = len(track["period"])
    records = []
    for i in range(0, n, downsample):
        period = int(track["period"][i])
        team = owner_lookup(period, i)
        if team is None:
            continue
        direction = attack_direction(team, period, track["meta"])

        home_xy_i, away_xy_i = track["home_xy"][i], track["away_xy"][i]
        home_valid = np.all(np.isfinite(home_xy_i), axis=1)
        away_valid = np.all(np.isfinite(away_xy_i), axis=1)
        home_pts, away_pts = home_xy_i[home_valid], away_xy_i[away_valid]
        if len(home_pts) + len(away_pts) < 2:
            continue

        pts = np.vstack([home_pts, away_pts])
        team_labels = np.array(["home"] * len(home_pts) + ["away"] * len(away_pts))
        player_idx = np.concatenate([np.where(home_valid)[0], np.where(away_valid)[0]])

        nearest = np.argmin(cdist(grid_points, pts), axis=1)
        cell_owner_team = team_labels[nearest]
        cell_owner_player = player_idx[nearest]

        att_mask = cell_owner_team == team
        danger = epv_dir[direction]
        das_score = float(np.sum(danger[att_mask]) * cell_area)
        das_area_m2 = float(np.sum(att_mask) * cell_area)

        bx, by = track["ball_xy"][i]
        carrier_player_idx = None
        if np.isfinite(bx) and np.isfinite(by):
            att_xy_full = home_xy_i if team == "home" else away_xy_i
            att_valid = home_valid if team == "home" else away_valid
            valid_idx = np.where(att_valid)[0]
            if len(valid_idx):
                dists = np.linalg.norm(att_xy_full[valid_idx] - np.array([bx, by]), axis=1)
                j = int(np.argmin(dists))
                if dists[j] <= carrier_radius:
                    carrier_player_idx = int(valid_idx[j])

        threat = xt_dir[direction]
        if carrier_player_idx is not None:
            offball_mask = att_mask & (cell_owner_player != carrier_player_idx)
            ball_carrier_xt = float(np.sum(threat[att_mask & (cell_owner_player == carrier_player_idx)]) * cell_area)
        else:
            offball_mask = att_mask
            ball_carrier_xt = np.nan
        off_ball_xt = float(np.sum(threat[offball_mask]) * cell_area)

        records.append({
            "period": period, "elapsed": float(track["elapsed"][i]), "team": team,
            "das_score": das_score, "das_area_m2": das_area_m2,
            "off_ball_xt": off_ball_xt, "ball_carrier_xt": ball_carrier_xt,
        })
    return pd.DataFrame(records)

def compute_signed_xt_series_vectorized(track, xt_grid, owner_arr):
    if xt_grid is None:
        return pd.DataFrame()

    pl, pw = track["pitch_length"], track["pitch_width"]
    n_rows, n_cols = xt_grid.shape
    ball_xy = track["ball_xy"]
    has_owner = np.array([o is not None for o in owner_arr])
    valid = np.all(np.isfinite(ball_xy), axis=1) & has_owner
    if not valid.any():
        return pd.DataFrame()

    idx = np.where(valid)[0]
    periods = track["period"][idx].astype(int)
    teams = np.array([owner_arr[k] for k in idx])
    meta = track["meta"]

    direction = np.empty(len(idx), dtype=np.int8)
    for team_val in ("home", "away"):
        for p_val in np.unique(periods):
            m = (teams == team_val) & (periods == p_val)
            if m.any():
                direction[m] = attack_direction(team_val, int(p_val), meta)

    bx, by = ball_xy[idx, 0], ball_xy[idx, 1]
    gx = np.where(direction == 1, bx, pl - bx)
    col = np.clip((gx / pl * n_cols).astype(int), 0, n_cols - 1)
    row = np.clip((by / pw * n_rows).astype(int), 0, n_rows - 1)
    xt_val = xt_grid[row, col]
    signed_xt = np.where(teams == "home", xt_val, -xt_val)

    return pd.DataFrame({
        "period": periods, "elapsed": track["elapsed"][idx].astype(float),
        "team": teams, "xt": xt_val, "signed_xt": signed_xt,
    })

# ==============================================================================
# 4. TRACKING LOADER
# ==============================================================================
def _pick_players(frame, key, smoothed_key):
    if not isinstance(frame, dict): return []
    if USE_SMOOTHED_TRACKING and smoothed_key in frame and frame[smoothed_key]:
        return frame[smoothed_key]
    return frame.get(key, [])

def _pick_ball(frame):
    if not isinstance(frame, dict): return []
    if USE_SMOOTHED_TRACKING and frame.get("ballsSmoothed"): return frame["ballsSmoothed"]
    return frame.get("balls", [])

def load_tracking(match_id, processed_dir):
    folder = Path(processed_dir) / str(match_id)
    with open(folder / "metadata.json", "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    meta = meta_raw[0] if isinstance(meta_raw, list) and meta_raw else meta_raw

    if "stadium" in meta and "pitches" in meta["stadium"]:
        pitch_data = meta["stadium"]["pitches"][0] if isinstance(meta["stadium"]["pitches"], list) else meta["stadium"]["pitches"]
        pl, pw = pitch_data.get("length", 105.0), pitch_data.get("width", 68.0)
    elif "pitch" in meta and isinstance(meta["pitch"], dict):
        pl, pw = meta["pitch"].get("length", 105.0), meta["pitch"].get("width", 68.0)
    else:
        pl, pw = meta.get("pitchLength", 105.0), meta.get("pitchWidth", 68.0)

    x_shift, y_shift = pl / 2.0, pw / 2.0
    logger.info(f"[{match_id}] Applying coordinate shift: x+={x_shift}, y+={y_shift}")

    rows = []
    with bz2.open(folder / "tracking.jsonl.bz2", "rt") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                continue

    if not rows:
        return None

    home_ids, away_ids = set(), set()
    for fr in rows:
        for p_ in _pick_players(fr, "homePlayers", "homePlayersSmoothed"):
            pid = _get(p_, PLAYER_ID_KEYS)
            if pid: home_ids.add(pid)
        for p_ in _pick_players(fr, "awayPlayers", "awayPlayersSmoothed"):
            pid = _get(p_, PLAYER_ID_KEYS)
            if pid: away_ids.add(pid)

    def _key(v):
        try: return (0, int(v))
        except: return (1, str(v))
    home_ids, away_ids = sorted(home_ids, key=_key), sorted(away_ids, key=_key)
    h_idx, a_idx = {pid: i for i, pid in enumerate(home_ids)}, {pid: i for i, pid in enumerate(away_ids)}

    n = len(rows)
    frame_num = np.zeros(n, np.int64)
    period    = np.zeros(n, np.int16)
    elapsed   = np.zeros(n, np.float32)
    game_clock = np.zeros(n, np.float32)
    home_xy = np.full((n, len(home_ids), 2), np.nan, np.float32)
    away_xy = np.full((n, len(away_ids), 2), np.nan, np.float32)
    home_rel = np.zeros((n, len(home_ids)), bool)
    away_rel = np.zeros((n, len(away_ids)), bool)
    ball_xy = np.full((n, 2), np.nan, np.float32)
    ball_z = np.full(n, np.nan, np.float32)
    possession_event_ids = np.full(n, -1, dtype=np.int64)
    game_event_ids = np.full(n, -1, dtype=np.int64)

    for i, fr in enumerate(rows):
        frame_num[i] = fr.get("frameNum", i)
        period[i] = fr.get("period", 0)
        elapsed[i] = fr.get("periodElapsedTime", 0.0)

        gc = fr.get("periodGameClockTime", fr.get("period_game_clock_time"))
        game_clock[i] = float(gc) if gc is not None else np.nan

        pe_id = fr.get("possession_event_id", fr.get("possessionEventId"))
        ge_id = fr.get("game_event_id", fr.get("gameEventId"))
        if pe_id is not None:
            try: possession_event_ids[i] = int(pe_id)
            except (TypeError, ValueError): pass
        if ge_id is not None:
            try: game_event_ids[i] = int(ge_id)
            except (TypeError, ValueError): pass

        for p_ in _pick_players(fr, "homePlayers", "homePlayersSmoothed"):
            pid, x, y = _get(p_, PLAYER_ID_KEYS), _get(p_, PLAYER_X_KEYS), _get(p_, PLAYER_Y_KEYS)
            if pid is not None and x is not None and y is not None:
                j = h_idx[pid]
                home_xy[i, j] = (float(x) + x_shift, float(y) + y_shift)
                home_rel[i, j] = (_get(p_, PLAYER_VIS_KEYS, "VISIBLE") in VISIBLE_VALUES) and \
                                 (_get(p_, PLAYER_CONF_KEYS, "HIGH") not in REJECT_CONFIDENCE_VALUES)

        for p_ in _pick_players(fr, "awayPlayers", "awayPlayersSmoothed"):
            pid, x, y = _get(p_, PLAYER_ID_KEYS), _get(p_, PLAYER_X_KEYS), _get(p_, PLAYER_Y_KEYS)
            if pid is not None and x is not None and y is not None:
                j = a_idx[pid]
                away_xy[i, j] = (float(x) + x_shift, float(y) + y_shift)
                away_rel[i, j] = (_get(p_, PLAYER_VIS_KEYS, "VISIBLE") in VISIBLE_VALUES) and \
                                 (_get(p_, PLAYER_CONF_KEYS, "HIGH") not in REJECT_CONFIDENCE_VALUES)

        balls = _pick_ball(fr)
        if balls:
            b = balls[0] if isinstance(balls, list) else balls
            bx, by, bz = _get(b, PLAYER_X_KEYS), _get(b, PLAYER_Y_KEYS), _get(b, PLAYER_Z_KEYS)
            if bx is not None and by is not None:
                ball_xy[i] = (float(bx) + x_shift, float(by) + y_shift)
            if bz is not None:
                ball_z[i] = float(bz)

    del rows

    order = np.argsort(frame_num)
    order_time = np.argsort(elapsed)

    period_time_index = {}
    for p in np.unique(period):
        idxs = np.where(period == p)[0]
        p_order = np.argsort(elapsed[idxs])
        period_time_index[int(p)] = (elapsed[idxs][p_order], idxs[p_order])

    gc_valid = np.isfinite(game_clock)
    if gc_valid.any():
        gc_idx = np.where(gc_valid)[0]
        gc_order = np.argsort(game_clock[gc_idx])
        gc_sorted = game_clock[gc_idx][gc_order]
        gc_frame_idx = gc_idx[gc_order]
    else:
        gc_sorted = np.array([], dtype=np.float32)
        gc_frame_idx = np.array([], dtype=np.int64)

    pe_to_frames = defaultdict(list)
    for i, pe in enumerate(possession_event_ids):
        if pe > 0: pe_to_frames[int(pe)].append(i)
    ge_to_frames = defaultdict(list)
    for i, ge in enumerate(game_event_ids):
        if ge > 0: ge_to_frames[int(ge)].append(i)
    n_ids_found = int((possession_event_ids > 0).sum()) + int((game_event_ids > 0).sum())
    if n_ids_found == 0:
        logger.warning(f"[{match_id}] No possession_event_id/game_event_id found on any tracking "
                        f"frame -- event-to-frame matching will fall back to game-clock lookup.")

    home_x_min = np.nanmin(home_xy[:, :, 0]); home_x_max = np.nanmax(home_xy[:, :, 0])
    home_y_min = np.nanmin(home_xy[:, :, 1]); home_y_max = np.nanmax(home_xy[:, :, 1])
    ball_x_min = np.nanmin(ball_xy[:, 0]);     ball_x_max = np.nanmax(ball_xy[:, 0])

    logger.info(f"[{match_id}] === DATA QUALITY REPORT ===")
    logger.info(f"  Total frames loaded: {n}")
    logger.info(f"  Home X range: [{home_x_min:.2f}, {home_x_max:.2f}]")
    logger.info(f"  Home Y range: [{home_y_min:.2f}, {home_y_max:.2f}]")
    logger.info(f"  Ball X range:  [{ball_x_min:.2f}, {ball_x_max:.2f}]")

    frames_with_home = np.sum(np.any(np.isfinite(home_xy), axis=(1, 2)))
    frames_with_away = np.sum(np.any(np.isfinite(away_xy), axis=(1, 2)))
    frames_with_ball = np.sum(np.all(np.isfinite(ball_xy), axis=1))

    logger.info(f"  Frames with >0 Home players: {frames_with_home} ({100*frames_with_home/n:.1f}%)")
    logger.info(f"  Frames with >0 Away players: {frames_with_away} ({100*frames_with_away/n:.1f}%)")
    logger.info(f"  Frames with valid Ball:      {frames_with_ball} ({100*frames_with_ball/n:.1f}%)")
    logger.info(f"  Frames with PE id:           {int((possession_event_ids > 0).sum())}")
    logger.info(f"  Frames with game-clock time: {int(gc_valid.sum())}")
    logger.info(f"===============================")

    return {"meta": meta, "pitch_length": pl, "pitch_width": pw, "fps": meta.get("fps", 25) or 25,
            "home_ids": home_ids, "away_ids": away_ids,
            "frame_num": frame_num, "period": period, "elapsed": elapsed,
            "game_clock": game_clock,
            "home_xy": home_xy, "away_xy": away_xy, "home_rel": home_rel, "away_rel": away_rel,
            "ball_xy": ball_xy, "ball_z": ball_z,
            "_frame_num_sorted": frame_num[order], "_frame_idx_sorted": order,
            "_elapsed_sorted": elapsed[order_time], "_frame_idx_sorted_time": order_time,
            "_period_time_index": period_time_index,
            "_game_clock_sorted": gc_sorted, "_game_clock_frame_idx": gc_frame_idx,
            "possession_event_ids": possession_event_ids, "game_event_ids": game_event_ids,
            "_pe_to_frames": dict(pe_to_frames), "_ge_to_frames": dict(ge_to_frames)}

def frame_range_mask(track, start_frame, end_frame):
    fns = track["_frame_num_sorted"]
    lo, hi = np.searchsorted(fns, start_frame, "left"), np.searchsorted(fns, end_frame, "right")
    return np.sort(track["_frame_idx_sorted"][lo:hi]) if lo < hi else np.array([], np.int64)

def frame_at_time(track, target_time, period=None):
    if period is not None:
        idx = track.get("_period_time_index", {}).get(period)
        if idx is not None:
            elapsed_arr, idx_arr = idx
            if len(elapsed_arr):
                pos = min(np.searchsorted(elapsed_arr, target_time, "left"), len(elapsed_arr) - 1)
                return int(idx_arr[pos])
    times = track["_elapsed_sorted"]
    pos = min(np.searchsorted(times, target_time, "left"), len(times) - 1)
    return int(track["_frame_idx_sorted_time"][pos])

def frame_at_game_clock(track, target_game_clock):
    gc = track.get("_game_clock_sorted")
    idx_arr = track.get("_game_clock_frame_idx")
    if gc is None or idx_arr is None or len(gc) == 0:
        return None
    if not np.isfinite(target_game_clock):
        return None
    pos = int(np.searchsorted(gc, target_game_clock, "left"))
    pos = max(0, min(pos, len(gc) - 1))
    return int(idx_arr[pos])

PE_ID_KEYS = ["possession_event_id", "possessionEventId"]
GE_CLOCK_KEYS = ["startGameClock", "start_game_clock", "gameClock", "game_clock"]

def _candidate_frames_for_event(track, ev):
    pe_id = _get(ev, PE_ID_KEYS)
    if pe_id is not None:
        cand = track.get("_pe_to_frames", {}).get(int(pe_id))
        if cand: return cand
    ge_id = _get(ev, GE_GAME_EVENT_ID_KEYS)
    if ge_id is not None:
        cand = track.get("_ge_to_frames", {}).get(int(ge_id))
        if cand: return cand
    return None

def frame_of_event(track, ev, prefer_last=False, max_scan=25):
    cand = _candidate_frames_for_event(track, ev)
    if cand:
        ball_xy = track["ball_xy"]
        n = len(track["period"])
        ordered = sorted(cand, reverse=prefer_last)
        for f in ordered:
            bx, by = ball_xy[f]
            if np.isfinite(bx) and np.isfinite(by):
                return f
        anchor = ordered[0]
        for off in range(1, max_scan + 1):
            for f in (anchor + off, anchor - off):
                if 0 <= f < n:
                    bx, by = ball_xy[f]
                    if np.isfinite(bx) and np.isfinite(by):
                        return f
    gc = _get(ev, GE_CLOCK_KEYS)
    if gc is not None:
        fi = frame_at_game_clock(track, float(gc))
        if fi is not None:
            return fi
    return None

# ==============================================================================
# 5. EVENT PARSING
# ==============================================================================
GE_SEQUENCE_KEYS = ["sequence"]
GE_START_TIME_KEYS = ["start_time", "startTime"]
GE_END_TIME_KEYS = ["end_time", "endTime"]
GE_EVENT_TIME_KEYS = ["eventTime", "event_time"]
GE_DURATION_KEYS = ["duration"]
GE_SETPIECE_KEYS = ["setpieceType", "setpiece_type"]
GE_VIDEO_MISSING_KEYS = ["videoMissing", "video_missing"]
GE_GAME_EVENT_ID_KEYS = ["game_event_id", "gameEventId"]
GE_HOME_BALL_KEYS = ["home_ball"]
GE_HOME_TEAM_KEYS = ["home_team", "homeTeam"]
PE_TYPE_KEYS = ["possession_event_type", "possessionEventType"]
PE_OUTCOME_KEYS = ["outcome", "outcomeType", "passOutcomeType", "crossOutcomeType"]
PE_LINES_BROKEN_KEYS = ["linesBrokenType"]
PE_PASS_TYPE_KEYS = ["passType"]
PE_ACCURACY_KEYS = ["accuracyType"]
PE_INCOMPLETION_KEYS = ["incompletionReasonType"]
PE_CARRY_SUCCESS_KEYS = ["carrySuccessful"]
PE_SHOT_OUTCOME_KEYS = ["shotOutcomeType"]
PE_PRESSURE_KEYS = ["pressureType"]
PE_OPPORTUNITY_KEYS = ["opportunityType"]
PE_INIT_TOUCH_TYPE_KEYS = ["initialTouchType"]
PE_INIT_PRESSURE_KEYS = ["initialPressureType"]
PE_FACING_KEYS = ["facingType"]
PE_CARRY_INTENT_KEYS = ["carryIntent"]
PE_CARRY_TYPE_KEYS = ["carryType"]
PE_CHALLENGE_OUTCOME_KEYS = ["challengeOutcomeType"]
PE_CHALLENGE_TYPE_KEYS = ["challengeType"]
PE_CLEARANCE_OUTCOME_KEYS = ["clearanceOutcomeType"]
PE_SHOT_TYPE_KEYS = ["shotType"]
GE_START_FRAME_KEYS = ["start_frame", "startFrame"]
GE_END_FRAME_KEYS = ["end_frame", "endFrame"]
PE_START_FRAME_KEYS = ["start_frame", "startFrame"]
PE_END_FRAME_KEYS = ["end_frame", "endFrame"]
GE_GAME_EVENT_TYPE_KEYS = ["game_event_type", "gameEventType"]

_SUBDICT_KEYS = ("gameEvents", "initialTouch", "possessionEvents", "fouls", "game_event", "possession_event")
def flatten_event(rec):
    if not isinstance(rec, dict): return {}
    flat = dict(rec)
    for key in _SUBDICT_KEYS:
        sub = rec.get(key)
        if isinstance(sub, dict): flat.update(sub)
    return flat

def load_flat_events(match_id, processed_dir):
    p = Path(processed_dir) / str(match_id) / "events.json"
    if not p.exists(): return []
    with open(p, "r", encoding="utf-8") as f: raw = json.load(f)
    if isinstance(raw, dict): raw = raw.get("data", raw.get("events", [raw]))
    if not isinstance(raw, list): raw = [raw]
    return [flatten_event(r) for r in raw if isinstance(r, dict)]

def _possession_frame_bounds(track, rows):
    all_frames = []
    for r in rows:
        cand = _candidate_frames_for_event(track, r)
        if cand: all_frames.extend(cand)
    if not all_frames:
        return None, None
    return min(all_frames), max(all_frames)

def _game_clock_bounds(track, rows):
    clocks = [_get(r, GE_CLOCK_KEYS) for r in rows if _get(r, GE_CLOCK_KEYS) is not None]
    if not clocks:
        return None, None
    start_gc = float(min(clocks))
    end_gc   = float(max(clocks))
    start_idx = frame_at_game_clock(track, start_gc)
    end_idx   = frame_at_game_clock(track, end_gc)
    return start_idx, end_idx

def build_possessions_from_events(events, track):
    groups = defaultdict(list)
    for r in events:
        seq = _get(r, GE_SEQUENCE_KEYS)
        if seq is not None: groups[seq].append(r)

    last_game_clock = None
    gc_arr = track.get("game_clock")
    if gc_arr is not None and np.isfinite(gc_arr).any():
        last_game_clock = float(np.nanmax(gc_arr))

    possessions = []
    for seq, rows in groups.items():
        home_ball_votes = [_get(r, GE_HOME_BALL_KEYS) for r in rows if _get(r, GE_HOME_BALL_KEYS) is not None]
        if home_ball_votes:
            team = "home" if sum(1 for v in home_ball_votes if v is True) >= sum(1 for v in home_ball_votes if v is False) else "away"
        else:
            home_team_votes = [_get(r, GE_HOME_TEAM_KEYS) for r in rows if _get(r, GE_HOME_TEAM_KEYS) is not None]
            if not home_team_votes: continue
            team = "home" if sum(1 for v in home_team_votes if v is True) >= sum(1 for v in home_team_votes if v is False) else "away"

        start_times = [_get(r, GE_START_TIME_KEYS) for r in rows if _get(r, GE_START_TIME_KEYS) is not None]
        end_times = [_get(r, GE_END_TIME_KEYS) for r in rows if _get(r, GE_END_TIME_KEYS) is not None]
        if not start_times or not end_times: continue

        start_sec, end_sec = float(min(start_times)), float(max(end_times))
        period_val = _get(rows[0], ["period"])
        period = int(period_val) if period_val is not None else (1 if start_sec < 2700 else 2)

        if last_game_clock is not None:
            row_clocks = [_get(r, GE_CLOCK_KEYS) for r in rows if _get(r, GE_CLOCK_KEYS) is not None]
            if row_clocks and float(min(row_clocks)) > last_game_clock + POST_END_GRACE_SECONDS:
                continue

        start_idx, end_idx = _possession_frame_bounds(track, rows)
        if start_idx is None or end_idx is None:
            gc_start, gc_end = _game_clock_bounds(track, rows)
            if gc_start is not None and gc_end is not None:
                start_idx, end_idx = gc_start, gc_end
        if start_idx is None or end_idx is None:
            start_idx = frame_at_time(track, start_sec, period)
            end_idx   = frame_at_time(track, end_sec,   period)

        if start_idx is None or end_idx is None:
            continue

        start_frame = int(track["frame_num"][start_idx])
        end_frame   = int(track["frame_num"][end_idx])

        start_sec_period = float(track["elapsed"][start_idx])
        end_sec_period   = float(track["elapsed"][end_idx])

        seen_ge, game_events = set(), []
        for r in rows:
            geid = _get(r, GE_GAME_EVENT_ID_KEYS)
            if geid not in seen_ge:
                seen_ge.add(geid)
                game_events.append(r)
        game_events.sort(key=lambda r: _get(r, GE_START_TIME_KEYS) or 0)

        duration = end_sec - start_sec

        possessions.append({
            "sequence_id": seq, "team": team, "period": period,
            "start_frame": start_frame, "end_frame": end_frame, "duration": duration,
            "start_sec": start_sec, "end_sec": end_sec,
            "start_sec_period": start_sec_period, "end_sec_period": end_sec_period,
            "setpiece_type": _get(game_events[0], GE_SETPIECE_KEYS) if game_events else None,
            "video_missing": any(bool(_get(r, GE_VIDEO_MISSING_KEYS, False)) for r in game_events),
            "possession_events": rows, "game_events": game_events,
        })
    return sorted(possessions, key=lambda p: p["start_frame"])

def _dist_dict(values):
    values = [v for v in values if v is not None]
    return json.dumps(pd.Series(values).value_counts().to_dict()) if values else "{}"

PE_CODE_NAMES = {"BC": "carry", "CH": "challenge", "CL": "clearance", "CR": "cross", "PA": "pass", "RE": "rebound", "SH": "shot"}

def aggregate_possession_events(poss, track, direction):
    rows = poss["possession_events"]
    codes = [_get(r, PE_TYPE_KEYS) for r in rows]
    names = [PE_CODE_NAMES.get(c) for c in codes]
    out = {"n_events": len(rows)}
    for code, name in PE_CODE_NAMES.items(): out[f"n_{name}s"] = names.count(name)

    passes = [r for r, n in zip(rows, names) if n in ("pass", "cross")]
    out["n_pass_attempts"] = len(passes)
    completed = [r for r in passes if _get(r, PE_OUTCOME_KEYS) == "C"]
    out["n_completed_passes"] = len(completed)
    out["pass_completion_rate"] = len(completed) / len(passes) if passes else np.nan
    out["lines_broken_count"] = sum(1 for r in passes if _get(r, PE_LINES_BROKEN_KEYS) is not None)
    out["pass_type_distribution"] = _dist_dict([_get(r, PE_PASS_TYPE_KEYS) for r in passes])
    out["accuracy_type_distribution"] = _dist_dict([_get(r, PE_ACCURACY_KEYS) for r in completed])
    out["incompletion_reason_distribution"] = _dist_dict([_get(r, PE_INCOMPLETION_KEYS) for r in passes if r not in completed])

    carries = [r for r, n in zip(rows, names) if n == "carry"]
    out["successful_carries"] = sum(1 for r in carries if _get(r, PE_CARRY_SUCCESS_KEYS) is True)
    out["carry_success_rate"] = out["successful_carries"] / len(carries) if carries else np.nan
    out["carry_intent_distribution"] = _dist_dict([_get(r, PE_CARRY_INTENT_KEYS) for r in carries])
    out["carry_type_distribution"] = _dist_dict([_get(r, PE_CARRY_TYPE_KEYS) for r in carries])

    prog_events = sorted([r for r, n in zip(rows, names) if n in ("pass", "cross", "carry")], key=lambda r: _get(r, GE_START_TIME_KEYS) or 0)
    dists, fwd = [], []
    prev_pos = None
    for r in prog_events:
        fi = frame_of_event(track, r)
        pos = track["ball_xy"][fi].astype(float) if fi is not None else np.array([np.nan, np.nan])
        if prev_pos is not None and np.all(np.isfinite(pos)) and np.all(np.isfinite(prev_pos)):
            dists.append(float(np.linalg.norm(pos - prev_pos)))
            fwd.append(float(direction * (pos[0] - prev_pos[0])))
        prev_pos = pos
    out["mean_progression_distance"] = float(np.mean(dists)) if dists else np.nan
    out["total_progression_distance"] = float(np.sum(dists)) if dists else np.nan
    out["mean_forward_progression"] = float(np.mean(fwd)) if fwd else np.nan
    out["total_forward_progression"] = float(np.sum(fwd)) if fwd else np.nan

    shots = [r for r, n in zip(rows, names) if n == "shot"]
    out["goals"] = sum(1 for r in shots if _get(r, PE_SHOT_OUTCOME_KEYS) == "G")
    out["shots_on_target"] = sum(1 for r in shots if _get(r, PE_SHOT_OUTCOME_KEYS) in ("G", "S", "B", "L"))

    pressures = [_get(r, PE_PRESSURE_KEYS) for r in rows if _get(r, PE_PRESSURE_KEYS) is not None]
    out["n_pressured_actions"] = sum(1 for p in pressures if p in ("A", "P", "L"))
    out["pressure_rate"] = out["n_pressured_actions"] / len(rows) if rows else np.nan

    opps = [_get(r, PE_OPPORTUNITY_KEYS) for r in rows if _get(r, PE_OPPORTUNITY_KEYS) is not None]
    out["n_chances_created"] = opps.count("C")
    out["n_dangerous_positions"] = opps.count("D")
    return out

# ==============================================================================
# 6. VECTORIZED SPATIAL FEATURES
# ==============================================================================
def compute_all_spatial_features(track):
    length = track["pitch_length"]
    home_xy, away_xy, ball_xy = track["home_xy"], track["away_xy"], track["ball_xy"]
    N = len(track["period"])

    def get_shapes(xy_arr):
        centroids = np.nanmean(xy_arr, axis=1)
        dist_to_centroid = np.linalg.norm(xy_arr - centroids[:, None, :], axis=2)
        compactness = np.nanmean(dist_to_centroid, axis=1)

        x_coords, y_coords = xy_arr[..., 0], xy_arr[..., 1]
        width = np.nanmax(y_coords, axis=1) - np.nanmin(y_coords, axis=1)
        depth = np.nanmax(x_coords, axis=1) - np.nanmin(x_coords, axis=1)
        elongation = np.where(width > 1e-6, depth / width, np.nan)

        area = np.full(N, np.nan)
        valid_frames = np.sum(np.all(np.isfinite(xy_arr), axis=2), axis=1) >= 3
        for i in np.where(valid_frames)[0]:
            pts = xy_arr[i][np.all(np.isfinite(xy_arr[i]), axis=1)]
            if len(pts) >= 3:
                try: area[i] = ConvexHull(pts).volume
                except Exception: area[i] = np.nan

        diff = xy_arr[:, :, None, :] - xy_arr[:, None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=3)
        P = xy_arr.shape[1]
        eye = np.eye(P, dtype=bool)
        dist_matrix[:, eye] = np.nan
        nearest_teammate = np.nanmin(dist_matrix, axis=2)
        nearest_teammate_mean = np.nanmean(nearest_teammate, axis=1)

        mask = np.triu(np.ones((P, P), dtype=bool), 1)
        pairwise_dist = np.where(mask[None, :, :], dist_matrix, np.nan)
        pairwise_mean = np.nanmean(pairwise_dist, axis=(1, 2))

        return {
            "centroid_x": centroids[:, 0], "centroid_y": centroids[:, 1],
            "compactness": compactness, "width": width, "depth": depth,
            "elongation": elongation, "area": area,
            "nearest_teammate_distance": nearest_teammate_mean,
            "pairwise_distance": pairwise_mean,
            "centroids": centroids, "_xy": xy_arr
        }

    home_shapes, away_shapes = get_shapes(home_xy), get_shapes(away_xy)

    def get_ball_relations(xy_arr, ball_xy, centroids, direction):
        dist = np.hypot(centroids[:, 0] - ball_xy[:, 0], centroids[:, 1] - ball_xy[:, 1])
        x_coords, ball_x = xy_arr[..., 0], ball_xy[:, 0:1]
        valid = np.all(np.isfinite(xy_arr), axis=2)
        ball_valid = np.isfinite(ball_xy[:, 0]) & np.isfinite(ball_xy[:, 1])
        ahead = x_coords > ball_x if direction == 1 else x_coords < ball_x
        ahead_valid = ahead & valid
        ahead_count = np.sum(ahead_valid, axis=1).astype(float)
        behind_count = np.sum(valid & ~ahead, axis=1).astype(float)
        ahead_count[~ball_valid] = np.nan
        behind_count[~ball_valid] = np.nan
        return {
            "centroid_to_ball_distance": dist,
            "players_ahead_of_ball": ahead_count,
            "players_behind_ball": behind_count
        }

    home_rel_dir1 = get_ball_relations(home_xy, ball_xy, home_shapes["centroids"], 1)
    home_rel_dirM1 = get_ball_relations(home_xy, ball_xy, home_shapes["centroids"], -1)
    away_rel_dir1 = get_ball_relations(away_xy, ball_xy, away_shapes["centroids"], 1)
    away_rel_dirM1 = get_ball_relations(away_xy, ball_xy, away_shapes["centroids"], -1)

    def get_interactions(att_xy, def_xy, att_shapes, def_shapes):
        centroid_dist = np.hypot(att_shapes["centroid_x"] - def_shapes["centroid_x"], att_shapes["centroid_y"] - def_shapes["centroid_y"])
        long_diff = att_shapes["centroid_x"] - def_shapes["centroid_x"]
        lat_diff = att_shapes["centroid_y"] - def_shapes["centroid_y"]
        width_diff = att_shapes["width"] - def_shapes["width"]
        width_ratio = np.where(def_shapes["width"] > 1e-6, att_shapes["width"] / def_shapes["width"], np.nan)
        depth_diff = att_shapes["depth"] - def_shapes["depth"]
        depth_ratio = np.where(def_shapes["depth"] > 1e-6, att_shapes["depth"] / def_shapes["depth"], np.nan)
        compactness_diff = att_shapes["compactness"] - def_shapes["compactness"]
        compactness_ratio = np.where(def_shapes["compactness"] > 1e-6, att_shapes["compactness"] / def_shapes["compactness"], np.nan)
        area_diff = att_shapes["area"] - def_shapes["area"]
        area_ratio = np.where(def_shapes["area"] > 1e-6, att_shapes["area"] / def_shapes["area"], np.nan)
        diff = att_xy[:, :, None, :] - def_xy[:, None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=3)
        nearest_opp = np.nanmin(dist_matrix, axis=2)
        return {
            "centroid_distance": centroid_dist, "longitudinal_centroid_difference": long_diff,
            "lateral_centroid_difference": lat_diff, "width_difference": width_diff, "width_ratio": width_ratio,
            "depth_difference": depth_diff, "depth_ratio": depth_ratio,
            "compactness_difference": compactness_diff, "compactness_ratio": compactness_ratio,
            "area_difference": area_diff, "area_ratio": area_ratio,
            "nearest_opponent_distance": np.nanmean(nearest_opp, axis=1)
        }

    inter_home_att = get_interactions(home_xy, away_xy, home_shapes, away_shapes)
    inter_away_att = get_interactions(away_xy, home_xy, away_shapes, home_shapes)

    return {
        "home": home_shapes, "away": away_shapes,
        "home_rel_dir1": home_rel_dir1, "home_rel_dirM1": home_rel_dirM1,
        "away_rel_dir1": away_rel_dir1, "away_rel_dirM1": away_rel_dirM1,
        "inter_home_att": inter_home_att, "inter_away_att": inter_away_att
    }

def agg_series(name, values, weights=None):
    arr = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(arr)
    out = {}
    if not finite_mask.any():
        for suf in ("mean", "std", "min", "max", "range", "start", "end", "change"): out[f"{name}_{suf}"] = np.nan
        return out
    finite = arr[finite_mask]
    if weights is not None:
        w = np.asarray(weights, dtype=float)[finite_mask]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() > 0:
            wmean = float(np.average(finite, weights=w))
            wvar = float(np.average((finite - wmean) ** 2, weights=w))
            out[f"{name}_mean"], out[f"{name}_std"] = wmean, math.sqrt(wvar)
        else:
            out[f"{name}_mean"], out[f"{name}_std"] = float(np.mean(finite)), float(np.std(finite))
    else:
        out[f"{name}_mean"], out[f"{name}_std"] = float(np.mean(finite)), float(np.std(finite))
    out[f"{name}_min"] = float(np.min(finite))
    out[f"{name}_max"] = float(np.max(finite))
    out[f"{name}_range"] = out[f"{name}_max"] - out[f"{name}_min"]
    idx = np.where(finite_mask)[0]
    out[f"{name}_start"] = float(arr[idx[0]])
    out[f"{name}_end"] = float(arr[idx[-1]])
    out[f"{name}_change"] = out[f"{name}_end"] - out[f"{name}_start"]
    return out

# ==============================================================================
# 7. FORMATION PIPELINE
# ==============================================================================
def goalkeepers_from_metadata(meta):
    gk_meta = meta.get("goalkeepers", {}) or {}
    return {side: {str(entry["shirtNumber"])} if entry and entry.get("shirtNumber") else None for side, entry in ((s, gk_meta.get(s)) for s in ("home", "away"))}

def identify_goalkeeper_by_distance(xy_arr, ids, min_frames=FORMATION_GK_MIN_FRAMES):
    n_players = xy_arr.shape[1]
    if n_players == 0: return None
    total_dist, frame_count = np.zeros(n_players), np.zeros(n_players, dtype=int)
    for j in range(n_players):
        traj = xy_arr[:, j, :]
        finite = np.all(np.isfinite(traj), axis=1)
        traj_f = traj[finite]
        frame_count[j] = len(traj_f)
        if len(traj_f) >= 2: total_dist[j] = float(np.sum(np.linalg.norm(np.diff(traj_f, axis=0), axis=1)))
    candidates = {j: total_dist[j] / frame_count[j] for j in range(n_players) if frame_count[j] >= min_frames}
    if not candidates: candidates = {j: total_dist[j] / frame_count[j] for j in range(n_players) if frame_count[j] > 0}
    return {str(ids[min(candidates, key=candidates.get)])} if candidates else None

def resolve_goalkeepers(track):
    result = goalkeepers_from_metadata(track["meta"])
    if not result.get("home"): result["home"] = identify_goalkeeper_by_distance(track["home_xy"], track["home_ids"])
    if not result.get("away"): result["away"] = identify_goalkeeper_by_distance(track["away_xy"], track["away_ids"])
    return result

def build_templates(pitch_length, pitch_width):
    pitch = Pitch(pitch_type="custom", pitch_length=pitch_length, pitch_width=pitch_width)
    df = pitch.formations_dataframe
    return {f: {"names": sub["name"].tolist(), "normal": sub[["x", "y"]].to_numpy(dtype=float), "flipped": sub[["x_flip", "y_flip"]].to_numpy(dtype=float)}
            for f in pitch.formations if len(sub := df[(df["formation"] == f) & (df["name"] != "GK")]) == 10}

def match_formation(player_xy, templates, orientation):
    best_formation, best_cost, best_names = None, np.inf, None
    for formation, tmpl in templates.items():
        cost_matrix = cdist(player_xy, tmpl[orientation])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cost = cost_matrix[row_ind, col_ind].sum()
        norm_cost = cost / len(row_ind)
        if norm_cost < best_cost: best_cost, best_formation, best_names = norm_cost, formation, [tmpl["names"][i] for i in col_ind]
    return best_formation, best_cost, best_names

def get_orientation(team, period, meta):
    home_start_left = meta.get("homeTeamStartLeftExtraTime" if period in (3, 4) else "homeTeamStartLeft", meta.get("homeTeamStartLeft", True))
    home_attacks_left_to_right = home_start_left if period % 2 == 1 else not home_start_left
    return "normal" if (team == "home" and home_attacks_left_to_right) or (team == "away" and not home_attacks_left_to_right) else "flipped"

def get_window_indices(elapsed_seconds, stride_seconds, window_seconds):
    return range(max(0, math.ceil((elapsed_seconds - window_seconds) / stride_seconds)), int(elapsed_seconds // stride_seconds) + 1)

def event_period_sec(track, ev):
    fi = frame_of_event(track, ev)
    if fi is not None:
        return int(track["period"][fi]), float(track["elapsed"][fi])

    sf = _get(ev, GE_START_FRAME_KEYS)
    if sf is not None:
        fi = np.searchsorted(track["_frame_num_sorted"], sf, side="left")
        fi = min(fi, len(track["_frame_num_sorted"]) - 1)
        idx = track["_frame_idx_sorted"][fi]
        return int(track["period"][idx]), float(track["elapsed"][idx])

    gc = _get(ev, GE_CLOCK_KEYS)
    if gc is not None:
        fi = frame_at_game_clock(track, float(gc))
        if fi is not None:
            return int(track["period"][fi]), float(track["elapsed"][fi])
    return None, None

def classify_event_disruptions(events, track):
    records = []
    for ev in events:
        gtype = _get(ev, GE_GAME_EVENT_TYPE_KEYS)
        period, sec = event_period_sec(track, ev)
        if period is None or sec is None: continue
        home = _get(ev, GE_HOME_TEAM_KEYS)
        team_key = "homePlayers" if home else "awayPlayers"
        if gtype in ("FOU", "FOUL"): records.append((period, sec, sec + FOUL_DECAY_SECONDS, team_key, 0.0))
        elif gtype == "SUB": records.append((period, sec, sec + SUB_DECAY_SECONDS, team_key, 0.0))
    return records

def detect_turnovers(track, threshold=2.5, stride=5):
    turnovers, prev_owner = [], None
    for i in range(0, len(track["period"]), stride):
        bx, by = track["ball_xy"][i]
        if not np.isfinite(bx): continue
        best_team, best_dist = None, threshold
        for team_key, xy in (("homePlayers", track["home_xy"][i]), ("awayPlayers", track["away_xy"][i])):
            pts = xy[np.all(np.isfinite(xy), axis=1)]
            if len(pts) == 0: continue
            d = float(np.min(np.linalg.norm(pts - [bx, by], axis=1)))
            if d < best_dist: best_dist, best_team = d, team_key
        if prev_owner is not None and best_team is not None and best_team != prev_owner: turnovers.append((int(track["period"][i]), float(track["elapsed"][i])))
        if best_team is not None: prev_owner = best_team
    return turnovers

def compute_frame_weights(track, events, stride=5):
    fps, n = track["fps"], len(track["period"])
    w_home, w_away = np.ones(n), np.ones(n)
    disruptions = classify_event_disruptions(events, track)
    for period, start, end, team_key, base in disruptions:
        mask = (track["period"] == period) & (track["elapsed"] >= start) & (track["elapsed"] <= end)
        if not mask.any() or end <= start: continue
        frac = np.clip((track["elapsed"][mask] - start) / (end - start), 0, 1)
        weight = base + (1.0 - base) * frac
        if team_key in ("both", "homePlayers"): w_home[mask] *= weight
        if team_key in ("both", "awayPlayers"): w_away[mask] *= weight
    return w_home, w_away

def accumulate_positions(track, goalkeepers, w_home, w_away, stride=5):
    buckets = defaultdict(lambda: defaultdict(list))
    gk = {"home": goalkeepers.get("home") or set(), "away": goalkeepers.get("away") or set()}
    for i in range(0, len(track["period"]), stride):
        period, elapsed = int(track["period"][i]), float(track["elapsed"][i])
        for side, xy_arr, ids, w_arr in (("home", track["home_xy"], track["home_ids"], w_home), ("away", track["away_xy"], track["away_ids"], w_away)):
            w = w_arr[i]
            for j, pid in enumerate(ids):
                if str(pid) in gk[side]: continue
                x, y = xy_arr[i, j]
                if not (np.isfinite(x) and np.isfinite(y)): continue
                for k in get_window_indices(elapsed, FORMATION_STRIDE_SECONDS, FORMATION_WINDOW_SECONDS):
                    buckets[(side, period, k)][pid].append((x, y, w))
    return buckets

def build_formation_windows(track, templates, goalkeepers, w_home, w_away):
    buckets = accumulate_positions(track, goalkeepers, w_home, w_away)
    rows = []
    for (side, period, k), players in sorted(buckets.items()):
        n_frames = sum(len(v) for v in players.values())
        if n_frames < FORMATION_MIN_FRAMES_PER_WINDOW: continue
        avg_xy, weight_sum = [], 0.0
        for pid, triples in players.items():
            arr = np.array(triples, dtype=float)
            wsum = arr[:, 2].sum()
            if wsum <= 0: continue
            avg_xy.append(((arr[:, 0] * arr[:, 2]).sum() / wsum, (arr[:, 1] * arr[:, 2]).sum() / wsum))
            weight_sum += wsum
        if not avg_xy: continue
        avg_xy = np.array(avg_xy)
        if avg_xy.shape[0] < FORMATION_MIN_OUTFIELD_PLAYERS: continue
        orientation = get_orientation(side, period, track["meta"])
        formation, cost, _ = match_formation(avg_xy, templates, orientation)
        fit_quality = 1.0 / (1.0 + float(cost))
        confidence = weight_sum * fit_quality
        if confidence < MIN_WINDOW_CONFIDENCE: continue
        window_start = k * FORMATION_STRIDE_SECONDS
        rows.append({"team": side, "period": period, "windowIndex": k, "windowStartSec": window_start, "windowEndSec": window_start + FORMATION_WINDOW_SECONDS,
                     "nOutfieldPlayers": int(avg_xy.shape[0]), "nFrames": int(n_frames), "formation": formation, "orientation": orientation,
                     "avgCostPerPlayer": round(float(cost), 3), "confidence": round(float(confidence), 4),
                     "mean_compactness": float(np.mean(np.linalg.norm(avg_xy - avg_xy.mean(axis=0), axis=1))),
                     "mean_width": float(np.ptp(avg_xy[:, 1])), "mean_depth": float(np.ptp(avg_xy[:, 0]))})
    df = pd.DataFrame(rows)
    return df.sort_values(["team", "period", "windowIndex"]) if not df.empty else df

def derive_hierarchy(raw_formation):
    variant = str(raw_formation).removesuffix("flat") if raw_formation else raw_formation
    family = {"pyramid": "back-2", "metodo": "back-3", "wm": "back-3"}.get(variant, f"back-{variant[0]}" if variant and variant[0].isdigit() else "other")
    return {"variant": variant, "family": family}

def build_formation_segments(windows_df, possessions, epv_bucket_df, das_df, obso_df):
    if windows_df.empty: return pd.DataFrame()
    segments = []
    for (team, period), grp in windows_df.groupby(["team", "period"], sort=False):
        grp = grp.sort_values("windowStartSec").reset_index(drop=True)
        cur, seg_start, seg_end = None, None, None
        for _, row in grp.iterrows():
            formation = str(row["formation"]).strip()
            start, end = float(row["windowStartSec"]), float(row["windowEndSec"])
            if formation == cur: seg_end = end
            else:
                if cur is not None: segments.append({"team": team, "period": period, "formation": cur, "variant": derive_hierarchy(cur)["variant"], "family": derive_hierarchy(cur)["family"],
                                                     "start_sec": round(seg_start, 2), "end_sec": round(seg_end, 2), "duration": round(seg_end - seg_start, 2),
                                                     "mean_confidence": float(grp[(grp["windowStartSec"] >= seg_start) & (grp["windowEndSec"] <= seg_end)]["confidence"].mean())})
                cur, seg_start, seg_end = formation, start, end
        if cur is not None: segments.append({"team": team, "period": period, "formation": cur, "variant": derive_hierarchy(cur)["variant"], "family": derive_hierarchy(cur)["family"],
                                             "start_sec": round(seg_start, 2), "end_sec": round(seg_end, 2), "duration": round(seg_end - seg_start, 2),
                                             "mean_confidence": float(grp[(grp["windowStartSec"] >= seg_start) & (grp["windowEndSec"] <= seg_end)]["confidence"].mean())})
    return pd.DataFrame(segments)

def evaluate_das(possessions, signed_epv_df, threshold=DAS_EPV_THRESHOLD, min_duration=DAS_MIN_DURATION_SECONDS):
    rows = []
    for p in possessions:
        if p["team"] not in ("home", "away"): continue
        dur = p.get("duration") or (p["end_sec"] - p["start_sec"])
        if dur is None or dur < min_duration: continue
        peak = 0.0
        if signed_epv_df is not None and not signed_epv_df.empty:
            m = (signed_epv_df["period"] == p["period"]) & \
                (signed_epv_df["elapsed"] >= p["start_sec_period"]) & \
                (signed_epv_df["elapsed"] <= p["end_sec_period"])
            sub = signed_epv_df[m]
            if not sub.empty:
                vals = sub["signed_epv"].to_numpy()
                peak = float(np.max(vals if p["team"] == "home" else -vals))
        rows.append({"sequence_id": p["sequence_id"], "team": p["team"], "period": p["period"],
                     "start_sec": p["start_sec"], "end_sec": p["end_sec"], "duration": dur, "peakEPV": peak, "isDAS": peak >= threshold})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["sequence_id", "team", "period", "start_sec", "end_sec", "duration", "peakEPV", "isDAS"])

def attach_formation(possession_row, segments_df):
    if segments_df.empty: return {}
    team, period = possession_row["team"], possession_row["period"]
    s = possession_row.get("start_sec_period")
    e = possession_row.get("end_sec_period")
    if s is None or e is None:
        return {}
    cand = segments_df[(segments_df["team"] == team) & (segments_df["period"] == period) &
                       (segments_df["start_sec"] < e) & (segments_df["end_sec"] > s)]
    if cand.empty: return {}
    overlaps = np.minimum(cand["end_sec"], e) - np.maximum(cand["start_sec"], s)
    best = cand.iloc[int(np.argmax(overlaps.values))]
    return {"dominant_formation": best["formation"], "formation_variant": best["variant"],
            "formation_family": best["family"], "formation_confidence": best["mean_confidence"]}

# ==============================================================================
# 8. BUILD ONE POSSESSION ROW
# ==============================================================================
def build_possession_row(match_id, poss, track, pc_df, obso_df, formation_segments, spatial_features,
                          das_offball_df=None, signed_xt_df=None):
    frame_idx = frame_range_mask(track, poss["start_frame"], poss["end_frame"])
    if len(frame_idx) == 0: return None

    meta = track["meta"]
    team, period = poss["team"], poss["period"]
    opp_team = "away" if team == "home" else "home"
    direction = attack_direction(team, period, meta)
    length = track["pitch_length"]

    t0_period = poss["start_sec_period"]
    t1_period = poss["end_sec_period"]

    rel_home = track["home_rel"][frame_idx]; rel_away = track["away_rel"][frame_idx]
    has_home = np.all(np.isfinite(track["home_xy"][frame_idx]), axis=2)
    has_away = np.all(np.isfinite(track["away_xy"][frame_idx]), axis=2)
    total_slots = has_home.sum() + has_away.sum()
    reliable_slots = (rel_home & has_home).sum() + (rel_away & has_away).sum()
    reliability_ratio = reliable_slots / total_slots if total_slots else np.nan

    per_frame_reliability = np.full(len(frame_idx), np.nan)
    for k in range(len(frame_idx)):
        tot = has_home[k].sum() + has_away[k].sum()
        rel = (rel_home[k] & has_home[k]).sum() + (rel_away[k] & has_away[k]).sum()
        per_frame_reliability[k] = rel / tot if tot else np.nan

    keep = np.ones(len(frame_idx), dtype=bool)
    weights = None
    if RELIABILITY_MODE == "filter":
        keep = per_frame_reliability >= RELIABILITY_THRESHOLD
        if not keep.any(): keep = np.ones(len(frame_idx), dtype=bool)
    elif RELIABILITY_MODE == "weighted":
        weights = np.where(np.isfinite(per_frame_reliability), per_frame_reliability, 0.0)

    used_idx = frame_idx[keep]
    weights = weights[keep] if weights is not None else None
    if len(used_idx) == 0: return None

    if team == "home":
        att_shapes, def_shapes = spatial_features["home"], spatial_features["away"]
        att_rel = spatial_features["home_rel_dir1"] if direction == 1 else spatial_features["home_rel_dirM1"]
        def_rel = spatial_features["away_rel_dir1"] if direction == 1 else spatial_features["away_rel_dirM1"]
        inter = spatial_features["inter_home_att"]
    else:
        att_shapes, def_shapes = spatial_features["away"], spatial_features["home"]
        att_rel = spatial_features["away_rel_dir1"] if direction == 1 else spatial_features["away_rel_dirM1"]
        def_rel = spatial_features["home_rel_dir1"] if direction == 1 else spatial_features["home_rel_dirM1"]
        inter = spatial_features["inter_away_att"]

    metrics = defaultdict(list)
    for prefix, shapes, rel in [("attacking", att_shapes, att_rel), ("defending", def_shapes, def_rel)]:
        metrics[f"{prefix}_centroid_x"] = (length - shapes["centroid_x"][used_idx]) if direction < 0 else shapes["centroid_x"][used_idx]
        metrics[f"{prefix}_centroid_y"] = shapes["centroid_y"][used_idx]
        metrics[f"{prefix}_compactness"] = shapes["compactness"][used_idx]
        metrics[f"{prefix}_width"] = shapes["width"][used_idx]
        metrics[f"{prefix}_depth"] = shapes["depth"][used_idx]
        metrics[f"{prefix}_elongation"] = shapes["elongation"][used_idx]
        metrics[f"{prefix}_area"] = shapes["area"][used_idx]
        metrics[f"{prefix}_nearest_teammate_distance"] = shapes["nearest_teammate_distance"][used_idx]
        metrics[f"{prefix}_pairwise_distance"] = shapes["pairwise_distance"][used_idx]
        metrics[f"{prefix}_centroid_to_ball_distance"] = rel["centroid_to_ball_distance"][used_idx]
        metrics[f"{prefix}_players_ahead_of_ball"] = rel["players_ahead_of_ball"][used_idx]
        metrics[f"{prefix}_players_behind_ball"] = rel["players_behind_ball"][used_idx]

    metrics["centroid_distance"] = inter["centroid_distance"][used_idx]
    metrics["longitudinal_centroid_difference"] = -inter["longitudinal_centroid_difference"][used_idx] if direction < 0 else inter["longitudinal_centroid_difference"][used_idx]
    metrics["lateral_centroid_difference"] = inter["lateral_centroid_difference"][used_idx]
    metrics["width_difference"] = inter["width_difference"][used_idx]
    metrics["width_ratio"] = inter["width_ratio"][used_idx]
    metrics["depth_difference"] = inter["depth_difference"][used_idx]
    metrics["depth_ratio"] = inter["depth_ratio"][used_idx]
    metrics["compactness_difference"] = inter["compactness_difference"][used_idx]
    metrics["compactness_ratio"] = inter["compactness_ratio"][used_idx]
    metrics["area_difference"] = inter["area_difference"][used_idx]
    metrics["area_ratio"] = inter["area_ratio"][used_idx]
    metrics["nearest_opponent_distance"] = inter["nearest_opponent_distance"][used_idx]

    ball_x = track["ball_xy"][used_idx, 0]
    metrics["ball_x"] = (length - ball_x) if direction < 0 else ball_x
    metrics["ball_y"] = track["ball_xy"][used_idx, 1]
    metrics["ball_z"] = track["ball_z"][used_idx]

    row = {
        "match_id": str(match_id), "sequence_id": poss["sequence_id"], "period": period,
        "team": team, "opponent_team": opp_team,
        "start_frame": poss["start_frame"], "end_frame": poss["end_frame"],
        "start_sec": poss["start_sec"], "end_sec": poss["end_sec"], "duration": poss["duration"],
        "start_sec_period": poss["start_sec_period"], "end_sec_period": poss["end_sec_period"],
        "setpiece_type": poss["setpiece_type"], "open_play_indicator": poss["setpiece_type"] in (None, "O"),
        "video_missing": poss["video_missing"],
        "n_unique_frames": int(len(frame_idx)), "reliability_ratio": reliability_ratio,
    }

    for key, vals in metrics.items():
        use_weights = weights if key not in ("players_ahead_of_ball", "players_behind_ball") else None
        row.update(agg_series(key, vals, use_weights))

    dur = row["duration"] if row["duration"] and row["duration"] > 0 else np.nan
    for prefix in ("attacking", "defending"):
        cx, cy = np.array(metrics[f"{prefix}_centroid_x"]), np.array(metrics[f"{prefix}_centroid_y"])
        finite = np.isfinite(cx) & np.isfinite(cy)
        if finite.sum() >= 2:
            fi = np.where(finite)[0]
            disp = float(np.hypot(cx[fi[-1]] - cx[fi[0]], cy[fi[-1]] - cy[fi[0]]))
        else: disp = np.nan
        row[f"{prefix}_centroid_displacement"] = disp
        row[f"{prefix}_centroid_velocity"] = disp / dur if (dur and np.isfinite(disp)) else np.nan
        for f in ("width", "depth", "compactness", "area"):
            row[f"{prefix}_{f}_change_rate"] = row.get(f"{prefix}_{f}_change", np.nan) / dur if dur else np.nan

    speeds = []
    own_arr = track["home_xy"] if team == "home" else track["away_xy"]
    sub = own_arr[used_idx]
    fps = track["fps"]
    for p_ in range(sub.shape[1]):
        traj = sub[:, p_, :]
        finite = np.all(np.isfinite(traj), axis=1)
        traj_f = traj[finite]
        if len(traj_f) >= 2: speeds.extend((np.linalg.norm(np.diff(traj_f, axis=0), axis=1) * fps).tolist())
    row["mean_player_speed"] = float(np.mean(speeds)) if speeds else np.nan
    row["std_player_speed"] = float(np.std(speeds)) if speeds else np.nan
    row["max_player_speed"] = float(np.max(speeds)) if speeds else np.nan

    row.update(aggregate_possession_events(poss, track, direction))
    row["events_per_second"] = row["n_events"] / dur if dur else np.nan

    # ---- Pitch control: per-phase mean ownership map ------------------------
    pc_map_flat = np.full(PC_MAP_SIZE, np.nan, dtype=PC_MAP_DTYPE)
    if pc_df is not None and not pc_df.empty:
        m = (pc_df["period"] == period) & (pc_df["elapsed"] >= t0_period) & (pc_df["elapsed"] <= t1_period)
        if m.any():
            frames_maps = np.stack(pc_df.loc[m, "pc_map"].values)          # (F, grid_x, grid_y)
            mean_map = frames_maps.mean(axis=0).astype(np.float32)         # in [0, 1]
            if team == "away":
                mean_map = 1.0 - mean_map
            if direction < 0:
                mean_map = mean_map[::-1, :]
            pc_map_flat = mean_map.ravel().astype(PC_MAP_DTYPE)
    row["pc_map"] = pc_map_flat

    if obso_df is not None and not obso_df.empty:
        m = (obso_df["period"] == period) & (obso_df["elapsed"] >= t0_period) & (obso_df["elapsed"] <= t1_period) & (obso_df["team"] == team)
        if m.any(): row.update(agg_series("obso", obso_df.loc[m, "obso"].to_numpy()))

    if das_offball_df is not None and not das_offball_df.empty:
        m = (das_offball_df["period"] == period) & (das_offball_df["elapsed"] >= t0_period) & \
            (das_offball_df["elapsed"] <= t1_period) & (das_offball_df["team"] == team)
        if m.any():
            sub = das_offball_df.loc[m]
            row.update(agg_series("das_score", sub["das_score"].to_numpy()))
            row.update(agg_series("das_area_m2", sub["das_area_m2"].to_numpy()))
            row.update(agg_series("off_ball_xt", sub["off_ball_xt"].to_numpy()))
            row.update(agg_series("ball_carrier_xt", sub["ball_carrier_xt"].to_numpy()))

    if signed_xt_df is not None and not signed_xt_df.empty:
        m = (signed_xt_df["period"] == period) & (signed_xt_df["elapsed"] >= t0_period) & \
            (signed_xt_df["elapsed"] <= t1_period) & (signed_xt_df["team"] == team)
        if m.any():
            row.update(agg_series("xt", signed_xt_df.loc[m, "xt"].to_numpy()))

    row.update(attach_formation(row, formation_segments))
    return row

# ==============================================================================
# 9. PER-MATCH DRIVER
# ==============================================================================
def build_match(match_id, processed_dir, epv_grid, xt_grid=None):
    logger.info(f"[{match_id}] Step 1/5: Loading tracking data...")
    track = load_tracking(match_id, processed_dir)
    if track is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info(f"[{match_id}] Step 2/5: Loading events and building possessions...")
    events = load_flat_events(match_id, processed_dir)
    possessions = build_possessions_from_events(events, track)
    if not possessions:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info(f"[{match_id}] Step 3/5: Computing pitch control, OBSO, EPV, DAS, and xT...")
    owner_arr = np.full(len(track["period"]), None, dtype=object)
    for p in possessions:
        owner_arr[frame_range_mask(track, p["start_frame"], p["end_frame"])] = p["team"]

    def owner_lookup(period, i):
        return owner_arr[i]

    pc_df = compute_pc_for_match(track)
    obso_df = compute_obso_for_match(track, epv_grid, owner_lookup) if epv_grid is not None else pd.DataFrame()
    signed_epv_df = compute_signed_epv_series(track, epv_grid, owner_lookup) if epv_grid is not None else pd.DataFrame()
    epv_bucket_df = bucket_epv_by_second(signed_epv_df)
    das_df = evaluate_das(possessions, signed_epv_df)

    effective_xt_grid = xt_grid if xt_grid is not None else epv_grid
    das_offball_df = compute_das_and_offball_xt_for_match(track, epv_grid, owner_lookup, xt_grid=xt_grid) \
        if epv_grid is not None else pd.DataFrame()
    signed_xt_df = compute_signed_xt_series_vectorized(track, effective_xt_grid, owner_arr) \
        if effective_xt_grid is not None else pd.DataFrame()

    logger.info(f"[{match_id}] Step 3.5/5: Precomputing spatial features (Vectorized)...")
    spatial_features = compute_all_spatial_features(track)

    logger.info(f"[{match_id}] Step 4/5: Building formation windows and segments...")
    templates = build_templates(track["pitch_length"], track["pitch_width"])
    goalkeepers = resolve_goalkeepers(track)
    w_home, w_away = compute_frame_weights(track, events)
    windows_df = build_formation_windows(track, templates, goalkeepers, w_home, w_away)
    formation_segments = build_formation_segments(windows_df, possessions, epv_bucket_df, das_df, obso_df)

    logger.info(f"[{match_id}] Step 5/5: Aggregating possession-level features...")
    rows = []
    failed_possessions = 0
    empty_frame_possessions = 0

    for poss in possessions:
        try:
            frame_idx = frame_range_mask(track, poss["start_frame"], poss["end_frame"])
            if len(frame_idx) == 0:
                empty_frame_possessions += 1
                continue

            row = build_possession_row(match_id, poss, track, pc_df, obso_df, formation_segments, spatial_features,
                                        das_offball_df=das_offball_df, signed_xt_df=signed_xt_df)
            if row:
                rows.append(row)
            else:
                failed_possessions += 1
        except Exception as ex:
            logger.error(f"Possession {poss.get('sequence_id')} in {match_id} failed: {ex}")
            failed_possessions += 1

    logger.info(f"[{match_id}] === POSSESSION HEALTH ===")
    logger.info(f"  Total possessions generated:   {len(possessions)}")
    logger.info(f"  Successfully processed:        {len(rows)}")
    logger.info(f"  Failed/Returned None:          {failed_possessions}")
    logger.info(f"  Empty frame mapping:           {empty_frame_possessions}")
    logger.info(f"===============================")

    poss_df = pd.DataFrame(rows)
    windows_df = windows_df.assign(match_id=str(match_id)) if not windows_df.empty else windows_df
    formation_segments = formation_segments.assign(match_id=str(match_id)) if not formation_segments.empty else formation_segments

    logger.info(f"[{match_id}] Match processing complete.")
    return poss_df, windows_df, formation_segments

# ==============================================================================
# 10. MULTICORE WORKER GLUE
# ==============================================================================
_WORKER = {}

def _init_worker(processed_dir, epv_grid_path, xt_grid_path):
    _WORKER["processed_dir"] = processed_dir
    _WORKER["epv_grid"] = load_epv_grid(epv_grid_path)
    _WORKER["xt_grid"] = load_grid(xt_grid_path)

def _process_one_match_worker(match_id):
    t0 = time.time()
    try:
        poss_df, windows_df, seg_df = build_match(
            match_id,
            _WORKER["processed_dir"],
            _WORKER["epv_grid"],
            _WORKER["xt_grid"],
        )
        return match_id, poss_df, windows_df, seg_df, None, time.time() - t0
    except Exception:
        import traceback
        return match_id, None, None, None, traceback.format_exc(), time.time() - t0

# ==============================================================================
# 11. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    root = Path(PROCESSED_DIR)
    match_ids = [d.name for d in root.iterdir() if d.is_dir() and (d / "metadata.json").exists() and (d / "tracking.jsonl.bz2").exists() and (d / "events.json").exists()] if root.exists() else []
    match_ids = sorted(match_ids)
    total_found = len(match_ids)
    if NUM_MATCHES is not None:
        match_ids = match_ids[:NUM_MATCHES]

    if not match_ids:
        print(f"No processed matches found under {PROCESSED_DIR}.")
    else:
        epv_grid = load_epv_grid(EPV_GRID_PATH)
        xt_grid = load_grid(XT_GRID_PATH)
        if epv_grid is None: logger.warning(f"No EPV grid at {EPV_GRID_PATH}")
        if xt_grid is None: logger.warning(f"No dedicated xT grid at {XT_GRID_PATH} -- falling back to the EPV grid for xT/off-ball xT.")

        auto = max(1, (os.cpu_count() or 2) // 2)
        workers = NUM_WORKERS or min(len(match_ids), auto)
        workers = max(1, int(workers))

        print(f"\n{'='*70}\nFOUND {total_found} MATCHES, PROCESSING {len(match_ids)} "
              f"(NUM_MATCHES={NUM_MATCHES}, NUM_WORKERS={workers})\n{'='*70}\n")

        all_poss, all_windows, all_segments = [], [], []
        t_start = time.time()
        done = [0]

        def _absorb(mid, poss_df, windows_df, seg_df, err, dt):
            done[0] += 1
            if err is not None:
                print(f"  [{done[0]}/{len(match_ids)}] {mid}  FAILED ({dt:.1f}s)")
                if done[0] == 1:
                    print(err)
                return
            n_poss = 0 if poss_df is None or poss_df.empty else len(poss_df)
            n_segs = 0 if seg_df is None or seg_df.empty else len(seg_df)
            if poss_df is not None and not poss_df.empty: all_poss.append(poss_df)
            if windows_df is not None and not windows_df.empty: all_windows.append(windows_df)
            if seg_df is not None and not seg_df.empty: all_segments.append(seg_df)
            print(f"  [{done[0]}/{len(match_ids)}] {mid}  possessions={n_poss:4d}  "
                  f"segments={n_segs:3d}  ({dt:.1f}s)")

        if workers == 1:
            _init_worker(PROCESSED_DIR, EPV_GRID_PATH, XT_GRID_PATH)
            for mid in match_ids:
                print(f"\n{'='*70}\nPROCESSING MATCH {done[0]+1}/{len(match_ids)}: {mid}\n{'='*70}")
                _absorb(*_process_one_match_worker(mid))
        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(PROCESSED_DIR, EPV_GRID_PATH, XT_GRID_PATH),
                ) as ex:
                    futs = [ex.submit(_process_one_match_worker, mid) for mid in match_ids]
                    for fut in as_completed(futs):
                        _absorb(*fut.result())
            except Exception as ex:
                print(f"  [WARN] parallel path failed ({ex!r}); falling back to single process")
                _init_worker(PROCESSED_DIR, EPV_GRID_PATH, XT_GRID_PATH)
                for mid in match_ids:
                    _absorb(*_process_one_match_worker(mid))

        print(f"\nAll matches done in {time.time() - t_start:.1f}s.")

        print(f"\n{'='*70}\nFINALIZING AND SAVING RESULTS...\n{'='*70}")
        suffix = "" if NUM_MATCHES == 64 else f"_test{NUM_MATCHES}"
        OUTPUT_PATH = OUTPUT_PATH_BASE.replace(".parquet", f"{suffix}.parquet")
        FORMATION_WINDOWS_OUTPUT = FORMATION_WINDOWS_BASE.replace(".parquet", f"{suffix}.parquet")
        FORMATION_SEGMENTS_OUTPUT = FORMATION_SEGMENTS_BASE.replace(".parquet", f"{suffix}.parquet")
        print(f"Saving with suffix '{suffix}': {OUTPUT_PATH}")

        # Note: to save with the pc_map as a fixed-size list column, use the
        # helper in build_tactical_phases.py or see _save_with_pc_map there.
        if all_poss:
            result = pd.concat(all_poss, ignore_index=True)
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                flat_size = PC_MAP_SIZE
                maps_list = []
                for m in result["pc_map"].values:
                    arr = np.asarray(m, dtype=np.float16).ravel() if m is not None else np.full(flat_size, np.nan, dtype=np.float16)
                    if arr.shape[0] != flat_size:
                        arr = np.full(flat_size, np.nan, dtype=np.float16)
                    maps_list.append(arr)
                flat = np.concatenate(maps_list)
                pc_map_col = pa.FixedSizeListArray.from_arrays(
                    pa.array(flat, type=pa.float16()), list_size=flat_size)
                df_no_pc = result.drop(columns=["pc_map"])
                table = pa.Table.from_pandas(df_no_pc, preserve_index=False)
                table = table.append_column("pc_map", pc_map_col)
                pq.write_table(table, OUTPUT_PATH, compression="zstd")
                print(f"SUCCESS: {OUTPUT_PATH} -- {len(result)} possessions (pc_map as fixed-size list)")
            except ImportError:
                result.to_parquet(OUTPUT_PATH, index=False)
                print(f"SUCCESS (fallback): {OUTPUT_PATH} -- {len(result)} possessions")
        if all_windows:
            pd.concat(all_windows, ignore_index=True).to_parquet(FORMATION_WINDOWS_OUTPUT, index=False)
        if all_segments:
            pd.concat(all_segments, ignore_index=True).to_parquet(FORMATION_SEGMENTS_OUTPUT, index=False)
        print(f"\n{'='*70}\nPIPELINE COMPLETE!\n{'='*70}\n")