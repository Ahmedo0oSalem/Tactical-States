"""
unified_prediction.py
=====================

TWO TASKS, ONE PIPELINE.

TASK A — predict the full 75×45 PC map:
    map (3375) → PCA → N_PCS components → per-PC regressors → inverse PCA → predicted map

TASK B — predict 4 scalar tactical targets:
    xt_mean, obso_mean, off_ball_xt_mean, das_score_mean

    NOTE: pitch control is NOT a scalar target anymore — it is only
    predicted as a map (Task A). The old `pitch_control_mean` column is
    not derived and not regressed against.

BOTH tasks use the same feature sets and the same train/test split.

Feature sets:
    Cat 1 — attack + speed + events + formation
    Cat 2 — Cat 1 + defending + interactions
    Cat 3 — formation only

Outputs (PNGs + JSON) go to the parquet's parent directory:
    pcmap_summary_bars.png
    pcmap_samples_<best>.png         true / predicted / error maps
    pcmap_cell_r2_<best>.png         per-cell R² across phases
    pcmap_mean_comparison.png        true mean map vs predicted mean map
    scalar_summary_bars.png
    scalar_r2_heatmap.png
    unified_results.json
"""

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ==============================================================================
# CONFIG
# ==============================================================================
INPUT_PATH = r"Outputs\SixthRun\tactical_phases_test.parquet"
OUTPUT_DIR = Path(r"ModelsOutput")

PC_GRID_X, PC_GRID_Y = 75, 45
PC_MAP_SIZE = PC_GRID_X * PC_GRID_Y
N_PCS = 60

# Pitch control is NOT here — it is only a map target (Task A).
SCALAR_TARGETS = [
    "xt_mean",
    "obso_mean",
    "off_ball_xt_mean",
    "das_score_mean",
]

TEST_FRACTION = 0.20
RANDOM_SEED = 42
N_WORKERS = 6
N_SAMPLE_VIZ = 4

# Set to True to skip Task A on re-runs.
SKIP_TASK_A = False

MODELS = ["hist"] + (["xgb"] if HAS_XGB else [])

HIST_PARAMS = dict(
    max_iter=500, learning_rate=0.05, max_depth=5, min_samples_leaf=30,
    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, random_state=RANDOM_SEED,
)
XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=30,
    reg_lambda=1.0, subsample=0.9, colsample_bytree=0.9,
    tree_method="hist", n_jobs=1, random_state=RANDOM_SEED,
    early_stopping_rounds=30, verbosity=0,
)


# ==============================================================================
# LOGGING
# ==============================================================================
_T0 = time.time()
def log(msg, level="INFO"):
    print(f"[{time.time()-_T0:7.1f}s] [{level:<5s}] {msg}", flush=True)
def section(t):
    log(""); log("=" * 70); log(t); log("=" * 70)


# ==============================================================================
# FEATURE REGISTRY
# ==============================================================================
LEAKING_PREFIXES = [
    "ball_x_", "ball_y_", "ball_z_",
    "attacking_centroid_to_ball_distance_", "defending_centroid_to_ball_distance_",
    "attacking_players_ahead_of_ball_", "attacking_players_behind_ball_",
    "defending_players_ahead_of_ball_", "defending_players_behind_ball_",
]
ID_COLUMNS = {"match_id", "sequence_id", "window_id"}
FORMATION_CAT_COLS = ["dominant_formation", "formation_variant", "formation_family"]
FORMATION_NUM_COLS = ["formation_confidence"]

def _numeric_cols(df, prefixes, exclude=()):
    return [c for c in df.columns
            if c not in ID_COLUMNS and df[c].dtype != object
            and not any(c.startswith(p) for p in exclude)
            and any(c == p or c.startswith(p + "_") for p in prefixes)]

def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

_FORM_CACHE = {}
def _formation_onehot(df):
    if "cols" in _FORM_CACHE:
        return _FORM_CACHE["cols"]
    cols = []
    for c in FORMATION_CAT_COLS:
        if c not in df.columns:
            continue
        df[c] = df[c].astype(str).str.strip()
        dummies = pd.get_dummies(df[c], prefix=f"fmt_{c}", drop_first=True)
        for d in dummies.columns:
            df[d] = dummies[d].astype(np.int8)
            cols.append(d)
    _FORM_CACHE["cols"] = cols
    return cols


def build_feature_sets(df):
    _FORM_CACHE.clear()
    reg = {
        "attacking_shape": lambda d: _numeric_cols(d, ["attacking"], LEAKING_PREFIXES),
        "defending_shape": lambda d: _numeric_cols(d, ["defending"], LEAKING_PREFIXES),
        "interactions":    lambda d: _numeric_cols(d, [
            "centroid_distance", "longitudinal_centroid_difference",
            "lateral_centroid_difference", "width_difference", "width_ratio",
            "depth_difference", "depth_ratio", "compactness_difference",
            "compactness_ratio", "area_difference", "area_ratio",
            "nearest_opponent_distance",
        ], LEAKING_PREFIXES),
        "speed":           lambda d: [c for c in (
            "mean_player_speed", "std_player_speed", "max_player_speed") if c in d.columns],
        "events":          lambda d: [c for c in (
            "duration", "n_events", "n_passs", "n_carrys", "n_crosss",
            "n_challenges", "n_clearances", "n_rebounds", "n_shots",
            "n_pass_attempts", "n_completed_passes", "pass_completion_rate",
            "successful_carries", "lines_broken_count", "events_per_second",
            "goals", "shots_on_target") if c in d.columns],
        "context":         lambda d: [c for c in ("open_play_indicator", "period")
                                       if c in d.columns],
        "phase_context":   lambda d: [c for c in (
            "n_passes_in_window", "threshold_mean", "threshold_max",
            "has_progression_data") if c in d.columns],
        "formation_onehot": _formation_onehot,
        "formation_confidence": lambda d: [c for c in FORMATION_NUM_COLS if c in d.columns],
    }

    def build(categories):
        cols = []
        for name in categories:
            cols += reg[name](df)
        return _dedup(cols)

    return {
        "Cat 1": build(["attacking_shape", "speed", "events", "context",
                         "formation_onehot", "formation_confidence"]),
        "Cat 2": build(["attacking_shape", "defending_shape", "interactions",
                         "speed", "events", "context",
                         "formation_onehot", "formation_confidence"]),
        "Cat 3": build(["formation_onehot", "formation_confidence"]),
    }


# ==============================================================================
# MAP LOADING
# ==============================================================================
def load_maps(df):
    """Return (maps_flat (N, 3375) with NaN→0.5, valid_mask (N,))."""
    raw = np.stack(df["pc_map"].values).astype(np.float32).reshape(-1, PC_GRID_X, PC_GRID_Y)
    all_nan = np.isnan(raw).all(axis=(1, 2))
    filled = np.nan_to_num(raw, nan=0.5)
    return filled.reshape(len(raw), -1), ~all_nan


# ==============================================================================
# MODEL FACTORY
# ==============================================================================
def make_model(name):
    if name == "hist":
        return HistGradientBoostingRegressor(**HIST_PARAMS)
    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return XGBRegressor(**XGB_PARAMS)
    raise ValueError(name)


def _fit_predict(X_tr, y_tr, X_te, model_name):
    """Fit and predict a scalar.

    For xgb we convert to numpy before fitting: passing pandas objects with
    a non-RangeIndex to xgb.fit() alongside eval_set triggers an internal
    index-alignment KeyError.
    """
    m = make_model(model_name)

    if model_name == "xgb":
        X_tr_np = np.asarray(X_tr, dtype=np.float32)
        y_tr_np = np.asarray(y_tr, dtype=np.float32)
        X_te_np = np.asarray(X_te, dtype=np.float32)

        n_val = max(50, int(0.1 * len(X_tr_np)))
        rng = np.random.RandomState(RANDOM_SEED)
        perm = rng.permutation(len(X_tr_np))
        v_idx, t_idx = perm[:n_val], perm[n_val:]

        m.fit(
            X_tr_np[t_idx], y_tr_np[t_idx],
            eval_set=[(X_tr_np[v_idx], y_tr_np[v_idx])],
            verbose=False,
        )
        y_hat = m.predict(X_te_np)
    else:
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)

    return y_hat, m


# ==============================================================================
# TASK A — MAP PREDICTION
# ==============================================================================
def _fit_one_pc(k, X_tr, y_tr, X_te, y_te, model_name):
    t0 = time.time()
    try:
        y_hat, _ = _fit_predict(X_tr, y_tr, X_te, model_name)
        return k, y_hat, float(r2_score(y_te, y_hat)), time.time() - t0, None
    except Exception as e:
        return k, None, float("nan"), time.time() - t0, f"{type(e).__name__}: {e}"


def predict_map_for_cat(df, feat_cols, train_idx, test_idx,
                        pca, pcs_train, pcs_test, model_name):
    # .iloc because train_idx / test_idx are positional (from GroupShuffleSplit)
    X_tr = df.iloc[train_idx][feat_cols].astype(np.float32)
    X_te = df.iloc[test_idx][feat_cols].astype(np.float32)

    pred_pcs = np.full((len(X_te), N_PCS), np.nan, dtype=np.float32)
    pc_r2s = np.full(N_PCS, np.nan)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_fit_one_pc, k, X_tr, pcs_train[:, k],
                        X_te, pcs_test[:, k], model_name): k
            for k in range(N_PCS)
        }
        for fut in as_completed(futures):
            k, y_hat, r2, _, err = fut.result()
            if err:
                log(f"    pc{k}: FAIL {err}", level="ERROR")
                continue
            pred_pcs[:, k] = y_hat
            pc_r2s[k] = r2

    for k in range(N_PCS):
        if np.isnan(pred_pcs[:, k]).any():
            pred_pcs[:, k] = np.nanmean(pcs_train[:, k])

    return pca.inverse_transform(pred_pcs), pc_r2s


def evaluate_map(true_flat, pred_flat):
    true = true_flat.reshape(-1, PC_GRID_X, PC_GRID_Y)
    pred = pred_flat.reshape(-1, PC_GRID_X, PC_GRID_Y)

    overall_r2 = float(r2_score(true.ravel(), pred.ravel()))
    overall_rmse = float(np.sqrt(mean_squared_error(true.ravel(), pred.ravel())))
    overall_mae = float(mean_absolute_error(true.ravel(), pred.ravel()))

    mean_true = true.mean(axis=(1, 2))
    mean_pred = pred.mean(axis=(1, 2))
    mean_r2 = float(r2_score(mean_true, mean_pred))
    mean_rmse = float(np.sqrt(mean_squared_error(mean_true, mean_pred)))

    cell_r2 = np.full((PC_GRID_X, PC_GRID_Y), np.nan, dtype=np.float32)
    for i in range(PC_GRID_X):
        for j in range(PC_GRID_Y):
            tv = true[:, i, j]
            if tv.std() < 1e-6:
                continue
            cell_r2[i, j] = r2_score(tv, pred[:, i, j])

    return {
        "overall_r2": overall_r2, "overall_rmse": overall_rmse, "overall_mae": overall_mae,
        "mean_r2": mean_r2, "mean_rmse": mean_rmse,
        "cell_r2_mean": float(np.nanmean(cell_r2)),
        "cell_r2_median": float(np.nanmedian(cell_r2)),
        "cell_r2_map": cell_r2,
    }


# ==============================================================================
# TASK B — SCALAR TARGETS
# ==============================================================================
def fit_scalar(df, target, feat_cols, train_idx, test_idx, model_name):
    t0 = time.time()
    try:
        # .iloc because train_idx / test_idx are positional
        X_tr = df.iloc[train_idx][feat_cols].astype(np.float32)
        y_tr = df.iloc[train_idx][target]
        X_te = df.iloc[test_idx][feat_cols].astype(np.float32)
        y_te = df.iloc[test_idx][target]

        m_tr, m_te = y_tr.notna(), y_te.notna()
        X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
        X_te, y_te = X_te[m_te], y_te[m_te]

        y_hat, _ = _fit_predict(X_tr, y_tr, X_te, model_name)
        return {
            "target": target, "model": model_name,
            "n_features": len(feat_cols), "n_train": len(X_tr), "n_test": len(X_te),
            "r2":   float(r2_score(y_te, y_hat)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, y_hat))),
            "mae":  float(mean_absolute_error(y_te, y_hat)),
            "fit_time_s": time.time() - t0,
            "error": None,
        }
    except Exception as e:
        return {"target": target, "model": model_name,
                "error": f"{type(e).__name__}: {e}", "fit_time_s": time.time() - t0}


# ==============================================================================
# VISUALISATION — figure builders (caller saves)
# ==============================================================================
def _draw_map(ax, data, title, cmap="RdBu_r", vmin=0, vmax=1):
    im = ax.imshow(data.T, origin="lower", aspect="auto",
                    cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 105, 0, 68])
    ax.set_title(title, fontsize=9); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    return im


def _make_map_summary_fig(results, baseline):
    labels  = [f"{r['cat']}\n{r['model']}" for r in results]
    overall = [r["overall_r2"] for r in results]
    meanr2  = [r["mean_r2"] for r in results]
    cellmed = [r["cell_r2_median"] for r in results]

    x = np.arange(len(results)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.4), 5))
    ax.bar(x - w, overall, w, label="Overall cell R²")
    ax.bar(x,     meanr2, w, label="Phase-mean R²")
    ax.bar(x + w, cellmed, w, label="Median per-cell R²")
    ax.axhline(baseline["overall_r2"], color="k", linestyle="--",
               label="Baseline (mean map)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("R²"); ax.set_ylim(min(0, baseline["overall_r2"] - 0.1), 1)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def _make_sample_predictions_fig(true_maps, pred_maps, n=N_SAMPLE_VIZ):
    idxs = np.random.RandomState(RANDOM_SEED).choice(
        len(true_maps), size=min(n, len(true_maps)), replace=False)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(12, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    for row, i in enumerate(idxs):
        t, p = true_maps[i], pred_maps[i]
        _draw_map(axes[row, 0], t, f"phase {i} — true")
        _draw_map(axes[row, 1], p, f"phase {i} — predicted")
        _draw_map(axes[row, 2], p - t, "error (pred − true)",
                  cmap="coolwarm", vmin=-0.5, vmax=0.5)
    plt.tight_layout()
    return fig


def _make_cell_r2_fig(cell_r2, title=""):
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(cell_r2.T, origin="lower", aspect="auto",
                    cmap="RdYlGn", vmin=-0.5, vmax=1.0, extent=[0, 105, 0, 68])
    ax.set_xlabel("Pitch length (m) — attacking direction →")
    ax.set_ylabel("Pitch width (m)")
    ax.set_title(f"Per-cell R² {title}")
    plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    return fig


def _make_mean_comparison_fig(true_maps, pred_maps):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _draw_map(axes[0], true_maps.mean(axis=0), "True mean map")
    _draw_map(axes[1], pred_maps.mean(axis=0), "Predicted mean map")
    _draw_map(axes[2], pred_maps.mean(axis=0) - true_maps.mean(axis=0),
              "Bias (pred − true)", cmap="coolwarm", vmin=-0.2, vmax=0.2)
    plt.tight_layout()
    return fig


def _make_scalar_bars_fig(scalar_df):
    targets = SCALAR_TARGETS
    cats = sorted(scalar_df["cat"].unique())
    models = sorted(scalar_df["model"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4.5),
                              sharey=True)
    if len(targets) == 1:
        axes = [axes]
    width = 0.8 / max(1, len(cats))
    for ax, t in zip(axes, targets):
        sub = scalar_df[scalar_df["target"] == t]
        for i, m in enumerate(models):
            for j, c in enumerate(cats):
                row = sub[(sub["model"] == m) & (sub["cat"] == c)]
                if row.empty:
                    continue
                ax.bar(i + j * width - 0.4 + width / 2,
                       row["r2"].iloc[0], width=width,
                       label=c if i == 0 else None)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models)
        ax.set_title(t, fontsize=10); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("R²")
    axes[-1].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


def _make_scalar_heatmap_fig(scalar_df):
    pivot = scalar_df.pivot_table(index="target", columns=["model", "cat"],
                                   values="r2", aggfunc="first")
    fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 3, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{m}\n{c}" for m, c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="w" if v < 0.6 else "k", fontsize=8)
    plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    return fig


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    section("Unified prediction: map (Task A) + 4 scalars (Task B)")
    log("Pitch control is a MAP target only — no scalar PC regression.")

    log(f"Loading {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    log(f"  {len(df):,} phases × {df.shape[1]} columns")

    maps_flat, valid = load_maps(df)
    log(f"  Map matrix: {maps_flat.shape}  | valid rows: {valid.sum():,}")

    missing = [t for t in SCALAR_TARGETS if t not in df.columns]
    if missing:
        log(f"  Missing scalar targets: {missing}", level="WARN")

    feature_sets = build_feature_sets(df)
    for k, v in feature_sets.items():
        log(f"  {k}: {len(v)} features")

    # ---- Split --------------------------------------------------------------
    section("Train / test split")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                             random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["match_id"]))
    log(f"  train: {len(train_idx):,} rows | "
        f"{df.iloc[train_idx]['match_id'].nunique()} matches")
    log(f"  test:  {len(test_idx):,} rows | "
        f"{df.iloc[test_idx]['match_id'].nunique()} matches")

    # ==========================================================================
    # TASK A — MAP PREDICTION
    # ==========================================================================
    section("TASK A — Map prediction (PCA + per-PC regressors)")

    map_results = []
    baseline_metrics = {}
    pca = None
    pcs_train = pcs_test = None
    cumvar = np.array([0.0])

    if SKIP_TASK_A:
        log("  SKIP_TASK_A=True — skipping Task A")
    else:
        pca = PCA(n_components=N_PCS, random_state=RANDOM_SEED)
        pcs_train = pca.fit_transform(maps_flat[train_idx])
        pcs_test  = pca.transform(maps_flat[test_idx])
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        log(f"  {N_PCS} PCs explain {cumvar[-1]:.4f} of map variance")
        log(f"  PCA reconstruction R² on train (upper bound): "
            f"{r2_score(maps_flat[train_idx].ravel(), pca.inverse_transform(pcs_train).ravel()):.4f}")

        mean_map_flat = maps_flat[train_idx].mean(axis=0, keepdims=True)
        baseline_pred = np.repeat(mean_map_flat, len(test_idx), axis=0)
        baseline_metrics = evaluate_map(maps_flat[test_idx], baseline_pred)
        log(f"  baseline (mean map):  overall R²={baseline_metrics['overall_r2']:.4f}  "
            f"mean R²={baseline_metrics['mean_r2']:.4f}")

        for cat_name, feat_cols in feature_sets.items():
            for model_name in MODELS:
                log(f"\n  {cat_name} | {model_name} | {len(feat_cols)} feats")
                t0 = time.time()
                pred_flat, _ = predict_map_for_cat(df, feat_cols,
                                                    train_idx, test_idx,
                                                    pca, pcs_train, pcs_test,
                                                    model_name)
                dt = time.time() - t0
                metrics = evaluate_map(maps_flat[test_idx], pred_flat)
                log(f"    {dt:.1f}s  overall R²={metrics['overall_r2']:.4f}  "
                    f"mean R²={metrics['mean_r2']:.4f}  "
                    f"cell-med R²={metrics['cell_r2_median']:.4f}")
                map_results.append({
                    "cat": cat_name, "model": model_name,
                    "n_features": len(feat_cols), "fit_time_s": dt,
                    **{k: v for k, v in metrics.items() if k != "cell_r2_map"},
                    "_cell_r2_map": metrics["cell_r2_map"],
                    "_pred_flat": pred_flat,
                })

        log("\n  MAP SUMMARY")
        log(f"  {'cat':<8} {'model':<6} {'overall R²':>11} {'mean R²':>10} "
            f"{'cell-med':>10} {'RMSE':>8}")
        for r in map_results:
            log(f"  {r['cat']:<8} {r['model']:<6} {r['overall_r2']:>11.4f} "
                f"{r['mean_r2']:>10.4f} {r['cell_r2_median']:>10.4f} "
                f"{r['overall_rmse']:>8.4f}")

    # ==========================================================================
    # TASK B — SCALAR TARGETS
    # ==========================================================================
    section("TASK B — Scalar targets (4 targets, no pitch control)")
    jobs = [(t, feature_sets[c], c, m)
            for t in SCALAR_TARGETS for c in feature_sets for m in MODELS]
    log(f"  jobs: {len(jobs)} | workers: {N_WORKERS}")

    scalar_results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for t, feats, c, m in jobs:
            f = pool.submit(fit_scalar, df, t, feats, train_idx, test_idx, m)
            futures[f] = (t, c, m)
        for i, fut in enumerate(as_completed(futures), 1):
            t, c, m = futures[fut]
            r = fut.result()
            r["cat"] = c
            if r.get("error"):
                log(f"  [{i:>2}/{len(jobs)}] FAIL {t:<22} {c} {m}: {r['error']}",
                    level="ERROR")
            else:
                log(f"  [{i:>2}/{len(jobs)}] {t:<22} {c} {m:<4} "
                    f"R²={r['r2']:>7.4f}  RMSE={r['rmse']:>8.4f}  ({r['fit_time_s']:.1f}s)")
            scalar_results.append(r)

    scalar_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                              for r in scalar_results if not r.get("error")])

    log("\n  SCALAR SUMMARY")
    for t in SCALAR_TARGETS:
        sub = scalar_df[scalar_df["target"] == t]
        if sub.empty:
            continue
        log(f"  {t}:")
        for c in feature_sets:
            line = f"    {c:<6} "
            for m in MODELS:
                row = sub[(sub["cat"] == c) & (sub["model"] == m)]
                line += (f"{m}={row['r2'].iloc[0]:.4f}  "
                         if not row.empty else f"{m}=--     ")
            log(line)

    # ==========================================================================
    # VISUALISATIONS
    # ==========================================================================
    section("Visualisations")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Output dir (absolute): {OUTPUT_DIR.resolve()}")

    def _save(fig, name):
        out_path = (OUTPUT_DIR / name).resolve()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved: {out_path}")

    # ---- Task A visuals -----------------------------------------------------
    if map_results:
        best = max(map_results, key=lambda r: r["overall_r2"])
        log(f"Best map model: {best['cat']} / {best['model']}  "
            f"overall R²={best['overall_r2']:.4f}")

        fig = _make_map_summary_fig(map_results, baseline_metrics)
        _save(fig, "pcmap_summary_bars.png")

        true_maps_test = maps_flat[test_idx].reshape(-1, PC_GRID_X, PC_GRID_Y)
        pred_maps_test = best["_pred_flat"].reshape(-1, PC_GRID_X, PC_GRID_Y)

        fig = _make_sample_predictions_fig(true_maps_test, pred_maps_test)
        _save(fig, f"pcmap_samples_{best['cat']}_{best['model']}.png".replace(" ", "_"))

        fig = _make_cell_r2_fig(best["_cell_r2_map"],
                                 title=f"({best['cat']} / {best['model']})")
        _save(fig, f"pcmap_cell_r2_{best['cat']}_{best['model']}.png".replace(" ", "_"))

        fig = _make_mean_comparison_fig(true_maps_test, pred_maps_test)
        _save(fig, "pcmap_mean_comparison.png")

    # ---- Task B visuals -----------------------------------------------------
    if not scalar_df.empty:
        fig = _make_scalar_bars_fig(scalar_df)
        _save(fig, "scalar_summary_bars.png")

        fig = _make_scalar_heatmap_fig(scalar_df)
        _save(fig, "scalar_r2_heatmap.png")

    # ==========================================================================
    # SAVE JSON
    # ==========================================================================
    out_json = (OUTPUT_DIR / "unified_results.json").resolve()
    map_clean = [{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in map_results]
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "n_pcs": N_PCS,
                "pc_grid": [PC_GRID_X, PC_GRID_Y],
                "explained_variance": float(cumvar[-1]),
                "test_fraction": TEST_FRACTION,
                "random_seed": RANDOM_SEED,
                "models": MODELS,
                "scalar_targets": SCALAR_TARGETS,
                "pitch_control_is_map_only": True,
                "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
                "skip_task_a": SKIP_TASK_A,
            },
            "task_a_map": {
                "baseline": {k: v for k, v in baseline_metrics.items()
                             if k != "cell_r2_map"},
                "results": map_clean,
            },
            "task_b_scalar": scalar_df.to_dict(orient="records"),
        }, f, indent=2)
    log(f"Saved: {out_json}")
    log(f"Total elapsed: {time.time()-_T0:.1f}s")


if __name__ == "__main__":
    main()