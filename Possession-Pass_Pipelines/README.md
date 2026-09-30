# Tactical Phase & Possession Pipelines

Two Python pipelines that turn PFF FC football tracking and event data into tactical
datasets. Every row is a segment of play, described by features like pitch-control maps,
EPV / xT, OBSO, DAS, formation fits, team shape metrics, and pass, shot and pressure
event counts.

There are two pipelines:

| Pipeline | Script | One row per |
|---|---|---|
| **Possession pipeline** | `possession_pipeline.py` | PFF possession **sequence** (a team's spell on the ball) |
| **Tactical phase pipeline** | `build_tactical_phases.py` | **Tactical phase** (a sequence cut into shorter, tactically coherent windows) |

The phase pipeline imports its feature machinery from the possession pipeline, so the two
share the same loaders, feature code, and output schema. They differ only in how play is
segmented.

## How the phase pipeline works

It runs in two steps.

First, it computes **per-pass movement scores**. For every completed pass and cross, it
measures how far the attacking team's outfielders moved between the pass snapshot and the
next event's snapshot, which gives each pass a mean and a max displacement.

Then it does **phase segmentation**. Within each sequence it accumulates events into a
phase and closes that phase as soon as one of these triggers fires:

- a stoppage or dead ball (a whistle or a score) falls inside it,
- player movement crosses the configured thresholds (`THRESHOLD_MEAN`, default 2 m, or
  `THRESHOLD_MAX`, default 4 m),
- a pass fails, or a shot is taken,
- the phase reaches the cap on completed passes (`MAX_WINDOW_PASSES`, default 8) or the
  cap on elapsed time (`MAX_WINDOW_SECONDS`, default 30 s),
- or the sequence simply ends.

All of these are just constants at the top of `build_tactical_phases.py`, so you can tune
them to get longer or shorter phases.

Each surviving phase's frames are then fed through the shared feature machinery and
aggregated into one output row, plus a formation fit for the phase.

## How to run

From the project root:

```bash
python possession_pipeline.py     # possession/sequence-level dataset
python build_tactical_phases.py   # tactical-phase dataset
```

Both scripts auto-discover matches and process the first `NUM_MATCHES` (default 64) using
`NUM_WORKERS` (default 3) parallel worker processes. Change those two constants at the top
of each script to scale up or down. The phase pipeline imports `possession_pipeline.py`,
so keep both files in the same folder.

You will need `numpy`, `pandas`, `scipy`, `matplotlib`, `mplsoccer`, and `pyarrow`.

## Options

There are no command-line flags. Everything is a plain constant at the top of the script,
so you change a value, save, and re-run. The phase pipeline inherits most of its settings
from `possession_pipeline.py`, since it imports them.

**Run scale** (both scripts)

| Option | Default | What it does |
|---|---|---|
| `NUM_MATCHES` | 64 | How many matches to process. |
| `NUM_WORKERS` | 3 | Number of parallel worker processes. |
| `PROCESSED_DIR` | `processed_tracking` | Folder holding the per-match data. |

**Phase segmentation** (`build_tactical_phases.py`)

| Option | Default | What it does |
|---|---|---|
| `THRESHOLD_MEAN` | 2.0 | Close a phase when the mean outfielder movement (m) on a completed pass exceeds this. |
| `THRESHOLD_MAX` | 4.0 | Same, but for the single most-moved player. |
| `MAX_WINDOW_PASSES` | 8 | Cap on completed passes before the phase is closed. |
| `MAX_WINDOW_SECONDS` | 30.0 | Cap on phase duration, in seconds. |
| `MIN_PHASE_FRAMES` | 3 | Phases that resolve to fewer tracking frames are dropped. |
| `MIN_VALID_OUTFIELDERS` | 7 | Players needing a position both before and after a pass for it to be scored. |
| `EXCLUDE_GK` | True | Leaves goalkeepers out of the movement scores. |
| `RECOMPUTE_MOVEMENT_SCORES` | True | Recomputes the pass scores every run instead of reusing the cached parquet. |

**Pitch control and value grids** (`possession_pipeline.py`)

| Option | Default | What it does |
|---|---|---|
| `PC_GRID_RES_X` / `PC_GRID_RES_Y` | 75 / 45 | Resolution of the pitch-control map. |
| `PC_DOWNSAMPLE` | 3 | Only every Nth frame gets a pitch-control map. |
| `PC_MAP_DTYPE` | `float16` | How the pitch-control map is stored in the parquet. |
| `EPV_GRID_PATH` / `XT_GRID_PATH` | `EPV_grid.csv` / `xT_grid.csv` | The value grids used to score pitch position. |
| `OBSO_RADIUS_M` | 5.0 | Radius, in metres, of the probe lattice around the ball for OBSO. |
| `OBSO_DOWNSAMPLE` | 5 | Only every Nth frame is scored for OBSO. |
| `DAS_GRID_RES` | 30 | Grid resolution for the danger and threat area scores. |
| `DAS_EPV_THRESHOLD` | 0.05 | EPV a cell must exceed to count as dangerous. |
| `DAS_MIN_DURATION_SECONDS` | 3.0 | Possessions shorter than this are skipped when flagging dangerous area situations. |
| `OFFBALL_CARRIER_RADIUS_M` | 3.0 | Distance, in metres, within which a player counts as the ball carrier. |

**Tracking quality**

| Option | Default | What it does |
|---|---|---|
| `USE_SMOOTHED_TRACKING` | True | Prefers the provider's Kalman-smoothed positions over the raw ones. |
| `RELIABILITY_MODE` | `all` | How low-quality frames are treated: `all` (use every frame), `filter` (drop frames below the threshold), or `weighted` (down-weight them). |
| `RELIABILITY_THRESHOLD` | 0.5 | Share of reliably tracked players a frame needs when in `filter` mode. |

**Formations**

| Option | Default | What it does |
|---|---|---|
| `FORMATION_WINDOW_SECONDS` | 180 | Length, in seconds, of a rolling formation window. |
| `FORMATION_STRIDE_SECONDS` | 60 | How far, in seconds, the window advances each step. |
| `FORMATION_MIN_FRAMES_PER_WINDOW` | 100 | Windows with fewer valid frames are discarded. |
| `FORMATION_MIN_OUTFIELD_PLAYERS` | 8 | Outfielders needed to fit a formation. |
| `FORMATION_GK_MIN_FRAMES` | 200 | Frames a goalkeeper must appear in to be resolved as the keeper. |
| `MIN_WINDOW_CONFIDENCE` | 0.0 | Minimum formation fit confidence to keep a window. |

**Frame weighting after disruptions**

| Option | Default | What it does |
|---|---|---|
| `FOUL_DECAY_SECONDS` | 15.0 | Frames after a foul are down-weighted for this many seconds. |
| `SUB_DECAY_SECONDS` | 30.0 | Same, for substitutions. |

A few constants in the config block (`SETPIECE_RECOVERY_SECONDS`, `TURNOVER_DECAY_SECONDS`,
`RELIABILITY_PERCENTILE`) are defined but not currently wired into the main path.

## Data used

The pipelines read **PFF FC tracking and event data** (Sportlogiq broadcast tracking merged
with PFF events), stored as one folder per match under `processed_tracking/<match_id>/`.
Each folder should contain:

- `metadata.json` for pitch dimensions, fps, and teams,
- `tracking.jsonl.bz2` for the per-frame player and ball positions (raw and smoothed),
- `events.json` for the game events and possession events.

They also use two value grids for scoring pitch position: `EPV_grid.csv` and `xT_grid.csv`.
Any match folder missing one of the three files above is skipped.


## License

The pipeline code is open source under the [MIT License](LICENSE), so you are free to use,
modify, and distribute it. The PFF FC data files under 