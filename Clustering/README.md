# Recurrence Map Pipeline: Usage Guide

This guide covers only the five Python files in the `clustering pipeline` commit:

- `possession_pipeline.py`
- `build_tactical_phases.py`
- `pca_recurrence.py`
- `cluster_pipeline_noball.py`
- `formation_recurrence_maps.py`

## What Each File Does

| File | Purpose | How to use it |
| --- | --- | --- |
| `possession_pipeline.py` | Loads processed tracking and event data and builds possession-level feature tables, formation windows, and formation segments. It also supplies shared tracking, spatial-feature, and formation functions used by the tactical-phase builder. | Run `python possession_pipeline.py` for the possession-level outputs. Adjust the configuration constants near the top of the file first if your input directory, match limit, grid files, or output names differ. |
| `build_tactical_phases.py` | Scores completed passes, divides open play into tactical phases, and computes phase-level features, formation labels, player recurrence maps, ball trajectories, and pitch-control maps. It imports shared functions from `possession_pipeline.py`. | Run `python build_tactical_phases.py`. Configure input/output paths, phase thresholds, and worker count near the top of the file. This is the required producer for the PCA, clustering, and formation-map workflows below. |
| `pca_recurrence.py` | Converts each phase's home/away recurrence maps into vectors, optionally orients them so possession attacks in the same direction, and computes PCA scores, loadings, and plots. | Run `python pca_recurrence.py <recurrence-parquet>`. See [PCA options](#pca-options). |
| `cluster_pipeline_noball.py` | Clusters tactical phases using non-ball features, including recurrence and pitch-control maps by default. Reports cluster assignments, sweep diagnostics, feature importance, phase profiles, and recurrence plots. | Run `python cluster_pipeline_noball.py` after producing its input files. Configure input/output directories and clustering options with environment variables; see [Clustering options](#clustering-options). |
| `formation_recurrence_maps.py` | Regrids phase recurrence maps onto a common pitch grid, sums and compares maps grouped by formation, and runs a subsampling/patchiness diagnostic. | Run `python formation_recurrence_maps.py` after producing tactical phases and per-side recurrence maps. Configure directories and formation grouping with environment variables; see [Formation-map options](#formation-map-options). |

## Prerequisites and Inputs

Use Python with the packages imported by these scripts:

```sh
python -m pip install numpy pandas pyarrow matplotlib scipy scikit-learn mplsoccer
```

The data pipelines expect each processed match under `processed_tracking/<match_id>/` with `metadata.json`, `tracking.jsonl.bz2`, and `events.json`. The possession pipeline also looks for `EPV_grid.csv` and `xT_grid.csv` in the working directory; if a grid is unavailable, some value-based features are omitted or use the documented fallback.

Run commands from the repository root so relative input and output paths resolve as expected. The main data-intensive jobs can use substantial memory and take time when run across many matches.

## Recommended Workflow

1. Run `python possession_pipeline.py` if you need its separate possession-level dataset and formation-window/segment tables.
2. Run `python build_tactical_phases.py`. This writes pass-movement scores, tactical-phase features, formation windows, per-side recurrence maps, ball trajectory/path data, pitch-control maps, and a movement histogram.
3. Run PCA if you want recurrence-map components: `python pca_recurrence.py '<recurrence-parquet-path>'`. By default, this writes `pca_output/`.
4. Run `python cluster_pipeline_noball.py` to cluster phases. It needs the tactical-phase, movement-score, formation-window, recurrence, pitch-control, and PCA-score parquet files. Its default results directory is `cluster_output_noball/`.
5. Run `python formation_recurrence_maps.py` to aggregate recurrence by formation. It needs the tactical-phase and recurrence parquet files. Its default results directory is `formation_recurrence_maps/`.

Steps 4 and 5 are separate analyses and can be run in either order once their inputs exist. For clustering, PCA scores are an expected input even though positional PCA is held out of clustering by default when recurrence-map features are enabled.

## PCA Options

The first argument is the per-side recurrence parquet created by `build_tactical_phases.py`:

```sh
python pca_recurrence.py '<recurrence-parquet-path>' --features poss_opp --n-components 20 --outdir pca_output
```

Useful options:

- `--features poss_opp|home_away|combined`: choose possession-relative, home/away, or blended map features. Default: `poss_opp`.
- `--transform sqrt|none`: apply the default square-root transform or leave occupancy values unchanged.
- `--orient auto|none`: automatically normalize attack direction by default, or disable orientation.
- `--match <id>`: restrict the analysis to one match.
- `--min-frames <count>`: discard shorter phases.
- `--n-components <count>` and `--show <count>`: set the number of saved components and displayed component maps.
- `--outdir <directory>`: choose the output directory.

The output includes `pca_scores.parquet`, `pca_loadings.parquet`, `pca_explained_variance.csv`, `pca_model.npz`, and scree, loading, and scatter plots.

## Clustering Options

Set environment variables before running the script. For example:

```sh
UPLOAD_DIR='.' OUT_DIR='./cluster_output_noball' FINAL_KS='2,4,6' python cluster_pipeline_noball.py
```

`UPLOAD_DIR` points to the generated input files (and may contain a `pca_output/` subdirectory); `OUT_DIR` selects the results directory. Other useful variables include `K_MAX`, `GROUP_ALPHA`, `HOLDOUT_GROUPS`, `PCA_VAR`, and `STAB_REPEATS`. `RECURRENCE_FEATURES` and `PITCH_CONTROL_FEATURES` default to `1`. Set them to `0` to exclude those maps from clustering. See the configuration block near the top of `cluster_pipeline_noball.py` for the full list and defaults.

The script produces a k-sweep table and plot, assignments for every tested k, and per-`FINAL_KS` feature-importance tables, cluster profiles, validation profiles, and recurrence-map plots. It does not choose a final k automatically.

## Formation-Map Options

For example:

```sh
UPLOAD_DIR='.' OUT_DIR='./formation_recurrence_maps' python formation_recurrence_maps.py
```

`UPLOAD_DIR` selects where the tactical-phase and recurrence parquet files are found. `OUT_DIR` selects where results are written. `FORMATION_COL` chooses the grouping column (`dominant_formation` by default; `formation_variant` and `formation_family` are alternatives). `MIN_PHASES_PER_FORMATION` sets the minimum sample count; `TARGET_N_COLS` and `TARGET_N_ROWS` set the common output grid size.

Outputs include summed per-formation maps, formation-correlation CSVs and plots, and the subsampling patchiness test results.

## Path Note for macOS

`build_tactical_phases.py` defines default output names such as `Outputs\4th_Run\tactical_phases.parquet`. On macOS, backslashes in these strings are ordinary filename characters, not directory separators. The downstream scripts account for these literal names when locating inputs. Keep the producer and consumer paths consistent if changing them; alternatively, update the paths in the scripts to use `pathlib.Path` with real directories.