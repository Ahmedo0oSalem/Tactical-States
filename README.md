# Tactical States

Football tactical-state analysis built on **PFF FC** tracking and event data. Turn
raw per-frame tracking into possession- and phase-level tactical datasets, then
cluster and visualise them.

## Layout

| Folder | Contents |
| --- | --- |
| [`Possession-Pass_Pipelines/`](Possession-Pass_Pipelines) | The data producers. `possession_pipeline.py` builds possession-level datasets; `build_tactical_phases.py` segments play into shorter tactical phases. Includes the `EPV_grid.csv` / `xT_grid.csv` value grids and the MIT license. |
| [`Clustering/`](Clustering) | The analysis layer. PCA of recurrence maps (`pca_recurrence.py`), clustering of tactical phases (`cluster_pipeline_noball.py`, `cluster_pipeline_v3.py`), formation recurrence aggregation (`formation_recurrence_maps.py`), and ball-trajectory animation (`animate_ball_trajectory.py`). |

## Quick start

```sh
python -m pip install numpy pandas pyarrow matplotlib scipy scikit-learn mplsoccer

# 1. Build the datasets (from the repo root)
python Possession-Pass_Pipelines/possession_pipeline.py
python Possession-Pass_Pipelines/build_tactical_phases.py

# 2. Analyse them
python Clustering/pca_recurrence.py '<recurrence-parquet>'
python Clustering/cluster_pipeline_noball.py
python Clustering/formation_recurrence_maps.py
```

Both pipelines expect one folder per match under `processed_tracking/<match_id>/`
containing `metadata.json`, `tracking.jsonl.bz2`, and `events.json`. See each
folder's README for the full option lists.

## License

Code is MIT licensed — see [`Possession-Pass_Pipelines/LICENSE`](Possession-Pass_Pipelines/LICENSE).

