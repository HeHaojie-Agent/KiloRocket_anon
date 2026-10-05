# KiloRocket

Code and per-run results for the anonymous submission
**"KiloRocket: Few-Shot Class-Incremental Time-Series Classification in a Few Kilobytes"**.

KiloRocket is a training-free classifier for few-shot class-incremental time-series classification:
standardized class prototypes over MiniRocket features selected on the base session, with all fitted
state (biases, feature means and standard deviations, feature indices) stored in half precision.
The repository also contains every baseline family of the paper, the byte account that charges all state
retained between sessions, and the scripts that produce every table and figure.

> Code comments are partly in Chinese; function names, arguments and outputs are in English.

## Contents

| Path | What it is |
|---|---|
| `datasets.py` | Scans the UCR/UEA archives through aeon and applies the selection rule of Section 4.1 |
| `incremental.py` | Protocol (base session, then `n_way` new classes with `n_shot` examples each), 1NN replay, HDC, CNN |
| `exp_analytic_prune.py` | MiniRocket features, base-session feature selection, analytic (recursive ridge) head |
| `exp_v5.py` | Main grid: all families x memory knobs x datasets x seeds x protocols |
| `exp_check.py`, `exp_fp16.py`, `exp_teen_sweep.py` | Analytic-head variants, fp16 storage, TEEN sweep |
| `cloud_baselines/` | ResNet1D re-implementations of Decoupled-Cosine, TEEN, ALICE and TS-ACL (GPU) |
| `analyze_v5.py`, `robust_v6.py` | Byte accounting, matched-memory comparison (budget envelopes on a common log grid), leave-one-dataset-out selection, CIs, TOST |
| `run_fp16_v6_summary.py` | Matched-memory comparison and crossover budget with every real number in fp16 |
| `efficiency_v6.py` | Convolutions needed by the selected features, inference latency, binary serialization of the state |
| `make_paper_assets_v5.py`, `make_session_fig.py` | All tables and figures |
| `results/` | Per-run results (one row per dataset x seed x protocol x configuration) and summaries |
| `run_all.sh` | The full pipeline in order |
| `LICENSE` | MIT |

## Datasets

`results/dataset_scan_incremental_full.csv` lists **all** equal-length UCR/UEA datasets that aeon provides,
with their number of channels, length, classes, train/test sizes and smallest class.
The selection rule (Section 4.1 of the paper) keeps datasets with 6-60 classes, length at most 500,
at most 20 channels, at least 5 training series per class, at least 60 test series and no missing values;
`results/datasets_selected_incremental_v6.json` holds the 27 selected names.
The 14 datasets used while designing KiloRocket and the 13 added afterwards are listed in `robust_v6.py`.

## Setup

The UCR/UEA archives are downloaded through aeon; to run offline, set `KILOROCKET_DATA_DIR` to a local copy or pass `--data-dir`.


```bash
pip install -r requirements.txt
bash run_all.sh                 # CPU part; several hours on one machine
```

The ResNet baselines need a GPU: `python cloud_baselines/pack_data.py` writes the 27 datasets as `.npz`,
then `bash cloud_baselines/run_cloud.sh start 4` and `python cloud_baselines/run_orig.py`.

Only the last two steps of `run_all.sh` are needed to regenerate the tables and figures from the
results shipped in `results/`:

```bash
python make_paper_assets_v5.py --csv results/results_v6.csv --check results/results_check_v6.csv --out paper_assets
python robust_v6.py
```

## Result files

| File | Rows | Content |
|---|---|---|
| `results/results_v6.csv` | 19 440 | Main grid, 3 protocols |
| `results/results_check_v6.csv` | 4 218 | Analytic heads |
| `results/results_fp16_v6.csv` | 3 915 | fp16 storage |
| `results/results_teen_v6.csv` | 17 820 | TEEN sweep |
| `results/results_deep_v6.csv`, `results_deep_orig_v6.csv` | 3 375 + 1 350 | ResNet baselines (lighter and original hyperparameters) |
| `results/session_curves.csv` | 30 | Accuracy after each session (Fig. 3) |

Columns: `final_acc` (after the last session), `avg_acc` (mean over sessions), `base_last`, `novel_last`,
`forgetting` (base-class accuracy after session 0 minus after the last session), `mean_update_time`
(seconds per incremental session, excluding feature extraction), `final_memory_bytes` and
`encoder_state_bytes` (retained bytes; the paper charges their sum). The accuracy after session 0 equals
`base_last + forgetting`, from which the performance drop (PD) in Table 4 is computed.
