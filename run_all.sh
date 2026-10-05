#!/bin/bash
# Full pipeline on CPU (all 27 datasets x 5 seeds). Finished rows are skipped, so the script can be re-run.
set -e
cd "$(dirname "$0")"
python -u datasets.py --profile incremental-v6                                   # 1. scan UCR/UEA, select 27 datasets
python -u exp_v5.py        --list v6 --seeds 0 1 2 3 4 --out results/results_v6.csv            # 2. main grid
python -u exp_check.py     --list v6 --seeds 0 1 2 3 4 --no-refs --out results/results_check_v6.csv  # 3. analytic heads
python -u exp_fp16.py      --list v6 --seeds 0 1 2 3 4 --out results/results_fp16_v6.csv       # 4. fp16 storage
python -u exp_teen_sweep.py --list v6 --seeds 0 1 2 3 4 --out results/results_teen_v6.csv      # 5. TEEN sweep
python -u session_curves.py --datasets PhonemeSpectra ArticularyWordRecognition Crop             # 6. per-session curves
# 7. ResNet baselines (GPU): python cloud_baselines/pack_data.py, then see cloud_baselines/run_cloud.sh and run_orig.py
python -u analyze_v5.py --csv results/results_v6.csv --check results/results_check_v6.csv --out results/v6_summaries/v6_summary.txt   # 8a. summaries, matched-memory comparison
python -u run_fp16_v6_summary.py --report results/v6_summaries/v6_fp16_matched_summary.txt   # 8b. fp16 matched comparison
python -u robust_v6.py                                                            # 8. LODO, CIs, external test set, TOST
python -u efficiency_v6.py cloud_baselines/data                                   # 9. inference cost and state serialization
python -u make_paper_assets_v5.py --csv results/results_v6.csv --check results/results_check_v6.csv --out paper_assets   # 10. tables and figures
python -u make_session_fig.py --datasets PhonemeSpectra ArticularyWordRecognition Crop --out paper_assets/figs/fig3_sessions_lncs.pdf --width 4.75 --height 1.6 --ncol 2
