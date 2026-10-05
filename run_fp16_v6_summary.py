"""fp16 storage: matched-memory comparison and crossover budget with every real number stored in fp16 (reads the v6 results). --old uses the earlier asymmetric interpolation for checking."""
import os, sys
import exp_minirocket_sweep as E
import exp_fp16 as X
if "--old" in sys.argv:
    E.matched = E.matched_interp
R = "results"
X.V5_CSV, X.CHECK_CSV = os.path.join(R, "results_v6.csv"), os.path.join(R, "results_check_v6.csv")
out = sys.argv[sys.argv.index("--report") + 1]
X.summary(out_csv=os.path.join(R, "results_fp16_v6.csv"), report=out)
