"""Runs ALICE and TS-ACL with the hyperparameters of the original papers."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUTD = RES / "deep_orig_parts"
LOGS = HERE / "logs"
MERGED = RES / "results_deep_orig_v6.csv"
KINDS = {
    "aliceo": ["--families", "deep-alice", "--alice-orig"],
    "tsacl8k": ["--families", "deep-tsacl", "--tsacl-E", "8192"],
}
WIDTHS = ["4", "8", "16", "32", "64"]


def names():
    p = json.loads((RES / "datasets_selected_incremental_v6.json").read_text(encoding="utf-8"))
    return p["univariate"] + p["multivariate"]


def part(kind, name, seed):
    return OUTD / f"{kind}__{name}__{seed}.csv"


def n_rows(f):
    if not f.exists():
        return 0
    try:
        d = pd.read_csv(f)
        return int((d["error"].isna() | (d["error"] == "")).sum())
    except Exception:  # noqa: BLE001
        return 0


def jobs():
    out = []
    meta = {}
    mp = HERE / "data" / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    for name in names():
        sh = meta.get(name, {}).get("shape_train", [1, 1, 1])
        for seed in range(5):
            for kind in KINDS:
                if n_rows(part(kind, name, seed)) >= len(WIDTHS):
                    continue
                out.append((sh[0] * sh[1] * sh[2], kind, name, seed))
    out.sort(reverse=True)          # 大数据集先跑
    return out


def run_job(job):
    _, kind, name, seed = job
    f = part(kind, name, seed)
    if f.exists() and n_rows(f) < len(WIDTHS):
        d = pd.read_csv(f)          # 去掉失败行，让它重跑
        d[d["error"].isna() | (d["error"] == "")].to_csv(f, index=False)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    cmd = [sys.executable, "-u", "fscil_deep.py", "--datasets", name, "--seeds", str(seed),
           "--widths", *WIDTHS, "--out", str(f), *KINDS[kind]]
    with (LOGS / f"orig_{kind}_{name}_{seed}.log").open("w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    return kind, name, seed, p.returncode, n_rows(f)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    OUTD.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    if cmd == "test":
        import torch
        print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
              "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU", flush=True)
        name = names()[0]
        for kind, extra in KINDS.items():
            t0 = time.time()
            subprocess.run([sys.executable, "-u", "fscil_deep.py", "--datasets", name, "--seeds", "0",
                            "--widths", "16", "--out", str(RES / f"results_deep_orig_quick_{kind}.csv"),
                            *extra], cwd=HERE)
            print(f"== {kind} 用时 {time.time() - t0:.0f}s", flush=True)
    elif cmd == "start":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        todo = jobs()
        print(f"待跑任务 {len(todo)} 个（共 {len(names()) * 5 * len(KINDS)} 个），并行 {n} 路", flush=True)
        with ThreadPoolExecutor(max_workers=n) as pool:
            for fu in as_completed([pool.submit(run_job, j) for j in todo]):
                kind, name, seed, code, rows = fu.result()
                print(f"{time.strftime('%H:%M:%S')} DONE {kind} {name} seed={seed} exit={code} rows={rows}/5",
                      flush=True)
        print("ALL_DONE", flush=True)
    elif cmd == "status":
        total = len(names()) * 5 * len(KINDS)
        left = jobs()
        print(f"已完成 {total - len(left)}/{total} 个任务（每个任务 = 1 数据集 × 1 seed × 5 宽度）")
        for kind in KINDS:
            print(f"  {kind}: 剩 {sum(1 for j in left if j[1] == kind)} 个")
        os.system("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null")
    elif cmd == "merge":
        fs = sorted(OUTD.glob("*.csv"))
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        d = d.drop_duplicates(["dataset", "seed", "protocol", "config"], keep="last")
        bad = (~(d["error"].isna() | (d["error"] == ""))).sum()
        d.to_csv(MERGED, index=False)
        print(f"合并 {len(fs)} 个文件 → {MERGED}：{len(d)} 行（失败 {bad} 行），"
              f"{d.dataset.nunique()} 个数据集，种子 {sorted(d.seed.unique())}")
        ok = d[d["error"].isna() | (d["error"] == "")].copy()
        ok["mem_kb"] = (ok["final_memory_bytes"] + ok["encoder_state_bytes"]) / 1024
        print(ok.groupby("config").agg(n=("final_acc", "size"), final_acc=("final_acc", "mean"),
                                       novel=("novel_last", "mean"), forgetting=("forgetting", "mean"),
                                       mem_kb_median=("mem_kb", "median"), epochs=("train_epochs", "mean"))
              .round(3).sort_values("mem_kb_median").to_string())
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
