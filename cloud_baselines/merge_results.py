"""Merges sharded GPU results."""

import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "results_deep_v6.csv")


def main():
    parts = sorted(glob.glob(os.path.join(RES, "results_deep_v6_shard*.csv")))
    frames = [pd.read_csv(p) for p in parts]
    if os.path.exists(OUT):
        frames.append(pd.read_csv(OUT))
    if not frames:
        print("没有找到结果文件。")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "seed", "protocol", "config"], keep="last")
    df.to_csv(OUT, index=False)
    ok = df[df["error"].isna()]
    print(f"合并 {len(parts)} 个分片 → {OUT}：{len(df)} 行（失败 {len(df) - len(ok)} 行），"
          f"{ok['dataset'].nunique()} 个数据集，种子 {sorted(int(s) for s in ok['seed'].unique())}")
    ok = ok.assign(mem_kb=(ok["final_memory_bytes"] + ok["encoder_state_bytes"]) / 1024)
    g = (ok.groupby("config")
           .agg(n=("final_acc", "size"), final_acc=("final_acc", "mean"), avg_acc=("avg_acc", "mean"),
                novel=("novel_last", "mean"), forgetting=("forgetting", "mean"),
                mem_kb_median=("mem_kb", "median"), epochs=("train_epochs", "mean"))
           .sort_values("mem_kb_median"))
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(g.round(3).to_string())


if __name__ == "__main__":
    main()
