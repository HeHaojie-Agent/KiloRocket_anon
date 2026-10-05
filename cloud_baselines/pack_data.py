"""Packs the 27 datasets as .npz files for the GPU runs."""

from __future__ import annotations

import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import datasets as D  # noqa: E402

OUT_DIR = os.path.join(HERE, "data")
ZIP = os.path.join(ROOT, "cloud_pack.zip")

CODE = ["incremental.py", "datasets.py",
        "cloud_baselines/fscil_deep.py", "cloud_baselines/pack_data.py",
        "cloud_baselines/merge_results.py", "cloud_baselines/run_cloud.sh",
        "cloud_baselines/README_云端运行.md",
        "results/datasets_selected_incremental_v6.json",
        "results/datasets_selected_incremental.json"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = D.selected_incremental_v6()
    print(f"导出 {len(names)} 个数据集到 {OUT_DIR}", flush=True)
    meta = {}
    for name in names:
        p = os.path.join(OUT_DIR, f"{name}.npz")
        X_tr, y_tr, X_te, y_te = D.load_split(name)
        if not os.path.exists(p):
            np.savez_compressed(p, X_tr=X_tr.astype(np.float32), y_tr=y_tr,
                                X_te=X_te.astype(np.float32), y_te=y_te)
        meta[name] = dict(shape_train=list(X_tr.shape), shape_test=list(X_te.shape),
                          n_classes=int(len(np.unique(y_tr))))
        print(f"  {name:28s} train {X_tr.shape}  test {X_te.shape}  "
              f"{os.path.getsize(p) / 1e6:6.1f} MB", flush=True)
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    print(f"\n打包 {ZIP} ...", flush=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:     # npz 已压缩，不再压
        for rel in CODE:
            src = os.path.join(ROOT, rel)
            if os.path.exists(src):
                z.write(src, "kilorocket/" + rel)
            else:
                print(f"  （缺少 {rel}，跳过）")
        for f in sorted(os.listdir(OUT_DIR)):
            z.write(os.path.join(OUT_DIR, f), "kilorocket/cloud_baselines/data/" + f)
    print(f"完成：{ZIP}（{os.path.getsize(ZIP) / 1e6:.0f} MB）。把它上传到。", flush=True)


if __name__ == "__main__":
    main()
