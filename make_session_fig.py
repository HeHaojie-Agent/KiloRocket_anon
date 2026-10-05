"""Per-session accuracy curves (Fig. 3)."""
import argparse
import json
import os

import numpy as np
import pandas as pd

import datasets as D
from make_paper_assets import _style, C_PRUNE, INK, INK2, GRID
from make_paper_assets_v5 import C_REP, C_DEEP, C_DEEP2

SERIES = [  # (来源, 配置, 标签, 颜色, 标记, 线型)
    ("sc", "prune-ncmzh-f84", "KiloRocket, $F'$=84 (4.4 KB)", C_PRUNE, "o", "-"),
    ("sc", "mrrep-f840-m8", "MR + ridge replay, $m$=8 (215 KB)", C_REP, "P", "-"),
    ("deep", "resnet-alice-w64", "ResNet + ALICE, $w$=64 (2.0 MB)", C_DEEP, "h", "--"),
    ("deep", "resnet-tsacl-E8192-w64", "ResNet + TS-ACL, $E$=8192 (130 MB)", C_DEEP2, "p", "--"),
]


def load(sc_csv, deep_csvs):
    sc = pd.read_csv(sc_csv)
    sc["acc"] = sc["acc"].map(json.loads)
    dp = pd.concat([pd.read_csv(p) for p in deep_csvs if os.path.exists(p)], ignore_index=True)
    dp = dp[(dp.protocol == "main") & dp.acc_per_session.notna()].copy()
    dp["acc"] = dp["acc_per_session"].map(json.loads)
    return sc[["dataset", "seed", "config", "acc"]], dp[["dataset", "seed", "config", "acc"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--sc", default=os.path.join(D.RESULTS_DIR, "session_curves.csv"))
    ap.add_argument("--deep", nargs="*", default=[os.path.join(D.RESULTS_DIR, "results_deep_v6.csv"),
                                                  os.path.join(D.RESULTS_DIR, "results_deep_orig_v6.csv")])
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=1.75)
    ap.add_argument("--ncol", type=int, default=4)
    a = ap.parse_args()
    sc, dp = load(a.sc, a.deep)
    src = {"sc": sc, "deep": dp}
    plt = _style()
    n = len(a.datasets)
    fig, axes = plt.subplots(1, n, figsize=(a.width, a.height), sharey=False)
    for ax, name in zip(np.atleast_1d(axes), a.datasets):
        for kind, cfg, lab, col, mk, ls in SERIES:
            d = src[kind]
            rows = d[(d.dataset == name) & (d.config == cfg)]
            if rows.empty:
                continue
            A = np.array(rows.sort_values("seed")["acc"].tolist())    # seed × 会话
            m, sd = A.mean(0), A.std(0)
            x = np.arange(A.shape[1])
            ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.12, lw=0)
            ax.plot(x, m, ls, color=col, lw=1.2, marker=mk, ms=3.6, mec="white", mew=0.5, label=lab)
        ax.set_title(name.replace("ArticularyWordRecognition", "ArticularyWordRec."), fontsize=8, color=INK, pad=3)
        ax.set_xlabel("Session")
        ax.set_xticks(x)
        ax.grid(True, color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy (all seen classes)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=a.ncol, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.2, fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", a.out)


if __name__ == "__main__":
    main()
