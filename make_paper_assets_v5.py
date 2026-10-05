"""Generates every table and figure of the paper from the result CSVs."""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import analyze_v5 as A
import exp_minirocket_sweep as E
from make_paper_assets import _style, _kb_axis, _fmt, _kb, C_PRUNE, C_NN, C_HDC, C_MR, C_AN, INK, INK2, GRID

C_REP, C_CNN = "#7a52c7", "#8a8984"
C_DEEP, C_DEEP2 = "#9c5a1a", "#5f7a1e"
HERE = os.path.dirname(os.path.abspath(__file__))


def drop_hdc_codebook(df):
    """统一规则：能由随机种子重新生成的状态一律不计。HDC 码本（通道、level、时间基向量）由种子生成，
    与 MiniRocket 的卷积核同理，不计入内存；幅值边界是拟合出来的，仍计（8C 字节）。
    exp_v5 记账时码本按 (C+L+1)·D/8 字节计入（L=64），这里从 final_memory_bytes − bytes_proto 里解出 C 再减去。"""
    d = df.copy()
    h = d.family.astype(str).str.startswith("hdc")
    if h.any():
        D_ = d.loc[h, "knob"].astype(float)
        extra = (d.loc[h, "final_memory_bytes"] - d.loc[h, "bytes_proto"]).astype(float)
        C = (extra - 65 * D_ / 8) / (D_ / 8 + 8)
        d.loc[h, "mem"] = d.loc[h, "mem"] - (C + 65) * D_ / 8
    return d


DEEP_CSV = os.path.join(os.path.dirname(A.V5_CSV), "results_deep_v6.csv")
DEEP_ORIG_CSV = os.path.join(os.path.dirname(A.V5_CSV), "results_deep_orig_v6.csv")  # 按原文超参补跑（ALICE 原设置、TS-ACL E=8192）
DEEP_GROUPS = {  # 已发表的深度 FSCIL 方法（ResNet1D 骨干，cloud_baselines/fscil_deep.py 复现）
    "Deep-proto": ["deep-proto"], "Deep-TEEN": ["deep-teen"], "Deep-ALICE": ["deep-alice"],
    "Deep-TSACL": ["deep-tsacl512", "deep-tsacl2048", "deep-tsacl8192"],
    "Deep-FSCIL": ["deep-proto", "deep-teen", "deep-alice", "deep-alice-orig",
                   "deep-tsacl512", "deep-tsacl2048", "deep-tsacl8192"],
}
A.GROUPS.update(DEEP_GROUPS)


def load_deep(path, datasets):
    """深度基线的字节记账与本文统一：
    骨干参数 + BN 统计量（fp32）全部计入；原型 4Kd；
    TS-ACL 的 R 是对称矩阵，与本文解析头一样只计上三角 E(E+1)/2 个 fp32；W 计 4EK；
    随机扩展层 W_E 可由种子重建，按统一规则不计（与 MiniRocket 卷积核、HDC 码本同理）。"""
    paths = [p for p in ([path] if isinstance(path, str) else list(path or [])) if p and os.path.exists(p)]
    if not paths:
        return None
    d = A._ok(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True, sort=False))
    d = d.drop_duplicates(["dataset", "seed", "protocol", "config"], keep="first")
    d = d[(d.protocol == "main") & d.dataset.isin(datasets)].copy()
    d["mem"] = d["final_memory_bytes"].astype(float) + d["encoder_state_bytes"].astype(float)
    ts = d.family == "deep-tsacl"
    E = d.loc[ts, "config"].str.extract(r"-E(\d+)-")[0].astype(float)
    d.loc[ts, "mem"] = (d.loc[ts, "mem"] - d.loc[ts, "expansion_bytes"].astype(float)
                        - 4 * E * E + 2 * E * (E + 1))
    d.loc[ts, "family"] = "deep-tsacl" + E.astype(int).astype(str)
    d["knob"] = d["knob"].astype(float)
    return d


def load(v5_csv, check_csv, allow_partial, charge_codebook=False, deep_csv=(DEEP_CSV, DEEP_ORIG_CSV)):
    df, _ = A.load(v5_csv, check_csv)
    df = df[df.protocol == "main"].copy()
    if not charge_codebook:
        df = drop_hdc_codebook(df)
    dp = load_deep(deep_csv, df.dataset.unique())
    if dp is not None and len(dp):
        df = pd.concat([df, dp], ignore_index=True, sort=False)
    df["mem_B"] = df["mem"]
    per = df.groupby(["dataset", "seed"]).config.nunique()
    full = per.max()
    if allow_partial:
        keep = per[per >= full - 20].index      # 解析头只有 12 个配置，允许差这些
        df = df.set_index(["dataset", "seed"]).loc[keep].reset_index()
    return df


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------

FIG1 = [  # family, color, label, marker, linestyle, filled
    ("prune-binzh", C_PRUNE, "KiloRocket, 1-bit protos (ours)", "o", "-", True),
    ("prune-ncmzh", C_PRUNE, "KiloRocket, fp32 protos (ours)", "o", "--", False),
    ("mr-ncm", C_MR, "MR fewer kernels, fp32 protos", "D", "--", True),
    ("mrrep-f840", C_REP, "MR selected ($F'$=840) + ridge replay", "P", "-", True),
    ("1nn", C_NN, "Raw series + 1NN replay", "^", "-", True),
    ("hdc-real", C_HDC, "HDC, real-valued protos", "s", ":", True),
    ("cnn-frozen", C_CNN, "CNN (trained on base), protos", "X", ":", True),
    ("analytic-cvbal", C_AN, "MR + analytic head (tuned, bal.)", "v", "-.", True),
    ("deep-alice", C_DEEP, "ResNet + ALICE (lighter)", "h", "-", True),
    ("deep-alice-orig", C_DEEP, "ResNet + ALICE (original)", "h", ":", False),
    ("deep-tsacl512", C_DEEP2, "ResNet + TS-ACL ($E$=512)", "p", "--", False),
    ("deep-tsacl8192", C_DEEP2, "ResNet + TS-ACL ($E$=8192)", "p", "-", True),
]


def _curve(df, fam, metric="final_acc"):
    return (df[df.family == fam].groupby("knob")
            .agg(mem=("mem_B", "mean"), acc=(metric, "mean")).sort_values("mem"))


def fig1(df, path, lncs=False):
    """lncs=True：v6（LNCS 单栏）版，按印刷尺寸出图（\\includegraphics 不缩放），图例放在右侧、7 pt。"""
    plt = _style()
    fig, ax = plt.subplots(figsize=(2.55, 2.05) if lncs else (3.5, 2.55))
    for fam, col, lab, mk, ls, filled in FIG1:
        g = _curve(df, fam)
        if g.empty:
            continue
        kw = dict(color=col, lw=1.3, zorder=3, label=lab, marker=mk, ms=4.0)
        kw.update(dict(mec="white", mew=0.6) if filled else dict(mfc="white", mec=col, mew=0.9))
        ax.plot(g.mem / 1024, g.acc, ls, **kw)
    ax.set_xscale("log")
    _kb_axis(ax)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1e3, 1e4, 1e5]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1k", "10k", "100k"]))
    ax.set_xlim(0.45, 2.5e5)
    ax.set_xlabel("Retained memory (KB, log scale; all fitted state charged)")
    ax.set_ylabel("Final accuracy")
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if lncs:
        ax.set_xlabel("Retained memory (KB, log scale)")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False,
                  handlelength=2.4, borderaxespad=0, labelspacing=0.45, handletextpad=0.5, fontsize=7)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False,
                  handlelength=2.2, borderaxespad=0, labelspacing=0.3, columnspacing=0.6,
                  handletextpad=0.4, fontsize=5.6)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


FIG2 = [
    ("prune-binzh", C_PRUNE, "KiloRocket, 1-bit", "o", True),
    ("prune-ncmzh", C_PRUNE, "KiloRocket, fp32", "o", False),
    ("mr-ncm", C_MR, "Fewer kernels, fp32", "D", False),
    ("mrrep-f840", C_REP, "MR + ridge replay", "P", True),
    ("1nn", C_NN, "1NN replay", "^", True),
    ("hdc-real", C_HDC, "HDC, real", "s", True),
    ("cnn-frozen", C_CNN, "CNN, protos", "X", True),
    ("deep-alice", C_DEEP, "ResNet + ALICE", "h", True),
    ("deep-tsacl8192", C_DEEP2, "ResNet + TS-ACL", "p", False),
    ("analytic-cvbal", C_AN, r"Analytic, tuned $\gamma$, bal.", "v", True),
    ("analytic-g1", C_AN, r"Analytic, $\gamma=1$", "v", False),
]


def fig2(df, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    allb, alln = [], []
    for fam, col, lab, mk, filled in FIG2:
        g = (df[(df.family == fam) & df.novel_last.notna()].groupby("knob")
             .agg(mem=("mem_B", "mean"), b=("base_last", "mean"), n=("novel_last", "mean"))
             .sort_values("mem"))
        if g.empty:
            continue
        allb += list(g.b); alln += list(g.n)
        ax.plot(g.b, g.n, "-", color=col, lw=0.7, alpha=0.8, zorder=2)
        kw = dict(color=col, ms=4.4, ls="none", zorder=3, label=lab, marker=mk)
        kw.update(dict(mec="white", mew=0.5) if filled else dict(mfc="white", mec=col, mew=1.0))
        ax.plot(g.b, g.n, **kw)
    lo = min(allb + alln) - 0.02
    hi = max(allb + alln) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#9a9994", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(min(allb) - 0.02, max(allb) + 0.02)
    ax.set_ylim(min(alln) - 0.02, max(alln) + 0.02)
    ax.set_xlabel("Base-class accuracy (last session)")
    ax.set_ylabel("Incremental-class accuracy (last session)")
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.6, labelspacing=0.3, fontsize=5.8)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# 表
# --------------------------------------------------------------------------

ROWS = [  # v6：压到 15 行；去掉内部版本号
    ("Raw series + 1NN replay", "$m=16$", "1nn-m16"),
    ("MR selected + ridge replay", "$F'=840$, $m=4$", "mrrep-f840-m4"),
    ("", "$F'=840$, $m=8$", "mrrep-f840-m8"),
    ("", "$F'=840$, $m=16$", "mrrep-f840-m16"),
    ("HDC, real-valued", "$D=8192$", "hdc-real-D8192"),
    ("CNN (base-trained), protos", "$w=64$", "cnn-frozen-w64"),
    ("MR fewer kernels, fp32", "$F=9996$", "mr-ncm-k10000"),
    ("MR selected, 1-bit, fp32 state", "$F'=840$", "prune-bin-f840"),
    ("KiloRocket, 1-bit (ours)", "$F'=84$", "prune-binzh-f84"),
    ("", "$F'=336$", "prune-binzh-f336"),
    ("KiloRocket, fp32 (ours)", "$F'=84$", "prune-ncmzh-f84"),
    ("", "$F'=840$", "prune-ncmzh-f840"),
    ("MR + analytic head, $\\gamma=1$", "$F=2520$", "analytic-g1-k2520"),
    ("MR + analytic, tuned $\\gamma$, bal.", "$F=2520$", "analytic-cvbal-k2520"),
    ("MiniRocket-retrain$^\\dagger$", "$F=9996$", "minirocket-retrain"),
]


def tab_configs(df, path, n_ds):
    cfgs = [c for _, _, c in ROWS if c in set(df.config)]
    sub = df[df.config.isin(cfgs)]
    rk = sub.pivot_table(index=["dataset", "seed"], columns="config", values="final_acc")
    ranks = rk.rank(axis=1, ascending=False, method="average").mean()
    agg = sub.groupby("config").agg(final=("final_acc", "mean"), avg=("avg_acc", "mean"),
                                    base=("base_last", "mean"), novel=("novel_last", "mean"),
                                    forget=("forgetting", "mean"), mem=("mem_B", "mean"))
    agg["rank"] = ranks
    best = {c: agg[c].max() for c in ["final", "avg", "base", "novel"]}
    best["forget"], best["rank"] = agg["forget"].min(), agg["rank"].min()

    def cell(c, col):
        v = agg.loc[c, col]
        s = _fmt(v) if col != "rank" else f"{v:.1f}"
        return f"\\textbf{{{s}}}" if abs(v - best[col]) < 1e-12 else s

    L = [r"\begin{table*}[t]", r"\centering",
         f"\\caption{{Representative configurations, mean over {n_ds} datasets $\\times$ 5 seeds. "
         r"Base and Incr.: accuracy on base-session and on incremental classes after the last session. "
         r"Memory charges all fitted state (Table~\ref{tab:mem}). Rank: mean rank of final accuracy "
         f"among the {len(cfgs)} listed configurations. Best value per column in bold.}}",
         r"\label{tab:main}", r"\small\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"Method & Config & Final & Avg. & Base & Incr. & Forget. & Memory (KB) & Rank \\", r"\midrule"]
    for grp, disp, c in ROWS:
        if c not in agg.index:
            continue
        L.append(f"{grp} & {disp} & {cell(c,'final')} & {cell(c,'avg')} & {cell(c,'base')} & "
                 f"{cell(c,'novel')} & {cell(c,'forget')} & {_kb(agg.loc[c,'mem'])} & {cell(c,'rank')} \\\\")
    L += [r"\bottomrule",
          r"\multicolumn{9}{@{}l}{\footnotesize $^\dagger$Stores every series seen so far; "
          r"serves as an upper reference for accuracy.}", r"\end{tabular}", r"\end{table*}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% 由 make_paper_assets_v5.py 生成，不要手改\n" + "\n".join(L) + "\n")
    return agg


MATCH = [  # (显示 A, 组 A, 显示 B, 组 B, 指标, 内存区间 KB 或 None)
    ("Protos", "Proto-selected", "1NN replay", "Replay-1NN", "final_acc", None),
    ("Ridge replay", "Replay-MR", "1NN replay", "Replay-1NN", "final_acc", None),
    ("Protos", "Proto-selected", "Ridge replay", "Replay-MR", "final_acc", None),
    ("\\quad $<$10\\,KB", "Proto-selected", "", "Replay-MR", "final_acc", (0, 10)),
    ("\\quad 10--100\\,KB", "Proto-selected", "", "Replay-MR", "final_acc", (10, 100)),
    ("\\quad $\\ge$100\\,KB", "Proto-selected", "", "Replay-MR", "final_acc", (100, np.inf)),
    ("Protos", "Proto-selected", "Fewer kernels", "Proto-fewer", "final_acc", None),
    ("Protos", "Proto-selected", "HDC", "HDC", "final_acc", None),
    ("Protos", "Proto-selected", "CNN protos", "CNN-frozen", "final_acc", None),
    ("Protos", "Proto-selected", "Analytic, bal.", "Analytic-bal", "final_acc", None),
    ("Protos", "Proto-selected", "ResNet FSCIL", "Deep-FSCIL", "final_acc", None),
    ("Protos", "Proto-selected", "Ridge replay", "Replay-MR", "novel_last", None),
    ("Protos", "Proto-selected", "Analytic, bal.", "Analytic-bal", "novel_last", None),
]


def _holm(ps):
    ps = np.asarray(ps, float)
    ok = ~np.isnan(ps)
    out = np.full_like(ps, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(ps[idx])]
    m, run = len(order), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        out[i] = run
    return out


def matched_rows(df):
    rows = []
    for da, ga, db, gb, metric, band in MATCH:
        d = df.copy()
        if metric != "final_acc":
            d["final_acc"] = d[metric]
        res, below, above = E.matched(d, A.GROUPS[ga], A.GROUPS[gb], "mem_B")
        if band is not None and not res.empty:
            res = res[(res.mem / 1024 >= band[0]) & (res.mem / 1024 < band[1])]
        if res.empty:
            rows.append((da, db, metric, band, 0, 0, np.nan, 0, 0, np.nan))
            continue
        per = res.groupby("dataset")["delta"].mean()
        rows.append((da, db, metric, band, int((res.delta > 1e-9).sum()), len(res), per.mean() * 100,
                     int((per > 0).sum()), len(per), A._wil(per.values)))
    r = pd.DataFrame(rows, columns=["A", "B", "metric", "band", "won", "n", "delta", "dpos", "dn", "p"])
    r["p_holm"] = np.nan
    for m in r.metric.unique():
        sel = r.metric == m
        r.loc[sel, "p_holm"] = _holm(r.loc[sel, "p"].values)
    return r


def tab_matched(r, path):
    def p_(x):
        return "--" if pd.isna(x) else (f"{x:.3f}" if x >= 0.001 else "$<$0.001")
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Matched-memory comparisons (Section~\ref{sec:matched}): budget envelopes of A and B "
         r"compared on a common log grid of budgets that both can meet. "
         r"Protos: prototypes over selected MiniRocket features (all variants). Ridge replay: stored series "
         r"re-encoded by the same MiniRocket features. ResNet FSCIL: front of the four re-implemented methods of Table~\ref{tab:main}. Won: grid budgets where A is more accurate. $\Delta$: mean "
         r"difference in accuracy points, averaged first within each dataset and then over datasets. Datasets: datasets with a positive mean difference. $p$: Wilcoxon "
         r"signed-rank test over per-dataset means, Holm-corrected within each block.}",
         r"\label{tab:matched}", r"\footnotesize\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
         r"A & B & Won & $\Delta$ & Datasets & $p$ \\", r"\midrule",
         r"\multicolumn{6}{@{}l}{\emph{All classes, final accuracy}} \\"]
    first_novel = True
    for _, x in r.iterrows():
        if x.metric == "novel_last" and first_novel:
            L += [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Incremental classes only}} \\"]
            first_novel = False
        if x.n == 0:
            L.append(f"{x.A} & {x.B} & \\multicolumn{{4}}{{c}}{{no overlap}} \\\\")
            continue
        L.append(f"{x.A} & {x.B} & {x.won}/{x.n} & ${x.delta:+.1f}$ & {x.dpos}/{x.dn} & {p_(x.p_holm)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% 由 make_paper_assets_v5.py 生成，不要手改\n" + "\n".join(L) + "\n")



# --------------------------------------------------------------------------
# v9：按内存预算的对比表（取代 tab_configs 作为 Table III）
# --------------------------------------------------------------------------

BUDGET_GROUPS = [  # 显示名, 包含的 family（每组在每个数据集上取预算内最好的配置）
    ("KiloRocket (ours)", ["prune-binzh", "prune-ncmzh"]),
    ("MR + ridge replay", ["mrrep-f84", "mrrep-f840", "mrrep-ffull"]),
    ("Raw series + 1NN replay", ["1nn"]),
    ("MR protos, fewer kernels", ["mr-ncm", "mr-bin", "mr-ncmc", "mr-ncmz"]),
    ("Selected protos + TEEN", ["prune-ncmc-teen", "prune-bin-teen"]),
    ("HDC protos", ["hdc-bin", "hdc-real"]),
    ("CNN (base-trained), protos", ["cnn-frozen"]),
    ("MR + analytic head", ["analytic-g1", "analytic-cv", "analytic-cvbal"]),
    ("ResNet + Decoupled-Cos.~\\cite{zhang2021cec}", ["deep-proto"]),
    ("ResNet + TEEN~\\cite{wang2023teen}", ["deep-teen"]),
    ("ResNet + ALICE~\\cite{peng2022alice}", ["deep-alice", "deep-alice-orig"]),
    ("ResNet + TS-ACL~\\cite{li2024tsacl}", ["deep-tsacl512", "deep-tsacl2048", "deep-tsacl8192"]),
]
BUDGETS_KB = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def budget_table(df):
    """每组、每个预算：先在每个 (数据集, seed) 上取预算内最好配置的最终精度（放不下时用 1/K，与 Eq. (10) 一致），
    再对 seed、数据集取平均。另算 1 KB–1 MB 的预算曲线面积，以及与「每个数据集上最好的其他组」的差。"""
    from scipy.stats import wilcoxon
    K = df.groupby("dataset").n_classes.max()
    grid = np.logspace(0, 3, 60)
    per, area, fits = {}, {}, {}
    for g, fams in BUDGET_GROUPS:
        sub = df[df.family.isin(fams)]
        rows = []
        for (d, s), x in sub.groupby(["dataset", "seed"]):
            m, a = x.mem_B.values / 1024, x.final_acc.values
            best = lambda b: a[m <= b].max() if (m <= b).any() else 1.0 / K[d]
            rows.append([d, s] + [best(b) for b in BUDGETS_KB] + [np.mean([best(b) for b in grid])])
        t = pd.DataFrame(rows, columns=["dataset", "seed"] + BUDGETS_KB + ["area"])
        t = t.groupby("dataset")[BUDGETS_KB + ["area"]].mean()
        per[g], area[g] = t[BUDGETS_KB], t["area"]
        mn = sub.groupby("dataset").mem_B.min() / 1024
        fits[g] = {b: int((mn <= b).sum()) for b in BUDGETS_KB}
    ours, others = BUDGET_GROUPS[0][0], [g for g, _ in BUDGET_GROUPS[1:]]
    delta = {}
    for b in BUDGETS_KB:
        best_other = pd.concat([per[g][b] for g in others], axis=1).max(axis=1)
        dl = per[ours][b] - best_other
        delta[b] = (100 * dl.mean(), int((dl > 0).sum()), wilcoxon(dl).pvalue)
    area_cmp = {g: (int((area[ours] - area[g] > 0).sum()), wilcoxon(area[ours] - area[g]).pvalue)
                for g in others}
    return per, area, fits, delta, area_cmp


def tab_budget(df, path, n_ds, lncs=False):
    """lncs=True：单栏 LNCS 版（v6），字号更小、列距更窄、用 table 环境。"""
    per, area, fits, delta, area_cmp = budget_table(df)
    n = len(BUDGETS_KB)
    means = {g: per[g].mean() for g, _ in BUDGET_GROUPS}
    best = {b: max(means[g][b] for g, _ in BUDGET_GROUPS) for b in BUDGETS_KB}
    best_area = max(area[g].mean() for g, _ in BUDGET_GROUPS)
    hdr = " & ".join(f"{b:,}".replace(",", "\\,") for b in BUDGETS_KB)
    env = "table" if lncs else "table*"
    L = [f"\\begin{{{env}}}[t]", r"\centering",
         f"\\caption{{Best final accuracy each family reaches within a memory budget (KB), mean over {n_ds} "
         r"datasets $\times$ 5 seeds. On each dataset and seed, the most accurate configuration of the family "
         r"whose retained memory fits the budget is taken; where none fits, chance level $1/K$ is used "
         r"(Section~\ref{sec:matched}). Area: mean over 60 log-spaced budgets from 1\,KB to 1\,MB. Best value per column "
         r"in bold. Last two rows: KiloRocket minus the best other family on each dataset, in points, and the "
         r"number of datasets where KiloRocket is ahead."
         + (r" The grid ends at 1\,MB, which penalizes families that need more memory: the ResNet methods "
            r"score chance level below about 10\,KB and TS-ACL over almost the whole grid." if lncs else "") + "}",
         r"\label{tab:main}",
         r"\scriptsize\setlength{\tabcolsep}{2.4pt}" if lncs else r"\small\setlength{\tabcolsep}{4.2pt}",
         r"\begin{tabular}{@{}l" + "r" * n + r"r@{}}", r"\toprule",
         f" & \\multicolumn{{{n}}}{{c}}{{Memory budget (KB)}} & \\\\",
         f"\\cmidrule(lr){{2-{n+1}}}",
         f"Family & {hdr} & Area \\\\", r"\midrule"]
    for g, _ in BUDGET_GROUPS:
        cells = []
        for b in BUDGETS_KB:
            v = means[g][b]
            c = _fmt(v)
            if abs(v - best[b]) < 1e-12:
                c = f"\\textbf{{{c}}}"
            if fits[g][b] < n_ds:
                c += "$^*$"
            cells.append(c)
        a = area[g].mean()
        ac = _fmt(a)
        ac = f"\\textbf{{{ac}}}" if abs(a - best_area) < 1e-12 else ac
        L.append(f"{g} & " + " & ".join(cells) + f" & {ac} \\\\")
        if g.startswith("KiloRocket"):
            L.append(r"\addlinespace[1pt]")
    L += [r"\midrule",
          r"$\Delta$ vs.\ best other & " + " & ".join(f"${d[0]:+.1f}" + (r"^\dagger" if d[2] < 0.05 else "") + "$" for d in (delta[b] for b in BUDGETS_KB)) + r" & \\",
          r"Datasets ahead & " + " & ".join(f"{delta[b][1]}/{n_ds}" for b in BUDGETS_KB) + r" & \\",
          r"\bottomrule", r"\end{tabular}",
          r"\par\vspace{2pt}{\footnotesize $^*$The family fits the budget on only some "
          r"datasets; chance level is used on the others. $^\dagger p < 0.05$, Wilcoxon signed-rank test over datasets, uncorrected.\par}",
          f"\\end{{{env}}}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% 由 make_paper_assets_v5.py 生成，不要手改\n" + "\n".join(L) + "\n")
    return per, area, fits, delta, area_cmp



# --------------------------------------------------------------------------
# v9：数据集表（Table 2），由扫描结果生成；lncs=True 时左右两栏并排
# --------------------------------------------------------------------------

def tab_datasets(df, path, lncs=False):
    import datasets as D
    scans = [pd.read_csv(f) for f in (getattr(D, "SCAN_INC_FULL_CSV", ""), D.SCAN_INC_CSV)
             if f and os.path.exists(f)]
    sc = pd.concat(scans).drop_duplicates("name").set_index("name")
    rows = []
    for d in df.dataset.unique():
        r = sc.loc[d]
        K = int(r.n_classes)
        S = 1 + int(np.ceil((K - K // 2) / 2))
        rows.append((int(r.n_channels), int(r.length), K, int(r.n_train), int(r.n_test), S, d))
    rows.sort(key=lambda x: (x[0] > 1, x[0], x[2], x[6]))   # 先单变量，再按通道、类别数
    cells = [f"{d} & {C} & {T} & {K} & {ntr} & {nte} & {S}" for C, T, K, ntr, nte, S, d in rows]
    hdr = r"Dataset & $C$ & $T$ & $K$ & Train & Test & $S$"
    if lncs:
        h = (len(cells) + 1) // 2
        body = [cells[k] + " & " + (cells[h + k] if h + k < len(cells) else "& & & & & &") + r" \\"
                for k in range(h)]
        L = [r"\begin{table}[tb]", r"\centering",
             r"\caption{Datasets. $S$: number of sessions under the main protocol.}", r"\label{tab:data}",
             r"\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrrrrrr@{\hspace{7pt}}lrrrrrr@{}}", r"\toprule",
             hdr + " & " + hdr + r" \\", r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    else:
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{Datasets. $S$: number of sessions under the main protocol.}", r"\label{tab:data}",
             r"\scriptsize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             hdr + r" \\", r"\midrule"] + [c + r" \\" for c in cells] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% 由 make_paper_assets_v5.py 生成，不要手改\n" + "\n".join(L) + "\n")
    return rows


# --------------------------------------------------------------------------
# v6新增：方法级主表（仿 FSR 表 2 + HBOP 表 1）与 KiloRocket 消融表
# --------------------------------------------------------------------------
METHOD_ROWS = [  # (显示名, 配置, 是否 CPU 计时可比)
    (r"KiloRocket, $F'=84$ (ours)", "prune-ncmzh-f84", True),
    (r"KiloRocket, 1-bit, $F'=336$ (ours)", "prune-binzh-f336", True),
    (r"MiniRocket protos, default", "mr-ncm-k10000", True),
    (r"MR + ridge replay, $m=8$", "mrrep-f840-m8", True),
    (r"Raw series + 1NN replay, $m=16$", "1nn-m16", True),
    (r"MR + analytic head, bal.", "analytic-cvbal-k2520", True),
    (r"ResNet + Decoupled-Cos.~\cite{zhang2021cec}", "resnet-proto-w64", False),
    (r"ResNet + TEEN~\cite{wang2023teen}", "resnet-teen-w64", False),
    (r"ResNet + ALICE~\cite{peng2022alice}", "resnet-alice-w64", False),
    (r"ResNet + TS-ACL~\cite{li2024tsacl}", "resnet-tsacl-E8192-w64", False),
]
ABL_ROWS = [  # (显示名, 配置)
    (r"KiloRocket, $F'=84$ (all components)", "prune-ncmzh-f84"),
    (r"\quad fixed state in fp32 instead of fp16", "prune-ncmz-f84"),
    (r"\quad centring only, no standardization", "prune-ncmch-f84"),
    (r"\quad fewer kernels ($F=84$) instead of selection$^\ddagger$", "mr-ncmz-k84"),
    (r"\quad 1-bit instead of fp32 prototypes", "prune-binzh-f84"),
]


def _kbfmt(x):
    return f"{x:.1f}" if x < 100 else f"{x:,.0f}".replace(",", "\\,")


def tab_methods(df, path, ref="prune-ncmzh-f84"):
    from scipy.stats import wilcoxon
    acc = df.groupby(["dataset", "config"]).final_acc.mean().unstack()
    n_ds = acc.shape[0]
    rows, ps = [], []
    for lab, c, cpu in METHOD_ROWS:
        g = df[df.config == c]
        if g.empty:
            continue
        kb = g.groupby("dataset").mem_B.mean() / 1024
        r = dict(lab=lab, med=kb.median(), lo=kb.min(), hi=kb.max(), fin=g.final_acc.mean(), avg=g.avg_acc.mean(),
                 base=g.base_last.mean(), nov=g.novel_last.mean(), fgt=g.forgetting.mean(),
                 pd=(g.base_last + g.forgetting - g.final_acc).mean(),  # PD：会话 0（只有 base 类）精度 − 最终精度
                 upd=1000 * g.mean_update_time.mean() if cpu else np.nan)
        if c != ref:
            d = (acc[ref] - acc[c]).dropna()
            r.update(w=int((d > 1e-9).sum()), t=int((d.abs() <= 1e-9).sum()), l=int((d < -1e-9).sum()))
            ps.append(wilcoxon(d).pvalue)
        rows.append(r)
    ph = _holm(ps)
    j = 0
    best = {k: max(r[k] for r in rows) for k in ("fin", "avg", "base", "nov")}
    L = ["% 由 make_paper_assets_v5.py 生成，不要手改", r"\begin{table}[t]", r"\centering",
         r"\caption{Main comparison, mean over " + str(n_ds) + r" datasets $\times$ 5 seeds. Memory: retained KB, median "
         r"[min, max] over datasets. Final, Avg.: accuracy after the last session and averaged over sessions; Base, "
         r"Incr.: final accuracy on base and incremental classes; PD: drop in accuracy from the base session to the last; Forg.: drop in base-class accuracy. Upd.: mean "
         r"update time per session on one CPU core (the ResNet methods ran on a GPU and are not timed). W/T/L: "
         r"datasets on which KiloRocket ($F'=84$) is more accurate / tied / less accurate; $p$: Wilcoxon "
         r"signed-rank test over datasets, Holm-corrected. ResNet methods at $w=64$; TS-ACL with $E=8192$; highest accuracy "
         r"per column in bold.}",
         r"\label{tab:methods}", r"\scriptsize\setlength{\tabcolsep}{1.8pt}",
         r"\begin{tabular}{@{}lrrrrrrrrcl@{}}", r"\toprule",
         r"Method & Memory (KB) & Final & Avg. & Base & Incr. & PD & Forg. & Upd.\,ms & W/T/L & $p$ \\", r"\midrule"]
    for r in rows:
        cells = []
        for k in ("fin", "avg", "base", "nov"):
            v = _fmt(r[k])
            cells.append(f"\\textbf{{{v}}}" if abs(r[k] - best[k]) < 1e-12 else v)
        mem = f"{_kbfmt(r['med'])} [{_kbfmt(r['lo'])}, {_kbfmt(r['hi'])}]" if r["hi"] / max(r["lo"], 1e-9) > 1.05 else _kbfmt(r["med"])
        upd = "--" if np.isnan(r["upd"]) else (f"{r['upd']:.2f}" if r["upd"] < 10 else f"{r['upd']:.0f}")
        if "w" in r:
            pv = ph[j]; j += 1
            wtl, pp = f"{r['w']}/{r['t']}/{r['l']}", ("$<$0.001" if pv < 0.001 else f"{pv:.3f}")
        else:
            wtl, pp = "--", "--"
        L.append(f"{r['lab']} & {mem} & " + " & ".join(cells) + f" & {_fmt(r['pd'])} & {_fmt(r['fgt'])} & {upd} & {wtl} & {pp} \\\\")
        if r["lab"].startswith("KiloRocket, 1-bit"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return rows


def tab_ablation(df, path, ref="prune-ncmzh-f84"):
    from scipy.stats import wilcoxon
    acc = df.groupby(["dataset", "config"]).final_acc.mean().unstack()
    n_ds = acc.shape[0]
    rows, ps = [], []
    for lab, c in ABL_ROWS:
        g = df[df.config == c]
        r = dict(lab=lab, kb=g.groupby("dataset").mem_B.mean().mean() / 1024, fin=g.final_acc.mean(),
                 nov=g.novel_last.mean())
        if c != ref:
            d = (acc[c] - acc[ref]).dropna()
            r.update(dm=100 * d.mean(), ahead=int((d < -1e-9).sum()))
            ps.append(wilcoxon(d).pvalue)
        rows.append(r)
    ph = _holm(ps)
    L = ["% 由 make_paper_assets_v5.py 生成，不要手改", r"\begin{table}[t]", r"\centering",
         r"\caption{Ablation of KiloRocket, changing one component at a time (mean over " + str(n_ds) +
         r" datasets $\times$ 5 seeds; memory is the mean over datasets). $\Delta$: change in final accuracy, "
         r"in points; KR ahead: datasets on which KiloRocket is more accurate; $p$: Wilcoxon signed-rank test, "
         r"Holm-corrected. $^\ddagger$This variant keeps its fixed state in fp32.}",
         r"\label{tab:ablation}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lrrrrrl@{}}", r"\toprule",
         r"Variant & KB & Final & Incr. & $\Delta$ & KR ahead & $p$ \\", r"\midrule"]
    j = 0
    for r in rows:
        if "dm" in r:
            pv = ph[j]; j += 1
            tail = f" & ${r['dm']:+.1f}$ & {r['ahead']}/{n_ds} & " + ("$<$0.001" if pv < 0.001 else f"{pv:.3f}")
        else:
            tail = " & -- & -- & --"
        L.append(f"{r['lab']} & {r['kb']:.1f} & {_fmt(r['fin'])} & {_fmt(r['nov'])}{tail} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=A.V5_CSV)
    ap.add_argument("--check", default=A.CHECK_CSV)
    ap.add_argument("--out", default=os.path.join(HERE, "paper_assets_v5"))
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--tables-only", action="store_true", help="只生成表和数字，不重画图")
    ap.add_argument("--charge-hdc-codebook", action="store_true", help="旧口径：HDC 码本按 1 bit/元素计入")
    ap.add_argument("--deep", nargs="*", default=[DEEP_CSV, DEEP_ORIG_CSV], help="深度 FSCIL 基线结果（不存在的跳过）")
    a = ap.parse_args()
    df = load(a.csv, a.check, a.allow_partial, a.charge_hdc_codebook, a.deep)
    n_ds = df.dataset.nunique()
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
    if not a.tables_only:
        fig1(df, os.path.join(a.out, "figs", "fig1_pareto.pdf"))
        fig2(df, os.path.join(a.out, "figs", "fig2_base_novel.pdf"))
        fig1(df, os.path.join(a.out, "figs", "fig1_pareto_lncs.pdf"), lncs=True)   # v6
    agg = tab_configs(df, os.path.join(a.out, "tab_configs.tex"), n_ds)
    _, b_area, _, b_delta, b_cmp = tab_budget(df, os.path.join(a.out, "tab_budget.tex"), n_ds)
    tab_budget(df, os.path.join(a.out, "tab_budget_lncs.tex"), n_ds, lncs=True)   # v6（LNCS 单栏）版
    tab_datasets(df, os.path.join(a.out, "tab_data.tex"))
    tab_datasets(df, os.path.join(a.out, "tab_data_lncs.tex"), lncs=True)
    r = matched_rows(df)
    tab_matched(r, os.path.join(a.out, "tab_matched.tex"))
    tab_methods(df, os.path.join(a.out, "tab_methods_lncs.tex"))      # v6主表
    tab_ablation(df, os.path.join(a.out, "tab_ablation_lncs.tex"))    # v6消融表
    with open(os.path.join(a.out, "numbers_v5.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# {n_ds} 个数据集，(数据集, seed) 对 {df.groupby(['dataset','seed']).ngroups} 个\n\n")
        fh.write("## Table III 配置均值\n" + agg.round(4).to_string() + "\n\n")
        fh.write("## Table IV 同内存对比\n" + r.round(4).to_string() + "\n\n")
        fh.write("## Table III（v9）预算表：KiloRocket − 每个数据集上最好的其他组（点, 领先数据集数, Wilcoxon p 未校正）\n")
        for b, (dm, w, p) in b_delta.items():
            fh.write(f"  {b:>5} KB: {dm:+.1f}  {w}/{n_ds}  p={p:.4f}\n")
        fh.write("## 预算曲线面积（1 KB–1 MB）：KiloRocket 对各组（领先数据集数, Wilcoxon p 未校正）\n")
        for g, (w, p) in b_cmp.items():
            fh.write(f"  {g}: area={b_area[g].mean():.3f}  {w}/{n_ds}  p={p:.4f}\n")
    print(f"完成：{n_ds} 个数据集 → {a.out}")


if __name__ == "__main__":
    main()
