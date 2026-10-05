"""Plot style and table helpers."""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import exp_minirocket_sweep as E

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
F_FULL = 9996

# 已通过 dataviz 调色板校验（light，白底）：CVD / 正常视觉间距均 PASS；
# 其中 3 色对白底对比度 < 3:1 → 每个系列另配不同标记和线型，并有表格兜底。
C_PRUNE, C_NN, C_HDC, C_MR, C_AN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#1f1f1e", "#52514e", "#e5e4e0"


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

def _ok(df):
    return df[df["error"].isna() | (df["error"] == "")].copy()


def load():
    par = _ok(pd.read_csv(os.path.join(RES, "results_pareto.csv")))
    par["encoder_state_bytes"] = 0
    par.loc[par.config.isin(["minirocket-ncm", "minirocket-retrain"]), "encoder_state_bytes"] = 4 * F_FULL
    mr = _ok(pd.read_csv(os.path.join(RES, "results_mr_sweep.csv")))
    ap = _ok(pd.read_csv(os.path.join(RES, "results_analytic_prune.csv")))
    ap.loc[ap.family == "analytic", "config"] = ap.loc[ap.family == "analytic", "config"].str.replace(
        "analytic-k", "analytic-g1-k")
    ap.loc[ap.family == "analytic", "family"] = "analytic-g1"
    ck = _ok(pd.read_csv(os.path.join(RES, "results_check.csv")))

    base = pd.concat([par, mr, ap], ignore_index=True, sort=False)
    base = base.drop_duplicates(["dataset", "seed", "config"])
    # check 独有的配置（解析头 CV / CV+平衡）
    extra = ck[~ck.set_index(["dataset", "seed", "config"]).index.isin(
        base.set_index(["dataset", "seed", "config"]).index)]
    df = pd.concat([base, extra[base.columns.intersection(extra.columns)]], ignore_index=True, sort=False)
    # 合并拆分列
    split = ck[["dataset", "seed", "config", "base_last", "novel_last", "novel_avg"]]
    df = df.drop(columns=[c for c in ["base_last", "novel_last", "novel_avg"] if c in df], errors="ignore")
    df = df.merge(split, on=["dataset", "seed", "config"], how="left")
    df["mem_B"] = df["final_memory_bytes"].astype(float) + df["encoder_state_bytes"].fillna(0).astype(float)
    return df, ck


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK,
    })
    return plt


def _curve(df, fam, metric="final_acc"):
    return (df[df.family == fam].groupby("knob")
            .agg(mem=("mem_B", "mean"), acc=(metric, "mean")).sort_values("mem"))


def _kb_axis(ax):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ticks = [1, 10, 100, 1e3, 1e4]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1k", "10k"]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig1(df, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    series = [  # family, color, label, marker, linestyle, filled
        ("prune-bin", C_PRUNE, "MR, selected, 1-bit protos", "o", "-", True),
        ("mr-ncm", C_MR, "MR, fewer kernels, fp32 protos", "D", "--", True),
        ("analytic-cvbal", C_AN, "MR, analytic head (tuned, bal.)", "v", "-.", True),
        ("hdc-real", C_HDC, "HDC, real-valued protos", "s", ":", True),
        ("1nn", C_NN, "1NN exemplar replay", "^", "-", True),
    ]
    for fam, col, lab, mk, ls, _ in series:
        g = _curve(df, fam)
        ax.plot(g.mem / 1024, g.acc, ls, color=col, lw=1.4, zorder=3, label=lab,
                marker=mk, ms=4.2, mec="white", mew=0.6)
    # 标出 9.6 KB 点
    g = _curve(df, "prune-bin")
    r = g.loc[840]
    ax.annotate(f"{r.mem/1024:.1f} KB, {r.acc:.3f}", (r.mem / 1024, r.acc), xytext=(10, 8),
                textcoords="offset points", fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    ax.set_xscale("log")
    ax.set_xlim(0.6, 2.2e4)
    _kb_axis(ax)
    ax.set_xlabel("Retained memory (KB, log scale)")
    ax.set_ylabel("Final accuracy")
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False,
              handlelength=2.4, borderaxespad=0, labelspacing=0.3, columnspacing=0.8,
              handletextpad=0.4, fontsize=6.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def fig2(df, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    series = [  # family, color, label, marker, filled
        ("prune-bin", C_PRUNE, "Selected, 1-bit", "o", True),
        ("prune-ncm", C_PRUNE, "Selected, fp32", "o", False),
        ("mr-bin", C_MR, "Fewer kernels, 1-bit", "D", True),
        ("mr-ncm", C_MR, "Fewer kernels, fp32", "D", False),
        ("analytic-cvbal", C_AN, r"Analytic, tuned $\gamma$, balanced", "v", True),
        ("analytic-g1", C_AN, r"Analytic, $\gamma=1$", "v", False),
        ("analytic-cv", C_AN, r"Analytic, tuned $\gamma$", "x", True),
        ("hdc-bin", C_HDC, "HDC, binary", "s", True),
        ("hdc-real", C_HDC, "HDC, real-valued", "s", False),
        ("1nn", C_NN, "1NN replay", "^", True),
    ]
    for fam, col, lab, mk, filled in series:
        g = (df[(df.family == fam) & df.novel_last.notna()].groupby("knob")
             .agg(mem=("mem_B", "mean"), b=("base_last", "mean"), n=("novel_last", "mean"))
             .sort_values("mem"))
        if g.empty:
            continue
        ax.plot(g.b, g.n, "-", color=col, lw=0.7, alpha=0.8, zorder=2)
        kw = dict(color=col, ms=4.6, ls="none", zorder=3, label=lab, marker=mk)
        if mk == "x":
            kw.update(mew=1.1)
        elif filled:
            kw.update(mec="white", mew=0.5)
        else:
            kw.update(mfc="white", mec=col, mew=1.0)
        ax.plot(g.b, g.n, **kw)
    lo, hi = 0.12, 0.86
    ax.plot([lo, hi], [lo, hi], color="#9a9994", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0.5, 0.86)
    ax.set_ylim(0.12, 0.66)
    ax.set_xlabel("Base-class accuracy (last session)")
    ax.set_ylabel("Incremental-class accuracy (last session)")
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.6, labelspacing=0.3, fontsize=6.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# 表
# --------------------------------------------------------------------------

ROWS = [  # (组名, 配置显示, config)
    ("1NN replay", "$m=4$", "1nn-m4"),
    ("", "$m=16$", "1nn-m16"),
    ("HDC, binary", "$D=2048$", "hdc-bin-D2048"),
    ("HDC, real-valued", "$D=512$", "hdc-real-D512"),
    ("", "$D=8192$", "hdc-real-D8192"),
    ("MiniRocket, fewer kernels, fp32", "$F=840$", "mr-ncm-k840"),
    ("", "$F=9996$ (default)", "mr-ncm-k10000"),
    ("MiniRocket, fewer kernels, 1-bit", "$F=840$", "mr-bin-k840"),
    ("MiniRocket, selected, fp32", "$F'=840$", "prune-ncm-f840"),
    ("MiniRocket, selected, 1-bit", "$F'=84$", "prune-bin-f84"),
    ("", "$F'=336$", "prune-bin-f336"),
    ("", "$F'=840$", "prune-bin-f840"),
    ("Analytic head, $\\gamma=1$", "$F=2520$", "analytic-g1-k2520"),
    ("Analytic head, tuned $\\gamma$, balanced", "$F=84$", "analytic-cvbal-k84"),
    ("", "$F=2520$", "analytic-cvbal-k2520"),
    ("MiniRocket-retrain$^\\dagger$", "$F=9996$", "minirocket-retrain"),
    ("1D-CNN", "frozen + NCM", "cnn-frozen-ncm"),
    ("", "fine-tune", "cnn-finetune"),
]


def _fmt(x, nd=3):
    if pd.isna(x):
        return "--"
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("0.") else s


def _kb(b):
    kb = b / 1024
    return f"{kb:,.0f}".replace(",", "\\,") if kb >= 100 else f"{kb:.1f}"


def tab_configs(df, path):
    cfgs = [c for _, _, c in ROWS]
    sub = df[df.config.isin(cfgs)]
    # 平均名次：每个 (数据集, seed) 内对列出的配置按最终精度排名
    rk = sub.pivot_table(index=["dataset", "seed"], columns="config", values="final_acc")
    ranks = rk.rank(axis=1, ascending=False, method="average").mean()
    agg = sub.groupby("config").agg(final=("final_acc", "mean"), avg=("avg_acc", "mean"),
                                    base=("base_last", "mean"), novel=("novel_last", "mean"),
                                    forget=("forgetting", "mean"), mem=("mem_B", "mean"),
                                    n=("final_acc", "size"))
    agg["rank"] = ranks
    best = {c: agg[c].max() for c in ["final", "avg", "base", "novel"]}
    best["forget"] = agg["forget"].min()
    best["rank"] = agg["rank"].min()

    def cell(c, col, nd=3):
        v = agg.loc[c, col]
        s = _fmt(v, nd) if col != "rank" else f"{v:.1f}"
        return f"\\textbf{{{s}}}" if not pd.isna(v) and abs(v - best[col]) < 1e-12 else s

    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Representative configurations, mean over 14 datasets $\times$ 3 seeds. "
         r"Base and Incr.: accuracy on base-session and on incremental classes after the last session "
         r"(``--'': not recorded). Memory includes MiniRocket's fitted biases. Rank: mean rank of final "
         f"accuracy among the {len(ROWS)} listed configurations. Best value per column in bold.}}",
         r"\label{tab:main}", r"\small\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"Method & Config & Final & Avg. & Base & Incr. & Forget. & Memory (KB) & Rank \\",
         r"\midrule"]
    prev = None
    for grp, disp, c in ROWS:
        fam = c.split("-")[0]
        if prev is not None and grp and fam != prev:
            pass
        if c not in agg.index:
            L.append(f"{grp} & {disp} & \\multicolumn{{7}}{{c}}{{(missing)}} \\\\")
            continue
        L.append(f"{grp} & {disp} & {cell(c,'final')} & {cell(c,'avg')} & {cell(c,'base')} & "
                 f"{cell(c,'novel')} & {cell(c,'forget')} & {_kb(agg.loc[c,'mem'])} & {cell(c,'rank')} \\\\")
        prev = fam
    L += [r"\bottomrule",
          r"\multicolumn{9}{@{}l}{\footnotesize $^\dagger$Stores every series seen so far; "
          r"an accuracy reference, not an incremental method.}",
          r"\end{tabular}", r"\end{table*}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% 由 make_paper_assets.py 生成，不要手改\n" + "\n".join(L) + "\n")
    return agg


FAMS = {
    "prune": ["prune-ncm", "prune-bin"], "mr": ["mr-ncm", "mr-bin"], "hdc": ["hdc-bin", "hdc-real"],
    "hdcbin": ["hdc-bin"], "nn": ["1nn"], "an": ["analytic-cvbal"],
}
MATCH_ROWS = [  # (A 名, A, B 名, B, 指标)
    ("Selected", "prune", "1NN replay", "nn", "final_acc"),
    ("Selected", "prune", "HDC", "hdc", "final_acc"),
    ("Selected", "prune", "Fewer kernels", "mr", "final_acc"),
    ("Selected", "prune", "Analytic, bal.", "an", "final_acc"),
    ("Fewer kernels", "mr", "1NN replay", "nn", "final_acc"),
    ("HDC", "hdc", "Fewer kernels", "mr", "final_acc"),
    ("HDC, binary", "hdcbin", "1NN replay", "nn", "final_acc"),
    ("Selected", "prune", "Fewer kernels", "mr", "novel_last"),
    ("Selected", "prune", "Analytic, bal.", "an", "novel_last"),
]


def matched_stats(df, a, b, metric):
    from scipy.stats import wilcoxon
    d = df.copy()
    if metric != "final_acc":
        d = d[d[metric].notna()].copy()
        d["final_acc"] = d[metric]
    res, below, above = E.matched(d, FAMS[a], FAMS[b], "mem_B")
    if res.empty:
        return None
    per = res.groupby("dataset")["delta"].mean()
    try:
        p = wilcoxon(per.values).pvalue
    except ValueError:
        p = np.nan
    return dict(w=int((res.delta > 1e-9).sum()), n=len(res), delta=res.delta.mean() * 100,
                pos=int((per > 0).sum()), nds=len(per), p=p, below=below, above=above)


def _p(p):
    if pd.isna(p):
        return "--"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def tab_matched(df, path):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Matched-memory comparisons (Section~\ref{sec:matched}). Each operating point of A is "
         r"compared with B's accuracy interpolated at the same memory; points outside B's range are "
         r"excluded. Won: points where A is more accurate. $\Delta$: mean difference in accuracy points. "
         r"Datasets: datasets with a positive mean difference. $p$: two-sided Wilcoxon signed-rank test "
         r"over per-dataset means.}",
         r"\label{tab:matched}", r"\footnotesize\setlength{\tabcolsep}{2.6pt}",
         r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
         r"A & B & Won & $\Delta$ & Datasets & $p$ \\", r"\midrule",
         r"\multicolumn{6}{@{}l}{\emph{All classes, final accuracy}} \\"]
    out = []
    novel_started = False
    for an, a, bn, b, m in MATCH_ROWS:
        if m == "novel_last" and not novel_started:
            L += [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Incremental classes only}} \\"]
            novel_started = True
        s = matched_stats(df, a, b, m)
        out.append((an, bn, m, s))
        if s is None:
            L.append(f"{an} & {bn} & \\multicolumn{{4}}{{c}}{{no overlap}} \\\\")
            continue
        L.append(f"{an} & {bn} & {s['w']}/{s['n']} & ${s['delta']:+.1f}$ & {s['pos']}/{s['nds']} & {_p(s['p'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% 由 make_paper_assets.py 生成，不要手改\n" + "\n".join(L) + "\n")
    return out


# --------------------------------------------------------------------------

def numbers(df, agg, mt, path):
    L = ["# 正文引用数字（由 make_paper_assets.py 生成）", ""]
    L.append("## Table II 各配置均值（final / avg / base / incr / forget / mem KB / rank）")
    for c, r in agg.iterrows():
        L.append(f"{c:24s} {r.final:.4f} {r.avg:.4f} {r.base:.4f} {r.novel:.4f} {r.forget:.4f} "
                 f"{r.mem/1024:10.1f} {r['rank']:.2f}")
    L += ["", "## Table III 同内存对比"]
    for an, bn, m, s in mt:
        if s:
            L.append(f"{an} vs {bn} [{m}]: won {s['w']}/{s['n']} ({s['w']/s['n']:.0%}), Δ={s['delta']:+.2f}, "
                     f"datasets {s['pos']}/{s['nds']}, p={s['p']:.4g}, excluded below/above {s['below']}/{s['above']}")
    L += ["", "## 平均 Pareto 前沿（final_acc，口径 B，全部配置）"]
    pa = E.pareto_avg(df, "mem_B")
    for _, r in pa[pa.front].iterrows():
        L.append(f"  {r.config:22s} {r.mem/1024:9.2f} KB  {r.acc:.4f}")
    # 各族曲线
    L += ["", "## 各族曲线（knob: mem KB, final）"]
    for fam in ["prune-bin", "prune-ncm", "mr-bin", "mr-ncm", "analytic-g1", "analytic-cv", "analytic-cvbal",
                "hdc-bin", "hdc-real", "1nn"]:
        g = _curve(df, fam)
        L.append(f"{fam:15s} " + "  ".join(f"{int(k)}:{r.mem/1024:.1f}KB/{r.acc:.3f}" for k, r in g.iterrows()))
    # 剪枝 vs 减核配对
    L += ["", "## 剪枝 vs 减核，同一 F 配对（Δ final / base / incr，点）"]
    from scipy.stats import wilcoxon
    for pf, mf in [("prune-ncm", "mr-ncm"), ("prune-bin", "mr-bin")]:
        for f in [2520, 840, 336, 84]:
            a = df[(df.family == pf) & (df.knob == f)].set_index(["dataset", "seed"])
            b = df[(df.family == mf) & (df.knob == f)].set_index(["dataset", "seed"])
            j = a.join(b, lsuffix="_p", rsuffix="_m", how="inner")
            parts = []
            for col in ["final_acc", "base_last", "novel_last"]:
                dd = (j[f"{col}_p"] - j[f"{col}_m"]).groupby(level=0).mean()
                parts.append(f"{col}: {dd.mean()*100:+.1f} ({int((dd>0).sum())}/{len(dd)}, p={wilcoxon(dd).pvalue:.4f})")
            L.append(f"  {pf} F={f}: " + " | ".join(parts))
    # 逐数据集：prune-bin-f840 vs 1nn-m16 / minirocket-ncm
    L += ["", "## 逐数据集 prune-bin-f840 减去对照（final，点）"]
    p = df[df.config == "prune-bin-f840"].groupby("dataset").final_acc.mean()
    for ref in ["1nn-m16", "mr-ncm-k10000", "hdc-real-D8192"]:
        q = df[df.config == ref].groupby("dataset").final_acc.mean()
        d = (p - q) * 100
        L.append(f"  vs {ref}: 胜 {(d>0).sum()}/14；" + ", ".join(f"{k} {v:+.1f}" for k, v in d.sort_values().items()))
    # 解析头 γ 中位数
    L += ["", "## 解析头选出的 γ（中位数）"]
    ck = df[df.family.isin(["analytic-cv", "analytic-cvbal"])]
    if "gamma" in ck:
        L.append(ck.groupby(["family", "knob"]).gamma.median().to_string())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "paper_assets"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
    df, ck = load()
    if "gamma" not in df and "gamma" in ck:
        df = df.merge(ck[["dataset", "seed", "config", "gamma"]], on=["dataset", "seed", "config"], how="left")
    fig1(df, os.path.join(a.out, "figs", "fig1_pareto.pdf"))
    fig2(df, os.path.join(a.out, "figs", "fig2_base_novel.pdf"))
    agg = tab_configs(df, os.path.join(a.out, "tab_configs.tex"))
    mt = tab_matched(df, os.path.join(a.out, "tab_matched.tex"))
    numbers(df, agg, mt, os.path.join(a.out, "numbers.txt"))
    print("已生成：", a.out)


if __name__ == "__main__":
    main()
