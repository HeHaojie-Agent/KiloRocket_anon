"""Loading, byte accounting and matched-memory comparison used by all analysis scripts."""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import datasets as D
import exp_minirocket_sweep as E

V5_CSV = os.path.join(D.RESULTS_DIR, "results_v5.csv")
CHECK_CSV = os.path.join(D.RESULTS_DIR, "results_check.csv")
OUT_TXT = os.path.join(D.RESULTS_DIR, "v5_summary.txt")
MEANS_CSV = os.path.join(D.RESULTS_DIR, "v5_config_means.csv")

# 方法族分组（同内存对比时，一组内的族先合并再取前沿）
GROUPS = {
    "Proto-selected": ["prune-bin", "prune-binh", "prune-binz", "prune-binzh", "prune-ncmc",
                       "prune-ncmz", "prune-ncmch", "prune-ncmzh"],
    "Proto-selected (v4: bin/ncm)": ["prune-bin", "prune-ncm"],
    "Proto-fewer": ["mr-bin", "mr-ncmc", "mr-ncmz", "mr-ncm"],
    "Replay-MR": ["mrrep-f84", "mrrep-f840", "mrrep-ffull"],
    "Replay-1NN": ["1nn"],
    "HDC": ["hdc-bin", "hdc-real"],
    "Analytic-bal": ["analytic-cvbal"],
    "CNN-frozen": ["cnn-frozen"],
}
PAIRS = [   # (A, B) 同内存对比
    ("Proto-selected", "Replay-1NN"),
    ("Proto-selected", "Replay-MR"),
    ("Replay-MR", "Proto-selected"),
    ("Replay-MR", "Replay-1NN"),
    ("Proto-selected", "HDC"),
    ("Proto-selected", "Proto-fewer"),
    ("Proto-selected", "Analytic-bal"),
    ("Proto-selected", "CNN-frozen"),
    ("Proto-selected", "Proto-selected (v4: bin/ncm)"),
]
BANDS = [(0, 10), (10, 100), (100, 1000), (1000, np.inf)]   # KB


def _ok(df):
    return df[df["error"].isna() | (df["error"] == "")].copy()


def _wil(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    try:
        return wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
    except ValueError:
        return np.nan


def fix_nn_bytes(df):
    """NNExemplar.memory_bytes() 用的是 ndarray.nbytes，而数据按 float64 读入，
    所以 1NN 回放的原始序列被按 8 字节/值记账；论文统一按 fp32（4 字节/值）计 → 减半。
    （同编码器回放和 MiniRocket-retrain 在 exp_v5 里已按 4·C·T 计，不受影响。）"""
    d = df.copy()
    nn = d.family.astype(str) == "1nn"
    for c in ("final_memory_bytes", "bytes_proto"):
        if c in d:
            d.loc[nn, c] = d.loc[nn, c].astype(float) / 2
    return d


def load(v5_csv, check_csv):
    v5 = fix_nn_bytes(_ok(pd.read_csv(v5_csv)))
    v5["mem"] = v5["final_memory_bytes"].astype(float) + v5["encoder_state_bytes"].astype(float)
    extra = None
    if os.path.exists(check_csv):
        ck = fix_nn_bytes(_ok(pd.read_csv(check_csv)))
        an = ck[ck.family.str.startswith("analytic")].copy()
        if len(an):
            an["protocol"] = "main"
            an["mem"] = an["final_memory_bytes"].astype(float) + an["encoder_state_bytes"].astype(float)
            an = an[an.dataset.isin(v5.dataset.unique())]
            extra = an
        ck["mem"] = ck["final_memory_bytes"].astype(float) + ck["encoder_state_bytes"].astype(float)
    else:
        ck = None
    df = pd.concat([v5, extra], ignore_index=True, sort=False) if extra is not None else v5
    return df, ck


def config_means(df):
    g = df.groupby(["protocol", "family", "config"])
    m = g.agg(final=("final_acc", "mean"), avg=("avg_acc", "mean"), base=("base_last", "mean"),
              novel=("novel_last", "mean"), forget=("forgetting", "mean"),
              kb=("mem", lambda x: x.mean() / 1024), kb_geo=("mem", lambda x: np.exp(np.log(x).mean()) / 1024),
              upd_ms=("mean_update_time", lambda x: 1000 * x.mean()), n=("final_acc", "size")).reset_index()
    return m


def pareto(m):
    m = m.sort_values("kb").copy()
    best, fr = -1, []
    for a in m["final"]:
        fr.append(a > best)
        best = max(best, a)
    m["front"] = fr
    return m


def matched(df, A, B):
    d = df.copy()
    d["family"] = d["family"].astype(str)
    fa, fb = GROUPS[A], GROUPS[B]
    res, below, above = E.matched(d, fa, fb, "mem")
    return res, below, above


def describe(res, below, above):
    if res.empty:
        return "无重叠", np.nan
    per = res.groupby("dataset")["delta"].mean()
    p = _wil(per.values)
    w = int((res.delta > 1e-9).sum())
    return (f"胜 {w}/{len(res)}  Δ={res.delta.mean()*100:+.1f} 点  数据集 {int((per > 0).sum())}/{len(per)}  "
            f"p={p:.4f}  [排除 低于 {below} / 高于 {above}]"), p


def budget_curve(g, grid, K):
    """给定预算 b，该族在 ≤ b 的配置中能达到的最高精度；放不下时记为随机猜测 1/K。"""
    pts = g[["mem", "final_acc"]].to_numpy()
    out = []
    for b in grid:
        ok = pts[pts[:, 0] <= b]
        out.append(ok[:, 1].max() if len(ok) else 1.0 / K)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=V5_CSV)
    ap.add_argument("--check", default=CHECK_CSV)
    ap.add_argument("--out", default=OUT_TXT)
    a = ap.parse_args()

    df, ck = load(a.csv, a.check)
    L = []
    P = L.append
    main_df = df[df.protocol == "main"].copy()
    # 每 (数据集, seed) 按 config 取均值（同一 config 不应重复）
    P(f"v5 结果：{df.dataset.nunique()} 个数据集，seed {sorted(df.seed.unique())}，"
      f"协议 {sorted(df.protocol.unique())}，{len(df)} 行")

    # ---------------------------------------------------------------- 0
    P("\n================ 0. 复现核对（main 协议 vs results_check.csv） ================")
    if ck is not None:
        j = main_df.merge(ck[["dataset", "seed", "config", "final_acc", "mem"]],
                          on=["dataset", "seed", "config"], suffixes=("", "_v4"))
        if len(j):
            diff = (j.final_acc - j.final_acc_v4).abs()
            dm = (j.mem - j.mem_v4).abs()
            P(f"  共同配置 {j.config.nunique()} 个、{len(j)} 行；final_acc 最大差 {diff.max():.2e}；"
              f"内存最大差 {dm.max():.0f} 字节")
            bad = j[diff > 1e-6].groupby("config").size()
            if len(bad):
                P("  ⚠ 不一致的配置：" + ", ".join(f"{k}({v})" for k, v in bad.items()))
        else:
            P("  没有共同的 (数据集, seed, 配置)。")
    else:
        P("  找不到 results_check.csv，跳过。")

    # ---------------------------------------------------------------- 1
    M = config_means(df)
    M.to_csv(MEANS_CSV, index=False)
    Mm = M[M.protocol == "main"].copy()
    # 名次
    piv = main_df.pivot_table(index=["dataset", "seed"], columns="config", values="final_acc")
    rank = piv.rank(axis=1, ascending=False).mean()
    Mm["rank"] = Mm["config"].map(rank)
    P("\n================ 1. 各配置均值（main；KB 为算术平均，KBgeo 为几何平均） ================")
    P(f"  {'config':26s} {'final':>6s} {'avg':>6s} {'base':>6s} {'novel':>6s} {'forget':>6s} "
      f"{'KB':>9s} {'KBgeo':>8s} {'rank':>6s} {'upd ms':>7s}")
    for _, r in Mm.sort_values("kb").iterrows():
        P(f"  {r.config:26s} {r.final:6.3f} {r.avg:6.3f} {r.base:6.3f} {r.novel:6.3f} {r.forget:6.3f} "
          f"{r.kb:9.1f} {r.kb_geo:8.1f} {r['rank']:6.1f} {r.upd_ms:7.1f}")

    # ---------------------------------------------------------------- 2
    P("\n================ 2. 配对效应（同一 F，Δ = A − B，点；p 为逐数据集均值的 Wilcoxon） ================")
    pairs = []
    for t in [84, 336, 840, 2520]:
        pairs += [(f"prune-ncmc-f{t}", f"prune-ncm-f{t}", "fp32 中心化 vs 不中心化"),
                  (f"prune-ncmz-f{t}", f"prune-ncmc-f{t}", "fp32 z-score vs 中心化"),
                  (f"prune-bin-f{t}", f"prune-ncmc-f{t}", "1-bit vs fp32 中心化"),
                  (f"prune-binz-f{t}", f"prune-bin-f{t}", "1-bit z-score vs 1-bit"),
                  (f"prune-binz-f{t}", f"prune-ncmz-f{t}", "1-bit z vs fp32 z"),
                  (f"prune-binh-f{t}", f"prune-bin-f{t}", "1-bit fp16 状态 vs fp32 状态"),
                  (f"prune-binzh-f{t}", f"prune-binz-f{t}", "1-bit z fp16 vs fp32"),
                  (f"prune-bin-teen-f{t}", f"prune-bin-f{t}", "TEEN 校准（1-bit）"),
                  (f"prune-ncmc-teen-f{t}", f"prune-ncmc-f{t}", "TEEN 校准（fp32 中心化）")]
    for k in [84, 336, 840, 2520, 10000]:
        pairs += [(f"mr-ncmc-k{k}", f"mr-ncm-k{k}", "减核：fp32 中心化 vs 不中心化"),
                  (f"mr-ncmz-k{k}", f"mr-ncmc-k{k}", "减核：fp32 z vs 中心化")]
    for A_, B_, tag in pairs:
        a_ = main_df[main_df.config == A_].set_index(["dataset", "seed"])
        b_ = main_df[main_df.config == B_].set_index(["dataset", "seed"])
        j = a_.join(b_, lsuffix="_a", rsuffix="_b", how="inner")
        if j.empty:
            continue
        s = []
        for col in ["final_acc", "base_last", "novel_last"]:
            d = (j[f"{col}_a"] - j[f"{col}_b"]).groupby(level=0).mean()
            s.append(f"{col.split('_')[0]} {d.mean()*100:+5.1f} ({int((d > 0).sum())}/{len(d)}, p={_wil(d):.3f})")
        P(f"  {tag:30s} {A_:22s} vs {B_:20s} | " + " | ".join(s))

    # ---------------------------------------------------------------- 3
    P("\n================ 3. 平均帕累托前沿（main，final_acc，算术平均内存） ================")
    fr = pareto(Mm[["config", "family", "final", "kb", "kb_geo", "novel"]].rename(columns={}))
    for _, r in fr[fr.front].iterrows():
        P(f"  {r.config:26s} {r.kb:9.2f} KB  final={r.final:.4f}  novel={r.novel:.4f}")
    P("  —— 用几何平均内存排序时的前沿 ——")
    fr2 = pareto(Mm[["config", "family", "final", "kb_geo", "novel"]].rename(columns={"kb_geo": "kb"}))
    for _, r in fr2[fr2.front].iterrows():
        P(f"  {r.config:26s} {r.kb:9.2f} KB(geo)  final={r.final:.4f}")
    P("  —— 各族曲线 ——")
    for fam, g in Mm.groupby("family"):
        g = g.sort_values("kb")
        P(f"  {fam:18s} " + "  ".join(f"{str(r.config).split('-')[-1]}:{r.kb:.1f}KB/{r.final:.3f}"
                                      for _, r in g.iterrows()))

    # ---------------------------------------------------------------- 4
    P("\n================ 4. 同内存对比（main；A 的每个点对 B 的前沿插值） ================")
    pvals = []
    for A_, B_ in PAIRS:
        res, below, above = matched(main_df, A_, B_)
        txt, pv = describe(res, below, above)
        pvals.append((f"{A_} vs {B_}", pv))
        P(f"  {A_:30s} vs {B_:30s}: {txt}")
        if not res.empty:
            for lo, hi in BANDS:
                r = res[(res.mem / 1024 >= lo) & (res.mem / 1024 < hi)]
                if len(r):
                    per = r.groupby("dataset")["delta"].mean()
                    P(f"      {lo:>5}–{hi:<5} KB: {len(r):4d} 点  Δ={r.delta.mean()*100:+6.1f}  "
                      f"数据集 {int((per > 0).sum())}/{len(per)}  p={_wil(per.values):.4f}")
    ok = [(n, v) for n, v in pvals if v == v]
    if ok:
        order = np.argsort([v for _, v in ok])
        m_ = len(ok)
        adj, run = [0.0] * m_, 0.0
        for rank_, i in enumerate(order):
            run = max(run, min(1.0, (m_ - rank_) * ok[i][1]))
            adj[i] = run
        P("  —— Holm 校正后的 p（上表整体对比） ——")
        for (n, v), a_ in zip(ok, adj):
            P(f"      {n:62s} p={v:.4f}  Holm p={a_:.4f}")
    P("  —— 只看新类精度（novel_last） ——")
    dn = main_df.copy()
    dn["final_acc"] = dn["novel_last"]
    for A_, B_ in [("Proto-selected", "Replay-MR"), ("Proto-selected", "Replay-1NN"),
                   ("Proto-selected", "Analytic-bal")]:
        res, below, above = matched(dn, A_, B_)
        P(f"  {A_:30s} vs {B_:30s}: {describe(res, below, above)[0]}")

    # ---------------------------------------------------------------- 5
    P("\n================ 5. 交叉点：同编码器回放何时超过原型 ================")
    grid = np.logspace(np.log10(256), np.log10(64 * 2**20), 200)   # 0.25 KB – 64 MB
    cross = []
    for (ds, sd), g in main_df.groupby(["dataset", "seed"]):
        K = int(g.n_classes.iloc[0])
        pa = budget_curve(g[g.family.isin(GROUPS["Proto-selected"])], grid, K)
        pb = budget_curve(g[g.family.isin(GROUPS["Replay-MR"])], grid, K)
        better = pb > pa + 1e-9
        cross.append((ds, sd, grid[np.argmax(better)] / 1024 if better.any() else np.nan,
                      pa[-1], pb[-1]))
    cr = pd.DataFrame(cross, columns=["dataset", "seed", "cross_kb", "proto_best", "replay_best"])
    per = cr.groupby("dataset").agg(cross_kb=("cross_kb", "median"), proto=("proto_best", "mean"),
                                    replay=("replay_best", "mean"))
    for ds, r in per.sort_values("cross_kb").iterrows():
        P(f"  {ds:28s} 回放首次超过原型的预算 ≈ {r.cross_kb:9.1f} KB   "
          f"最高精度 原型 {r.proto:.3f} / 回放 {r.replay:.3f}")
    P(f"  中位数交叉预算 {np.nanmedian(per.cross_kb):.1f} KB；回放从未超过的数据集 {int(per.cross_kb.isna().sum())} 个；"
      f"回放最高精度更高的数据集 {int((per.replay > per.proto).sum())}/{len(per)}")

    # ---------------------------------------------------------------- 6
    P("\n================ 6. AUC（MEMO 式）：预算-精度曲线下的平均精度 ================")
    P("  A_f(b) = 该族在内存 ≤ b 的配置中能达到的最高 final_acc；一个都放不下时记为 1/K。")
    for lo_kb, hi_kb in [(1, 1024), (1, 16), (16, 1024)]:
        g_ = np.logspace(np.log10(lo_kb * 1024), np.log10(hi_kb * 1024), 60)
        rows = []
        for (ds, sd), g in main_df.groupby(["dataset", "seed"]):
            K = int(g.n_classes.iloc[0])
            for name, fams in GROUPS.items():
                if name.startswith("Proto-selected (v4"):
                    continue
                gg = g[g.family.isin(fams)]
                if gg.empty:
                    continue
                rows.append((ds, sd, name, budget_curve(gg, g_, K).mean()))
        au = pd.DataFrame(rows, columns=["dataset", "seed", "group", "auc"])
        tab = au.groupby(["group", "dataset"]).auc.mean().unstack(0)
        P(f"  预算区间 {lo_kb}–{hi_kb} KB：" + "  ".join(
            f"{c}={tab[c].mean():.3f}" for c in tab.mean().sort_values(ascending=False).index))
        if "Proto-selected" in tab:
            for c in tab.columns:
                if c == "Proto-selected":
                    continue
                d = (tab["Proto-selected"] - tab[c]).dropna()
                P(f"      Proto-selected − {c:14s}: {d.mean()*100:+5.1f} 点  {int((d > 0).sum())}/{len(d)}  p={_wil(d):.4f}")

    # ---------------------------------------------------------------- 7
    P("\n================ 7. 内存构成（main，均值） ================")
    P("  fixed = 总内存 − 随类数增长的部分（bias、μ、σ、下标、码本、scaler 等）")
    comp = main_df.groupby("config").agg(tot=("mem", "mean"), proto=("bytes_proto", "mean"))
    comp["fixed_frac"] = 1 - comp.proto / comp.tot
    for c in list(fr[fr.front].config) + ["prune-bin-f840", "prune-binzh-f840", "prune-binh-f840",
                                           "mr-ncm-k10000", "mrrep-f840-m4", "1nn-m4"]:
        if c in comp.index:
            r = comp.loc[c]
            P(f"  {c:26s} 总 {r.tot/1024:8.2f} KB  其中随类增长 {r.proto/1024:8.2f} KB  固定状态占 {r.fixed_frac*100:5.1f}%")

    # ---------------------------------------------------------------- 8
    P("\n================ 8. 敏感性协议 ================")
    for prot in [p for p in df.protocol.unique() if p != "main"]:
        sub = df[df.protocol == prot]
        common = sorted(set(sub.config) & set(main_df.config))
        P(f"  —— {prot}：{sub.dataset.nunique()} 个数据集，{len(common)} 个共同配置 ——")
        mp = M[M.protocol == prot].set_index("config")
        mm = Mm.set_index("config")
        for c in sorted(common, key=lambda c: mp.loc[c, "kb"]):
            P(f"    {c:24s} {prot}: final={mp.loc[c,'final']:.3f} novel={mp.loc[c,'novel']:.3f}  "
              f"main: final={mm.loc[c,'final']:.3f}   {mp.loc[c,'kb']:8.1f} KB")
        piv = sub.pivot_table(index=["dataset", "seed"], columns="config", values="final_acc")
        rk = piv.rank(axis=1, ascending=False).mean().sort_values()
        P("    平均名次：" + ", ".join(f"{c} {v:.1f}" for c, v in rk.items()))
        fr_p = pareto(mp.reset_index()[["config", "family", "final", "kb", "novel"]])
        P("    前沿：" + " → ".join(f"{r.config}({r.kb:.1f}KB, {r.final:.3f})" for _, r in fr_p[fr_p.front].iterrows()))
        for A_, B_ in [("Proto-selected", "Replay-1NN"), ("Proto-selected", "Replay-MR")]:
            res, below, above = matched(sub, A_, B_)
            P(f"    {A_} vs {B_}: {describe(res, below, above)[0]}")

    # ---------------------------------------------------------------- 9
    P("\n================ 9. 逐数据集（main，seed 平均，final_acc × 100） ================")
    wide = main_df.groupby(["dataset", "config"]).final_acc.mean().unstack()
    best_small = [c for c in ["prune-binzh-f840", "prune-binz-f840", "prune-bin-f840", "prune-binh-f840"]
                  if c in wide]
    refs = [c for c in ["mr-ncm-k10000", "mr-ncmz-k10000", "1nn-m16", "mrrep-f840-m4", "mrrep-f840-m16",
                        "hdc-real-D8192", "analytic-cvbal-k2520"] if c in wide]
    for a_ in best_small:
        for b_ in refs:
            d = (wide[a_] - wide[b_]).dropna() * 100
            if d.empty:
                continue
            P(f"  {a_:18s} − {b_:22s}: 均值 {d.mean():+5.1f}  胜 {int((d > 0).sum())}/{len(d)}  "
              f"p={_wil(d):.4f}   " + ", ".join(f"{k} {v:+.1f}" for k, v in d.sort_values().items()))

    # ---------------------------------------------------------------- 10
    P("\n================ 10. 更新耗时（main，ms，均值） ================")
    for fam, g in Mm.groupby("family"):
        P(f"  {fam:20s} {g.upd_ms.min():8.1f} – {g.upd_ms.max():8.1f} ms")

    txt = "\n".join(L)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"\n已写入 {a.out}，配置均值表 {MEANS_CSV}")


if __name__ == "__main__":
    main()
