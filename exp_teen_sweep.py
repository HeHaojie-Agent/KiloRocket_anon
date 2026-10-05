"""TEEN calibration sweep over alpha, tau and prototype type."""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import datasets as D
import exp_analytic_prune as AP
import exp_v5 as V
from exp_check import run_split
from incremental import IncProtocol, session_data

OUT_CSV = os.path.join(D.RESULTS_DIR, "results_teen.csv")
REPORT = os.path.join(D.RESULTS_DIR, "teen_summary.txt")
V5_CSV = os.path.join(D.RESULTS_DIR, "results_v5.csv")

ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
TAUS = [4.0, 16.0]
VARIANTS = [("ncmc", False), ("bin", False), ("ncmz", True)]   # (名字, 是否 z-score)
PRUNE_T = V.PRUNE_T


class TeenHead(V.ProtoHead):
    """与 exp_v5.ProtoHead 相同，只是 TEEN 的 α、τ 可以改。"""

    def __init__(self, variant, mu, alpha, tau, sd=None):
        super().__init__(variant, mu, teen=True, sd=sd)
        self.alpha, self.tau = alpha, tau

    def _calibrate(self, v):
        B = V._unit(np.stack([self.P[k].astype(np.float64) for k in self.base_keys]))
        vn = V._unit(v)
        s = self.tau * (B @ vn)
        w = np.exp(s - s.max())
        w /= w.sum()
        return self.alpha * vn + (1 - self.alpha) * (w @ B)


def _cfg(vname, t, alpha=None, tau=None):
    if alpha is None:
        return f"teensweep-{vname}-none-f{t}"
    return f"teensweep-{vname}-a{alpha:g}-t{tau:g}-f{t}"


def run(names, seeds, out_csv, n_jobs):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = V._done(out_csv)
    kw = V.PROTOCOLS["main"]
    t_all = time.time()
    for name in names:
        try:
            X_tr, y_tr, X_te, y_te = D.load_split(name)
            X_tr, X_te = D.znormalize(X_tr), D.znormalize(X_te)
        except Exception as e:  # noqa: BLE001
            print(f"!! {name} 加载失败：{type(e).__name__}: {e}")
            continue
        K = len(np.unique(y_tr))
        print(f"\n### {name}  K={K}", flush=True)
        for seed in seeds:
            todo = [(vn, t, a, tau) for vn, _ in VARIANTS for t in PRUNE_T
                    for a, tau in [(None, None)] + [(a, tau) for a in ALPHAS for tau in TAUS]
                    if (name, seed, "main", _cfg(vn, t, a, tau)) not in done]
            if not todo:
                continue
            t0 = time.time()
            proto = IncProtocol(seed=seed, **kw)
            sess = proto.sessions(np.unique(y_tr))
            nb = len(sess[0])
            idx_tr = np.arange(len(y_tr))[:, None]
            base_idx = session_data(idx_tr, y_tr, sess[0], None, seed)[0][:, 0]
            tf, F_tr, F_te = V.fit_mr(X_tr, X_te, base_idx, V.F_FULL_KERNELS, seed, n_jobs)
            sel = AP.sfd_select(F_tr[base_idx], y_tr[base_idx], PRUNE_T)
            print(f"  seed={seed}  特征 + SFD {time.time()-t0:.1f}s，{len(todo)} 个配置", flush=True)
            for vname, z in VARIANTS:
                v = "ncmc" if vname == "ncmz" else vname
                for t in PRUNE_T:
                    cols = sel[t]
                    Ftr, Fte = F_tr[:, cols], F_te[:, cols]
                    mu = Ftr[base_idx].mean(axis=0).astype(np.float64)
                    sd = Ftr[base_idx].std(axis=0).astype(np.float64) + 1e-8
                    for a, tau in [(None, None)] + [(a, tau) for a in ALPHAS for tau in TAUS]:
                        cfg = _cfg(vname, t, a, tau)
                        if (name, seed, "main", cfg) in done:
                            continue
                        fam = f"teensweep-{vname}"
                        try:
                            head = (V.ProtoHead(v, mu, False, sd if z else None) if a is None
                                    else TeenHead(v, mu, a, tau, sd if z else None))
                            res = run_split(lambda Fs, ys, si: head.update(Fs, ys, si),
                                            lambda m: head.predict(Fte[m]),
                                            y_tr, y_te, sess, proto, lambda ii: Ftr[ii])
                            bp = head.proto_bytes(t)
                            row = V._row(name, seed, "main", fam, cfg, t, K, nb, len(sess), res, t, bp, bp, 4 * t)
                        except Exception as e:  # noqa: BLE001
                            row = V._row(name, seed, "main", fam, cfg, t, K, nb, len(sess),
                                         err=f"{type(e).__name__}: {e}"[:200])
                        V._append(out_csv, row)
                        done.add((name, seed, "main", cfg))
            print(f"  seed={seed}  完成（{time.time()-t0:.1f}s，累计 {(time.time()-t_all)/60:.1f} 分钟）", flush=True)


# --------------------------------------------------------------------------
# 分析
# --------------------------------------------------------------------------

def _wil(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    try:
        return wilcoxon(x).pvalue if len(x) >= 5 and np.any(np.abs(x) > 1e-12) else np.nan
    except ValueError:
        return np.nan


def summary(out_csv=OUT_CSV, report=REPORT):
    df = pd.read_csv(out_csv)
    df = df[df.error.isna() | (df.error == "")]
    L = []
    P = L.append
    P(f"TEEN α/τ 扫描：{df.dataset.nunique()} 个数据集，{df.seed.nunique()} 个 seed，{len(df)} 行")

    # 0. 复现核对
    if os.path.exists(V5_CSV):
        v5 = pd.read_csv(V5_CSV)
        v5 = v5[(v5.protocol == "main") & (v5.error.isna())]
        P("\n== 0. 复现核对：α=0.5, τ=16 与 results_v5.csv 的 prune-*-teen，以及不校准基线与 prune-ncmc / prune-bin / prune-ncmz ==")
        pairs = []
        for vn, src in [("ncmc", "prune-ncmc-teen"), ("bin", "prune-bin-teen")]:
            for t in PRUNE_T:
                pairs.append((_cfg(vn, t, 0.5, 16.0), f"{src}-f{t}"))
        for vn, src in [("ncmc", "prune-ncmc"), ("bin", "prune-bin"), ("ncmz", "prune-ncmz")]:
            for t in PRUNE_T:
                pairs.append((_cfg(vn, t), f"{src}-f{t}"))
        diffs = []
        for mine, theirs in pairs:
            a = df[df.config == mine].set_index(["dataset", "seed"]).final_acc
            b = v5[v5.config == theirs].set_index(["dataset", "seed"]).final_acc
            j = pd.concat([a, b], axis=1, join="inner")
            if len(j):
                diffs.append((mine, len(j), float((j.iloc[:, 0] - j.iloc[:, 1]).abs().max())))
        if diffs:
            mx = max(d[2] for d in diffs)
            P(f"  可对照 {sum(d[1] for d in diffs)} 行，最大差 {mx:.2e}" + ("（完全一致）" if mx < 1e-9 else "（不一致，请检查！）"))

    # 1. 每个 (原型, F′, α, τ) 相对不校准基线的配对差
    P("\n== 1. TEEN − 不校准（final accuracy，点）：先对 seed 求均值，再逐数据集配对；Wilcoxon 未校正 ==")
    g = df.groupby(["dataset", "config"])[["final_acc", "base_last", "novel_last"]].mean()
    acc = g.final_acc.unstack()
    base = g.base_last.unstack()
    nov = g.novel_last.unstack()
    rows = []
    for vn, _ in VARIANTS:
        for t in PRUNE_T:
            c0 = _cfg(vn, t)
            if c0 not in acc:
                continue
            for a in ALPHAS:
                for tau in TAUS:
                    c = _cfg(vn, t, a, tau)
                    if c not in acc:
                        continue
                    d = (acc[c] - acc[c0]).dropna()
                    rows.append(dict(variant=vn, F=t, alpha=a, tau=tau, delta=100 * d.mean(),
                                     win=int((d > 1e-9).sum()), loss=int((d < -1e-9).sum()), n=len(d), p=_wil(d),
                                     d_base=100 * (base[c] - base[c0]).mean(), d_novel=100 * (nov[c] - nov[c0]).mean()))
    R = pd.DataFrame(rows)
    for vn, _ in VARIANTS:
        sub = R[R.variant == vn]
        if sub.empty:
            continue
        P(f"\n  -- 原型 {vn} --   （每格：Δfinal 点 [赢/输]）")
        for tau in TAUS:
            P(f"   τ={tau:g}")
            for t in PRUNE_T:
                s = sub[(sub.F == t) & (sub.tau == tau)].sort_values("alpha")
                cells = "  ".join(f"α={r.alpha:<4g} {r.delta:+5.1f} [{r.win}/{r.loss}]" for r in s.itertuples())
                P(f"     F′={t:<5d} {cells}")
    # 2. 结论用的几个数
    P("\n== 2. 汇总 ==")
    if len(R):
        best = R.sort_values("delta", ascending=False).iloc[0]
        P(f"  所有 {len(R)} 个 (原型, F′, α, τ) 组合里，TEEN 为正的有 {int((R.delta > 0).sum())} 个，"
          f"显著为正（p<0.05 且 Δ>0）的有 {int(((R.p < 0.05) & (R.delta > 0)).sum())} 个")
        P(f"  最好的组合：{best.variant} F′={int(best.F)} α={best.alpha:g} τ={best.tau:g}  "
          f"Δ={best.delta:+.2f} 点 [{best.win}/{best.loss}]  p={best.p:.3f}  "
          f"(Δbase {best.d_base:+.1f}, Δnovel {best.d_novel:+.1f})")
        for vn, _ in VARIANTS:
            s = R[R.variant == vn]
            if len(s):
                P(f"  {vn}: Δ 范围 {s.delta.min():+.1f} 到 {s.delta.max():+.1f} 点；"
                  f"新类精度变化 {s.d_novel.min():+.1f} 到 {s.d_novel.max():+.1f}；base 变化 {s.d_base.min():+.1f} 到 {s.d_base.max():+.1f}")
        R.round(4).to_csv(os.path.join(D.RESULTS_DIR, "teen_sweep_table.csv"), index=False)
    text = "\n".join(L)
    print(text)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n报告：{report}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--list", default="v5", choices=["v5", "v6"],
                    help="v5=原 14 个数据集；v6=v6全量清单（datasets.py --profile incremental-v6）")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()
    if a.summary:
        summary(a.out)
        return
    if a.data_dir:
        D.set_data_dir(a.data_dir)
    names = a.datasets or (D.selected_incremental_v6() if a.list == "v6" else D.selected_incremental())
    seeds = a.seeds
    if a.quick:
        names, seeds = names[:2], seeds[:1]
    t0 = time.time()
    run(names, seeds, a.out, a.n_jobs)
    print(f"\n实验完成，用时 {(time.time()-t0)/60:.1f} 分钟。")
    summary(a.out)


if __name__ == "__main__":
    main()
