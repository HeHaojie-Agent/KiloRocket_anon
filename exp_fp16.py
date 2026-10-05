"""Storing every real number in fp16 (series, ridge weights, biases, prototypes)."""
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
from exp_check import run_split, method_runner
from incremental import IncProtocol, session_data, HDCPrototype, NNExemplar

OUT_CSV = os.path.join(D.RESULTS_DIR, "results_fp16.csv")
REPORT = os.path.join(D.RESULTS_DIR, "fp16_summary.txt")
V5_CSV = os.path.join(D.RESULTS_DIR, "results_v5.csv")
CHECK_CSV = os.path.join(D.RESULTS_DIR, "results_check.csv")


def h16(a):
    return np.asarray(a).astype(np.float16).astype(np.float32 if np.asarray(a).dtype == np.float32 else np.float64)


class ReplayRidge16(V.ReplayRidge):
    """同编码器回放，岭模型的全部参数在拟合后舍入到 fp16。"""

    def update(self, ii, ys, si):
        super().update(ii, ys, si)
        self.sc.mean_ = h16(self.sc.mean_)
        self.sc.scale_ = h16(self.sc.scale_)
        self.clf.coef_ = h16(self.clf.coef_)
        self.clf.intercept_ = h16(self.clf.intercept_)


class Proto16(V.ProtoHead):
    """原型在每次更新后舍入到 fp16。"""

    def update(self, Fs, ys, si):
        super().update(Fs, ys, si)
        for c in np.unique(ys):
            self.P[c] = self.P[c].astype(np.float16)

    def proto_bytes(self, nF):
        return 2 * len(self.P) * nF


class HDC16(HDCPrototype):
    """HDC 实值原型存成 fp16：存类均值（余弦分类对缩放不敏感），避免大类求和溢出。"""

    def update(self, X, y):
        H = self._encode(X)
        for c in np.unique(y):
            self.proto[c] = H[y == c].mean(axis=0).astype(np.float16).astype(np.float32)


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
        C, T = X_tr.shape[1], X_tr.shape[2]
        X16_tr = X_tr.astype(np.float16).astype(X_tr.dtype)
        print(f"\n### {name}  K={K}  C={C}  T={T}", flush=True)
        for seed in seeds:
            t0 = time.time()
            proto = IncProtocol(seed=seed, **kw)
            sess = proto.sessions(np.unique(y_tr))
            nb = len(sess[0])
            idx_tr = np.arange(len(y_tr))[:, None]
            base_idx = session_data(idx_tr, y_tr, sess[0], None, seed)[0][:, 0]

            def emit(row):
                V._append(out_csv, row)
                done.add((name, seed, "main", row["config"]))
                tot = (row["final_memory_bytes"] or 0) + (row["encoder_state_bytes"] or 0)
                print(f"    {row['config']:26s} " + (f"FAILED {row['error']}" if row["error"] else
                      f"acc={row['final_acc']:.4f} mem={tot/1024:9.1f}KB"), flush=True)

            def want(cfg):
                return (name, seed, "main", cfg) not in done

            # ------------- MiniRocket：选特征与 v5 相同（fp32 特征上做 SFD）
            mr_cfgs = ([f"fp16-mrrep-f{f}-m{m}" for f in V.REPLAY_F for m in V.REPLAY_M] +
                       [f"fp16-prune-ncmzh-f{t}" for t in V.PRUNE_T])
            if any(want(c) for c in mr_cfgs):
                tf, F_tr, F_te = V.fit_mr(X_tr, X_te, base_idx, V.F_FULL_KERNELS, seed, n_jobs)
                sel = AP.sfd_select(F_tr[base_idx], y_tr[base_idx], V.PRUNE_T)
                del F_tr, F_te
                tf16 = V.fp16_biases(tf)
                G_te = V._transform(tf16, X_te)            # 测试：fp32 序列，fp16 bias
                G16_tr = V._transform(tf16, X16_tr)        # 回放：fp16 序列，fp16 bias
                nF_full = G_te.shape[1]
                print(f"  seed={seed}  特征完成（{time.time()-t0:.1f}s）", flush=True)

                for fsel in V.REPLAY_F:
                    cols = None if fsel == "full" else sel[fsel]
                    Ftr = G16_tr if cols is None else G16_tr[:, cols]
                    Fte = G_te if cols is None else G_te[:, cols]
                    nF = nF_full if cols is None else fsel
                    idx_bytes = 0 if cols is None else 2 * nF
                    for m in V.REPLAY_M:
                        cfg = f"fp16-mrrep-f{fsel}-m{m}"
                        if not want(cfg):
                            continue
                        try:
                            head = ReplayRidge16(Ftr, m, seed)
                            res = run_split(lambda ii, ys, si: head.update(ii, ys, si),
                                            lambda msk: head.predict(Fte[msk]),
                                            y_tr, y_te, sess, proto, lambda ii: ii)
                            stored = len(head.mem_idx) * 2 * C * T
                            weights = 2 * (nF * head.K + head.K + 2 * nF)
                            emit(V._row(name, seed, "main", f"fp16-mrrep-f{fsel}", cfg, m, K, nb, len(sess), res, nF,
                                        stored + 2 * nF * head.K, stored + weights + idx_bytes, 2 * nF))
                        except Exception as e:  # noqa: BLE001
                            emit(V._row(name, seed, "main", f"fp16-mrrep-f{fsel}", cfg, m, K, nb, len(sess),
                                        err=f"{type(e).__name__}: {e}"[:200]))
                del G16_tr

                # 原型不存训练序列 → 训练特征用 fp32 序列 + fp16 bias（与 v5 的 prune-ncmzh 相同）
                union = np.unique(np.concatenate([sel[t] for t in V.PRUNE_T]))
                G_tr = V._transform(tf16, X_tr, union)
                pos = {c: i for i, c in enumerate(union)}
                for t in V.PRUNE_T:
                    cfg = f"fp16-prune-ncmzh-f{t}"
                    if not want(cfg):
                        continue
                    Ftr = G_tr[:, [pos[c] for c in sel[t]]]
                    Fte = G_te[:, sel[t]]
                    mu16 = Ftr[base_idx].mean(axis=0).astype(np.float16).astype(np.float64)
                    sd16 = (Ftr[base_idx].std(axis=0) + 1e-3).astype(np.float16).astype(np.float64)
                    try:
                        head = Proto16("ncmc", mu16, sd=sd16)
                        res = run_split(lambda Fs, ys, si: head.update(Fs, ys, si),
                                        lambda msk: head.predict(Fte[msk]),
                                        y_tr, y_te, sess, proto, lambda ii: Ftr[ii])
                        bp = head.proto_bytes(t)
                        emit(V._row(name, seed, "main", "fp16-prune-ncmzh", cfg, t, K, nb, len(sess), res, t,
                                    bp, bp + 6 * t, 2 * t))
                    except Exception as e:  # noqa: BLE001
                        emit(V._row(name, seed, "main", "fp16-prune-ncmzh", cfg, t, K, nb, len(sess),
                                    err=f"{type(e).__name__}: {e}"[:200]))
                del G_tr, G_te

            # ------------- 1NN 回放（fp16 序列）与 HDC 实值原型（fp16）
            refs = [("fp16-1nn", f"fp16-1nn-m{m}", m, lambda m=m: NNExemplar(m_per_class=m, seed=seed))
                    for m in V.NN_M]
            refs += [("fp16-hdc-real", f"fp16-hdc-real-D{Dm}", Dm, lambda Dm=Dm: HDC16(dim=Dm, binary=False, seed=seed))
                     for Dm in V.HDC_D]
            for fam, cfg, knob, make in refs:
                if not want(cfg):
                    continue
                try:
                    mth = make()
                    Xtr_ = X16_tr if fam == "fp16-1nn" else X_tr
                    res = method_runner(mth, Xtr_, X_te, y_tr, y_te, sess, proto)
                    if fam == "fp16-1nn":
                        n = len(mth.mem_y)
                        bp = mem = n * 2 * C * T
                    else:
                        bp = 2 * len(mth.proto) * knob
                        mem = bp + 8 * C
                    emit(V._row(name, seed, "main", fam, cfg, knob, K, nb, len(sess), res, 0, bp, mem, 0))
                except Exception as e:  # noqa: BLE001
                    emit(V._row(name, seed, "main", fam, cfg, knob, K, nb, len(sess),
                                err=f"{type(e).__name__}: {e}"[:200]))
            print(f"  seed={seed}  完成（{time.time()-t0:.1f}s，累计 {(time.time()-t_all)/60:.1f} 分钟）", flush=True)


# --------------------------------------------------------------------------
# 分析
# --------------------------------------------------------------------------

def summary(out_csv=OUT_CSV, report=REPORT):
    import analyze_v5 as A
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        return np.percentile(rng.choice(x, size=(10000, len(x)), replace=True).mean(1), [2.5, 97.5])

    def pw(x):
        try:
            return wilcoxon(x).pvalue if np.any(np.abs(np.asarray(x)) > 1e-12) else 1.0
        except ValueError:
            return np.nan

    L = []
    P = L.append
    f16 = pd.read_csv(out_csv)
    f16 = f16[f16.error.isna() | (f16.error == "")]
    f16["mem"] = f16.final_memory_bytes.astype(float) + f16.encoder_state_bytes.astype(float)
    base, _ = A.load(V5_CSV, CHECK_CSV)
    base = base[base.protocol == "main"].copy()
    # 统一规则：HDC 码本不计
    h = base.family.astype(str).str.startswith("hdc")
    Dk = base.knob.astype(float)
    Cc = ((base.final_memory_bytes - base.bytes_proto) - 65 * Dk / 8) / (Dk / 8 + 8)
    base.loc[h, "mem"] = (base.final_memory_bytes - (Cc + 65) * Dk / 8)[h] + base.encoder_state_bytes[h]
    P(f"fp16 实测：{f16.dataset.nunique()} 个数据集，{f16.seed.nunique()} 个 seed，{len(f16)} 行")

    # 1. 精度变化：fp16 − fp32（同一配置）
    P("\n== 1. fp16 存储对精度的影响（fp16 − fp32，点；先对 seed 求均值再逐数据集配对） ==")
    a32 = base.groupby(["dataset", "config"]).final_acc.mean().unstack()
    a16 = f16.groupby(["dataset", "config"]).final_acc.mean().unstack()
    m32 = base.groupby("config").mem.mean() / 1024
    m16 = f16.groupby("config").mem.mean() / 1024
    for c16 in sorted(a16.columns):
        c32 = c16.replace("fp16-", "")
        if c32 not in a32:
            continue
        d = (a16[c16] - a32[c32]).dropna()
        P(f"  {c32:22s} {100*d.mean():+5.2f} 点  最大 |Δ| {100*d.abs().max():.2f}  {int((d>1e-9).sum())}/{int((d<-1e-9).sum())}/{len(d)}"
          f"   内存 {m32[c32]:8.1f} → {m16[c16]:8.1f} KB")

    # 2. 用 fp16 版本替换后重做关键对比
    P("\n== 2. 全部实数状态 fp16 时的同内存对比（HDC 码本不计；1-bit 原型族不变） ==")
    repl = {"1nn": "fp16-1nn", "mrrep-f84": "fp16-mrrep-f84", "mrrep-f840": "fp16-mrrep-f840",
            "mrrep-ffull": "fp16-mrrep-ffull", "prune-ncmzh": "fp16-prune-ncmzh", "hdc-real": "fp16-hdc-real"}
    d = base[~base.family.isin(list(repl))].copy()
    g16 = f16.copy()
    g16["family"] = g16.family.map({v: k for k, v in repl.items()})
    g16["config"] = g16.config.str.replace("fp16-", "", regex=False)
    d = pd.concat([d, g16], ignore_index=True, sort=False)
    for ga, gb in [("Proto-selected", "Replay-1NN"), ("Proto-selected", "Replay-MR"),
                   ("Replay-MR", "Replay-1NN"), ("Proto-selected", "HDC")]:
        res, _, _ = A.matched(d, ga, gb)
        per = res.groupby("dataset").delta.mean()
        lo, hi = boot(per.values)
        P(f"  {ga:15s} − {gb:12s}: {100*per.mean():+5.1f} 点 [{100*lo:+.1f}, {100*hi:+.1f}]  "
          f"{int((per>0).sum())}/{len(per)}  p={pw(per.values):.4f}")
        if (ga, gb) == ("Proto-selected", "Replay-MR"):
            for lo_kb, hi_kb in [(0, 10), (10, 100), (100, np.inf)]:
                r = res[(res.mem / 1024 >= lo_kb) & (res.mem / 1024 < hi_kb)]
                if r.empty:
                    continue
                pr = r.groupby("dataset").delta.mean()
                a_, b_ = boot(pr.values)
                P(f"      [{lo_kb}–{hi_kb} KB] {100*pr.mean():+5.1f} 点 [{100*a_:+.1f}, {100*b_:+.1f}]  {int((pr>0).sum())}/{len(pr)}")
    grid = np.logspace(np.log10(256), np.log10(64 * 2**20), 200)
    cross = []
    for (ds, sd), g in d.groupby(["dataset", "seed"]):
        Kc = int(g.n_classes.iloc[0])
        pa = A.budget_curve(g[g.family.isin(A.GROUPS["Proto-selected"])], grid, Kc)
        pb = A.budget_curve(g[g.family.isin(A.GROUPS["Replay-MR"])], grid, Kc)
        better = pb > pa + 1e-9
        cross.append((ds, grid[np.argmax(better)] / 1024 if better.any() else np.nan))
    cr = pd.DataFrame(cross, columns=["dataset", "kb"]).groupby("dataset").kb.median()
    P(f"  回放首次超过原型的中位预算：{np.nanmedian(cr):.1f} KB（从未超过 {int(cr.isna().sum())} 个数据集）")
    cm = d.groupby("config").agg(kb=("mem", lambda x: x.mean() / 1024), acc=("final_acc", "mean")).sort_values("kb")
    best, fr = -1, []
    for c, r in cm.iterrows():
        if r.acc > best and r.kb <= 1024:
            fr.append(f"{c}({r.kb:.1f}KB,{r.acc:.3f})")
            best = r.acc
    P("  平均前沿（≤1 MB）：" + " → ".join(fr))
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
    seeds = a.seeds[:1] if a.quick else a.seeds
    if a.quick:
        names = names[:2]
    t0 = time.time()
    run(names, seeds, a.out, a.n_jobs)
    print(f"\n实验完成，用时 {(time.time()-t0)/60:.1f} 分钟。")
    summary(a.out)


if __name__ == "__main__":
    main()
