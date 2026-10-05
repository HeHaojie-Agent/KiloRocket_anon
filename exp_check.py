"""Protocol runner that records base/incremental accuracy (run_split) and the analytic-head variants."""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import datasets as D
import exp_minirocket_sweep as E
import exp_analytic_prune as AP
from incremental import IncProtocol, session_data, pad_for_rocket

OUT_CSV = os.path.join(D.RESULTS_DIR, "results_check.csv")
REPORT_TXT = os.path.join(D.RESULTS_DIR, "check_summary.txt")
FIG_PARETO = os.path.join(D.RESULTS_DIR, "pareto_final.png")
FIG_SPLIT = os.path.join(D.RESULTS_DIR, "base_vs_novel.png")

KERNELS = [84, 336, 840, 2520, 10000]
ANALYTIC_K = [84, 336, 840, 2520]
GAMMA_GRID = np.logspace(-3, 5, 17)

COLS = ["dataset", "seed", "family", "config", "knob", "n_classes", "n_base_classes",
        "final_acc", "avg_acc", "forgetting", "base_last", "novel_last", "novel_avg",
        "mean_update_time", "final_memory_bytes", "n_features", "encoder_state_bytes",
        "gamma", "error"]


# --------------------------------------------------------------------------
# 带正则选择 / 类平衡的解析头
# --------------------------------------------------------------------------

class AnalyticHead2(AP.AnalyticHead):
    def __init__(self, mu, gamma=1.0, balanced=False):
        super().__init__(mu, gamma)
        self.balanced = balanced

    def update(self, Fs, ys):
        X = Fs.astype(np.float64) - self.mu
        if self.A is None:
            self.A = self.gamma * np.eye(X.shape[1])
        if self.balanced:
            cls, cnt = np.unique(ys, return_counts=True)
            w = np.array([1.0 / cnt[np.searchsorted(cls, c)] for c in ys])
        else:
            w = np.ones(len(ys))
        self.A += (X * w[:, None]).T @ X
        for c in np.unique(ys):
            m = ys == c
            self.B[c] = self.B.get(c, 0) + (X[m] * w[m, None]).sum(axis=0)
        self.ks = np.array(sorted(self.B))
        Bm = np.stack([self.B[k] for k in self.ks], axis=1)
        self.W = np.linalg.solve(self.A, Bm)


def select_gamma(Fb, yb, mu, balanced):
    """只用 base 会话数据选 γ（GCV）。与 AnalyticHead2 的目标函数同形：无截距、μ 中心化。"""
    from sklearn.linear_model import RidgeClassifierCV
    X = Fb.astype(np.float64) - mu
    sw = None
    if balanced:
        cls, cnt = np.unique(yb, return_counts=True)
        sw = np.array([1.0 / cnt[np.searchsorted(cls, c)] for c in yb])
    cv = RidgeClassifierCV(alphas=GAMMA_GRID, fit_intercept=False).fit(X, yb, sample_weight=sw)
    return float(cv.alpha_)


# --------------------------------------------------------------------------
# 带拆分的协议
# --------------------------------------------------------------------------

def run_split(update, predict, y_tr, y_te, sess, proto, X_for_update):
    """update(Xs_or_idx, ys, si)、predict(mask) -> 预测。X_for_update 给出每会话的输入。"""
    idx_tr = np.arange(len(y_tr))[:, None]
    base_cls = sess[0]
    accs, base_accs, novel_accs, times, seen = [], [], [], [], []
    for si, cls in enumerate(sess):
        if si == 0:
            ii, yy = session_data(idx_tr, y_tr, cls, None, proto.seed)
        else:
            ii, yy = session_data(idx_tr, y_tr, cls, proto.n_shot, proto.seed + si)
        ii = ii[:, 0]
        t0 = time.perf_counter()
        update(X_for_update(ii), yy, si)
        times.append(time.perf_counter() - t0)
        seen.extend(cls.tolist())
        m = np.isin(y_te, seen)
        yp = predict(m)
        acc = float(np.mean(yp == y_te[m]))
        yt = y_te[m]
        isb = np.isin(yt, base_cls)
        base_accs.append(float(np.mean(yp[isb] == yt[isb])))
        if si > 0 and (~isb).any():
            novel_accs.append(float(np.mean(yp[~isb] == yt[~isb])))
        accs.append(acc)
    # forgetting 与 incremental.evaluate 一致：base 类在 base 会话后与最后会话后的精度差
    return dict(final_acc=accs[-1], avg_acc=float(np.mean(accs)),
                forgetting=float(base_accs[0] - base_accs[-1]),
                base_last=base_accs[-1],
                novel_last=novel_accs[-1] if novel_accs else np.nan,
                novel_avg=float(np.mean(novel_accs)) if novel_accs else np.nan,
                mean_update_time=float(np.mean(times[1:])) if len(times) > 1 else 0.0)


def head_runner(head, F_tr, F_te, y_tr, y_te, sess, proto):
    return run_split(lambda Fs, ys, si: head.update(Fs, ys), lambda m: head.predict(F_te[m]),
                     y_tr, y_te, sess, proto, lambda ii: F_tr[ii])


def method_runner(method, X_tr, X_te, y_tr, y_te, sess, proto):
    def upd(Xs, ys, si):
        (method.fit_base if si == 0 else method.update)(Xs, ys)
    return run_split(upd, lambda m: method.predict(X_te[m]),
                     y_tr, y_te, sess, proto, lambda ii: X_tr[ii])


# --------------------------------------------------------------------------

def _row(name, seed, fam, cfg, knob, K, nb, res=None, nF=0, enc=0, mem=0, gamma=np.nan, err=""):
    row = dict.fromkeys(COLS)
    row.update(dataset=name, seed=seed, family=fam, config=cfg, knob=knob,
               n_classes=K, n_base_classes=nb, error=err, gamma=gamma)
    if res is not None:
        row.update(res)
        row.update(final_memory_bytes=int(mem), n_features=int(nF), encoder_state_bytes=int(enc))
    return row


def _append(path, row):
    hdr = not os.path.exists(path)
    pd.DataFrame([row], columns=COLS).to_csv(path, mode="a", index=False, header=hdr)


def _say(row):
    if row["error"]:
        print(f"    {row['config']:22s} FAILED {row['error']}")
    else:
        print(f"    {row['config']:22s} acc={row['final_acc']:.4f}  base={row['base_last']:.4f}  "
              f"novel={row['novel_last']:.4f}  mem={row['final_memory_bytes']/1024:9.1f}KB"
              + (f"  γ={row['gamma']:.3g}" if row['gamma'] == row['gamma'] else ""))


def run(names, seeds, proto_kw, out_csv=OUT_CSV, n_jobs=1, refs=True):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = E._done(out_csv)
    for name in names:
        try:
            X_tr, y_tr, X_te, y_te = D.load_split(name)
            X_tr, X_te = D.znormalize(X_tr), D.znormalize(X_te)
        except Exception as e:  # noqa: BLE001
            print(f"!! {name} 加载失败：{type(e).__name__}: {e}")
            continue
        K = len(np.unique(y_tr))
        print(f"\n### {name}  K={K}  C={X_tr.shape[1]}  T={X_tr.shape[2]}")
        for seed in seeds:
            proto = IncProtocol(seed=seed, **proto_kw)
            sess = proto.sessions(np.unique(y_tr))
            nb = len(sess[0])
            idx_tr = np.arange(len(y_tr))[:, None]
            base_idx = session_data(idx_tr, y_tr, sess[0], None, proto.seed)[0][:, 0]
            print(f"  seed={seed}  base 类 {nb} 个，增量会话 {len(sess)-1} 个")

            def emit(row):
                _append(out_csv, row)
                done.add((name, seed, row["config"]))
                _say(row)

            for k in KERNELS:
                want = [f"mr-ncm-k{k}", f"mr-bin-k{k}"]
                if k in ANALYTIC_K:
                    want += [f"analytic-g1-k{k}", f"analytic-cv-k{k}", f"analytic-cvbal-k{k}"]
                if k == AP.PRUNE_FROM:
                    want += [f"prune-{v}-f{t}" for t in AP.PRUNE_TARGETS for v in ("ncm", "bin")]
                if all((name, seed, c) in done for c in want):
                    continue
                try:
                    tf, F_tr, F_te = AP.fit_features(X_tr, X_te, base_idx, k, seed, n_jobs)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:200]
                    for c in want:
                        if (name, seed, c) not in done:
                            emit(_row(name, seed, c.rsplit("-", 1)[0], c, k, K, nb, err=err))
                    continue
                nF = F_tr.shape[1]
                mu = F_tr[base_idx].mean(axis=0).astype(np.float64)

                # 减核原型
                for v in E.VARIANTS:
                    fam = E.fam_of(v)
                    cfg = f"{fam}-k{k}"
                    if (name, seed, cfg) in done:
                        continue
                    head = AP.NCMHead(v, mu)
                    res = head_runner(head, F_tr, F_te, y_tr, y_te, sess, proto)
                    emit(_row(name, seed, fam, cfg, k, K, nb, res, nF, 4 * nF, head.mem(nF)))

                # 解析头三种
                if k in ANALYTIC_K:
                    for tag, bal, cvsel in [("g1", False, False), ("cv", False, True),
                                            ("cvbal", True, True)]:
                        fam = f"analytic-{tag}"
                        cfg = f"{fam}-k{k}"
                        if (name, seed, cfg) in done:
                            continue
                        try:
                            g = select_gamma(F_tr[base_idx], y_tr[base_idx], mu, bal) if cvsel else 1.0
                            head = AnalyticHead2(mu, g, bal)
                            res = head_runner(head, F_tr, F_te, y_tr, y_te, sess, proto)
                            emit(_row(name, seed, fam, cfg, k, K, nb, res, nF, 4 * nF, head.mem(nF), g))
                        except Exception as e:  # noqa: BLE001
                            emit(_row(name, seed, fam, cfg, k, K, nb, err=f"{type(e).__name__}: {e}"[:200]))

                # 剪枝（从 10000 核出发，与 exp_analytic_prune 相同）
                if k == AP.PRUNE_FROM:
                    todo = [(v, t) for t in AP.PRUNE_TARGETS for v in E.VARIANTS
                            if (name, seed, f"prune-{'ncm' if v == 'fp32' else 'bin'}-f{t}") not in done]
                    if todo:
                        sel = AP.sfd_select(F_tr[base_idx], y_tr[base_idx], AP.PRUNE_TARGETS)
                        for v, t in todo:
                            fam = "prune-ncm" if v == "fp32" else "prune-bin"
                            cols = sel[t]
                            Ftr, Fte = F_tr[:, cols], F_te[:, cols]
                            mu_p = Ftr[base_idx].mean(axis=0).astype(np.float64)
                            head = AP.NCMHead(v, mu_p)
                            res = head_runner(head, Ftr, Fte, y_tr, y_te, sess, proto)
                            emit(_row(name, seed, fam, f"{fam}-f{t}", t, K, nb, res, len(cols),
                                      4 * len(cols), head.mem(len(cols)) + 2 * len(cols)))

            # 参照：HDC / 1NN（原始方法类，逐条对应 results_pareto.csv）
            if refs:
                from incremental import HDCPrototype, NNExemplar
                for fam, cfg, knob, make in [
                        ("hdc-bin", "hdc-bin-D2048", 2048, lambda: HDCPrototype(dim=2048, binary=True, seed=seed)),
                        ("hdc-real", "hdc-real-D512", 512, lambda: HDCPrototype(dim=512, binary=False, seed=seed)),
                        ("1nn", "1nn-m4", 4, lambda: NNExemplar(m_per_class=4, seed=seed)),
                        ("1nn", "1nn-m16", 16, lambda: NNExemplar(m_per_class=16, seed=seed))]:
                    if (name, seed, cfg) in done:
                        continue
                    try:
                        m = make()
                        res = method_runner(m, X_tr, X_te, y_tr, y_te, sess, proto)
                        emit(_row(name, seed, fam, cfg, knob, K, nb, res, 0, 0, m.memory_bytes()))
                    except Exception as e:  # noqa: BLE001
                        emit(_row(name, seed, fam, cfg, knob, K, nb, err=f"{type(e).__name__}: {e}"[:200]))


# --------------------------------------------------------------------------
# 分析
# --------------------------------------------------------------------------

def _wil(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    try:
        return wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
    except ValueError:
        return np.nan


def summary(out_csv=OUT_CSV, report=REPORT_TXT, mr_csv=E.OUT_CSV, ap_csv=AP.OUT_CSV,
            pareto_csv=E.PARETO_CSV):
    if not os.path.exists(out_csv):
        print("还没有结果：", out_csv)
        return None
    df = pd.read_csv(out_csv)
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    df["mem_B"] = df["final_memory_bytes"].astype(float) + df["encoder_state_bytes"].fillna(0).astype(float)
    df["mem_A"] = df["final_memory_bytes"].astype(float)
    L = []
    P = L.append
    P(f"核查：{df['dataset'].nunique()} 个数据集 × {df['seed'].nunique()} 个种子，"
      f"{df['config'].nunique()} 个配置，{len(df)} 行")

    # ---- 0. 复现核对
    P("\n================ 0. 复现核对（final_acc 应与之前的结果逐条一致） ================")
    olds = []
    for p in (mr_csv, ap_csv, pareto_csv):
        if os.path.exists(p):
            o = pd.read_csv(p)
            olds.append(o[o["error"].isna() | (o["error"] == "")][["dataset", "seed", "config", "final_acc"]])
    if olds:
        old = pd.concat(olds).rename(columns={"final_acc": "old"})
        old["config"] = old["config"].str.replace(r"^analytic-k", "analytic-g1-k", regex=True)
        j = df.merge(old, on=["dataset", "seed", "config"])
        if len(j):
            d = (j.final_acc - j.old).abs()
            bad = j[d > 1e-9]
            P(f"  可对照 {len(j)} 行；最大差 {d.max():.2e}；不一致 {len(bad)} 行")
            if len(bad):
                P("  不一致的配置：" + ", ".join(sorted(bad.config.unique())[:12]))
                P("  （HDC/1NN 若有微小差异，多半来自 torch 版本；MiniRocket 系应完全为 0）")

    # ---- 1. 拆分表
    P("\n================ 1. base 类 / 新类精度拆分（最后会话后，均值） ================")
    t = df.groupby(["family", "knob"]).agg(
        final=("final_acc", "mean"), base=("base_last", "mean"), novel=("novel_last", "mean"),
        novel_avg=("novel_avg", "mean"), forget=("forgetting", "mean"),
        mem_B_KB=("mem_B", lambda x: x.mean() / 1024))
    P(t.round(4).to_string())

    # ---- 2. 剪枝是否牺牲新类
    P("\n================ 2. 剪枝 vs 减核：同一 F 配对（按数据集平均后 Wilcoxon） ================")
    P("  Δ>0 表示剪枝更好。关键看 novel 列：若 novel 显著为负，说明剪枝在拿新类换 base 类。")
    for v, pf, mf in [("fp32", "prune-ncm", "mr-ncm"), ("bin", "prune-bin", "mr-bin")]:
        for f in AP.PRUNE_TARGETS:
            a = df[(df.family == pf) & (df.knob == f)].set_index(["dataset", "seed"])
            b = df[(df.family == mf) & (df.knob == f)].set_index(["dataset", "seed"])
            j = a[["base_last", "novel_last", "final_acc"]].join(
                b[["base_last", "novel_last", "final_acc"]], lsuffix="_p", rsuffix="_m", how="inner")
            if j.empty:
                continue
            out = []
            for col in ["final_acc", "base_last", "novel_last"]:
                diff = (j[f"{col}_p"] - j[f"{col}_m"]).groupby(level=0).mean()
                out.append(f"{col.split('_')[0]:6s} Δ={diff.mean()*100:+5.1f} ({int((diff>0).sum())}/{len(diff)}, p={_wil(diff):.3f})")
            P(f"  {v:4s} F={f:5d}: " + " | ".join(out))

    # ---- 3. 解析头调参
    P("\n================ 3. 解析头：γ=1 vs CV 选 γ vs CV + 类平衡 ================")
    an = df[df.family.str.startswith("analytic")]
    if len(an):
        t = an.groupby(["family", "knob"]).agg(
            final=("final_acc", "mean"), avg=("avg_acc", "mean"), base=("base_last", "mean"),
            novel=("novel_last", "mean"), forget=("forgetting", "mean"),
            gamma_med=("gamma", "median"), mem_B_KB=("mem_B", lambda x: x.mean() / 1024))
        P(t.round(4).to_string())
        best = an.groupby("family").final_acc.mean().idxmax()
        P(f"  最终精度（各档平均）最好的解析头变体：{best}")

    # ---- 4. 同预算对比
    P("\n================ 4. 同内存预算对比（口径 B，B 侧取前沿插值，不外推） ================")
    PROTO = ["mr-ncm", "mr-bin", "prune-ncm", "prune-bin"]
    for tag, fa, fb in [
            ("Analytic-cvbal vs 原型-任一 ", ["analytic-cvbal"], PROTO),
            ("Analytic-cv    vs 原型-任一 ", ["analytic-cv"], PROTO),
            ("原型-任一 vs Analytic-任一   ", PROTO, ["analytic-g1", "analytic-cv", "analytic-cvbal"]),
            ("剪枝-任一 vs 减核-任一       ", ["prune-ncm", "prune-bin"], ["mr-ncm", "mr-bin"])]:
        P(E.describe_matched(tag, E.matched(df, fa, fb, "mem_B")))

    # 用新类精度做同预算对比（把 final_acc 暂时换成 novel_last）
    P("\n  —— 以新类精度（novel_last）为指标的同预算对比 ——")
    dn = df.copy()
    dn["final_acc"] = dn["novel_last"]
    for tag, fa, fb in [
            ("剪枝-任一 vs 减核-任一       ", ["prune-ncm", "prune-bin"], ["mr-ncm", "mr-bin"]),
            ("原型-任一 vs Analytic-cvbal  ", PROTO, ["analytic-cvbal"])]:
        P(E.describe_matched(tag, E.matched(dn, fa, fb, "mem_B")))

    # ---- 5. 前沿
    P("\n================ 5. 平均帕累托前沿（口径 B；final_acc 与 novel_last 各一份） ================")
    for metric in ["final_acc", "novel_last"]:
        dd = df.copy()
        dd["final_acc"] = dd[metric]
        pa = E.pareto_avg(dd, "mem_B")
        fr = pa[pa.front]
        P(f"  [{metric}] 前沿：" + " → ".join(f"{r.config}({r.mem/1024:.1f}KB,{r.acc:.3f})"
                                           for _, r in fr.iterrows()))

    # ---- 6. 判定
    P("\n================ 6. 判定 ================")
    pb = df[df.family == "prune-bin"].set_index(["dataset", "seed", "knob"])
    mb = df[df.family == "mr-bin"].set_index(["dataset", "seed", "knob"])
    j = pb[["novel_last"]].join(mb[["novel_last"]], lsuffix="_p", rsuffix="_m", how="inner")
    if len(j):
        dnov = (j.novel_last_p - j.novel_last_m).groupby(level=0).mean()
        p = _wil(dnov)
        if dnov.mean() >= 0 or p > 0.05:
            P(f"  ✓ 剪枝没有显著牺牲新类（二值，新类 Δ={dnov.mean()*100:+.1f} 点，p={p:.3f}）。主结论成立。")
        else:
            P(f"  ✗ 剪枝在新类上显著更差（二值，新类 Δ={dnov.mean()*100:+.1f} 点，p={p:.3f}）。"
              "论文必须报告这一点，并把剪枝写成'以新类为代价换 base 类'的取舍。")
    if len(an):
        fin = an.groupby("family").final_acc.mean()
        protos = df[df.family.isin(PROTO)]
        P(f"  解析头最好变体 final={fin.max():.4f}；原型最好配置 final={protos.groupby('config').final_acc.mean().max():.4f}")

    text = "\n".join(L)
    print(text)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n报告已保存：{report}")
    return df


# --------------------------------------------------------------------------

SERIES = [("hdc-bin", "#2a78d6", "HDC binary", "o", "-"),
          ("hdc-real", "#1baf7a", "HDC real-valued", "s", "-"),
          ("1nn", "#eb6834", "1NN exemplar replay", "^", "-"),
          ("mr-ncm", "#7a52c7", "MiniRocket NCM fp32 (fewer kernels)", "D", "--"),
          ("mr-bin", "#b8418a", "MiniRocket NCM binary (fewer kernels)", "P", "--"),
          ("prune-ncm", "#7a52c7", "MiniRocket NCM fp32 (SFD pruning)", "d", ":"),
          ("prune-bin", "#b8418a", "MiniRocket NCM binary (SFD pruning)", "X", ":"),
          ("analytic-cvbal", "#52514e", "Analytic head, tuned γ + class-balanced", "v", "-.")]


def plot(df, path=FIG_PARETO, split_path=FIG_SPLIT, memcol="mem_B"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg = "#fcfcfb"

    # 图 1：只含本次结果的前沿图（HDC/1NN 仅代表点）
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=200, sharey=False)
    fig.patch.set_facecolor(bg)
    for ax, metric, title in [(axes[0], "final_acc", "All classes (final session)"),
                              (axes[1], "novel_last", "Incremental classes only")]:
        ax.set_facecolor(bg)
        for fam, col, lab, mk, ls in SERIES:
            g = df[df.family == fam].groupby("knob").agg(mem=(memcol, "mean"), acc=(metric, "mean")).sort_values("mem")
            if g.empty:
                continue
            ax.plot(g.mem / 1024, g.acc, ls, color=col, lw=1.8, label=lab, zorder=3)
            ax.plot(g.mem / 1024, g.acc, mk, color=col, ms=5.5, mec=bg, mew=1.0, zorder=4)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("Retained memory (KB, log scale)", fontsize=9, color="#52514e")
        ax.set_ylabel("Accuracy", fontsize=9, color="#52514e")
        ax.grid(True, color="#e5e4e0", lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"图已保存：{path}")

    # 图 2：base vs novel 散点（每个配置一个点，均值）
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=200)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    for fam, col, lab, mk, _ in SERIES:
        g = df[df.family == fam].groupby("knob").agg(b=("base_last", "mean"), n=("novel_last", "mean"))
        if g.empty:
            continue
        ax.plot(g.b, g.n, mk, color=col, ms=7, mec=bg, mew=1.0, label=lab, ls="none")
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#8a8985", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("Base-class accuracy (final session)", fontsize=9, color="#52514e")
    ax.set_ylabel("Incremental-class accuracy (final session)", fontsize=9, color="#52514e")
    ax.grid(True, color="#e5e4e0", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(split_path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"图已保存：{split_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--list", default="v5", choices=["v5", "v6"],
                    help="v5=原 14 个数据集；v6=v6全量清单（datasets.py --profile incremental-v6）")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n-way", type=int, default=2)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--no-refs", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--report", default=REPORT_TXT)
    ap.add_argument("--mr", default=E.OUT_CSV)
    ap.add_argument("--ap", default=AP.OUT_CSV)
    ap.add_argument("--pareto", default=E.PARETO_CSV)
    ap.add_argument("--fig", default=FIG_PARETO)
    ap.add_argument("--fig-split", default=FIG_SPLIT)
    a = ap.parse_args()
    if a.data_dir:
        D.set_data_dir(a.data_dir)
    if not a.summary:
        names = a.datasets or (D.selected_incremental_v6() if a.list == "v6" else D.selected_incremental())
        seeds = a.seeds
        if a.quick:
            names, seeds = names[:4], [0]
        run(names, seeds, dict(n_base=None, n_way=a.n_way, n_shot=a.n_shot),
            a.out, a.n_jobs, not a.no_refs)
    df = summary(a.out, a.report, a.mr, a.ap, a.pareto)
    if df is not None:
        plot(df, a.fig, a.fig_split)


if __name__ == "__main__":
    main()
