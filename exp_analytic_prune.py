"""MiniRocket feature extraction, base-session feature selection (sequential feature detachment) and the analytic (recursive ridge) head."""

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
from incremental import IncProtocol, session_data, pad_for_rocket

OUT_CSV = os.path.join(D.RESULTS_DIR, "results_analytic_prune.csv")
REPORT_TXT = os.path.join(D.RESULTS_DIR, "analytic_prune_summary.txt")
FIG_PNG = os.path.join(D.RESULTS_DIR, "pareto_all_baselines.png")

ANALYTIC_K = [84, 336, 840, 2520]
PRUNE_TARGETS = [2520, 840, 336, 84]
PRUNE_FROM = 10000
PRUNE_STEP = 0.10          # 每轮删掉 10%
GAMMA = 1.0                # 岭正则

COLS = E.COLS


# --------------------------------------------------------------------------
# 分类头
# --------------------------------------------------------------------------

class NCMHead:
    def __init__(self, variant, mu):
        self.variant, self.mu, self.P = variant, mu, {}

    def update(self, Fs, ys):
        for c in np.unique(ys):
            if self.variant == "fp32":
                self.P[c] = Fs[ys == c].mean(axis=0)
            else:
                self.P[c] = (Fs[ys == c].astype(np.float64) - self.mu).sum(axis=0)

    def predict(self, F):
        return E._predict(F, self.P, self.variant, self.mu)

    def mem(self, nF):
        K = len(self.P)
        return 4 * K * nF if self.variant == "fp32" else int(np.ceil(K * nF / 8)) + 4 * nF


class AnalyticHead:
    """递推岭回归；与在已见全部数据上做岭回归等价。"""

    def __init__(self, mu, gamma=GAMMA):
        self.mu, self.gamma = mu, gamma
        self.A = None
        self.B = {}          # 类 -> F 维向量（B 的一列）
        self.W = None
        self.ks = None

    def update(self, Fs, ys):
        X = Fs.astype(np.float64) - self.mu
        if self.A is None:
            self.A = self.gamma * np.eye(X.shape[1])
        self.A += X.T @ X
        for c in np.unique(ys):
            self.B[c] = self.B.get(c, 0) + X[ys == c].sum(axis=0)
        self.ks = np.array(sorted(self.B))
        Bm = np.stack([self.B[k] for k in self.ks], axis=1)
        self.W = np.linalg.solve(self.A, Bm)

    def predict(self, F):
        return self.ks[np.argmax((F.astype(np.float64) - self.mu) @ self.W, axis=1)]

    def mem(self, nF):
        K = len(self.B)
        return 4 * nF * (nF + 1) // 2 + 4 * nF * K + 4 * nF


# --------------------------------------------------------------------------
# 协议
# --------------------------------------------------------------------------

def run_protocol(head, F_tr, F_te, y_tr, y_te, sess, proto, transform_fn, X_tr):
    idx_tr = np.arange(len(y_tr))[:, None]
    accs, base_accs, times, seen = [], [], [], []
    for si, cls in enumerate(sess):
        if si == 0:
            ii, yy = session_data(idx_tr, y_tr, cls, None, proto.seed)
        else:
            ii, yy = session_data(idx_tr, y_tr, cls, proto.n_shot, proto.seed + si)
        ii = ii[:, 0]
        t0 = time.perf_counter()
        if si > 0:
            transform_fn(X_tr[ii])            # 计入真实特征提取耗时
        head.update(F_tr[ii], yy)
        times.append(time.perf_counter() - t0)
        seen.extend(cls.tolist())
        m = np.isin(y_te, seen)
        accs.append(float(np.mean(head.predict(F_te[m]) == y_te[m])))
        mb = np.isin(y_te, sess[0])
        base_accs.append(float(np.mean(head.predict(F_te[mb]) == y_te[mb])))
    return dict(final_acc=accs[-1], avg_acc=float(np.mean(accs)),
                forgetting=float(base_accs[0] - base_accs[-1]),
                mean_update_time=float(np.mean(times[1:])) if len(times) > 1 else 0.0)


def fit_features(X_tr, X_te, base_idx, n_kernels, seed, n_jobs):
    from aeon.transformations.collection.convolution_based import MiniRocket
    tf = MiniRocket(n_kernels=n_kernels, random_state=seed, n_jobs=n_jobs)
    tf.fit(pad_for_rocket(X_tr[base_idx]))
    F_tr = np.asarray(tf.transform(pad_for_rocket(X_tr)), dtype=np.float32)
    F_te = np.asarray(tf.transform(pad_for_rocket(X_te)), dtype=np.float32)
    return tf, F_tr, F_te


def sfd_select(Fb, yb, targets):
    """Sequential Feature Detachment（只用 base 数据）。返回 {F': 保留列下标}。"""
    from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
    mu, sd = Fb.mean(axis=0), Fb.std(axis=0) + 1e-8
    Z = (Fb - mu) / sd
    alpha = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Z, yb).alpha_
    keep = np.arange(Z.shape[1])
    out = {}
    for tgt in sorted(targets, reverse=True):
        while True:
            coef = np.atleast_2d(RidgeClassifier(alpha=alpha).fit(Z[:, keep], yb).coef_)
            imp = np.abs(coef).sum(axis=0)
            nxt = int(np.floor(len(keep) * (1 - PRUNE_STEP)))
            if nxt <= tgt:
                keep = keep[np.sort(np.argsort(-imp)[:tgt])]
                break
            keep = keep[np.sort(np.argsort(-imp)[:nxt])]
        out[tgt] = keep.copy()
    return out


# --------------------------------------------------------------------------

def _row(name, seed, fam, cfg, knob, K, res=None, nF=None, enc=None, mem=None, err=""):
    row = dict.fromkeys(COLS)
    row.update(dataset=name, seed=seed, family=fam, config=cfg, knob=knob,
               n_classes=K, error=err)
    if res is not None:
        row.update(res)
        row.update(final_memory_bytes=int(mem), n_features=int(nF),
                   encoder_state_bytes=int(enc))
    return row


def run(names, seeds, proto_kw, out_csv=OUT_CSV, n_jobs=1, only=None, analytic_k=ANALYTIC_K):
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
        print(f"\n### {name}  K={K}  C={X_tr.shape[1]}  T={X_tr.shape[2]}  n_train={len(y_tr)}")
        for seed in seeds:
            proto = IncProtocol(seed=seed, **proto_kw)
            sess = proto.sessions(np.unique(y_tr))
            idx_tr = np.arange(len(y_tr))[:, None]
            base_idx = session_data(idx_tr, y_tr, sess[0], None, proto.seed)[0][:, 0]

            # ---------- A. 解析式头
            if only in (None, "analytic"):
                for k in analytic_k:
                    cfg = f"analytic-k{k}"
                    if (name, seed, cfg) in done:
                        continue
                    t0 = time.time()
                    try:
                        tf, F_tr, F_te = fit_features(X_tr, X_te, base_idx, k, seed, n_jobs)
                        nF = F_tr.shape[1]
                        mu = F_tr[base_idx].mean(axis=0).astype(np.float64)
                        head = AnalyticHead(mu)
                        res = run_protocol(head, F_tr, F_te, y_tr, y_te, sess, proto,
                                           lambda X: tf.transform(pad_for_rocket(X)), X_tr)
                        row = _row(name, seed, "analytic", cfg, k, K, res, nF, 4 * nF, head.mem(nF))
                        print(f"  s={seed} {cfg:22s} F={nF:5d} acc={res['final_acc']:.4f} "
                              f"mem={row['final_memory_bytes']/1024:9.1f}KB ({time.time()-t0:.1f}s)")
                    except Exception as e:  # noqa: BLE001
                        row = _row(name, seed, "analytic", cfg, k, K, err=f"{type(e).__name__}: {e}"[:200])
                        print(f"  s={seed} {cfg} FAILED {row['error']}")
                    E._append(out_csv, row)

            # ---------- B. SFD 剪枝 + 原型
            if only in (None, "prune"):
                cfgs = [(v, t) for t in PRUNE_TARGETS for v in E.VARIANTS]
                todo = [(v, t) for v, t in cfgs
                        if (name, seed, f"prune-{'ncm' if v=='fp32' else 'bin'}-f{t}") not in done]
                if not todo:
                    continue
                t0 = time.time()
                try:
                    tf, F_tr, F_te = fit_features(X_tr, X_te, base_idx, PRUNE_FROM, seed, n_jobs)
                    sel = sfd_select(F_tr[base_idx], y_tr[base_idx], PRUNE_TARGETS)
                    print(f"  s={seed} SFD 剪枝完成（{time.time()-t0:.1f}s）")
                    for v, t in todo:
                        fam = "prune-ncm" if v == "fp32" else "prune-bin"
                        cfg = f"{fam}-f{t}"
                        cols = sel[t]
                        Ftr, Fte = F_tr[:, cols], F_te[:, cols]
                        mu = Ftr[base_idx].mean(axis=0).astype(np.float64)
                        head = NCMHead(v, mu)
                        # 部署时只算保留的核，这里用全量变换计时，更新耗时会偏高（对剪枝不利、保守）
                        res = run_protocol(head, Ftr, Fte, y_tr, y_te, sess, proto,
                                           lambda X: tf.transform(pad_for_rocket(X)), X_tr)
                        nF = len(cols)
                        mem = head.mem(nF) + 2 * nF
                        row = _row(name, seed, fam, cfg, t, K, res, nF, 4 * nF, mem)
                        print(f"  s={seed} {cfg:22s} acc={res['final_acc']:.4f} mem={mem/1024:8.1f}KB")
                        E._append(out_csv, row)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:200]
                    print(f"  s={seed} prune FAILED {err}")
                    for v, t in todo:
                        fam = "prune-ncm" if v == "fp32" else "prune-bin"
                        E._append(out_csv, _row(name, seed, fam, f"{fam}-f{t}", t, K, err=err))


# --------------------------------------------------------------------------
# 分析
# --------------------------------------------------------------------------

FAM_NAMES = {"hdc-bin": "HDC-bin", "hdc-real": "HDC-real", "1nn": "1NN replay",
             "mr-ncm": "MR-fp32(减核)", "mr-bin": "MR-bin(减核)",
             "prune-ncm": "MR-fp32(剪枝)", "prune-bin": "MR-bin(剪枝)",
             "analytic": "Analytic(RLS)"}


def load_everything(out_csv, mr_csv, pareto_csv):
    df, _ = E.load_all(mr_csv, pareto_csv)
    new = pd.read_csv(out_csv)
    new = new[new["error"].isna() | (new["error"] == "")].copy()
    new["mem_A"] = new["final_memory_bytes"].astype(float)
    new["mem_B"] = new["mem_A"] + new["encoder_state_bytes"].astype(float)
    return pd.concat([df, new], ignore_index=True, sort=False), new


def summary(out_csv=OUT_CSV, mr_csv=E.OUT_CSV, pareto_csv=E.PARETO_CSV, report=REPORT_TXT):
    if not os.path.exists(out_csv):
        print("还没有结果：", out_csv)
        return None
    df, new = load_everything(out_csv, mr_csv, pareto_csv)
    L = []
    P = L.append
    P(f"新基线：{new['dataset'].nunique()} 个数据集 × {new['seed'].nunique()} 个种子，"
      f"{new['config'].nunique()} 个配置，{len(new)} 行")

    P("\n================ 1. 各配置均值 ================")
    t = new.groupby(["family", "knob"]).agg(
        F=("n_features", "mean"), acc=("final_acc", "mean"), avg_acc=("avg_acc", "mean"),
        forget=("forgetting", "mean"),
        mem_B_KB=("mem_B", lambda x: x.mean() / 1024), upd_s=("mean_update_time", "mean"))
    P(t.round(4).to_string())

    P("\n  对照：同样 F 下减核（mr-*）的均值")
    ref = df[df.family.isin(["mr-ncm", "mr-bin"])].groupby(["family", "knob"]).agg(
        acc=("final_acc", "mean"), mem_B_KB=("mem_B", lambda x: x.mean() / 1024))
    P(ref.round(4).to_string())

    P("\n================ 2. 剪枝 vs 减核：同一 F 逐对比较（配对，口径 B 内存几乎相同） ================")
    from scipy.stats import wilcoxon
    for v, pf, mf in [("fp32", "prune-ncm", "mr-ncm"), ("bin", "prune-bin", "mr-bin")]:
        for f in PRUNE_TARGETS:
            a = df[(df.family == pf) & (df.knob == f)].set_index(["dataset", "seed"])["final_acc"]
            b = df[(df.family == mf) & (df.knob == f)].set_index(["dataset", "seed"])["final_acc"]
            j = pd.concat([a.rename("p"), b.rename("m")], axis=1).dropna()
            if j.empty:
                continue
            per = (j.p - j.m).groupby(level=0).mean()
            try:
                p = wilcoxon(per.values).pvalue if np.any(per.values != 0) else np.nan
            except ValueError:
                p = np.nan
            P(f"  {v:4s} F={f:5d}: 剪枝 {j.p.mean():.4f} vs 减核 {j.m.mean():.4f}  "
              f"Δ={(j.p-j.m).mean()*100:+.1f} 点  数据集 {int((per>0).sum())}/{len(per)} 剪枝更好  p={p:.4f}")

    P("\n================ 3. 同内存预算对比（口径 B，B 侧取前沿插值，不外推） ================")
    MR_ANY = ["mr-ncm", "mr-bin"]
    PR_ANY = ["prune-ncm", "prune-bin"]
    for tag, fa, fb in [
        ("剪枝-任一   vs 减核-任一  ", PR_ANY, MR_ANY),
        ("减核-任一   vs 剪枝-任一  ", MR_ANY, PR_ANY),
        ("Analytic    vs MR原型-任一", ["analytic"], MR_ANY + PR_ANY),
        ("MR原型-任一 vs Analytic   ", MR_ANY + PR_ANY, ["analytic"]),
        ("Analytic    vs 1NN replay ", ["analytic"], ["1nn"]),
        ("Analytic    vs HDC-任一   ", ["analytic"], ["hdc-bin", "hdc-real"]),
    ]:
        P(E.describe_matched(tag, E.matched(df, fa, fb, "mem_B")))

    fams = ["hdc-bin", "hdc-real", "1nn", "mr-ncm", "mr-bin", "prune-ncm", "prune-bin", "analytic",
            "cnn-frozen-ncm", "minirocket-retrain"]
    P("\n================ 4. 平均帕累托前沿（口径 B，所有方法） ================")
    pa = E.pareto_avg(df[df.family.isin(fams)], "mem_B")
    for _, r in pa.iterrows():
        P(f"    {r['config']:22s} {r['acc']:.4f}  {r['mem']/1024:10.1f}KB  {'← 前沿' if r['front'] else ''}")

    P("\n================ 5. 读法 ================")
    fr = pa[pa.front]
    P("  前沿成员按族计数：" + ", ".join(f"{FAM_NAMES.get(k,k)}={v}" for k, v in
                                  fr.family.value_counts().items()))
    an = new[new.family == "analytic"]
    if len(an):
        P(f"  Analytic 最省内存的配置平均 {an.groupby('knob').mem_B.mean().min()/1024:.1f} KB；"
          f"F=2520 时 {an[an.knob==2520].mem_B.mean()/1024/1024:.1f} MB（F² 增长）。")

    text = "\n".join(L)
    print(text)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n报告已保存：{report}")
    return df


def plot(df, path=FIG_PNG, memcol="mem_B"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg = "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=200)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    series = [("hdc-bin", "#2a78d6", "HDC binary", "o", "-"),
              ("hdc-real", "#1baf7a", "HDC real-valued", "s", "-"),
              ("1nn", "#eb6834", "1NN exemplar replay", "^", "-"),
              ("mr-ncm", "#7a52c7", "MiniRocket NCM fp32 (fewer kernels)", "D", "--"),
              ("mr-bin", "#b8418a", "MiniRocket NCM binary (fewer kernels)", "P", "--"),
              ("prune-ncm", "#7a52c7", "MiniRocket NCM fp32 (SFD pruning)", "d", ":"),
              ("prune-bin", "#b8418a", "MiniRocket NCM binary (SFD pruning)", "X", ":"),
              ("analytic", "#52514e", "Analytic head (RLS) on MiniRocket", "v", "-.")]
    for fam, col, lab, mk, ls in series:
        g = df[df.family == fam].groupby("knob").agg(mem=(memcol, "mean"), acc=("final_acc", "mean")).sort_values("mem")
        if g.empty:
            continue
        ax.plot(g.mem / 1024, g.acc, ls, color=col, lw=1.8, label=lab, zorder=3)
        ax.plot(g.mem / 1024, g.acc, mk, color=col, ms=5.5, mec=bg, mew=1.0, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Retained memory (KB, log scale), incl. MiniRocket biases", fontsize=9.5, color="#52514e")
    ax.set_ylabel("Final accuracy", fontsize=9.5, color="#52514e")
    ax.grid(True, color="#e5e4e0", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.5, frameon=False)
    fig.savefig(path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"图已保存：{path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n-way", type=int, default=2)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--only", choices=["analytic", "prune"], default=None)
    ap.add_argument("--analytic-10000", action="store_true", help="也跑 F≈10000 的解析头（约 1.6 GB 内存）")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--mr", default=E.OUT_CSV, help="exp_minirocket_sweep 的结果")
    ap.add_argument("--pareto", default=E.PARETO_CSV)
    ap.add_argument("--report", default=REPORT_TXT)
    ap.add_argument("--fig", default=FIG_PNG)
    a = ap.parse_args()
    if a.data_dir:
        D.set_data_dir(a.data_dir)
    if not a.summary:
        names = a.datasets or D.selected_incremental()
        seeds = a.seeds
        if a.quick:
            names, seeds = names[:4], [0]
        ak = ANALYTIC_K + ([10000] if a.analytic_10000 else [])
        run(names, seeds, dict(n_base=None, n_way=a.n_way, n_shot=a.n_shot),
            a.out, a.n_jobs, a.only, ak)
    df = summary(a.out, a.mr, a.pareto, a.report)
    if df is not None:
        plot(df, a.fig)


if __name__ == "__main__":
    main()
