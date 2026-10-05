"""MiniRocket prototype classifiers with fewer kernels (memory sweep) and shared helpers."""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import datasets as D
from incremental import IncProtocol, session_data, pad_for_rocket

OUT_CSV = os.path.join(D.RESULTS_DIR, "results_mr_sweep.csv")
PARETO_CSV = os.path.join(D.RESULTS_DIR, "results_pareto.csv")
REPORT_TXT = os.path.join(D.RESULTS_DIR, "mr_sweep_summary.txt")
FIG_PNG = os.path.join(D.RESULTS_DIR, "pareto_with_mr_sweep.png")

KERNELS = [84, 336, 840, 2520, 10000]
VARIANTS = ["fp32", "bin"]

COLS = ["dataset", "seed", "family", "config", "knob", "n_classes",
        "final_acc", "avg_acc", "forgetting", "mean_update_time",
        "final_memory_bytes", "n_features", "encoder_state_bytes", "error"]


def fam_of(variant):
    return "mr-ncm" if variant == "fp32" else "mr-bin"


def cfg_name(variant, k):
    return f"{fam_of(variant)}-k{k}"


# --------------------------------------------------------------------------
# 单次运行：缓存特征后，两个变体一起跑
# --------------------------------------------------------------------------

def _predict(F, protos, variant, mu):
    ks = np.array(sorted(protos))
    M = np.stack([protos[k] for k in ks])
    if variant == "bin":
        M = M.astype(np.float64)
        M = np.sign(M)
        M[M == 0] = 1.0
        Q = F - mu
    else:
        Q = F
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    return ks[np.argmax(Q @ M.T, axis=1)]


def run_one(X_tr, y_tr, X_te, y_te, seed, n_kernels, proto, n_jobs=1):
    """返回 {variant: 指标 dict}。完全镜像 incremental.evaluate 的流程。"""
    from aeon.transformations.collection.convolution_based import MiniRocket

    sess = proto.sessions(np.unique(y_tr))
    idx_tr = np.arange(len(y_tr))[:, None]          # 用下标代替 X 调 session_data

    # ---- base 会话：拟合 MiniRocket（只看 base 数据），然后变换全部数据并缓存
    base_idx, _ = session_data(idx_tr, y_tr, sess[0], None, proto.seed)
    base_idx = base_idx[:, 0]
    t0 = time.perf_counter()
    tf = MiniRocket(n_kernels=n_kernels, random_state=seed, n_jobs=n_jobs)
    tf.fit(pad_for_rocket(X_tr[base_idx]))
    fit_time = time.perf_counter() - t0
    F_tr = np.asarray(tf.transform(pad_for_rocket(X_tr)), dtype=np.float32)
    F_te = np.asarray(tf.transform(pad_for_rocket(X_te)), dtype=np.float32)
    nF = F_tr.shape[1]
    mu = F_tr[base_idx].mean(axis=0).astype(np.float64)

    out = {}
    for variant in VARIANTS:
        protos = {}
        seen, accs, base_accs, times = [], [], [], []
        for si, cls in enumerate(sess):
            if si == 0:
                ii, yy = session_data(idx_tr, y_tr, cls, None, proto.seed)
            else:
                ii, yy = session_data(idx_tr, y_tr, cls, proto.n_shot, proto.seed + si)
            ii = ii[:, 0]
            t0 = time.perf_counter()
            if si > 0:        # 计入真实的特征提取耗时（缓存只是为了省实验时间）
                tf.transform(pad_for_rocket(X_tr[ii]))
            Fs = F_tr[ii]
            for c in np.unique(yy):
                if variant == "fp32":       # 与 incremental.MiniRocketNCM 完全一致（float32）
                    protos[c] = Fs[yy == c].mean(axis=0)
                else:
                    protos[c] = (Fs[yy == c].astype(np.float64) - mu).sum(axis=0)
            dt = time.perf_counter() - t0
            times.append(fit_time + dt if si == 0 else dt)

            seen.extend(cls.tolist())
            m = np.isin(y_te, seen)
            acc = float(np.mean(_predict(F_te[m], protos, variant, mu) == y_te[m]))
            mb = np.isin(y_te, sess[0])
            bacc = float(np.mean(_predict(F_te[mb], protos, variant, mu) == y_te[mb]))
            accs.append(acc)
            base_accs.append(bacc)

        K = len(protos)
        if variant == "fp32":
            mem = 4 * K * nF
        else:
            mem = int(np.ceil(K * nF / 8)) + 4 * nF
        out[variant] = dict(
            final_acc=accs[-1], avg_acc=float(np.mean(accs)),
            forgetting=float(base_accs[0] - base_accs[-1]),
            mean_update_time=float(np.mean(times[1:])) if len(times) > 1 else 0.0,
            final_memory_bytes=int(mem), n_features=int(nF),
            encoder_state_bytes=int(4 * nF))
    return out


# --------------------------------------------------------------------------

def _done(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df["dataset"], df["seed"], df["config"]))


def _append(path, row):
    hdr = not os.path.exists(path)
    pd.DataFrame([row], columns=COLS).to_csv(path, mode="a", index=False, header=hdr)


def run(names, seeds, proto_kw, out_csv=OUT_CSV, n_jobs=1, kernels=KERNELS):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = _done(out_csv)
    total = len(names) * len(seeds) * len(kernels)
    i = 0
    for name in names:
        try:
            X_tr, y_tr, X_te, y_te = D.load_split(name)
            X_tr, X_te = D.znormalize(X_tr), D.znormalize(X_te)
        except Exception as e:  # noqa: BLE001
            print(f"!! {name} 加载失败：{type(e).__name__}: {e}")
            i += len(seeds) * len(kernels)
            continue
        K = len(np.unique(y_tr))
        print(f"\n### {name}  K={K}  C={X_tr.shape[1]}  T={X_tr.shape[2]}")
        for seed in seeds:
            proto = IncProtocol(seed=seed, **proto_kw)
            for k in kernels:
                i += 1
                if all((name, seed, cfg_name(v, k)) in done for v in VARIANTS):
                    continue
                t0 = time.time()
                try:
                    res = run_one(X_tr, y_tr, X_te, y_te, seed, k, proto, n_jobs)
                    err = ""
                except Exception as e:  # noqa: BLE001
                    res, err = None, f"{type(e).__name__}: {e}"[:200]
                for v in VARIANTS:
                    if (name, seed, cfg_name(v, k)) in done:
                        continue
                    row = dict.fromkeys(COLS)
                    row.update(dataset=name, seed=seed, family=fam_of(v),
                               config=cfg_name(v, k), knob=k, n_classes=K, error=err)
                    if res is not None:
                        row.update(res[v])
                    _append(out_csv, row)
                if res is None:
                    print(f"  [{i:3d}/{total}] s={seed} k={k:5d} FAILED {err}")
                else:
                    a, b = res["fp32"], res["bin"]
                    print(f"  [{i:3d}/{total}] s={seed} k={k:5d} F={a['n_features']:5d} "
                          f"fp32 acc={a['final_acc']:.4f} {a['final_memory_bytes']/1024:7.1f}KB | "
                          f"bin acc={b['final_acc']:.4f} {b['final_memory_bytes']/1024:7.1f}KB "
                          f"({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# 分析
# --------------------------------------------------------------------------

def load_all(out_csv=OUT_CSV, pareto_csv=PARETO_CSV):
    old = pd.read_csv(pareto_csv)
    old = old[old["error"].isna() | (old["error"] == "")].copy()
    new = pd.read_csv(out_csv)
    new = new[new["error"].isna() | (new["error"] == "")].copy()
    old["encoder_state_bytes"] = 0
    # 原稿 minirocket-ncm 在口径 B 下同样要加 bias：F=9996
    mr = old["config"] == "minirocket-ncm"
    old.loc[mr, "encoder_state_bytes"] = 4 * 9996
    df = pd.concat([old, new], ignore_index=True, sort=False)
    df["mem_A"] = df["final_memory_bytes"].astype(float)
    df["mem_B"] = df["mem_A"] + df["encoder_state_bytes"].fillna(0).astype(float)
    return df, new


def _front_curve(g, memcol):
    """一个 (数据集,seed) 内若干配置 -> 按内存排序的帕累托前沿 (mem, acc)。"""
    pts = g[[memcol, "final_acc"]].to_numpy()
    pts = pts[np.argsort(pts[:, 0])]
    front, best = [], -np.inf
    for m, a in pts:
        if a > best:
            front.append((m, a))
            best = a
    return np.array(front)


def matched_interp(df, fams_a, fams_b, memcol, use_front_b=True):
    """A 的每个点 vs B（B 取若干族合并后的前沿）在同内存下插值的精度。

    返回逐点记录 DataFrame（dataset, seed, delta）。超出 B 内存范围的点不外推。
    """
    recs, below, above = [], 0, 0
    for (ds, sd), g in df.groupby(["dataset", "seed"]):
        A = g[g["family"].isin(fams_a)]
        Bg = g[g["family"].isin(fams_b)]
        if len(A) == 0 or len(Bg) < 2:
            continue
        if use_front_b:
            B = _front_curve(Bg, memcol)
        else:
            B = Bg.sort_values(memcol)[[memcol, "final_acc"]].to_numpy()
        if len(B) < 2:
            # 前沿只有一个点：只有正好落在该点内存上的才可比，基本等于不可比
            lb = ub = B[0, 0]
        else:
            lb, ub = B[:, 0].min(), B[:, 0].max()
        for _, r in A.iterrows():
            m = r[memcol]
            if m < lb:
                below += 1
                continue
            if m > ub:
                above += 1
                continue
            if len(B) < 2:
                b = B[0, 1]
            else:
                b = np.interp(np.log(m), np.log(B[:, 0]), B[:, 1])
            recs.append((ds, sd, r["config"], m, r["final_acc"] - b))
    return pd.DataFrame(recs, columns=["dataset", "seed", "config", "mem", "delta"]), below, above


# 统一的对数预算网格：0.25 KB – 64 MB，200 点（与「回放首次超过原型」的交叉点网格相同）
MATCH_GRID = np.logspace(np.log10(256), np.log10(64 * 2**20), 200)


def _envelope(mem, acc, grid):
    """预算包络：预算 b 内（mem <= b）能达到的最高精度；没有配置放得下时为 NaN。"""
    o = np.argsort(mem)
    m, a = np.asarray(mem, float)[o], np.maximum.accumulate(np.asarray(acc, float)[o])
    idx = np.searchsorted(m, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = idx >= 0
    out[ok] = a[idx[ok]]
    return out


def matched(df, fams_a, fams_b, memcol, grid=None):
    """对称的同内存对比：A、B 各自取预算包络，在统一对数网格上逐点相减。

    只用两组都有配置放得下、且都不超过各自最大内存的预算，即
    max(A 最小内存, B 最小内存) <= b <= min(A 最大内存, B 最大内存)，不外推、不插值。
    交换 A、B 只改变符号：delta_AB(b) = -delta_BA(b)。
    返回逐网格点记录（dataset, seed, config="grid", mem=b, delta），与旧版列相同，下游按 dataset 平均即可。
    below / above：A 中内存低于 B 最小值 / 高于 B 最大值的配置数（仅作信息）。"""
    grid = MATCH_GRID if grid is None else np.asarray(grid, float)
    recs, below, above = [], 0, 0
    for (ds, sd), g in df.groupby(["dataset", "seed"]):
        A = g[g["family"].isin(fams_a)]
        B = g[g["family"].isin(fams_b)]
        if len(A) == 0 or len(B) == 0:
            continue
        ma, mb = A[memcol].to_numpy(float), B[memcol].to_numpy(float)
        below += int((ma < mb.min()).sum())
        above += int((ma > mb.max()).sum())
        lo, hi = max(ma.min(), mb.min()), min(ma.max(), mb.max())
        sel = (grid >= lo) & (grid <= hi)
        if not sel.any():
            continue
        b = grid[sel]
        d = _envelope(ma, A["final_acc"].to_numpy(float), b) - _envelope(mb, B["final_acc"].to_numpy(float), b)
        recs.extend((ds, sd, "grid", bb, dd) for bb, dd in zip(b, d))
    return pd.DataFrame(recs, columns=["dataset", "seed", "config", "mem", "delta"]), below, above


def describe_matched(tag, res):
    from scipy.stats import wilcoxon
    d, below, above = res
    if d.empty:
        return f"  {tag}: 没有重叠的内存区间（{below} 点低于对方最小内存，{above} 点高于最大）"
    w, l = int((d.delta > 1e-9).sum()), int((d.delta < -1e-9).sum())
    per = d.groupby("dataset")["delta"].mean()
    try:
        p = wilcoxon(per.values).pvalue if len(per) >= 5 and np.any(per.values != 0) else np.nan
    except ValueError:
        p = np.nan
    return (f"  {tag}: W/L={w}/{l} ({w/len(d):.0%})  Δ={d.delta.mean()*100:+.1f} 点  "
            f"数据集 {int((per>0).sum())}/{len(per)} 为正  Wilcoxon p={p:.4f}  "
            f"[排除：{below} 点低于对方最小内存，{above} 点高于最大]")


def pareto_avg(df, memcol):
    m = df.groupby(["family", "config"]).agg(mem=(memcol, "mean"), acc=("final_acc", "mean")).reset_index()
    m["front"] = [not any((m2 < a_m and a2 > a_a) for m2, a2 in zip(m["mem"], m["acc"]))
                  for a_m, a_a in zip(m["mem"], m["acc"])]
    return m.sort_values("mem")


def summary(out_csv=OUT_CSV, pareto_csv=PARETO_CSV, report=REPORT_TXT):
    if not os.path.exists(out_csv):
        print("还没有结果：", out_csv)
        return None
    df, new = load_all(out_csv, pareto_csv)
    lines = []
    P = lines.append

    P(f"MiniRocket 扫描：{new['dataset'].nunique()} 个数据集 × {new['seed'].nunique()} 个种子，"
      f"{new['config'].nunique()} 个配置，{len(new)} 行")

    # ---- 1. 复现核对
    P("\n================ 1. 复现核对：mr-ncm-k10000 vs 原稿 minirocket-ncm ================")
    a = df[df.config == "minirocket-ncm"].set_index(["dataset", "seed"])["final_acc"]
    b = df[df.config == "mr-ncm-k10000"].set_index(["dataset", "seed"])["final_acc"]
    j = pd.concat([a.rename("orig"), b.rename("new")], axis=1).dropna()
    if len(j):
        diff = (j["new"] - j["orig"]).abs()
        P(f"  {len(j)} 个 (数据集,seed) 对；最大绝对差 {diff.max():.4f}，均值差 {(j['new']-j['orig']).mean():+.4f}")
        P("  （≈0 说明协议与原实验一致；若差很大，先查 aeon 版本是否与跑主实验时相同）")

    # ---- 2. 各配置均值
    P("\n================ 2. 各配置均值（所有数据集 × 种子） ================")
    t = new.groupby(["family", "knob"]).agg(
        F=("n_features", "mean"), acc=("final_acc", "mean"), avg_acc=("avg_acc", "mean"),
        forget=("forgetting", "mean"),
        mem_A_KB=("final_memory_bytes", lambda x: x.mean() / 1024),
        bias_KB=("encoder_state_bytes", lambda x: x.mean() / 1024),
        upd_s=("mean_update_time", "mean"))
    P(t.round(4).to_string())
    P("\n  参照（原稿 results_pareto.csv）：")
    ref = df[df.family.isin(["hdc-bin", "hdc-real", "1nn"])].groupby("config").agg(
        acc=("final_acc", "mean"), mem_KB=("mem_A", lambda x: x.mean() / 1024)).sort_values("mem_KB")
    P(ref.round(4).to_string())

    # ---- 3. 同预算对比：核心问题
    for memcol, label in [("mem_A", "口径 A（与现稿一致，不计 MiniRocket bias）"),
                          ("mem_B", "口径 B（计入 MiniRocket bias 4F 字节）")]:
        P(f"\n================ 3. 同内存预算对比 —— {label} ================")
        P("  B 侧取该族在每个 (数据集,seed) 上的帕累托前沿，线性插值于 log 内存；不外推。")
        for tag, fa, fb in [
            ("HDC-bin  vs MR-fp32   ", ["hdc-bin"], ["mr-ncm"]),
            ("HDC-bin  vs MR-bin    ", ["hdc-bin"], ["mr-bin"]),
            ("HDC-bin  vs MR-任一    ", ["hdc-bin"], ["mr-ncm", "mr-bin"]),
            ("HDC-real vs MR-任一    ", ["hdc-real"], ["mr-ncm", "mr-bin"]),
            ("HDC-任一 vs MR-任一    ", ["hdc-bin", "hdc-real"], ["mr-ncm", "mr-bin"]),
            ("MR-任一  vs HDC-任一   ", ["mr-ncm", "mr-bin"], ["hdc-bin", "hdc-real"]),
            ("MR-任一  vs 1NN replay ", ["mr-ncm", "mr-bin"], ["1nn"]),
            ("HDC-bin  vs 1NN（原稿）", ["hdc-bin"], ["1nn"]),
        ]:
            P(describe_matched(tag, matched(df, fa, fb, memcol)))

        # 按预算分段看 HDC-任一 vs MR-任一
        d, _, _ = matched(df, ["hdc-bin", "hdc-real"], ["mr-ncm", "mr-bin"], memcol)
        if not d.empty:
            P("  按预算分段（HDC-任一 vs MR-任一）：")
            bins = [0, 10e3, 30e3, 100e3, 300e3, 1e9]
            labs = ["<10KB", "10-30KB", "30-100KB", "100-300KB", ">300KB"]
            d["bin"] = pd.cut(d["mem"], bins, labels=labs)
            for lb_, g in d.groupby("bin", observed=True):
                P(f"    {lb_:>9s}: n={len(g):3d}  HDC 胜 {int((g.delta>1e-9).sum()):3d}  "
                  f"Δ={g.delta.mean()*100:+.1f} 点")

        P(f"\n  ---- 平均帕累托前沿（{memcol}）----")
        pa = pareto_avg(df, memcol)
        for _, r in pa.iterrows():
            P(f"    {r['config']:20s} {r['acc']:.4f}  {r['mem']/1024:9.1f}KB  {'← 前沿' if r['front'] else ''}")

    # ---- 4. 判定
    P("\n================ 4. 判定（以口径 B 为准，更保守） ================")
    d, below, above = matched(df, ["hdc-bin", "hdc-real"], ["mr-ncm", "mr-bin"], "mem_B")
    pa = pareto_avg(df, "mem_B")
    front = pa[pa.front]
    low = front[front.mem < 250 * 1024]
    hdc_low = low["family"].str.startswith("hdc").sum()
    mr_low = low["family"].str.startswith("mr").sum()
    P(f"  250 KB 以下的平均前沿点：HDC {hdc_low} 个，MiniRocket {mr_low} 个")
    if not d.empty:
        P(f"  同预算 HDC 胜率 {(d.delta>0).mean():.0%}，平均 Δ={d.delta.mean()*100:+.1f} 点")
    if mr_low == 0 and (d.empty or d.delta.mean() > 0):
        P("  → 情形 1：HDC 仍然占据低预算前沿。主线不变，把 MiniRocket 曲线加进 Fig.1。")
    elif hdc_low == 0:
        P("  → 情形 2：MiniRocket 在低预算区也支配 HDC。主线需改写为"
          "『随机冻结编码器 + 原型 vs replay』，HDC 降为该类的一员。")
    else:
        P("  → 介于两者之间：前沿由两族分段占据。按分段结果改写 'which method to use' 规则，"
          "HDC 的主张收窄到它仍在前沿上的预算区间。")

    text = "\n".join(lines)
    print(text)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n报告已保存：{report}")
    return df


# --------------------------------------------------------------------------

def plot(df, path=FIG_PNG, memcol="mem_B"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    bg = "#fcfcfb"
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    series = [("hdc-bin", "#2a78d6", "HDC binary", "o", "-"),
              ("hdc-real", "#1baf7a", "HDC real-valued", "s", "-"),
              ("1nn", "#eb6834", "1NN exemplar replay", "^", "-"),
              ("mr-ncm", "#7a52c7", "MiniRocket-NCM, fp32 (kernel sweep)", "D", "--"),
              ("mr-bin", "#b8418a", "MiniRocket-NCM, binary (kernel sweep)", "P", "--")]
    for fam, col, lab, mk, ls in series:
        g = df[df.family == fam].groupby("knob").agg(mem=(memcol, "mean"), acc=("final_acc", "mean")).sort_values("mem")
        if g.empty:
            continue
        ax.plot(g.mem / 1024, g.acc, ls, color=col, lw=2, label=lab, zorder=3)
        ax.plot(g.mem / 1024, g.acc, mk, color=col, ms=6, mec=bg, mew=1.2, zorder=4)
    pa = pareto_avg(df[df.family.isin([s[0] for s in series])], memcol)
    fr = pa[pa.front].sort_values("mem")
    ax.step(fr.mem / 1024, fr.acc, where="post", color="#52514e", lw=1, ls=":", label="Pareto front", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(f"Retained memory (KB, log scale){' incl. MiniRocket biases' if memcol=='mem_B' else ''}",
                  fontsize=9.5, color="#52514e")
    ax.set_ylabel("Final accuracy", fontsize=9.5, color="#52514e")
    ax.grid(True, color="#e5e4e0", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor=bg, edgecolor="#e5e4e0")
    fig.savefig(path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"图已保存：{path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--kernels", nargs="*", type=int, default=KERNELS)
    ap.add_argument("--n-way", type=int, default=2)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--pareto", default=PARETO_CSV, help="主实验结果（HDC/1NN 行从这里读）")
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
        print(f"数据集 {len(names)} × 种子 {len(seeds)} × 核数 {len(a.kernels)} "
              f"（每次同时产出 fp32 与二值两个配置）")
        run(names, seeds, dict(n_base=None, n_way=a.n_way, n_shot=a.n_shot),
            a.out, a.n_jobs, a.kernels)
    df = summary(a.out, a.pareto, a.report)
    if df is not None:
        plot(df, a.fig)


if __name__ == "__main__":
    main()
