"""Leave-one-dataset-out selection (budget enforced per held-out dataset), bootstrap CIs, external test set (13 datasets added after the design), non-inferiority/TOST tests."""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import analyze_v5 as A

OUT = os.path.join("results", "v6_summaries", "v6_robust_heldout.txt")
L = []
P = L.append
rng = np.random.default_rng(0)
NB = 10000


def boot(x):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(NB, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def pw(x):
    x = np.asarray(x, float)
    try:
        return wilcoxon(x).pvalue if np.any(np.abs(x) > 1e-12) else 1.0
    except ValueError:
        return np.nan


def fmt_pair(lab, d):
    lo, hi = boot(d)
    return (f"  {lab:58s} {100*d.mean():+5.2f} 点  95% CI [{100*lo:+5.2f}, {100*hi:+5.2f}]  "
            f"{int((d > 1e-9).sum())}/{int((d < -1e-9).sum())}/{len(d)}  p={pw(d):.4f}")


df, _ = A.load(os.path.join("results", "results_v6.csv"), os.path.join("results", "results_check_v6.csv"))
main = df[df.protocol == "main"].copy()
# 每个 (数据集, 配置) 对 seed 求均值
acc = main.groupby(["dataset", "config"]).final_acc.mean().unstack()
mem = main.groupby(["dataset", "config"]).mem.mean().unstack()
fam = main.groupby("config").family.first()
dsets = list(acc.index)
P(f"数据集 {len(dsets)} 个；配置 {acc.shape[1]} 个（main 协议，含解析头）")

# 只把增量方法放进候选池：MiniRocket-retrain 存全部数据、CNN fine-tune 是遗忘参照，不参与选择
pool_all = [c for c in acc.columns if fam[c] not in ("minirocket-retrain", "cnn-finetune")]
pool_proto = [c for c in pool_all if fam[c] in A.GROUPS["Proto-selected"]]

# ---------------------------------------------------------------- 1. LODO
P("\n================ 1. 留一数据集选择（LODO） ================")
P("  每折：在其余 26 个数据集上按平均精度选配置；预算在留出数据集上判断（配置在该数据集上的内存 ≤ 预算），在留出数据集上报告精度。")
REF = {"默认 MiniRocket 原型 (575 KB)": "mr-ncm-k10000", "1NN 回放 m=16 (788 KB)": "1nn-m16",
       "v4 配置 1-bit F'=840 (9.6 KB)": "prune-bin-f840"}
for tag, pool, budget_kb in [("原型族，预算 ≤ 10 KB", pool_proto, 10),
                             ("所有增量方法，预算 ≤ 10 KB", pool_all, 10),
                             ("所有增量方法，不限预算", pool_all, np.inf)]:
    chosen, held = {}, {}
    for d in dsets:
        rest = [x for x in dsets if x != d]
        mkb = mem.loc[d, pool] / 1024          # 留出数据集上的内存（不再用其余数据集的平均）
        ok = [c for c in pool if mkb[c] <= budget_kb and acc.loc[rest, c].notna().all() and pd.notna(acc.loc[d, c])]
        c = acc.loc[rest, ok].mean().idxmax()
        chosen[d], held[d] = c, acc.loc[d, c]
    held = pd.Series(held)
    cnt = pd.Series(chosen).value_counts()
    P(f"\n  -- {tag} --")
    P(f"  留出精度均值 {held.mean():.4f}；各折选中的配置：" + ", ".join(f"{k}×{v}" for k, v in cnt.items()))
    P("  各折选中：" + ", ".join(f"{d_}→{c_}" for d_, c_ in chosen.items() if c_ != cnt.index[0]))
    for lab, ref in REF.items():
        P(fmt_pair(f"LODO 选中 − {lab}", (held - acc[ref]).values))
    if budget_kb == 10:
        P(fmt_pair("LODO 选中 − 事后最优单点 prune-ncmzh-f84 (5.2 KB)", (held - acc["prune-ncmzh-f84"]).values))

# ---------------------------------------------------------------- 2. 关键配对 CI
P("\n================ 2. 关键配对差值（95% bootstrap CI，数据集为单位） ================")
mkb_all = mem.mean() / 1024
def lab(c):
    return f"{c} ({mkb_all[c]:.1f} KB, {acc[c].mean():.3f})"
base = "prune-ncmzh-f84"
P(f"  基准：{lab(base)}")
rep_by_mem = [c for c in acc.columns if fam[c] in A.GROUPS["Replay-MR"]]
big = ["prune-ncmzh-f840"] + sorted(rep_by_mem, key=lambda c: -acc[c].mean())[:3] + \
      ["mr-ncm-k10000", "prune-bin-f840", "analytic-cvbal-k2520", "1nn-m16"]
for c in big:
    if c in acc.columns:
        P(fmt_pair(f"{base} − {lab(c)}", (acc[base] - acc[c]).values))

# ---------------------------------------------------------------- 3. 同内存区间 CI
P("\n================ 3. 同内存对比：原型 vs 同编码器回放（逐数据集均值 + bootstrap CI） ================")
res, below, above = A.matched(main, "Proto-selected", "Replay-MR")
for lo, hi in [(0, np.inf)] + A.BANDS[:3]:
    r = res[(res.mem / 1024 >= lo) & (res.mem / 1024 < hi)] if np.isfinite(hi) or lo > 0 else res
    if lo == 100:
        r = res[res.mem / 1024 >= 100]
    if r.empty:
        continue
    per = r.groupby("dataset").delta.mean()
    name = "全部" if (lo == 0 and not np.isfinite(hi)) else (f"≥{lo} KB" if lo == 100 else f"{lo}–{hi} KB")
    P(fmt_pair(f"原型 − 同编码器回放 [{name}]", per.values))
    if lo == 100:
        break

# ---------------------------------------------------------------- 4. 整族
P("\n================ 4. 整族稳健性：standardized 原型（fp16 状态）各 F′ ================")
for c in ["prune-binzh-f84", "prune-binzh-f336", "prune-binzh-f840", "prune-binzh-f2520",
          "prune-ncmzh-f84", "prune-ncmzh-f336", "prune-ncmzh-f840", "prune-ncmzh-f2520"]:
    d = (acc[c] - acc["mr-ncm-k10000"]).values
    lo, hi = boot(d)
    P(f"  {c:20s} {mkb_all[c]:7.1f} KB  精度 {acc[c].mean():.4f}   对默认 MiniRocket {100*d.mean():+5.2f} 点 "
      f"[{100*lo:+5.2f}, {100*hi:+5.2f}]  {int((d>0).sum())}/{len(d)}  p={pw(d):.4f}")

# ---------------------------------------------------------------- 5–7（独立随机数）
rng = np.random.default_rng(1)
KR = "prune-ncmzh-f84"
P("\n================ 5. KiloRocket 各配置的逐数据集内存（KB = 1024 B） ================")
for c in ["prune-binzh-f84", "prune-binzh-f336", "prune-ncmzh-f84", "prune-ncmzh-f336", "prune-binzh-f840"]:
    m = mem[c] / 1024
    P(f"  {c:18s} 均值 {m.mean():5.2f}  中位 {m.median():5.2f}  最小 {m.min():5.2f} ({m.idxmin()})  "
      f"最大 {m.max():5.2f} ({m.idxmax()})  >5 KB 的数据集 {int((m > 5).sum())}  >10 KB {int((m > 10).sum())}")

NEW13 = ["CricketX", "CricketY", "CricketZ", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
         "UWaveGestureLibraryZ", "Fish", "OSULeaf", "MedicalImages", "Plane", "Lightning7", "LSST", "ERing"]
old14 = [x for x in dsets if x not in NEW13]
P("\n================ 6. 设计时用过的 14 个 vs v6 新加的 13 个（外部测试集） ================")
P("  旧 14：" + ", ".join(old14))
for ref, name in [("mr-ncm-k10000", "默认 MiniRocket 原型"), ("prune-bin-f840", "1-bit F'=840, fp32 状态"),
                  ("mrrep-f840-m8", "同编码器回放 m=8"), ("1nn-m16", "1NN 回放 m=16")]:
    for tag, sub in [("旧 14", old14), ("新 13", NEW13)]:
        P(fmt_pair(f"{KR} − {name} [{tag}]", (acc.loc[sub, KR] - acc.loc[sub, ref]).values))

P("\n================ 7. 非劣效 / 等价性（预设 margin，Wilcoxon 单侧，差值平移） ================")
P("  非劣效：H0 差值 ≤ −M；等价（TOST）：两个单侧检验 p 取大者。差值 = KiloRocket − 对比方，单位：点。")
import importlib
deep = []
for f in ["results_deep_v6.csv", "results_deep_orig_v6.csv"]:
    pth = os.path.join("results", f)
    if os.path.exists(pth):
        deep.append(A._ok(pd.read_csv(pth)))
dacc = pd.concat(deep).query("protocol == 'main'").groupby(["dataset", "config"]).final_acc.mean().unstack() if deep else None
def ni_row(name, d):
    d = 100 * np.asarray(d, float)
    lo95, hi95 = np.percentile(rng.choice(d, size=(NB, len(d))).mean(1), [2.5, 97.5])
    out = f"  {name:44s} {d.mean():+5.2f} [{lo95:+5.2f}, {hi95:+5.2f}]  双侧 p={wilcoxon(d).pvalue:.4f}"
    for M in (1, 2):
        p_lo = wilcoxon(d + M, alternative="greater").pvalue
        p_hi = wilcoxon(d - M, alternative="less").pvalue
        out += f" | M={M}: 非劣效 p={p_lo:.4f}, TOST p={max(p_lo, p_hi):.4f}"
    P(out)
for c, nm in [("prune-ncmzh-f336", "标准化 fp32 F'=336"), ("prune-ncmzh-f840", "标准化 fp32 F'=840"),
              ("mrrep-f840-m4", "同编码器回放 m=4"), ("mrrep-f840-m8", "同编码器回放 m=8"),
              ("analytic-cvbal-k2520", "平衡解析头")]:
    ni_row(f"− {nm} ({mkb_all[c]:.0f} KB)", (acc[KR] - acc[c]).values)
if dacc is not None:
    for c, nm in [("resnet-tsacl-E8192-w64", "ResNet + TS-ACL E=8192 w=64"), ("resnet-alice-w64", "ResNet + ALICE w=64（较轻）")]:
        if c in dacc.columns:
            ni_row(f"− {nm}", (acc[KR] - dacc.loc[acc.index, c]).values)

text = "\n".join(L)
print(text)
open(OUT, "w", encoding="utf-8").write(text + "\n")
