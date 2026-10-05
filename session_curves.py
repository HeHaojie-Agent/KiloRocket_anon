"""Re-runs KiloRocket and same-encoder replay recording the accuracy after each session."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import datasets as D
import exp_analytic_prune as AP
import exp_v5 as V
from incremental import IncProtocol, session_data


def run_split_acc(update, predict, y_tr, y_te, sess, proto, X_for_update):
    """与 exp_check.run_split 逐行相同，另返回每个会话后的精度。"""
    idx_tr = np.arange(len(y_tr))[:, None]
    accs = []
    seen = []
    for si, cls in enumerate(sess):
        if si == 0:
            ii, yy = session_data(idx_tr, y_tr, cls, None, proto.seed)
        else:
            ii, yy = session_data(idx_tr, y_tr, cls, proto.n_shot, proto.seed + si)
        ii = ii[:, 0]
        update(X_for_update(ii), yy, si)
        seen.extend(cls.tolist())
        m = np.isin(y_te, seen)
        accs.append(float(np.mean(predict(m) == y_te[m])))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="npz 目录（只在没有 aeon 缓存时用）。注意 npz 是 float32，与主实验的 float64 原始数据不完全相同，"
                         "MiniRocket 的偏置会略有变化；要与主实验逐行一致，请不加此参数，直接用 aeon 缓存")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=os.path.join(D.RESULTS_DIR, "session_curves.csv"))
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--ref", default=None, help="主实验结果，用于比对 final_acc（默认 results_v6.csv，没有则 results_v5.csv）")
    a = ap.parse_args()
    t_keep, f_rep, m_rep = 84, 840, 8
    rows = []
    for name in a.datasets:
        if a.data:
            z = np.load(os.path.join(a.data, f"{name}.npz"))
            X_tr, y_tr, X_te, y_te = z["X_tr"], z["y_tr"], z["X_te"], z["y_te"]
        else:
            X_tr, y_tr, X_te, y_te = D.load_split(name)      # 与 exp_v5.run 完全相同（float64）
        X_tr, X_te = D.znormalize(X_tr), D.znormalize(X_te)
        for seed in a.seeds:
            proto = IncProtocol(seed=seed, **V.PROTOCOLS["main"])
            sess = proto.sessions(np.unique(y_tr))
            idx_tr = np.arange(len(y_tr))[:, None]
            base_idx = session_data(idx_tr, y_tr, sess[0], None, seed)[0][:, 0]
            tf, F_tr, F_te = V.fit_mr(X_tr, X_te, base_idx, V.F_FULL_KERNELS, seed, a.n_jobs)
            sel = AP.sfd_select(F_tr[base_idx], y_tr[base_idx], V.PRUNE_T)

            # KiloRocket：fp16 bias 特征 + fp16 μ、σ，标准化 fp32 原型
            cols = sel[t_keep]
            tf16 = V.fp16_biases(tf)
            F16_tr, F16_te = V._transform(tf16, X_tr, cols), V._transform(tf16, X_te, cols)
            mu16 = F16_tr[base_idx].mean(axis=0).astype(np.float16).astype(np.float64)
            sd16 = (F16_tr[base_idx].std(axis=0) + 1e-3).astype(np.float16).astype(np.float64)
            head = V.ProtoHead("ncmc", mu16, sd=sd16)
            acc_k = run_split_acc(lambda Fs, ys, si: head.update(Fs, ys, si), lambda m: head.predict(F16_te[m]),
                                  y_tr, y_te, sess, proto, lambda ii: F16_tr[ii])
            rows.append(dict(dataset=name, seed=seed, config=f"prune-ncmzh-f{t_keep}", acc=json.dumps(acc_k)))

            # 同编码器回放：F′=840 选中特征（fp32 bias）+ 岭分类器，每类最多 m 条
            cr = sel[f_rep]
            Ftr, Fte = F_tr[:, cr], F_te[:, cr]
            rep = V.ReplayRidge(Ftr, m_rep, seed)
            acc_r = run_split_acc(lambda ii, ys, si: rep.update(ii, ys, si), lambda msk: rep.predict(Fte[msk]),
                                  y_tr, y_te, sess, proto, lambda ii: ii)
            rows.append(dict(dataset=name, seed=seed, config=f"mrrep-f{f_rep}-m{m_rep}", acc=json.dumps(acc_r)))
            print(f"{name} seed={seed}  KiloRocket {acc_k[-1]:.4f}  replay {acc_r[-1]:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    # 与主实验比对
    ref_path = a.ref or next(p for p in (os.path.join(D.RESULTS_DIR, "results_v6.csv"),
                                         os.path.join(D.RESULTS_DIR, "results_v5.csv")) if os.path.exists(p))
    ref = pd.read_csv(ref_path)
    ref = ref[ref.protocol == "main"].set_index(["dataset", "seed", "config"]).final_acc
    diff = [abs(json.loads(r.acc)[-1] - ref.loc[(r.dataset, r.seed, r.config)]) for r in out.itertuples()]
    print(f"与 {os.path.basename(ref_path)} 比对：{len(diff)} 行，最大差 {max(diff):.2e}，"
          f"完全一致 {sum(d < 1e-9 for d in diff)} 行")


if __name__ == "__main__":
    main()
