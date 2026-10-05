"""Inference cost: convolutions needed by the selected features, per-series latency, and a binary serialization of KiloRocket's state checked against the byte account."""
import os, sys, json, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np, pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from incremental import IncProtocol, pad_for_rocket
import exp_analytic_prune as AP

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join("cloud_baselines", "data")
OUT = os.path.join("results", "v6_summaries", "efficiency_v6")
N_TIME = 60

rows = []
meta = json.load(open(os.path.join(DATA, "meta.json")))
for name in sorted(meta):
    z = np.load(os.path.join(DATA, name + ".npz"))
    Xtr, ytr, Xte = z["X_tr"].astype(np.float32), z["y_tr"], z["X_te"].astype(np.float32)
    sess = IncProtocol(seed=0).sessions(np.unique(ytr))
    K = len(np.unique(ytr))
    mb = np.isin(ytr, sess[0])
    tf = MiniRocket(n_kernels=10000, random_state=0, n_jobs=1)
    tf.fit(pad_for_rocket(Xtr[mb]))
    Fb = np.asarray(tf.transform(pad_for_rocket(Xtr[mb])), dtype=np.float32)
    keep = AP.sfd_select(Fb, ytr[mb], [336, 84])
    dil, nfpd = tf.parameters[2], tf.parameters[3]
    starts = np.concatenate([[0], np.cumsum(84 * nfpd)])
    def pairs(idx):
        d = np.searchsorted(starts, idx, side="right") - 1
        k = (idx - starts[d]) // nfpd[d]
        return len(set(zip(d.tolist(), k.tolist())))
    n_conv_total = int(84 * len(dil))
    mu, sd = Fb.mean(0), Fb.std(0) + 1e-3
    for Fp in (84, 336):
        c = keep[Fp]
        P = np.random.default_rng(0).standard_normal((K, Fp)).astype(np.float32)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        Xq = pad_for_rocket(Xte[:N_TIME])
        tf.transform(Xq[:2])                      # 预热
        ts = []
        for i in range(len(Xq)):
            t0 = time.perf_counter()
            f = np.asarray(tf.transform(Xq[i:i + 1]), dtype=np.float32)[0, c]
            q = (f - mu[c]) / sd[c]
            int(np.argmax(P @ (q / (np.linalg.norm(q) + 1e-9))))
            ts.append(time.perf_counter() - t0)
        # 序列化：fp16 bias、μ、σ，uint16 下标，fp32 原型（每类一个，用训练数据均值），写成二进制文件核对字节数
        Ftr = np.asarray(tf.transform(pad_for_rocket(Xtr)), dtype=np.float32)[:, c]
        Z = (Ftr - mu[c]) / sd[c]
        protos = np.stack([Z[ytr == k].mean(0) for k in np.unique(ytr)]).astype(np.float32)
        bias_all = np.asarray(tf.parameters[4], dtype=np.float32)
        blob = (bias_all[c].astype(np.float16).tobytes() + mu[c].astype(np.float16).tobytes()
                + sd[c].astype(np.float16).tobytes() + c.astype(np.uint16).tobytes() + protos.tobytes())
        fn = os.path.join(OUT + "_state", f"{name}_F{Fp}.bin")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        open(fn, "wb").write(blob)
        rows.append(dict(dataset=name, C=Xtr.shape[1], T=Xtr.shape[2], K=K, F=Fp,
                         conv_needed=pairs(c), conv_total=n_conv_total,
                         infer_ms_full=1000 * float(np.median(ts)),
                         state_file_bytes=os.path.getsize(fn), state_account_bytes=8 * Fp + 4 * K * Fp))
    print(name, rows[-2]["conv_needed"], rows[-1]["conv_needed"], n_conv_total, f"{rows[-2]['infer_ms_full']:.2f} ms", flush=True)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT + ".csv", index=False)
with open(OUT + ".txt", "w", encoding="utf-8") as fh:
    for Fp, g in df.groupby("F"):
        fr = g.conv_needed / g.conv_total
        fh.write(f"F'={Fp}: 用到的卷积对 中位 {g.conv_needed.median():.0f}（{g.conv_needed.min()}–{g.conv_needed.max()}），"
                 f"占全部的 {100*fr.median():.1f}%（{100*fr.min():.1f}–{100*fr.max():.1f}%）；"
                 f"完整变换推理 中位 {g.infer_ms_full.median():.2f} ms（{g.infer_ms_full.min():.2f}–{g.infer_ms_full.max():.2f}）；"
                 f"卷积对总数 中位 {g.conv_total.median():.0f}（{g.conv_total.min()}–{g.conv_total.max()}）；"
                 f"状态文件字节 = 账本（fp32 原型）在 {(g.state_file_bytes == g.state_account_bytes).sum()}/{len(g)} 个数据集上成立\n")
print(open(OUT + ".txt", encoding="utf-8").read())
