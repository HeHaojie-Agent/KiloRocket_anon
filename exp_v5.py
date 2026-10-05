"""Main experiment grid: all families x memory knobs x datasets x seeds x protocols. Writes results_v6.csv."""

from __future__ import annotations

import argparse
import copy
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import datasets as D
import exp_analytic_prune as AP
from exp_check import run_split
from incremental import IncProtocol, session_data, pad_for_rocket

OUT_CSV = os.path.join(D.RESULTS_DIR, "results_v5.csv")

F_FULL_KERNELS = 10000
KERNELS_FEW = [84, 336, 840, 2520]
PRUNE_T = [2520, 840, 336, 84]
REPLAY_M = [1, 2, 4, 8, 16]
REPLAY_F = [84, 840, "full"]
HDC_D = [512, 1024, 2048, 4096, 8192]
NN_M = [1, 2, 4, 8, 16]
CNN_W = [8, 16, 32, 64]
TEEN_ALPHA, TEEN_TAU = 0.5, 16.0
CHUNK = 2000

PROTOCOLS = {
    "main": dict(n_way=2, n_shot=5),
    "way1": dict(n_way=1, n_shot=5),
    "shot1": dict(n_way=2, n_shot=1),
}
# 敏感性协议只跑这些配置
ALT_CONFIGS = {
    "prune-bin-f84", "prune-bin-f336", "prune-bin-f840",
    "prune-binh-f84", "prune-binh-f336", "prune-binh-f840",
    "prune-binz-f840", "prune-binzh-f84", "prune-binzh-f336", "prune-binzh-f840",
    "prune-ncmc-f840", "prune-ncmz-f840", "prune-bin-teen-f840", "mr-ncmc-k10000",
    "mrrep-f840-m1", "mrrep-f840-m4", "mrrep-f840-m16",
    "1nn-m1", "1nn-m4", "1nn-m16", "hdc-real-D512", "hdc-real-D8192",
}

COLS = ["dataset", "seed", "protocol", "family", "config", "knob", "n_classes",
        "n_base_classes", "n_sessions", "final_acc", "avg_acc", "forgetting", "base_last",
        "novel_last", "novel_avg", "mean_update_time", "bytes_proto", "final_memory_bytes",
        "n_features", "encoder_state_bytes", "error"]


# ==========================================================================
# 特征
# ==========================================================================

def _transform(tf, X, cols=None):
    """分块变换，避免 Crop / ElectricDevices 上一次性占用过多内存。"""
    out = []
    for s in range(0, len(X), CHUNK):
        F = np.asarray(tf.transform(pad_for_rocket(X[s:s + CHUNK])), dtype=np.float32)
        out.append(F if cols is None else F[:, cols])
    return np.concatenate(out, axis=0)


def fit_mr(X_tr, X_te, base_idx, n_kernels, seed, n_jobs):
    from aeon.transformations.collection.convolution_based import MiniRocket
    tf = MiniRocket(n_kernels=n_kernels, random_state=seed, n_jobs=n_jobs)
    tf.fit(pad_for_rocket(X_tr[base_idx]))
    return tf, _transform(tf, X_tr), _transform(tf, X_te)


def fp16_biases(tf):
    """复制一份 MiniRocket，把 bias 舍入到 fp16（再存回 fp32 数组供 numba 使用）。

    aeon 的 MiniRocket.parameters 是一个 tuple，其中唯一的浮点数组就是 bias。
    """
    tf16 = copy.deepcopy(tf)
    params = list(tf16.parameters)
    hit = [i for i, p in enumerate(params)
           if isinstance(p, np.ndarray) and np.issubdtype(p.dtype, np.floating)]
    if len(hit) != 1:
        raise RuntimeError(f"找不到唯一的 bias 数组（浮点数组 {len(hit)} 个），请检查 aeon 版本")
    i = hit[0]
    params[i] = params[i].astype(np.float16).astype(params[i].dtype)
    tf16.parameters = tuple(params)
    return tf16


# ==========================================================================
# 分类头
# ==========================================================================

def _unit(M):
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-9)


class ProtoHead:
    """原型分类头。

    variant  ncm  : fp32，不中心化（与 v4 完全一致，用于复现核对）
             ncmc : fp32，减 base 均值 μ 后再算余弦
             bin  : 1-bit，P_k = sign(Σ(f − μ))，查询用实值 f − μ
    teen     True 时对增量会话的新类原型做 TEEN 校准（只用已存的 base 原型）
    sd       给定时先按 base 会话的均值/标准差 z-score（ncmz / binz，σ 另计 4F 字节）。
             用来检验「1-bit 原型胜过 fp32 原型」是不是因为 sign 相当于按特征归一化。
    """

    def __init__(self, variant, mu, teen=False, sd=None):
        self.variant, self.mu, self.teen, self.sd = variant, mu, teen, sd
        self.P, self.base_keys = {}, None

    def _c(self, F):
        """中心化；给了 sd 时再按 base 标准差缩放（z-score）。"""
        Z = F.astype(np.float64) - self.mu
        return Z if self.sd is None else Z / self.sd

    def _calibrate(self, v):
        B = _unit(np.stack([self.P[k].astype(np.float64) for k in self.base_keys]))
        vn = _unit(v)
        s = TEEN_TAU * (B @ vn)
        w = np.exp(s - s.max())
        w /= w.sum()
        return TEEN_ALPHA * vn + (1 - TEEN_ALPHA) * (w @ B)

    def update(self, Fs, ys, si):
        for c in np.unique(ys):
            Fc = Fs[ys == c]
            if self.variant == "ncm":
                v = Fc.mean(axis=0)
            else:
                v = self._c(Fc).sum(axis=0)
            if self.teen and si > 0:
                v = self._calibrate(v)
            if self.variant == "bin":
                v = np.sign(v)
                v[v == 0] = 1.0
                v = v.astype(np.int8)
            elif self.variant == "ncmc":
                v = v / len(Fc)
            self.P[c] = v
        if si == 0:
            self.base_keys = sorted(self.P)

    def predict(self, F):
        ks = np.array(sorted(self.P))
        M = _unit(np.stack([self.P[k].astype(np.float64) for k in ks]))
        Q = F if self.variant == "ncm" else self._c(F)
        return ks[np.argmax(_unit(Q) @ M.T, axis=1)]

    def proto_bytes(self, nF):
        K = len(self.P)
        return int(np.ceil(K * nF / 8)) if self.variant == "bin" else 4 * K * nF


class ReplayRidge:
    """同编码器样本回放：每类存 m 条原始序列，每会话重拟合 scaler + RidgeClassifierCV。

    base 会话在全部 base 数据上拟合，然后每类只留 m 条；增量会话在
    「已存样本 ∪ 本会话全部数据」上重拟合，再把本会话每类留 m 条。
    """

    def __init__(self, F_tr, m, seed):
        self.F_tr, self.m, self.rng = F_tr, m, np.random.default_rng(seed)
        self.mem_idx = np.array([], dtype=int)
        self.mem_y = np.array([], dtype=int)
        self.clf = self.sc = None
        self.K = 0

    def update(self, ii, ys, si):
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.preprocessing import StandardScaler
        idx = np.concatenate([self.mem_idx, ii])
        yy = np.concatenate([self.mem_y, ys])
        F = self.F_tr[idx]
        self.sc = StandardScaler().fit(F)
        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(self.sc.transform(F), yy)
        keep_i, keep_y = [], []
        for c in np.unique(ys):
            pool = ii[ys == c]
            k = min(self.m, len(pool))
            keep_i.append(self.rng.choice(pool, size=k, replace=False))
            keep_y.append(np.full(k, c))
        self.mem_idx = np.concatenate([self.mem_idx] + keep_i)
        self.mem_y = np.concatenate([self.mem_y] + keep_y)
        self.K = len(np.unique(self.mem_y))

    def predict(self, F):
        return self.clf.predict(self.sc.transform(F))


class RetrainRidge:
    """MiniRocket-retrain：存全部数据，每会话 RidgeClassifier(alpha=1)（与 v4 相同）。"""

    def __init__(self):
        self.F, self.y, self.clf = None, None, None

    def update(self, Fs, ys, si):
        from sklearn.linear_model import RidgeClassifier
        self.F = Fs if self.F is None else np.concatenate([self.F, Fs])
        self.y = ys if self.y is None else np.concatenate([self.y, ys])
        self.clf = RidgeClassifier(alpha=1.0).fit(self.F, self.y)

    def predict(self, F):
        return self.clf.predict(F)


# ==========================================================================
# 行
# ==========================================================================

def _row(name, seed, prot, fam, cfg, knob, K, nb, ns, res=None, nF=0, bproto=0,
         mem=0, enc=0, err=""):
    row = dict.fromkeys(COLS)
    row.update(dataset=name, seed=seed, protocol=prot, family=fam, config=cfg, knob=knob,
               n_classes=K, n_base_classes=nb, n_sessions=ns, error=err)
    if res is not None:
        row.update(res)
        row.update(bytes_proto=int(bproto), final_memory_bytes=int(mem), n_features=int(nF),
                   encoder_state_bytes=int(enc))
    return row


def _done(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df["dataset"], df["seed"], df["protocol"], df["config"]))


def _append(path, row):
    hdr = not os.path.exists(path)
    pd.DataFrame([row], columns=COLS).to_csv(path, mode="a", index=False, header=hdr)


# ==========================================================================
# 主循环
# ==========================================================================

def run(names, seeds, out_csv, n_jobs, skip, protocols):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = _done(out_csv)
    for name in names:
        try:
            X_tr, y_tr, X_te, y_te = D.load_split(name)
            X_tr, X_te = D.znormalize(X_tr), D.znormalize(X_te)
        except Exception as e:  # noqa: BLE001
            print(f"!! {name} 加载失败：{type(e).__name__}: {e}")
            continue
        K = len(np.unique(y_tr))
        C, T = X_tr.shape[1], X_tr.shape[2]
        series_bytes = 4 * C * T
        print(f"\n### {name}  K={K}  C={C}  T={T}", flush=True)

        for seed in seeds:
            t_seed = time.time()
            sess_of = {p: IncProtocol(seed=seed, **kw).sessions(np.unique(y_tr))
                       for p, kw in PROTOCOLS.items() if p in protocols}
            proto_of = {p: IncProtocol(seed=seed, **kw) for p, kw in PROTOCOLS.items() if p in protocols}
            nb = len(sess_of["main"][0]) if "main" in sess_of else len(next(iter(sess_of.values()))[0])
            idx_tr = np.arange(len(y_tr))[:, None]
            base_cls = next(iter(sess_of.values()))[0]
            base_idx = session_data(idx_tr, y_tr, base_cls, None, seed)[0][:, 0]

            def wanted(prot, cfg):
                if (name, seed, prot, cfg) in done:
                    return False
                return prot == "main" or cfg in ALT_CONFIGS

            def emit(row):
                _append(out_csv, row)
                done.add((name, seed, row["protocol"], row["config"]))
                if row["error"]:
                    print(f"    [{row['protocol']:5s}] {row['config']:24s} FAILED {row['error']}", flush=True)
                else:
                    tot = (row["final_memory_bytes"] + row["encoder_state_bytes"]) / 1024
                    print(f"    [{row['protocol']:5s}] {row['config']:24s} acc={row['final_acc']:.4f} "
                          f"base={row['base_last']:.3f} novel={row['novel_last']:.3f} "
                          f"mem={tot:9.1f}KB", flush=True)

            def run_head(prot, fam, cfg, knob, make, F_tr_, F_te_, nF, mem_fn, enc):
                """make() -> head；mem_fn(head) -> (bytes_proto, final_memory_bytes)。"""
                if not wanted(prot, cfg):
                    return
                sess, proto = sess_of[prot], proto_of[prot]
                try:
                    head = make()
                    res = run_split(lambda Fs, ys, si: head.update(Fs, ys, si),
                                    lambda m: head.predict(F_te_[m]),
                                    y_tr, y_te, sess, proto, lambda ii: F_tr_[ii])
                    bp, mem = mem_fn(head)
                    emit(_row(name, seed, prot, fam, cfg, knob, K, nb, len(sess), res, nF, bp, mem, enc))
                except Exception as e:  # noqa: BLE001
                    emit(_row(name, seed, prot, fam, cfg, knob, K, nb, len(sess),
                              err=f"{type(e).__name__}: {e}"[:200]))

            def run_replay(prot, fam, cfg, m, F_tr_, F_te_, nF, fixed, enc):
                """回放头的 update 需要下标，所以单独写。fixed = bias 以外的固定状态字节。"""
                if not wanted(prot, cfg):
                    return
                sess, proto = sess_of[prot], proto_of[prot]
                try:
                    head = ReplayRidge(F_tr_, m, seed)
                    res = run_split(lambda ii, ys, si: head.update(ii, ys, si),
                                    lambda msk: head.predict(F_te_[msk]),
                                    y_tr, y_te, sess, proto, lambda ii: ii)
                    stored = len(head.mem_idx) * series_bytes
                    weights = 4 * nF * head.K + 4 * head.K + 8 * nF      # 岭权重 + 截距 + scaler 均值/方差
                    bp = stored + 4 * nF * head.K
                    emit(_row(name, seed, prot, fam, cfg, m, K, nb, len(sess), res, nF, bp,
                              stored + weights + fixed, enc))
                except Exception as e:  # noqa: BLE001
                    emit(_row(name, seed, prot, fam, cfg, m, K, nb, len(sess),
                              err=f"{type(e).__name__}: {e}"[:200]))

            # ---------------- 全量 MiniRocket（约 10k 特征）+ SFD
            need_full = bool({"a", "b", "c", "d", "e"} - skip)
            if need_full:
                t0 = time.time()
                tf, F_tr, F_te = fit_mr(X_tr, X_te, base_idx, F_FULL_KERNELS, seed, n_jobs)
                nF_full = F_tr.shape[1]
                mu = F_tr[base_idx].mean(axis=0).astype(np.float64)
                sd_full = F_tr[base_idx].std(axis=0).astype(np.float64) + 1e-8
                sel = AP.sfd_select(F_tr[base_idx], y_tr[base_idx], PRUNE_T)
                print(f"  seed={seed}  全量特征 F={nF_full} + SFD 完成（{time.time()-t0:.1f}s）", flush=True)

                # fp16 bias 版本：只保留 SFD 选中的列
                F16 = None
                if "b" not in skip:
                    t0 = time.time()
                    try:
                        union = np.unique(np.concatenate([sel[t] for t in PRUNE_T]))
                        tf16 = fp16_biases(tf)
                        F16_tr = _transform(tf16, X_tr, union)
                        F16_te = _transform(tf16, X_te, union)
                        pos = {c: i for i, c in enumerate(union)}
                        F16 = (F16_tr, F16_te, pos)
                        print(f"  seed={seed}  fp16-bias 特征完成（{time.time()-t0:.1f}s）", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"  !! fp16-bias 失败，跳过 B 部分：{type(e).__name__}: {e}", flush=True)

                for prot in protocols:
                    # A / 复现：全量特征上的原型
                    for v in (["ncm", "bin", "ncmc", "ncmz"] if "a" not in skip else []):
                        fam = f"mr-{v}"
                        vv = "ncmc" if v == "ncmz" else v
                        extra = {"ncm": 0, "bin": 4, "ncmc": 4, "ncmz": 8}[v] * nF_full
                        run_head(prot, fam, f"{fam}-k10000", 10000,
                                 lambda vv=vv, v=v: ProtoHead(vv, mu, sd=sd_full if v == "ncmz" else None),
                                 F_tr, F_te, nF_full,
                                 lambda h, extra=extra: (h.proto_bytes(nF_full), h.proto_bytes(nF_full) + extra),
                                 4 * nF_full)
                    # A / D：选中特征上的原型（含 TEEN）
                    for t in PRUNE_T:
                        cols = sel[t]
                        Ftr, Fte = F_tr[:, cols], F_te[:, cols]
                        mu_p = Ftr[base_idx].mean(axis=0).astype(np.float64)
                        sd_p = Ftr[base_idx].std(axis=0).astype(np.float64) + 1e-8
                        heads = []      # (variant, teen, z-score, family)
                        if "a" not in skip:
                            heads += [("ncm", False, False, "prune-ncm"), ("bin", False, False, "prune-bin"),
                                      ("ncmc", False, False, "prune-ncmc"), ("ncmc", False, True, "prune-ncmz"),
                                      ("bin", False, True, "prune-binz")]
                        if "d" not in skip:
                            heads += [("ncmc", True, False, "prune-ncmc-teen"),
                                      ("bin", True, False, "prune-bin-teen")]
                        for v, teen, z, fam in heads:
                            extra = (0 if v == "ncm" else 4 * t) + (4 * t if z else 0) + 2 * t
                            run_head(prot, fam, f"{fam}-f{t}", t,
                                     lambda v=v, teen=teen, z=z: ProtoHead(v, mu_p, teen, sd_p if z else None),
                                     Ftr, Fte, t,
                                     lambda h, extra=extra: (h.proto_bytes(t), h.proto_bytes(t) + extra),
                                     4 * t)
                        # B：fp16 bias + fp16 μ
                        if F16 is not None:
                            F16_tr, F16_te, pos = F16
                            c16 = [pos[c] for c in cols]
                            Ftr16, Fte16 = F16_tr[:, c16], F16_te[:, c16]
                            mu16 = Ftr16[base_idx].mean(axis=0).astype(np.float16).astype(np.float64)
                            sd16 = (Ftr16[base_idx].std(axis=0) + 1e-3).astype(np.float16).astype(np.float64)
                            for v, z, fam in [("bin", False, "prune-binh"), ("ncmc", False, "prune-ncmch"),
                                              ("bin", True, "prune-binzh"), ("ncmc", True, "prune-ncmzh")]:
                                extra = 2 * t + (2 * t if z else 0) + 2 * t      # μ (+σ) fp16 + 下标
                                run_head(prot, fam, f"{fam}-f{t}", t,
                                         lambda v=v, z=z: ProtoHead(v, mu16, sd=sd16 if z else None),
                                         Ftr16, Fte16, t,
                                         lambda h, extra=extra: (h.proto_bytes(t), h.proto_bytes(t) + extra),
                                         2 * t)
                    # C：同编码器回放
                    if "c" not in skip:
                        for fsel in REPLAY_F:
                            if fsel == "full":
                                Ftr, Fte, nF, fixed = F_tr, F_te, nF_full, 0
                            else:
                                cols = sel[fsel]
                                Ftr, Fte, nF, fixed = F_tr[:, cols], F_te[:, cols], fsel, 2 * fsel
                            for m in REPLAY_M:
                                fam = f"mrrep-f{fsel}"
                                run_replay(prot, fam, f"{fam}-m{m}", m, Ftr, Fte, nF, fixed, 4 * nF)
                    # E：MiniRocket-retrain（计入岭权重）
                    if "e" not in skip:
                        run_head(prot, "minirocket-retrain", "minirocket-retrain", 0,
                                 lambda: RetrainRidge(), F_tr, F_te, nF_full,
                                 lambda h: (len(h.y) * series_bytes,
                                            len(h.y) * series_bytes + 4 * nF_full * len(np.unique(h.y))
                                            + 4 * len(np.unique(h.y))),
                                 4 * nF_full)
                del F_tr, F_te, F16

            # ---------------- A：减核，中心化 fp32（以及复现 v4 的 ncm / bin）
            if "a" not in skip and "main" in protocols:
                for k in KERNELS_FEW:
                    want = [f"mr-{v}-k{k}" for v in ("ncm", "bin", "ncmc", "ncmz")]
                    if not any(wanted("main", c) for c in want):
                        continue
                    tfk, Ftr, Fte = fit_mr(X_tr, X_te, base_idx, k, seed, n_jobs)
                    nF = Ftr.shape[1]
                    mu_k = Ftr[base_idx].mean(axis=0).astype(np.float64)
                    sd_k = Ftr[base_idx].std(axis=0).astype(np.float64) + 1e-8
                    for v in ("ncm", "bin", "ncmc", "ncmz"):
                        fam = f"mr-{v}"
                        vv = "ncmc" if v == "ncmz" else v
                        extra = {"ncm": 0, "bin": 4, "ncmc": 4, "ncmz": 8}[v] * nF
                        run_head("main", fam, f"{fam}-k{k}", k,
                                 lambda vv=vv, v=v: ProtoHead(vv, mu_k, sd=sd_k if v == "ncmz" else None),
                                 Ftr, Fte, nF,
                                 lambda h, extra=extra: (h.proto_bytes(nF), h.proto_bytes(nF) + extra),
                                 4 * nF)

            # ---------------- E：原始方法类参照（HDC / 1NN / CNN）
            if "e" not in skip:
                from exp_check import method_runner
                refs = [("1nn", f"1nn-m{m}", m, lambda m=m: _nn(m, seed)) for m in NN_M]
                if "hdc" not in skip:
                    for Dm in HDC_D:
                        refs.append(("hdc-bin", f"hdc-bin-D{Dm}", Dm, lambda Dm=Dm: _hdc(Dm, True, seed)))
                        refs.append(("hdc-real", f"hdc-real-D{Dm}", Dm, lambda Dm=Dm: _hdc(Dm, False, seed)))
                if "cnn" not in skip:
                    for w in CNN_W:
                        refs.append(("cnn-frozen", f"cnn-frozen-w{w}", w, lambda w=w: _cnn_frozen(w, seed)))
                    refs.append(("cnn-finetune", "cnn-finetune-w32", 32, lambda: _cnn_ft(seed)))
                for prot in protocols:
                    for fam, cfg, knob, make in refs:
                        if not wanted(prot, cfg):
                            continue
                        sess, proto = sess_of[prot], proto_of[prot]
                        try:
                            mth = make()
                            res = method_runner(mth, X_tr, X_te, y_tr, y_te, sess, proto)
                            mem = mth.memory_bytes()
                            if fam.startswith("hdc"):
                                mem += 8 * C          # 幅值上下界（fp32，每通道 2 个），按「拟合出来的都计入」
                            bp = _proto_bytes_ref(fam, mth)
                            emit(_row(name, seed, prot, fam, cfg, knob, K, nb, len(sess), res, 0, bp, mem, 0))
                        except Exception as e:  # noqa: BLE001
                            emit(_row(name, seed, prot, fam, cfg, knob, K, nb, len(sess),
                                      err=f"{type(e).__name__}: {e}"[:200]))
            print(f"  seed={seed} 用时 {time.time()-t_seed:.0f}s", flush=True)


# ---------------- 参照方法的构造（延迟导入 torch / torchhd）

def _nn(m, seed):
    from incremental import NNExemplar
    return NNExemplar(m_per_class=m, seed=seed)


def _hdc(Dm, binary, seed):
    from incremental import HDCPrototype
    return HDCPrototype(dim=Dm, binary=binary, seed=seed)


def _cnn_frozen(w, seed):
    from incremental import CNNFrozenNCM, _TinyCNN

    class _W(CNNFrozenNCM):
        name = f"cnn-frozen-w{w}"

        def fit_base(self, X, y):
            self.net = _TinyCNN(X.shape[1], seed=self.seed, hidden=(w, 2 * w), device=self.device)
            self.net.grow_head(sorted(np.unique(y).tolist()))
            self.net.fit(X, y, self.base_epochs)
            self.update(X, y)
    return _W(seed=seed)


def _cnn_ft(seed):
    from incremental import CNNFinetune
    return CNNFinetune(seed=seed)


def _proto_bytes_ref(fam, m):
    if fam == "1nn":
        return m.memory_bytes()
    if fam.startswith("hdc"):
        per = m.dim / 8 if m.binary else m.dim * 4
        return int(len(m.proto) * per)
    if fam == "cnn-frozen":
        return int(len(m.proto) * m.net.feat_dim * 4)
    if fam == "cnn-finetune":
        return int((m.net.head.weight.numel() + m.net.head.bias.numel()) * 4)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--list", default="v5", choices=["v5", "v6"],
                    help="v5=原 14 个数据集；v6=v6全量清单（先运行 python datasets.py --profile incremental-v6）")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--protocols", nargs="*", default=list(PROTOCOLS))
    ap.add_argument("--skip", nargs="*", default=[],
                    help="a=中心化 b=fp16 c=回放 d=TEEN e=参照 f=敏感性协议 cnn hdc")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()
    if a.data_dir:
        D.set_data_dir(a.data_dir)
    names = a.datasets or (D.selected_incremental_v6() if a.list == "v6"
                           else D.selected_incremental())
    seeds = a.seeds
    protocols = [p for p in a.protocols if p in PROTOCOLS]
    if "f" in a.skip:
        protocols = [p for p in protocols if p == "main"]
    if "main" not in protocols:
        protocols = ["main"] + protocols
    if a.quick:
        names, seeds = names[:2], seeds[:1]
    t0 = time.time()
    run(names, seeds, a.out, a.n_jobs, set(a.skip), protocols)
    print(f"\n全部完成，用时 {(time.time()-t0)/60:.1f} 分钟。接着运行：python analyze_v5.py")


if __name__ == "__main__":
    main()
