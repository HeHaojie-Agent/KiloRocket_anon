"""Re-implementations of Decoupled-Cosine, TEEN, ALICE and TS-ACL on a ResNet1D backbone (GPU)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # repository root
sys.path.insert(0, ROOT)

from incremental import IncProtocol, session_data  # noqa: E402  与 exp_v5 共用协议代码

import torch                                        # noqa: E402
import torch.nn as nn                               # noqa: E402
import torch.nn.functional as Fn                    # noqa: E402

DATA_NPZ_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_CSV = os.path.join(RESULTS_DIR, "results_deep_v6.csv")

PROTOCOLS = {"main": dict(n_way=2, n_shot=5)}
WIDTHS = [4, 8, 16, 32, 64]
TSACL_E = [512, 2048]
TSACL_GAMMAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
TEEN_ALPHA, TEEN_TAU = 0.5, 16.0

TRAIN = dict(epochs=200, lr=1e-3, wd=5e-4, batch=64, patience=40, val_frac=0.1, scale=16.0)
ALICE = dict(margin=0.1, scale=16.0, mix_frac=0.5, jitter=0.03, scale_sd=0.1, shift=0.1)
# 按原文/官方代码的 ALICE（--alice-orig）：CosFace s=30、m=0.4（官方脚本 run_base_*.sh），
# 两层 MLP 投影头（隐层 2048，训练后丢弃），每个样本两次增强、损失取平均，
# base 原型只用与增量会话同样数量（n_shot）的样本生成（原文 4.3 节，缓解不平衡）。
ALICE_ORIG = dict(ALICE, margin=0.4, scale=30.0, proj_hidden=2048, n_views=2)
TSACL_E_ORIG = 8192   # TS-ACL 原文的扩展宽度（"growth size"）；用 torch 在 GPU 上算

COLS = ["dataset", "seed", "protocol", "family", "config", "knob", "n_classes",
        "n_base_classes", "n_sessions", "final_acc", "avg_acc", "forgetting", "base_last",
        "novel_last", "novel_avg", "mean_update_time", "bytes_proto", "final_memory_bytes",
        "n_features", "encoder_state_bytes", "error",
        # 以下为本脚本新增列（合并时 pandas 会自动补空）
        "acc_per_session", "expansion_bytes", "gamma", "train_epochs", "base_val_acc",
        "n_params", "device"]


# ==========================================================================
# 数据
# ==========================================================================

def znormalize(X, eps=1e-8):
    """与 datasets.znormalize 完全相同：逐样本、逐通道。"""
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return (X - mu) / (sd + eps)


def load(name):
    """优先读 pack_data.py 在本机导出的 npz（与本机 aeon 缓存逐字节一致），否则用 aeon 下载。"""
    p = os.path.join(DATA_NPZ_DIR, f"{name}.npz")
    if os.path.exists(p):
        z = np.load(p)
        X_tr, y_tr, X_te, y_te = z["X_tr"], z["y_tr"], z["X_te"], z["y_te"]
    else:
        import datasets as D
        X_tr, y_tr, X_te, y_te = D.load_split(name)
    return (znormalize(X_tr.astype(np.float64)).astype(np.float32), y_tr.astype(np.int64),
            znormalize(X_te.astype(np.float64)).astype(np.float32), y_te.astype(np.int64))


def synthetic(name, seed=0):
    """测代码用的假数据：K 类正弦波 + 噪声。"""
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    K, C, T = (8, 1, 64) if name.endswith("A") else (12, 3, 40)
    def make(n):
        y = np.repeat(np.arange(K), n)
        t = np.linspace(0, 1, T)
        X = np.stack([np.stack([np.sin(2 * np.pi * (1 + k % 5) * t + 0.3 * c * k)
                                for c in range(C)]) for k in y])
        X += 0.6 * rng.standard_normal(X.shape)
        return X.astype(np.float32), y
    X_tr, y_tr = make(30)
    X_te, y_te = make(20)
    return znormalize(X_tr), y_tr, znormalize(X_te), y_te


# ==========================================================================
# 骨干：ResNet1D（Wang et al. 2017），宽度可调
# ==========================================================================

class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.b1 = nn.Conv1d(cin, cout, 8, padding="same"), nn.BatchNorm1d(cout)
        self.c2, self.b2 = nn.Conv1d(cout, cout, 5, padding="same"), nn.BatchNorm1d(cout)
        self.c3, self.b3 = nn.Conv1d(cout, cout, 3, padding="same"), nn.BatchNorm1d(cout)
        self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))

    def forward(self, x):
        h = Fn.relu(self.b1(self.c1(x)))
        h = Fn.relu(self.b2(self.c2(h)))
        h = self.b3(self.c3(h))
        return Fn.relu(h + self.sc(x))


class ResNet1D(nn.Module):
    def __init__(self, c_in, w=64):
        super().__init__()
        self.blocks = nn.Sequential(_Block(c_in, w), _Block(w, 2 * w), _Block(2 * w, 2 * w))
        self.feat_dim = 2 * w

    def forward(self, x):
        return self.blocks(x).mean(dim=-1)


def backbone_bytes(net):
    """参数 + BN running mean/var，全部 fp32。num_batches_tracked 是整数计数器，推理不需要，不计。"""
    n = sum(p.numel() for p in net.parameters())
    n += sum(b.numel() for name, b in net.named_buffers() if "running" in name)
    return 4 * n, n


class CosineHead(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.W = nn.Parameter(torch.randn(k, d) * 0.01)

    def forward(self, f):
        return Fn.normalize(f, dim=-1) @ Fn.normalize(self.W, dim=-1).T


# ==========================================================================
# base 会话训练
# ==========================================================================

def _strat_split(y, frac, rng):
    """分层留出 frac 做验证；某类样本太少（<3）就不留。"""
    tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = int(round(frac * len(idx))) if len(idx) >= 3 else 0
        k = max(1, k) if len(idx) >= 3 else 0
        va.append(idx[:k]); tr.append(idx[k:])
    return np.concatenate(tr), np.concatenate(va)


def _augment(x, rng_t):
    """ALICE 用的数据增强：高斯抖动、逐样本缩放、循环时间平移。"""
    a = ALICE
    B, C, T = x.shape
    x = x * (1 + a["scale_sd"] * torch.randn(B, 1, 1, device=x.device, generator=rng_t))
    x = x + a["jitter"] * torch.randn(x.shape, device=x.device, generator=rng_t)
    s = int(a["shift"] * T)
    if s > 0:
        shifts = torch.randint(-s, s + 1, (B,), device=x.device, generator=rng_t)
        ar = torch.arange(T, device=x.device)
        idx = (ar[None, :] - shifts[:, None]) % T
        x = torch.gather(x, 2, idx[:, None, :].expand(B, C, T))
    return x


def train_base(X, y, w, seed, device, alice=False, orig=False):
    """返回 (冻结的骨干, 训练 epoch 数, base 验证准确率, 训练/验证下标)。"""
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    g = torch.Generator(device=device); g.manual_seed(seed)
    tr, va = _strat_split(y, TRAIN["val_frac"], np_rng)
    classes = np.unique(y)
    lut = {c: i for i, c in enumerate(classes)}
    yi = np.array([lut[v] for v in y])
    Kb = len(classes)

    pairs = [(i, j) for i in range(Kb) for j in range(i + 1, Kb)] if alice else []
    pair_id = {p: Kb + n for n, p in enumerate(pairs)}

    net = ResNet1D(X.shape[1], w).to(device)
    A_ = ALICE_ORIG if (alice and orig) else ALICE
    if alice and orig:
        hd = A_["proj_hidden"]
        proj = nn.Sequential(nn.Linear(net.feat_dim, hd), nn.ReLU(), nn.Linear(hd, hd)).to(device)
        head_in = hd
    else:
        proj, head_in = nn.Identity(), net.feat_dim
    head = CosineHead(head_in, Kb + len(pairs)).to(device)
    params = list(net.parameters()) + list(proj.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=TRAIN["lr"], weight_decay=TRAIN["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN["epochs"])
    Xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(yi, device=device)
    tr_t = torch.as_tensor(tr, device=device)
    s, m = (A_["scale"], A_["margin"]) if alice else (TRAIN["scale"], 0.0)
    n_views = A_.get("n_views", 1) if alice else 1

    best, best_state, bad, ep_run = -1.0, None, 0, 0
    for ep in range(TRAIN["epochs"]):
        ep_run = ep + 1
        net.train(); head.train(); proj.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device, generator=g)]
        for b0 in range(0, len(perm), TRAIN["batch"]):
            b = perm[b0:b0 + TRAIN["batch"]]
            if len(b) < 2:
                continue
            xb, yb = Xt[b], yt[b]
            if alice:
                if n_views == 1:
                    xb = _augment(xb, g)
                # 类别增强：一半样本与另一类样本 mixup，标签为该无序类对的虚拟类
                n_mix = int(ALICE["mix_frac"] * len(b))
                if n_mix > 0 and Kb > 1:
                    j = torch.randperm(len(b), device=device, generator=g)[:n_mix]
                    k = torch.randperm(len(b), device=device, generator=g)[:n_mix]
                    ok = yb[j] != yb[k]
                    j, k = j[ok], k[ok]
                    if len(j):
                        lam = 0.4 + 0.2 * torch.rand(len(j), 1, 1, device=device, generator=g)
                        xm = lam * xb[j] + (1 - lam) * xb[k]
                        a, c = torch.minimum(yb[j], yb[k]), torch.maximum(yb[j], yb[k])
                        ym = torch.as_tensor([pair_id[(int(p), int(q))] for p, q in zip(a, c)],
                                             device=device)
                        xb = torch.cat([xb, xm]); yb = torch.cat([yb, ym])
            loss = 0.0
            for _v in range(n_views):
                xv = _augment(xb, g) if n_views > 1 else xb
                cos = head(proj(net(xv)))
                logits = s * (cos - m * Fn.one_hot(yb, cos.shape[1]))
                loss = loss + Fn.cross_entropy(logits, yb) / n_views
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        # 早停：在 base 验证集上只看 base 类（不含虚拟类）
        if len(va):
            net.eval(); head.eval(); proj.eval()
            with torch.no_grad():
                pv = head(proj(net(Xt[va])))[:, :Kb].argmax(1).cpu().numpy()
            acc = float(np.mean(pv == yi[va]))
        else:
            acc = -loss.item()
        if acc > best:
            best, bad = acc, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= TRAIN["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, ep_run, best, tr, va


@torch.no_grad()
def features(net, X, device, bs=512):
    out = []
    for s in range(0, len(X), bs):
        out.append(net(torch.as_tensor(X[s:s + bs], device=device)).double().cpu().numpy())
    return np.concatenate(out)


# ==========================================================================
# 增量分类头（都在冻结骨干的特征上工作）
# ==========================================================================

def _unit(M):
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-12)


class ProtoHead:
    """Decoupled-Cosine；teen=True 时为 TEEN。"""

    def __init__(self, teen=False):
        self.teen, self.P, self.base = teen, {}, None

    def update(self, F, y, si):
        for c in np.unique(y):
            v = F[y == c].mean(axis=0)
            if self.teen and si > 0:
                B = _unit(np.stack([self.P[k] for k in self.base]))
                vn = _unit(v)
                z = TEEN_TAU * (B @ vn)
                wgt = np.exp(z - z.max()); wgt /= wgt.sum()
                v = TEEN_ALPHA * vn + (1 - TEEN_ALPHA) * (wgt @ B)
            self.P[int(c)] = v
        if si == 0:
            self.base = sorted(self.P)

    def predict(self, F):
        ks = np.array(sorted(self.P))
        M = _unit(np.stack([self.P[k] for k in ks]))
        return ks[np.argmax(_unit(F) @ M.T, axis=1)]

    def class_bytes(self, d):
        return 4 * len(self.P) * d


class TSACLHead:
    """TS-ACL / ACIL 的解析头：随机扩展层 + 递归最小二乘。全程 float64。"""

    def __init__(self, d, E, gamma, seed):
        rng = np.random.default_rng(10_000 + seed)
        self.WE = rng.standard_normal((d, E)) / np.sqrt(d)
        self.E, self.gamma = E, gamma
        self.R = self.W = None
        self.classes = []

    def _h(self, F):
        return np.maximum(F @ self.WE, 0.0)

    def _Y(self, y):
        idx = {c: i for i, c in enumerate(self.classes)}
        Y = np.zeros((len(y), len(self.classes)))
        Y[np.arange(len(y)), [idx[int(v)] for v in y]] = 1.0
        return Y

    def update(self, F, y, si):
        new = [int(c) for c in np.unique(y) if int(c) not in self.classes]
        self.classes += new
        H = self._h(F)
        if self.R is None:
            self.R = np.linalg.inv(H.T @ H + self.gamma * np.eye(self.E))
            self.W = self.R @ H.T @ self._Y(y)
            return
        self.W = np.concatenate([self.W, np.zeros((self.E, len(new)))], axis=1)
        RHt = self.R @ H.T
        K = np.linalg.inv(np.eye(len(H)) + H @ RHt)
        self.R = self.R - RHt @ K @ RHt.T
        self.W = self.W + self.R @ H.T @ (self._Y(y) - H @ self.W)

    def predict(self, F):
        return np.array(self.classes)[np.argmax(self._h(F) @ self.W, axis=1)]


class TSACLHeadTorch(TSACLHead):
    """与 TSACLHead 相同的递推，float64，在 torch 设备上算（E=8192 时 numpy 太慢）。
    扩展层用与 TSACLHead 相同的 numpy 随机数生成，再搬到设备上。"""

    def __init__(self, d, E, gamma, seed, device):
        super().__init__(d, E, gamma, seed)
        self.dev = device
        self.WE = torch.as_tensor(self.WE, dtype=torch.float64, device=device)

    def _h(self, F):
        F = torch.as_tensor(F, dtype=torch.float64, device=self.dev)
        return torch.clamp(F @ self.WE, min=0.0)

    def _Y(self, y):
        return torch.as_tensor(super()._Y(y), dtype=torch.float64, device=self.dev)

    def update(self, F, y, si):
        new = [int(c) for c in np.unique(y) if int(c) not in self.classes]
        self.classes += new
        H = self._h(F)
        I = torch.eye(self.E, dtype=torch.float64, device=self.dev)
        if self.R is None:
            self.R = torch.linalg.inv(H.T @ H + self.gamma * I)
            self.W = self.R @ H.T @ self._Y(y)
            return
        self.W = torch.cat([self.W, torch.zeros(self.E, len(new), dtype=torch.float64, device=self.dev)], 1)
        RHt = self.R @ H.T
        K = torch.linalg.inv(torch.eye(len(H), dtype=torch.float64, device=self.dev) + H @ RHt)
        self.R = self.R - RHt @ K @ RHt.T
        self.W = self.W + self.R @ H.T @ (self._Y(y) - H @ self.W)

    def predict(self, F):
        idx = torch.argmax(self._h(F) @ self.W, dim=1).cpu().numpy()
        return np.array(self.classes)[idx]


def make_tsacl(d, E, gamma, seed, device=None):
    if E >= 4096 and device is not None:
        return TSACLHeadTorch(d, E, gamma, seed, device)
    return TSACLHead(d, E, gamma, seed)


def select_gamma(Ftr, ytr, Fva, yva, E, seed, device=None):
    """在 base 训练/验证划分上选 γ（只用 base 数据，不碰测试集）。"""
    if len(yva) == 0:
        return 1.0
    best, bg = -1, 1.0
    for g in TSACL_GAMMAS:
        h = make_tsacl(Ftr.shape[1], E, g, seed, device)
        h.update(Ftr, ytr, 0)
        acc = float(np.mean(h.predict(Fva) == yva))
        if acc > best:
            best, bg = acc, g
    return bg


# ==========================================================================
# 评测（与 exp_check.run_split 逐行一致）
# ==========================================================================

def run_split(update, predict, y_tr, y_te, sess, proto):
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
        update(ii, yy, si)
        times.append(time.perf_counter() - t0)
        seen.extend(cls.tolist())
        m = np.isin(y_te, seen)
        yp = predict(m)
        yt = y_te[m]
        accs.append(float(np.mean(yp == yt)))
        isb = np.isin(yt, base_cls)
        base_accs.append(float(np.mean(yp[isb] == yt[isb])))
        if si > 0 and (~isb).any():
            novel_accs.append(float(np.mean(yp[~isb] == yt[~isb])))
    return dict(final_acc=accs[-1], avg_acc=float(np.mean(accs)),
                forgetting=float(base_accs[0] - base_accs[-1]),
                base_last=base_accs[-1],
                novel_last=novel_accs[-1] if novel_accs else np.nan,
                novel_avg=float(np.mean(novel_accs)) if novel_accs else np.nan,
                mean_update_time=float(np.mean(times[1:])) if len(times) > 1 else 0.0,
                acc_per_session=json.dumps([round(a, 5) for a in accs]))


# ==========================================================================
# 主循环
# ==========================================================================

def _done(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df["dataset"], df["seed"], df["protocol"], df["config"]))


def _append(path, row):
    hdr = not os.path.exists(path)
    pd.DataFrame([row], columns=COLS).to_csv(path, mode="a", index=False, header=hdr)


def run(names, seeds, widths, out_csv, device, use_synth, families, alice_orig=False, tsacl_E=None):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = _done(out_csv)
    for name in names:
        try:
            X_tr, y_tr, X_te, y_te = synthetic(name) if use_synth else load(name)
        except Exception as e:  # noqa: BLE001
            print(f"!! {name} 加载失败：{type(e).__name__}: {e}", flush=True)
            continue
        K, C, T = len(np.unique(y_tr)), X_tr.shape[1], X_tr.shape[2]
        print(f"\n### {name}  K={K}  C={C}  T={T}  train={len(y_tr)}  test={len(y_te)}", flush=True)
        for seed in seeds:
            for prot, kw in PROTOCOLS.items():
                proto = IncProtocol(seed=seed, **kw)
                sess = proto.sessions(np.unique(y_tr))
                nb = len(sess[0])
                idx_tr = np.arange(len(y_tr))[:, None]
                base_idx = session_data(idx_tr, y_tr, sess[0], None, seed)[0][:, 0]

                for w in widths:
                    cfgs = {
                        "deep-proto": [f"resnet-proto-w{w}"],
                        "deep-teen": [f"resnet-teen-w{w}"],
                        "deep-tsacl": [f"resnet-tsacl-E{E}-w{w}" for E in (tsacl_E or TSACL_E)],
                        "deep-alice": [f"resnet-aliceo-w{w}" if alice_orig else f"resnet-alice-w{w}"],
                    }
                    cfgs = {f: c for f, c in cfgs.items() if f in families}
                    todo_ce = [c for f in ("deep-proto", "deep-teen", "deep-tsacl")
                               for c in cfgs.get(f, []) if (name, seed, prot, c) not in done]
                    todo_al = [c for c in cfgs.get("deep-alice", []) if (name, seed, prot, c) not in done]

                    def emit(fam, cfg, res, nF, bproto, mem, enc, extra, err=""):
                        row = dict.fromkeys(COLS)
                        row.update(dataset=name, seed=seed, protocol=prot, family=fam, config=cfg,
                                   knob=w, n_classes=K, n_base_classes=nb, n_sessions=len(sess),
                                   error=err, device=str(device))
                        if res is not None:
                            row.update(res)
                            row.update(bytes_proto=int(bproto), final_memory_bytes=int(mem),
                                       n_features=int(nF), encoder_state_bytes=int(enc))
                        row.update(extra)
                        _append(out_csv, row)
                        done.add((name, seed, prot, cfg))
                        if err:
                            print(f"    {cfg:26s} FAILED {err}", flush=True)
                        else:
                            print(f"    {cfg:26s} acc={row['final_acc']:.4f} base={row['base_last']:.3f} "
                                  f"novel={row['novel_last']:.3f} mem={(mem + enc) / 1024:9.1f}KB", flush=True)

                    for alice, todo in ((False, todo_ce), (True, todo_al)):
                        if not todo:
                            continue
                        t0 = time.time()
                        try:
                            net, ep, vacc, tr_i, va_i = train_base(
                                X_tr[base_idx], y_tr[base_idx], w, seed, device, alice=alice,
                                orig=alice_orig)
                        except Exception as e:  # noqa: BLE001
                            for cfg in todo:
                                fam = "deep-alice" if alice else next(f for f, c in cfgs.items() if cfg in c)
                                emit(fam, cfg, None, 0, 0, 0, 0, {}, err=f"{type(e).__name__}: {e}"[:200])
                            continue
                        enc, n_par = backbone_bytes(net)
                        d = net.feat_dim
                        print(f"  seed={seed} w={w} {'ALICE' if alice else 'CE'} 训练 {ep} epoch，"
                              f"base 验证 {vacc:.3f}，骨干 {enc / 1024:.1f}KB（{time.time() - t0:.0f}s）",
                              flush=True)
                        F_tr = features(net, X_tr, device)
                        F_te = features(net, X_te, device)
                        base_extra = dict(train_epochs=ep, base_val_acc=vacc, n_params=n_par)

                        def go(fam, cfg, head, extra_fn, upd=None):
                            try:
                                res = run_split(upd or (lambda ii, ys, si: head.update(F_tr[ii], ys, si)),
                                                lambda msk: head.predict(F_te[msk]),
                                                y_tr, y_te, sess, proto)
                                bp, mem, extra = extra_fn(head)
                                emit(fam, cfg, res, d, bp, mem, enc, {**base_extra, **extra})
                            except Exception as e:  # noqa: BLE001
                                emit(fam, cfg, None, 0, 0, 0, 0, base_extra,
                                     err=f"{type(e).__name__}: {e}"[:200])

                        for cfg in todo:
                            if "-aliceo-" in cfg:
                                # 原文：base 原型只用 n_shot 个样本（与增量会话等量），其余同 Decoupled-Cosine
                                hd_ = ProtoHead(False)
                                rng_b = np.random.default_rng(20_000 + seed)

                                def upd_bal(ii, ys, si, hd_=hd_, rng_b=rng_b):
                                    if si == 0:
                                        keep = np.concatenate([rng_b.permutation(np.flatnonzero(ys == c))[:proto.n_shot]
                                                               for c in np.unique(ys)])
                                        ii, ys = ii[keep], ys[keep]
                                    hd_.update(F_tr[ii], ys, si)
                                go("deep-alice-orig", cfg, hd_,
                                   lambda h: (h.class_bytes(d), h.class_bytes(d), {}), upd=upd_bal)
                            elif "-proto-" in cfg or "-alice-" in cfg:
                                fam = "deep-alice" if alice else "deep-proto"
                                go(fam, cfg, ProtoHead(False),
                                   lambda h: (h.class_bytes(d), h.class_bytes(d), {}))
                            elif "-teen-" in cfg:
                                go("deep-teen", cfg, ProtoHead(True),
                                   lambda h: (h.class_bytes(d), h.class_bytes(d), {}))
                            elif "-tsacl-" in cfg:
                                E = int(cfg.split("-E")[1].split("-")[0])
                                # γ 只用 base 会话的训练/验证划分来选（下标相对 base_idx）
                                Fb, yb = F_tr[base_idx], y_tr[base_idx]
                                gam = select_gamma(Fb[tr_i], yb[tr_i], Fb[va_i], yb[va_i], E, seed, device)

                                def ts_mem(h, E=E, gam=gam):
                                    Kc = len(h.classes)
                                    exp_b = 4 * d * E
                                    return (4 * E * Kc, 4 * E * E + 4 * E * Kc + exp_b,
                                            dict(expansion_bytes=exp_b, gamma=gam))
                                go("deep-tsacl", cfg, make_tsacl(d, E, gam, seed, device), ts_mem)
                        del net, F_tr, F_te
                        if device.type == "cuda":
                            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--list", default="v6", choices=["v5", "v6"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--widths", nargs="*", type=int, default=WIDTHS)
    ap.add_argument("--families", nargs="*", default=["deep-proto", "deep-teen", "deep-tsacl", "deep-alice"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="覆盖最大 epoch（只用于测试）")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alice-orig", action="store_true",
                    help="ALICE 按原文超参（s=30, m=0.4, 投影头, 两次增强, 等量 base 原型）；配置名 resnet-aliceo-w*")
    ap.add_argument("--tsacl-E", nargs="*", type=int, default=None,
                    help="TS-ACL 扩展宽度（默认 512 2048；原文为 8192）")
    a = ap.parse_args()

    if a.epochs:
        TRAIN["epochs"] = a.epochs
        TRAIN["patience"] = min(TRAIN["patience"], a.epochs)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if a.synthetic:
        names = a.datasets or ["SynthA", "SynthB"]
    else:
        js = "datasets_selected_incremental_v6.json" if a.list == "v6" else "datasets_selected_incremental.json"
        with open(os.path.join(RESULTS_DIR, js), encoding="utf-8") as fh:
            p = json.load(fh)
        names = a.datasets or (p["univariate"] + p["multivariate"])
    seeds = a.seeds
    if a.quick:
        names, seeds = names[:2], seeds[:1]
    names = names[a.shard::a.n_shards]
    out = a.out or (os.path.join(RESULTS_DIR, "results_deep_synth.csv") if a.synthetic else OUT_CSV)
    if a.n_shards > 1 and not a.out:
        out = out.replace(".csv", f"_shard{a.shard}.csv")
    print(f"device={device}  数据集 {len(names)} 个：{names}\n种子 {seeds}  宽度 {a.widths}\n输出 {out}", flush=True)
    t0 = time.time()
    run(names, seeds, a.widths, out, device, a.synthetic, set(a.families), a.alice_orig, a.tsacl_E)
    print(f"\n全部完成，用时 {(time.time() - t0) / 60:.1f} 分钟。", flush=True)


if __name__ == "__main__":
    main()
