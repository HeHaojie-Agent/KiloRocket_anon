"""Class-incremental protocol (base session + n_way/n_shot sessions), session sampling and the simple reference learners (1NN replay, HDC, CNN)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# ==========================================================================
# 协议
# ==========================================================================

@dataclass
class IncProtocol:
    n_base: int | None = None   # base 会话类数；None = 总类数的一半
    n_way: int = 2              # 每个增量会话新增类数
    n_shot: int = 5             # 增量会话每类样本数
    seed: int = 0

    def sessions(self, classes: np.ndarray) -> list[np.ndarray]:
        """把类别切成 [base, inc1, inc2, ...]。类顺序按 seed 打乱，保证可复现。"""
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(np.asarray(classes))
        nb = self.n_base if self.n_base is not None else max(2, len(order) // 2)
        nb = min(nb, len(order) - self.n_way) if len(order) > self.n_way else len(order)
        out = [order[:nb]]
        i = nb
        while i < len(order):
            out.append(order[i:i + self.n_way])
            i += self.n_way
        return [s for s in out if len(s)]


def session_data(X, y, cls, n_shot=None, seed=0):
    """取出属于 cls 的样本；n_shot 非 None 时每类只取 n_shot 个。"""
    mask = np.isin(y, cls)
    Xs, ys = X[mask], y[mask]
    if n_shot is None:
        return Xs, ys
    rng = np.random.default_rng(seed)
    idx = []
    for c in cls:
        pool = np.flatnonzero(ys == c)
        idx.append(rng.choice(pool, size=min(n_shot, len(pool)), replace=False))
    idx = np.sort(np.concatenate(idx))
    return Xs[idx], ys[idx]


def pad_for_rocket(X: np.ndarray, min_T: int = 9) -> np.ndarray:
    """MiniRocket 要求 n_timepoints >= 9，短序列补零。

    PenDigits 的 T=8，不补零会让两个 MiniRocket 方法整个数据集失败。
    """
    T = X.shape[2]
    if T >= min_T:
        return X
    pad = np.zeros((X.shape[0], X.shape[1], min_T - T), dtype=X.dtype)
    return np.concatenate([X, pad], axis=2)


# ==========================================================================
# 方法接口
# ==========================================================================

class IncMethod:
    """fit_base 一次，之后每个增量会话 update 一次，随时 predict。"""

    name = "base"

    def fit_base(self, X, y): raise NotImplementedError
    def update(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def memory_bytes(self) -> int: raise NotImplementedError


# ---------------------------------------------------------------- HDC 原型

class HDCPrototype(IncMethod):
    """冻结 HDC 编码器 + 类原型（叠加）。新类只加一个原型，旧原型不动。

    binary=True 时原型按 sign 二值化，每类只占 D/8 字节 —— 这是本文的核心卖点。
    """

    def __init__(self, dim=2048, n_levels=64, binary=True, seed=0):
        self.dim, self.n_levels, self.binary, self.seed = dim, n_levels, binary, seed
        self.name = f"hdc-proto{'-bin' if binary else ''}"
        self.proto: dict[int, np.ndarray] = {}
        self.enc = None

    def _encode(self, X):
        return self.enc.encode(X)

    def fit_base(self, X, y):
        from encode import HDCSequenceEncoder
        self.enc = HDCSequenceEncoder(dim=self.dim, n_levels=self.n_levels,
                                      seed=self.seed)
        self.enc.fit(X)                       # 编码器只在 base 会话拟合，之后永久冻结
        self.update(X, y)

    def update(self, X, y):
        H = self._encode(X)
        for c in np.unique(y):
            v = H[y == c].sum(axis=0)
            if c in self.proto:               # 同类再次出现就累加（本协议里不会发生）
                self.proto[c] = self.proto[c] + v
            else:
                self.proto[c] = v

    def _matrix(self):
        ks = sorted(self.proto)
        M = np.stack([self.proto[k] for k in ks])
        if self.binary:
            M = np.sign(M)
            M[M == 0] = 1.0
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return np.array(ks), M

    def predict(self, X):
        ks, M = self._matrix()
        H = self._encode(X)
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        return ks[np.argmax(H @ M.T, axis=1)]

    def memory_bytes(self):
        per = self.dim / 8 if self.binary else self.dim * 4
        # 编码器码本也要算：通道 HV + level HV + 时间 HV 基向量
        book = (self.enc.n_channels + self.n_levels + 1) * self.dim / 8
        return int(len(self.proto) * per + book)


# ------------------------------------------------- MiniRocket 最近类均值（关键对照）

class MiniRocketNCM(IncMethod):
    """MiniRocket 特征（base 会话拟合后冻结）+ 最近类均值。

    ⚠️ 这个方法同样零遗忘、同样 O(1) 更新。它存在的意义就是逼问：
       HDC 除了省内存，还有别的优势吗？
    """

    name = "minirocket-ncm"

    def __init__(self, n_kernels=10000, seed=0):
        self.n_kernels, self.seed = n_kernels, seed
        self.tf = None
        self.proto: dict[int, np.ndarray] = {}
        self.n_feat = 0

    def _feat(self, X):
        return np.asarray(self.tf.transform(pad_for_rocket(X)), dtype=np.float32)

    def fit_base(self, X, y):
        from aeon.transformations.collection.convolution_based import MiniRocket
        self.tf = MiniRocket(n_kernels=self.n_kernels, random_state=self.seed)
        self.tf.fit(pad_for_rocket(X))
        self.n_feat = self._feat(X[:1]).shape[1]
        self.update(X, y)

    def update(self, X, y):
        F = self._feat(X)
        for c in np.unique(y):
            self.proto[c] = F[y == c].mean(axis=0)

    def predict(self, X):
        ks = np.array(sorted(self.proto))
        M = np.stack([self.proto[k] for k in ks])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        F = self._feat(X)
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        return ks[np.argmax(F @ M.T, axis=1)]

    def memory_bytes(self):
        return int(len(self.proto) * self.n_feat * 4)


# ------------------------------------------- MiniRocket 全量重训（精度上界，成本无界）

class MiniRocketRetrain(IncMethod):
    """存下见过的全部数据，每个会话用 ridge 从头重训。

    精度上界，但存储随会话线性增长、更新成本不断变大 —— 正是类增量要避免的。
    """

    name = "minirocket-retrain"

    def __init__(self, n_kernels=10000, seed=0):
        self.n_kernels, self.seed = n_kernels, seed
        self.tf = self.clf = None
        self.X = self.y = None
        self.n_feat = 0

    def _refit(self):
        from sklearn.linear_model import RidgeClassifier
        F = np.asarray(self.tf.transform(pad_for_rocket(self.X)), dtype=np.float32)
        self.n_feat = F.shape[1]
        self.clf = RidgeClassifier(alpha=1.0).fit(F, self.y)

    def fit_base(self, X, y):
        from aeon.transformations.collection.convolution_based import MiniRocket
        self.tf = MiniRocket(n_kernels=self.n_kernels, random_state=self.seed)
        self.tf.fit(pad_for_rocket(X))
        self.X, self.y = X.copy(), y.copy()
        self._refit()

    def update(self, X, y):
        self.X = np.concatenate([self.X, X], axis=0)
        self.y = np.concatenate([self.y, y], axis=0)
        self._refit()

    def predict(self, X):
        return self.clf.predict(
            np.asarray(self.tf.transform(pad_for_rocket(X)), dtype=np.float32))

    def memory_bytes(self):
        return int(self.X.nbytes)      # 原始数据全存着，这是它的真实代价


# ------------------------------------------------------- 1NN + 样本回放（同预算对照）

class NNExemplar(IncMethod):
    """每类存 m 条原始样本，1NN 欧氏距离。

    m 由 budget_bytes 决定，用来和 HDC 做**同内存预算**对比（研究问题 Q1）。
    """

    def __init__(self, budget_bytes: int | None = None, m_per_class: int = 1, seed=0):
        self.budget, self.m, self.seed = budget_bytes, m_per_class, seed
        self.name = "1nn-exemplar"
        self.mem_X: list[np.ndarray] = []
        self.mem_y: list[int] = []
        self._per_sample = None

    def _add(self, X, y):
        rng = np.random.default_rng(self.seed)
        if self._per_sample is None:
            self._per_sample = X[0].nbytes
            if self.budget is not None:      # 按预算反推每类能存几条
                self.m = max(1, int(self.budget // max(1, self._per_sample)))
        for c in np.unique(y):
            pool = X[y == c]
            k = min(self.m, len(pool))
            sel = rng.choice(len(pool), size=k, replace=False)
            self.mem_X.append(pool[sel])
            self.mem_y.extend([c] * k)

    def fit_base(self, X, y): self._add(X, y)
    def update(self, X, y):   self._add(X, y)

    def predict(self, X):
        M = np.concatenate(self.mem_X, axis=0).reshape(len(self.mem_y), -1)
        Q = X.reshape(len(X), -1)
        # 分块算距离，避免 N×M 矩阵过大
        out = np.empty(len(Q), dtype=np.int64)
        ys = np.asarray(self.mem_y)
        for s in range(0, len(Q), 256):
            q = Q[s:s + 256]
            d = ((q[:, None, :] - M[None, :, :]) ** 2).sum(-1)
            out[s:s + len(q)] = ys[np.argmin(d, axis=1)]
        return out

    def memory_bytes(self):
        return int(sum(a.nbytes for a in self.mem_X))


# ==========================================================================
# 评测
# ==========================================================================

def evaluate(method: IncMethod, X_tr, y_tr, X_te, y_te,
             proto: IncProtocol, verbose=False) -> dict:
    """跑完整的类增量流程，返回每会话指标。"""
    from sklearn.metrics import accuracy_score

    sess = proto.sessions(np.unique(y_tr))
    seen: list[int] = []
    accs, base_accs, times, mems = [], [], [], []
    base_cls = sess[0]

    for si, cls in enumerate(sess):
        if si == 0:
            Xs, ys = session_data(X_tr, y_tr, cls, None, proto.seed)
            t0 = time.perf_counter()
            method.fit_base(Xs, ys)
            dt = time.perf_counter() - t0
        else:
            Xs, ys = session_data(X_tr, y_tr, cls, proto.n_shot, proto.seed + si)
            t0 = time.perf_counter()
            method.update(Xs, ys)
            dt = time.perf_counter() - t0

        seen.extend(cls.tolist())
        m = np.isin(y_te, seen)
        yp = method.predict(X_te[m])
        acc = accuracy_score(y_te[m], yp)

        mb = np.isin(y_te, base_cls)
        yb = method.predict(X_te[mb])
        bacc = accuracy_score(y_te[mb], yb)

        accs.append(acc); base_accs.append(bacc)
        times.append(dt); mems.append(method.memory_bytes())
        if verbose:
            print(f"    session {si}: |seen|={len(seen):3d} acc={acc:.4f} "
                  f"base_acc={bacc:.4f} t={dt:.3f}s mem={mems[-1]/1024:.1f}KB")

    return {
        "method": method.name,
        "n_sessions": len(sess),
        "acc_per_session": accs,
        "final_acc": accs[-1],
        "avg_acc": float(np.mean(accs)),
        "base_acc_first": base_accs[0],
        "base_acc_last": base_accs[-1],
        "forgetting": float(base_accs[0] - base_accs[-1]),
        "base_fit_time": times[0],
        "mean_update_time": float(np.mean(times[1:])) if len(times) > 1 else 0.0,
        "final_memory_bytes": mems[-1],
    }


# ------------------------------------------------------- 深度基线（灾难性遗忘对照）

class _TinyCNN:
    """一维卷积骨干 + 可增长的线性分类头。

    骨干：Conv(C→32,k=7) → ReLU → Conv(32→64,k=5) → ReLU → 全局平均池化
    头部：Linear(64 → 已见类数)，每个会话按新类数扩行，旧行权重原样拷贝。
    """

    def __init__(self, n_channels, seed=0, hidden=(32, 64), device="cpu"):
        import torch, torch.nn as nn
        # 模型很小，多线程反而因争用变慢（实测 1 线程 2.3s vs 4 线程 3.7s）
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        h1, h2 = hidden
        self.torch, self.nn = torch, nn
        self.device = torch.device(device)
        self.backbone = nn.Sequential(
            nn.Conv1d(n_channels, h1, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(h1, h2, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        ).to(self.device)
        self.feat_dim = h2
        self.head = None
        self.classes: list[int] = []

    def grow_head(self, new_classes):
        """扩展分类头，旧类的权重原样保留。"""
        nn, torch = self.nn, self.torch
        old = self.head
        self.classes = self.classes + [c for c in new_classes if c not in self.classes]
        head = nn.Linear(self.feat_dim, len(self.classes)).to(self.device)
        if old is not None:
            with torch.no_grad():
                head.weight[:old.out_features] = old.weight
                head.bias[:old.out_features] = old.bias
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))

    def fit(self, X, y, epochs, lr=1e-3, batch=32):
        torch, nn = self.torch, self.nn
        idx = {c: i for i, c in enumerate(self.classes)}
        Xt = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        yt = torch.as_tensor([idx[int(v)] for v in y], device=self.device)
        params = list(self.backbone.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        lossf = nn.CrossEntropyLoss()
        n = len(yt)
        self.backbone.train(); self.head.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, batch):
                b = perm[s:s + batch]
                opt.zero_grad()
                loss = lossf(self.forward(Xt[b]), yt[b])
                loss.backward()
                opt.step()

    @property
    def n_params(self):
        return (sum(p.numel() for p in self.backbone.parameters())
                + (self.head.weight.numel() + self.head.bias.numel()
                   if self.head is not None else 0))


class CNNFinetune(IncMethod):
    """1D-CNN，每个增量会话**只用新类的少量样本**微调整个网络。

    这是类增量里的朴素基线，存在的意义是给出**灾难性遗忘**的对照：
    它不存任何旧数据，所以旧类的知识只能靠权重保留，而权重正在被新数据覆盖。
    冻结原型类方法之所以"不遗忘"，参照物就是它。

    超参（论文里要如实报告）：Adam lr=1e-3，batch=32，base 60 epoch、增量 20 epoch。
    """

    name = "cnn-finetune"

    def __init__(self, seed=0, base_epochs=60, inc_epochs=20, device="cpu"):
        self.seed, self.base_epochs, self.inc_epochs = seed, base_epochs, inc_epochs
        self.device = device
        self.net = None

    def fit_base(self, X, y):
        self.net = _TinyCNN(X.shape[1], seed=self.seed, device=self.device)
        self.net.grow_head(sorted(np.unique(y).tolist()))
        self.net.fit(X, y, self.base_epochs)

    def update(self, X, y):
        self.net.grow_head(sorted(np.unique(y).tolist()))
        self.net.fit(X, y, self.inc_epochs)      # 只见新数据 —— 遗忘就发生在这里

    def predict(self, X):
        torch = self.net.torch
        self.net.backbone.eval(); self.net.head.eval()
        out = []
        with torch.no_grad():
            Xt = torch.as_tensor(np.asarray(X, np.float32), device=self.net.device)
            for s in range(0, len(Xt), 256):
                out.append(self.net.forward(Xt[s:s + 256]).argmax(1).cpu().numpy())
        cls = np.array(self.net.classes)
        return cls[np.concatenate(out)]

    def memory_bytes(self):
        return int(self.net.n_params * 4)        # 只存模型，不存数据


class CNNFrozenNCM(IncMethod):
    """1D-CNN 在 base 会话训练后**冻结骨干**，之后用最近类均值增量。

    这是 MiniRocket-NCM 的"学习版"对照：同样冻结、同样原型，但特征是**学出来的**
    而不是随机/确定性的。用来回答"冻结编码器该不该学"。
    """

    name = "cnn-frozen-ncm"

    def __init__(self, seed=0, base_epochs=60, device="cpu"):
        self.seed, self.base_epochs, self.device = seed, base_epochs, device
        self.net = None
        self.proto: dict[int, np.ndarray] = {}

    def _feat(self, X):
        torch = self.net.torch
        self.net.backbone.eval()
        out = []
        with torch.no_grad():
            Xt = torch.as_tensor(np.asarray(X, np.float32), device=self.net.device)
            for s in range(0, len(Xt), 256):
                out.append(self.net.backbone(Xt[s:s + 256]).cpu().numpy())
        return np.concatenate(out)

    def fit_base(self, X, y):
        self.net = _TinyCNN(X.shape[1], seed=self.seed, device=self.device)
        self.net.grow_head(sorted(np.unique(y).tolist()))
        self.net.fit(X, y, self.base_epochs)
        self.update(X, y)

    def update(self, X, y):
        F = self._feat(X)
        for c in np.unique(y):
            self.proto[int(c)] = F[y == c].mean(axis=0)

    def predict(self, X):
        ks = np.array(sorted(self.proto))
        M = np.stack([self.proto[k] for k in ks])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        F = self._feat(X)
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        return ks[np.argmax(F @ M.T, axis=1)]

    def memory_bytes(self):
        backbone = sum(p.numel() for p in self.net.backbone.parameters())
        return int((backbone + len(self.proto) * self.net.feat_dim) * 4)
