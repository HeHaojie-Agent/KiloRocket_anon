"""HDC encoder: binding of position, channel and quantized-amplitude codes."""

from __future__ import annotations

import numpy as np
import torch
import torchhd


# --------------------------------------------------------------------------

def _plain(t: torch.Tensor) -> torch.Tensor:
    """把 torchhd 的 MAPTensor 降级成普通 torch.Tensor，避免后续算子行为意外。"""
    return t.as_subclass(torch.Tensor).clone()


class HDCSequenceEncoder:
    """时间-通道-数值绑定的序列编码器。

    Parameters
    ----------
    dim : int
        超向量维度 D。消融/敏感性实验的主要扫描对象。
    n_levels : int
        数值量化的档数 L。
    value_encoding : {"level", "quantize", "none"}
        level    —— level hypervector，相邻档相似（默认，论文主设定）
        quantize —— 每档独立随机 HV，相邻档不相似（敏感性分析对照组）
        none     —— 直接用标量幅值调制 p_t⊗c_c，不做数值 HV（最弱对照）
    time_encoding : {"permutation", "random"}
        permutation —— p_t = ρ^t(p_0)，循环移位，内存 O(D)（默认）
        random      —— 每个时间位置一个独立随机 HV，内存 O(T·D)
    gamma : float
        时间衰减 w_t = γ^{T-t}。1.0 表示不衰减。
    binarize : bool
        True 则对叠加结果取 sign（双极输出）；False 保留实值（默认）。
        储池初始状态用实值通常更稳，双极版本留作对照。
    clip_sigma : float
        z-normalize 之后按 ±clip_sigma 截断再量化，抑制离群点吃掉量化档位。
    """

    def __init__(
        self,
        dim: int = 2048,
        n_levels: int = 64,
        value_encoding: str = "level",
        time_encoding: str = "permutation",
        gamma: float = 1.0,
        binarize: bool = False,
        clip_sigma: float = 3.0,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        assert value_encoding in ("level", "quantize", "none")
        assert time_encoding in ("permutation", "random")
        self.dim = dim
        self.n_levels = n_levels
        self.value_encoding = value_encoding
        self.time_encoding = time_encoding
        self.gamma = gamma
        self.binarize = binarize
        self.clip_sigma = clip_sigma
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype

        self._fitted = False
        self.n_channels: int | None = None
        self.length: int | None = None

    # ---------------------------------------------------------------- 内部

    def _build_codebook(self, n_channels: int, length: int) -> None:
        g = torch.Generator(device="cpu").manual_seed(self.seed)

        # 通道 HV：独立随机，通道之间近似正交
        self.channel_hv = _plain(
            torchhd.random(n_channels, self.dim, vsa="MAP", generator=g)
        ).to(self.device, self.dtype)

        # 数值 HV
        if self.value_encoding == "level":
            self.level_hv = _plain(
                torchhd.level(self.n_levels, self.dim, vsa="MAP", generator=g)
            ).to(self.device, self.dtype)
        elif self.value_encoding == "quantize":
            self.level_hv = _plain(
                torchhd.random(self.n_levels, self.dim, vsa="MAP", generator=g)
            ).to(self.device, self.dtype)
        else:
            self.level_hv = None

        # 时间 HV
        if self.time_encoding == "random":
            self.time_hv = _plain(
                torchhd.random(length, self.dim, vsa="MAP", generator=g)
            ).to(self.device, self.dtype)
        else:
            base = _plain(torchhd.random(1, self.dim, vsa="MAP", generator=g))[0]
            self.time_hv = torch.stack(
                [torch.roll(base, shifts=t) for t in range(length)]
            ).to(self.device, self.dtype)

        # 时间权重 w_t = γ^{T-t}
        if self.gamma == 1.0:
            w = torch.ones(length)
        else:
            w = torch.tensor(
                [self.gamma ** (length - 1 - t) for t in range(length)]
            )
        self.time_w = w.to(self.device, self.dtype)

    # ---------------------------------------------------------------- 公开

    def fit(self, X_train: np.ndarray) -> "HDCSequenceEncoder":
        """在训练集上确定量化边界并生成码本。

        量化边界用训练集的分位数（每通道独立），避免用到测试集信息。
        X_train : (N, C, T)，已经 z-normalize 过。
        """
        X = np.asarray(X_train, dtype=np.float64)
        assert X.ndim == 3, f"期望 (N, C, T)，拿到 {X.shape}"
        n, c, t = X.shape
        self.n_channels, self.length = c, t
        self._fit_bounds(X, c)
        self._build_codebook(c, t)
        self._fitted = True
        return self

    def _fit_bounds(self, X: np.ndarray, c: int) -> None:
        """按训练集分位数确定每通道的量化上下界（不触碰测试集）。"""
        lo = np.full(c, -self.clip_sigma, dtype=np.float64)
        hi = np.full(c, self.clip_sigma, dtype=np.float64)
        for j in range(c):
            vals = X[:, j, :].ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                q_lo, q_hi = np.quantile(vals, [0.005, 0.995])
                lo[j] = max(q_lo, -self.clip_sigma)
                hi[j] = min(q_hi, self.clip_sigma)
            if hi[j] - lo[j] < 1e-6:          # 常数通道兜底
                lo[j], hi[j] = lo[j] - 0.5, hi[j] + 0.5

        self.lo = torch.tensor(lo, device=self.device, dtype=self.dtype)
        self.hi = torch.tensor(hi, device=self.device, dtype=self.dtype)

    def _to_level_idx(self, Xb: torch.Tensor) -> torch.Tensor:
        """(B, C, T) 实值 -> (B, C, T) 档位下标 0..L-1。"""
        lo = self.lo.view(1, -1, 1)
        hi = self.hi.view(1, -1, 1)
        u = (Xb - lo) / (hi - lo)
        u = torch.nan_to_num(u, nan=0.5).clamp_(0.0, 1.0)
        idx = (u * (self.n_levels - 1)).round().long()
        return idx

    @torch.no_grad()
    def encode(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """(N, C, T) -> (N, D)。"""
        if not self._fitted:
            raise RuntimeError("先调用 fit()")
        X = np.asarray(X, dtype=np.float64)
        n, c, t = X.shape
        if c != self.n_channels or t != self.length:
            raise ValueError(
                f"形状与 fit 时不一致：fit 是 (C={self.n_channels}, T={self.length})，"
                f"现在是 (C={c}, T={t})"
            )

        out = torch.empty(n, self.dim, device=self.device, dtype=self.dtype)
        chan = self.channel_hv.unsqueeze(0)                      # (1, C, D)

        for s in range(0, n, batch_size):
            Xb = torch.as_tensor(X[s:s + batch_size], device=self.device,
                                 dtype=self.dtype)               # (B, C, T)
            b = Xb.shape[0]
            h = torch.zeros(b, self.dim, device=self.device, dtype=self.dtype)

            if self.value_encoding == "none":
                # 数值不编码成 HV，只作为标量幅值
                for ti in range(t):
                    amp = Xb[:, :, ti].unsqueeze(-1)             # (B, C, 1)
                    m = (chan * amp).sum(dim=1)                  # (B, D)
                    h += self.time_w[ti] * (self.time_hv[ti] * m)
            else:
                idx = self._to_level_idx(Xb)                     # (B, C, T)
                for ti in range(t):
                    V = self.level_hv[idx[:, :, ti]]             # (B, C, D)
                    m = (chan * V).sum(dim=1)                    # (B, D)  Σ_c c_c⊙v
                    h += self.time_w[ti] * (self.time_hv[ti] * m)

            if self.binarize:
                h = torch.sign(h)
                h[h == 0] = 1.0
            else:
                # 按 sqrt(T·C) 归一，使不同长度/通道数的数据集尺度可比
                h = h / float(np.sqrt(t * c))
            out[s:s + b] = h

        return out.cpu().numpy()

    def fit_encode(self, X_train, X_test=None, batch_size: int = 64):
        self.fit(X_train)
        if X_test is None:
            return self.encode(X_train, batch_size)
        return self.encode(X_train, batch_size), self.encode(X_test, batch_size)


# --------------------------------------------------------------------------
# 分段编码（出路 B）：编码时就按时间窗切分，直接构成元胞网格
# --------------------------------------------------------------------------

class SegmentedHDCEncoder(HDCSequenceEncoder):
    """把序列切成 N_c 个时间段，每段独立编码成 d_cell 维超向量。

    与父类的区别
    ------------
    父类：整条序列 -> 一个 D 维超向量 -> reshape 成 (N_c, d_cell) 元胞网格。
          元胞之间没有关系，邻域无语义（已被 dim-shuffle 实验证实）。

    本类：序列 -> N_c 个时间段 -> 每段一个 d_cell 维超向量 -> (N_c, d_cell)。
          **相邻元胞 = 相邻时间窗**，邻域有真实时序语义，方向置换才有东西可保留。

    段内用的是**相对位置** HV（每段共用同一套局部码本），所以不同元胞的表示
    落在同一个空间里，可比、可做邻域传播。元胞身份由它在网格里的位置给出，
    不再编进超向量。

    附带好处：每个超向量叠加的项数从 T·C 降到 seg_len·C，缓解 crosstalk。
    """

    def __init__(self, n_cells: int = 64, d_cell: int = 32, overlap: float = 0.0,
                 **kwargs):
        kwargs.pop("dim", None)
        super().__init__(dim=d_cell, **kwargs)
        self.n_cells = n_cells
        self.d_cell = d_cell
        self.total_dim = n_cells * d_cell
        self.overlap = float(overlap)

    def _segment_starts(self, T: int):
        """返回 (starts, seg_len)。T 不能整除时允许相邻段重叠，保证覆盖全序列。"""
        seg_len = max(1, int(np.ceil(T / self.n_cells)))
        seg_len = min(T, int(round(seg_len * (1.0 + self.overlap))))
        if self.n_cells == 1:
            return np.array([0]), T
        starts = np.round(np.linspace(0, T - seg_len, self.n_cells)).astype(int)
        return starts, seg_len

    def fit(self, X_train: np.ndarray):
        X = np.asarray(X_train, dtype=np.float64)
        assert X.ndim == 3
        n, c, t = X.shape
        self.starts, self.seg_len = self._segment_starts(t)
        # 父类的码本按「段内相对位置」建，长度是 seg_len 而不是 T
        self._fit_bounds(X, c)
        self.n_channels, self.length = c, t
        self._build_codebook(c, self.seg_len)
        self._fitted = True
        return self

    @torch.no_grad()
    def encode(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """(N, C, T) -> (N, N_c * d_cell)，展平后的元胞网格。

        用 grid = encode(X).reshape(N, n_cells, d_cell) 拿回网格形状。
        """
        if not self._fitted:
            raise RuntimeError("先调用 fit()")
        X = np.asarray(X, dtype=np.float64)
        n, c, t = X.shape
        if c != self.n_channels or t != self.length:
            raise ValueError(f"形状与 fit 不一致：(C={self.n_channels},T={self.length}) vs (C={c},T={t})")

        starts = torch.as_tensor(self.starts, device=self.device, dtype=torch.long)
        chan = self.channel_hv.view(1, c, 1, self.d_cell)     # (1,C,1,d)
        out = torch.empty(n, self.n_cells * self.d_cell,
                          device=self.device, dtype=self.dtype)

        for s in range(0, n, batch_size):
            Xb = torch.as_tensor(X[s:s + batch_size], device=self.device, dtype=self.dtype)
            b = Xb.shape[0]
            grid = torch.zeros(b, self.n_cells, self.d_cell,
                               device=self.device, dtype=self.dtype)
            idx_all = None if self.value_encoding == "none" else self._to_level_idx(Xb)

            for j in range(self.seg_len):                     # 段内相对位置
                ts = (starts + j).clamp_(max=t - 1)           # (N_c,)
                if self.value_encoding == "none":
                    amp = Xb[:, :, ts].unsqueeze(-1)          # (B,C,N_c,1)
                    m = (chan * amp).sum(dim=1)               # (B,N_c,d)
                else:
                    V = self.level_hv[idx_all[:, :, ts]]      # (B,C,N_c,d)
                    m = (chan * V).sum(dim=1)                 # (B,N_c,d)
                grid += self.time_w[j] * (self.time_hv[j] * m)

            if self.binarize:
                grid = torch.sign(grid)
                grid[grid == 0] = 1.0
            else:
                grid = grid / float(np.sqrt(self.seg_len * c))
            out[s:s + b] = grid.reshape(b, -1)

        return out.cpu().numpy()


def shuffle_cells(flat: np.ndarray, n_cells: int, d_cell: int,
                  seed: int = 0) -> np.ndarray:
    """随机打乱元胞顺序（不动元胞内部）。

    这是分段编码版本的对照实验：如果打乱元胞顺序后精度**不掉**，说明时序邻域
    依然没起作用；如果精度明显下降，说明邻域传播确实用上了时间上的相邻关系。
    """
    perm = np.random.default_rng(seed).permutation(n_cells)
    g = flat.reshape(len(flat), n_cells, d_cell)[:, perm, :]
    return g.reshape(len(flat), -1)


# --------------------------------------------------------------------------
# 风险 1 的验证工具（见冲刺计划第 3 节）
# --------------------------------------------------------------------------

def random_dim_permutation(dim: int, seed: int = 0) -> np.ndarray:
    """生成一个维度置换。

    用途：把 h 的维度随机打乱再 reshape 成元胞网格。如果下游分类器的精度基本不变，
    说明元胞邻域没有语义，储池只是随机非线性展开 —— 此时方法章节的措辞必须
    改成"随机展开"，不能暗示邻域保留了结构。这个实验务必在第 2 周做。
    """
    return np.random.default_rng(seed).permutation(dim)


# --------------------------------------------------------------------------
# 自测：HDC-only 基线（编码 + ridge）
# --------------------------------------------------------------------------

def _demo(name: str = "GunPoint", dim: int = 2048, n_shot: int | None = None):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score
    import datasets as D

    X_tr, y_tr, X_te, y_te = D.load_split(name)
    X_tr, X_te = D.znormalize(X_tr), D.znormalize(X_te)
    if n_shot:
        X_tr, y_tr = D.few_shot_split(X_tr, y_tr, n_shot, seed=0)

    enc = HDCSequenceEncoder(dim=dim, seed=0)
    H_tr, H_te = enc.fit_encode(X_tr, X_te)

    clf = RidgeClassifier(alpha=1.0).fit(H_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(H_te))
    print(f"{name:24s} shot={str(n_shot):>4s}  ntr={len(y_tr):<5d} "
          f"D={dim:<6d} HDC-only acc={acc:.4f}")
    return acc


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GunPoint")
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--shot", type=int, default=None)
    a = ap.parse_args()
    _demo(a.dataset, a.dim, a.shot)
