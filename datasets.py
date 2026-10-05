"""Dataset scan and selection (UCR/UEA via aeon). `python datasets.py --profile incremental-v6` writes the candidate list and the selected 27 datasets."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

SEEDS = [0, 1, 2, 3, 4]
SHOTS = [1, 5, 10, 20]

# 数据目录。aeon 会先在这里找 <DATA_DIR>/<name>/<name>_TRAIN.ts，找不到才联网下载。
# 设环境变量 KILOROCKET_DATA_DIR 或用 --data-dir 指定，可完全离线运行。
DATA_DIR = os.environ.get("KILOROCKET_DATA_DIR") or None


def set_data_dir(path: str | None) -> None:
    global DATA_DIR
    DATA_DIR = path

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SCAN_CSV = os.path.join(RESULTS_DIR, "dataset_scan.csv")
SELECTED_JSON = os.path.join(RESULTS_DIR, "datasets_selected.json")
SCAN_INC_CSV = os.path.join(RESULTS_DIR, "dataset_scan_incremental.csv")
SELECTED_INC_JSON = os.path.join(RESULTS_DIR, "datasets_selected_incremental.json")
# v6（v6）：aeon 全量等长数据集 + 每类下限放宽到 5。旧的 14 个清单文件保持不动。
SCAN_INC_FULL_CSV = os.path.join(RESULTS_DIR, "dataset_scan_incremental_full.csv")
SELECTED_INC_V6_JSON = os.path.join(RESULTS_DIR, "datasets_selected_incremental_v6.json")


# 类增量实验的筛选档：要多类别，类别数上限放宽，每类样本数要求降低
# （类增量的 few-shot 发生在增量会话里，不需要每类都有 20 个）
INCREMENTAL_FILTERS_KW = dict(
    max_length=500, min_classes=6, max_classes=60,
    min_per_class_train=10, max_channels=20, min_test_size=60,
)

# v6 筛选档（2026-10-01，v6）。与上面唯一的区别：每类下限 10 → 5。
# 理由：main 协议是 2-way 5-shot，新类每类只需要 5 条训练样本；base 类用全部数据。
# 10 是早期随手设的阈值，没有协议上的依据。旧的 14 个数据集在新阈值下全部仍然入选。
INCREMENTAL_V6_FILTERS_KW = dict(INCREMENTAL_FILTERS_KW, min_per_class_train=5)

# v6 扫描时不下载的 UEA 数据集。按 UEA 档案公布的元数据（Bagnall et al., 2018），它们必然被上面的
# 筛选规则排除（通道数 C > 20、类别数 K < 6 或长度 T > 500），而且文件很大（DuckDuckGeese、
# FaceDetection、MotorImagery、PEMS-SF 合计上 GB），下载只会白白卡住扫描。
# 跳过它们不改变入选结果；扫描表里会记 ok=False 和原因。
V6_SKIP_KNOWN_EXCLUDED = {
    "DuckDuckGeese": "C=1345, K=5", "FaceDetection": "C=144, K=2", "FingerMovements": "C=28, K=2",
    "HandMovementDirection": "K=4", "Heartbeat": "C=61, K=2", "MotorImagery": "C=64, T=3000, K=2",
    "PEMS-SF": "C=963", "SelfRegulationSCP1": "T=896, K=2", "SelfRegulationSCP2": "T=1152, K=2",
    "StandWalkJump": "T=2500, K=3", "EthanolConcentration": "T=1751, K=4",
}


@dataclass
class Filters:
    """筛选阈值。改动这里等于改实验协议，改完必须重跑 scan + select 并记录。"""

    max_length: int = 300          # 序列长度上限：控制 HDC 叠加的 crosstalk（见风险 3）
    min_classes: int = 0           # 类别数下限（类增量实验要多类别，设 >=6）
    max_classes: int = 10          # 类别数上限
    min_per_class_train: int = 20  # 训练集每类最少样本数：保证能做到 20-shot
    max_channels: int = 20         # 通道数上限（多变量用）
    min_test_size: int = 40        # 测试集最小规模，太小的话精度方差没法看


# 候选池。univariate_equal_length 有 112 个，全扫一遍要下载约 1-2GB、耗时较长。
# 这里给一个偏短序列的候选池先扫；想扫全量把 use_full_pool=True 传给 scan()。
UCR_CANDIDATE_POOL = [
    "ArrowHead", "BME", "CBF", "Chinatown", "ChlorineConcentration", "Coffee",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
    "ECG200", "ECG5000", "ECGFiveDays", "Earthquakes", "FaceFour",
    "FreezerRegularTrain", "FreezerSmallTrain", "GunPoint", "GunPointAgeSpan",
    "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham", "Herring",
    "InsectWingbeatSound", "ItalyPowerDemand", "Lightning7", "MedicalImages",
    "MelbournePedestrian", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW", "MoteStrain", "PhalangesOutlinesCorrect", "Plane", "PowerCons",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW",
    "ShapeletSim", "SmoothSubspace", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2",
    "Strawberry", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace",
    "TwoLeadECG", "TwoPatterns", "UMD", "Wafer", "Wine", "Symbols",
]

# 类增量实验用的候选池：类别数多的 UCR/UEA 数据集
INCREMENTAL_POOL = [
    # 单变量，类别多
    "Crop", "ElectricDevices", "FaceAll", "FacesUCR", "FiftyWords", "Fungi",
    "Handwriting", "InsectWingbeatSound", "Lightning7", "MedicalImages",
    "MelbournePedestrian", "MiddlePhalanxTW", "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2", "Plane", "ProximalPhalanxTW", "SwedishLeaf",
    "Symbols", "SyntheticControl", "Trace", "TwoPatterns", "UWaveGestureLibraryAll",
    "WordSynonyms", "Adiac", "ChlorineConcentration", "DistalPhalanxTW",
    # 多变量，类别多
    "ArticularyWordRecognition", "Cricket", "Handwriting", "Libras", "LSST",
    "NATOPS", "PenDigits", "PhonemeSpectra", "SpokenArabicDigits",
    "UWaveGestureLibrary", "EigenWorms", "Epilepsy", "RacketSports",
]

UEA_CANDIDATE_POOL = [
    "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "Epilepsy",
    "ERing", "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat",
    "Libras", "LSST", "NATOPS", "PenDigits", "RacketSports", "SelfRegulationSCP1",
    "SelfRegulationSCP2", "StandWalkJump", "UWaveGestureLibrary",
]


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------

def load_split(name: str, extract_path: str | None = None):
    """加载一个数据集的 train/test split。

    返回 (X_train, y_train, X_test, y_test)：
      X: float64 ndarray, shape (n_samples, n_channels, n_timepoints)
      y: int64 ndarray, shape (n_samples,)   —— 原始字符串标签已映射成 0..K-1
    """
    from aeon.datasets import load_classification

    path = extract_path if extract_path is not None else DATA_DIR
    kw = {} if path is None else {"extract_path": path}
    X_tr, y_tr = load_classification(name, split="train", **kw)
    X_te, y_te = load_classification(name, split="test", **kw)

    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)

    classes = sorted(set(np.asarray(y_tr).tolist()) | set(np.asarray(y_te).tolist()))
    lut = {c: i for i, c in enumerate(classes)}
    y_tr = np.array([lut[v] for v in np.asarray(y_tr).tolist()], dtype=np.int64)
    y_te = np.array([lut[v] for v in np.asarray(y_te).tolist()], dtype=np.int64)
    return X_tr, y_tr, X_te, y_te


def znormalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """逐样本、逐通道 z-normalize。UCR/UEA 分类的标准预处理。

    注意：这是 per-sample 归一化，不涉及 train/test 信息泄漏。
    """
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return (X - mu) / (sd + eps)


# --------------------------------------------------------------------------
# 扫描与筛选
# --------------------------------------------------------------------------

def profile_dataset(name: str, extract_path: str | None = None) -> dict:
    """下载并统计一个数据集的元信息。失败时返回 ok=False 而不是抛异常。"""
    rec = {"name": name, "ok": False, "error": ""}
    try:
        X_tr, y_tr, X_te, y_te = load_split(name, extract_path)
        if X_tr.ndim != 3 or X_te.ndim != 3:
            raise ValueError("非等长序列，跳过")
        counts = np.bincount(y_tr)
        rec.update(
            ok=True,
            n_channels=int(X_tr.shape[1]),
            length=int(X_tr.shape[2]),
            n_classes=int(len(counts)),
            n_train=int(X_tr.shape[0]),
            n_test=int(X_te.shape[0]),
            min_per_class_train=int(counts.min()),
            has_nan=bool(np.isnan(X_tr).any() or np.isnan(X_te).any()),
        )
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    return rec


def scan(names: Iterable[str], out_csv: str = SCAN_CSV, verbose: bool = True,
         resume: bool = False, skip: dict | None = None):
    """逐个下载并 profile，结果写 CSV。这是一次性成本，跑完就别再跑。

    resume=True 时，out_csv 里已经 ok 的数据集直接沿用，不重新下载；
    每扫完一个就落盘一次，中途关掉再运行会接着扫。
    """
    import pandas as pd

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = {}
    if resume and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = {r["name"]: r.to_dict() for _, r in prev.iterrows() if bool(r["ok"])}
    rows = []
    names = list(dict.fromkeys(names))  # 去重，保持顺序
    for i, n in enumerate(names, 1):
        if n in done:
            rows.append(done[n])
            continue
        if skip and n in skip:
            rec = {"name": n, "ok": False, "error": f"skipped: excluded by UEA metadata ({skip[n]})"}
            rows.append(rec)
            if verbose:
                print(f"[{i:3d}/{len(names)}] {n:38s} SKIPPED ({skip[n]})", flush=True)
            if resume:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
            continue
        rec = profile_dataset(n)
        rows.append(rec)
        if resume:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        if verbose:
            if rec["ok"]:
                print(f"[{i:3d}/{len(names)}] {n:38s} C={rec['n_channels']:<3d} "
                      f"T={rec['length']:<5d} K={rec['n_classes']:<3d} "
                      f"ntr={rec['n_train']:<5d} min/cls={rec['min_per_class_train']}", flush=True)
            else:
                print(f"[{i:3d}/{len(names)}] {n:38s} FAILED  {rec['error']}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def select(df, filters: Filters | None = None, n_univariate: int = 30,
           n_multivariate: int = 8, out_json: str = SELECTED_JSON):
    """按阈值筛选，并按确定性规则取前 N 个。

    排序规则：先按 min_per_class_train 降序（样本越充裕，few-shot 曲线越能画到 20-shot），
    同分按名称升序。规则写死，保证可复现。
    """
    f = filters or Filters()
    ok = df[df["ok"]].copy()
    mask = (
        (ok["length"] <= f.max_length)
        & (ok["n_classes"] >= f.min_classes)
        & (ok["n_classes"] <= f.max_classes)
        & (ok["min_per_class_train"] >= f.min_per_class_train)
        & (ok["n_channels"] <= f.max_channels)
        & (ok["n_test"] >= f.min_test_size)
        & (~ok["has_nan"])
    )
    kept = ok[mask].sort_values(["min_per_class_train", "name"],
                                ascending=[False, True])

    uni = kept[kept["n_channels"] == 1]["name"].tolist()[:n_univariate]
    mul = kept[kept["n_channels"] > 1]["name"].tolist()[:n_multivariate]

    payload = {
        "filters": asdict(f),
        "univariate": sorted(uni),
        "multivariate": sorted(mul),
        "n_univariate": len(uni),
        "n_multivariate": len(mul),
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def selected_incremental(path: str = SELECTED_INC_JSON) -> list[str]:
    """读回类增量实验的数据集清单（v5 及以前的 14 个）。"""
    return selected_datasets(path)


def selected_incremental_v6(path: str = SELECTED_INC_V6_JSON) -> list[str]:
    """读回 v6（v6）的类增量数据集清单。"""
    return selected_datasets(path)


def selected_datasets(path: str = SELECTED_JSON) -> list[str]:
    """读回固定的数据集清单。主实验一律走这个函数，不要在别处硬编码名字。"""
    with open(path, encoding="utf-8") as fh:
        p = json.load(fh)
    return p["univariate"] + p["multivariate"]


# --------------------------------------------------------------------------
# few-shot 采样
# --------------------------------------------------------------------------

def few_shot_indices(y: np.ndarray, n_shot: int, seed: int) -> np.ndarray:
    """分层抽样：每类抽 n_shot 个下标。

    若某类样本不足 n_shot，取该类全部（并不会报错，但 scan 的 min_per_class_train
    阈值已经保证了正常情况下不会触发）。返回的下标已排序，保证可复现。
    """
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        k = min(n_shot, len(pool))
        idx.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(idx))


def few_shot_split(X: np.ndarray, y: np.ndarray, n_shot: int, seed: int):
    """返回 few-shot 训练子集 (X_sub, y_sub)。"""
    idx = few_shot_indices(y, n_shot, seed)
    return X[idx], y[idx]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="扫描并固定数据集清单")
    ap.add_argument("--full-pool", action="store_true",
                    help="扫 aeon 的全部等长数据集（慢，约 1-2GB 下载）")
    ap.add_argument("--n-uni", type=int, default=30)
    ap.add_argument("--n-mul", type=int, default=8)
    ap.add_argument("--data-dir", default=None,
                    help="本地 .ts 数据目录，结构为 <dir>/<数据集名>/<数据集名>_TRAIN.ts")
    ap.add_argument("--profile", default="batch",
                    choices=["batch", "incremental", "incremental-v6"],
                    help="batch=原批量实验；incremental=类增量实验（要多类别数据集）；"
                         "incremental-v6=扫 aeon 全量等长数据集，每类下限 5（v6）")
    args = ap.parse_args()
    if args.data_dir:
        set_data_dir(args.data_dir)

    if args.profile == "incremental-v6":
        from aeon.datasets import tsc_datasets as td
        # 旧候选池放最前面，保证旧的 14 个先扫到；再接 aeon 全量等长清单
        pool = (INCREMENTAL_POOL + list(td.univariate_equal_length)
                + list(td.multivariate_equal_length))
        df = scan(pool, SCAN_INC_FULL_CSV, resume=True, skip=V6_SKIP_KNOWN_EXCLUDED)
        payload = select(df, Filters(**INCREMENTAL_V6_FILTERS_KW),
                         n_univariate=1000, n_multivariate=1000,  # 不截断，全部入选
                         out_json=SELECTED_INC_V6_JSON)
        old = set(selected_incremental())
        new = set(payload["univariate"] + payload["multivariate"])
        print("\n==== v6 类增量数据集清单 ====")
        print(f"单变量 {payload['n_univariate']} 个：{payload['univariate']}")
        print(f"多变量 {payload['n_multivariate']} 个：{payload['multivariate']}")
        print(f"新增 {len(new - old)} 个：{sorted(new - old)}")
        if old - new:
            print(f"[警告] 旧清单里有 {len(old - new)} 个没有入选：{sorted(old - new)}")
        print(f"\n已写入 {SELECTED_INC_V6_JSON}")
        return

    if args.profile == "incremental":
        df = scan(INCREMENTAL_POOL, SCAN_INC_CSV)
        payload = select(df, Filters(**INCREMENTAL_FILTERS_KW),
                         n_univariate=20, n_multivariate=8,
                         out_json=SELECTED_INC_JSON)
        print("\n==== 类增量数据集清单 ====")
        print(f"单变量 {payload['n_univariate']} 个：{payload['univariate']}")
        print(f"多变量 {payload['n_multivariate']} 个：{payload['multivariate']}")
        print(f"\n已写入 {SELECTED_INC_JSON}")
        return

    if args.full_pool:
        from aeon.datasets import tsc_datasets as td
        pool = list(td.univariate_equal_length) + list(td.multivariate_equal_length)
    else:
        pool = UCR_CANDIDATE_POOL + UEA_CANDIDATE_POOL

    df = scan(pool)
    payload = select(df, n_univariate=args.n_uni, n_multivariate=args.n_mul)
    print("\n==== 选定清单 ====")
    print(f"单变量 {payload['n_univariate']} 个：{payload['univariate']}")
    print(f"多变量 {payload['n_multivariate']} 个：{payload['multivariate']}")
    print(f"\n已写入 {SELECTED_JSON}，之后不要再改。")


if __name__ == "__main__":
    main()
