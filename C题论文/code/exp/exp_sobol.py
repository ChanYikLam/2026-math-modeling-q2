# -*- coding: utf-8 -*-
"""
全局灵敏度分析（Sobol 指数，Saltelli/Jansen 估计）——在典型日（问题 1）模型上做。
参数 8 个：往返效率、SOC 上限、SOC 下限、充放功率上限、电价水平、峰谷价差、光伏出力、负载水平。
输出：一阶指数 S_i、总效应指数 S_Ti、以及各参数对"全天购电费"与"相对基线节省率"的影响。

用法：python3 exp_sobol.py [N]   （默认 N=256 → 2560 次 LP 求解）
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # -> code/
import _pathsetup  # noqa: E402,F401  子目录引导：把 common/ 与各问目录加入 sys.path

import json
import os
import sys

import numpy as np
from scipy.stats import qmc

import config as C
import exp_scan as ES

PARAMS = [
    ("eta_rt", "往返效率", 0.75, 0.95),
    ("s_max", "储电量上限 kWh", 8400.0, 14400.0),
    ("s_min", "储电量下限 kWh", 600.0, 2400.0),
    ("p_max", "充放功率上限 kW", 2500.0, 7500.0),
    ("price_lv", "电价水平倍数", 0.8, 1.2),
    ("price_pv", "峰谷价差倍数", 0.7, 1.3),
    ("pv_scale", "光伏出力倍数", 0.8, 1.2),
    ("load_scale", "负载水平倍数", 0.9, 1.1),
]


def model(x, ret_series=False):
    """给定参数向量，返回（全天购电费 元，相对无储能基线节省率）。"""
    d = dict(zip([p[0] for p in PARAMS], x))
    a1 = C.read_attachment1()
    price = a1["price"].to_numpy(float).copy()
    load = a1["load"].to_numpy(float).copy() * d["load_scale"]
    pv = a1["pv"].to_numpy(float).copy() * d["pv_scale"]
    # 电价：水平缩放 + 峰谷价差缩放（围绕均值）
    pm = price.mean()
    price = pm * d["price_lv"] + (price - pm) * d["price_pv"]
    eta = float(np.sqrt(d["eta_rt"]))
    smin = float(d["s_min"]); smax = float(d["s_max"])
    if smin >= smax - 100:                 # 保证可行
        smin, smax = 0.1 * smax, smax
    cost, E, c, dd = ES.q1_with(pmax=float(d["p_max"]), eta_c=eta, eta_d=eta,
                                s_min=smin, s_max=smax,
                                s_init=float(np.clip(6000.0, smin, smax)),
                                price=price, load=load, pv=pv)
    base = C.baseline_cost(price, load, pv)
    save = (base - cost) / base
    return (cost, save) if ret_series else cost


def sobol_indices(N=256, seed=42, target=0):
    d = len(PARAMS)
    rng = np.random.default_rng(seed)
    A = qmc.Sobol(d=d, scramble=True, seed=seed).random(N).T
    B = qmc.Sobol(d=d, scramble=True, seed=seed + 1000).random(N).T
    lo = np.array([p[2] for p in PARAMS]); hi = np.array([p[3] for p in PARAMS])
    scale = lambda U: lo[:, None] + (hi - lo)[:, None] * U
    A, B = scale(A), scale(B)
    fA = np.array([model(A[:, j], ret_series=True)[target] for j in range(N)])
    fB = np.array([model(B[:, j], ret_series=True)[target] for j in range(N)])
    var = np.var(np.concatenate([fA, fB]), ddof=1)
    Si, STi = [], []
    for i in range(d):
        AB = A.copy(); AB[i] = B[i]
        fAB = np.array([model(AB[:, j], ret_series=True)[target] for j in range(N)])
        Si.append(float(np.mean(fB * (fAB - fA)) / var))
        STi.append(float(0.5 * np.mean((fA - fAB) ** 2) / var))
    return dict(N=N, names=[p[1] for p in PARAMS], keys=[p[0] for p in PARAMS],
                bounds=[[p[2], p[3]] for p in PARAMS],
                var=float(var), mean=float(np.mean(np.concatenate([fA, fB]))),
                S1=Si, ST=STi)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    out = dict(cost=sobol_indices(N=N, target=0),
               saving=sobol_indices(N=N, target=1, seed=43))
    with open(os.path.join(C.INTERMEDIATE_DIR, "exp_sobol.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for key, tag in (("cost", "全天购电费（元）"), ("saving", "相对基线节省率")):
        r = out[key]
        print(f"--- {tag}  N={r['N']}  均值={r['mean']:.4f}  方差={r['var']:.4g} ---")
        print(f"{'参数':<16}{'S1':>10}{'ST':>10}")
        for n, s, st in zip(r["names"], r["S1"], r["ST"]):
            print(f"{n:<16}{s:>10.4f}{st:>10.4f}")
        print("sum(S1) = %.4f" % sum(r["S1"]))


if __name__ == "__main__":
    main()
