# -*- coding: utf-8 -*-
"""
基准实验：全年单一 LP（完美信息、单一储电量轨迹、无逐日决策解耦）
=================================================================
作为"信息与协调"的上界基准：给出 2025-02-01 至 12-31 共 334 天、每日 144 槽
（合计 48 096 槽）的单一线性规划最优值，用以量化"逐日决策解耦 + 每日终端不亏空"
这两项建模选择的代价。

输出 intermediate/exp_benchmark.json
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # -> code/
import _pathsetup  # noqa: E402,F401  子目录引导：把 common/ 与各问目录加入 sys.path

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import config as C


def full_year_lp():
    panel = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D"))
    load = panel.L(); pv = panel.PV()
    price = C.read_attachment1()["price"].to_numpy(float)
    win = pd.date_range(C.WIN_START, C.WIN_END, freq="D")
    n = C.N_SLOT; T = len(win); nv = 4 * T * n
    iE, iC, iD, iS = 0, T * n, 2 * T * n, 3 * T * n
    obj = np.zeros(nv); obj[iE:iE + T * n] = np.tile(price, T)
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for k in range(T):
        L = load.loc[win[k]].to_numpy(float); P = pv.loc[win[k]].to_numpy(float)
        for t in range(n):
            rows += [r, r, r]; cols += [iE + k * n + t, iC + k * n + t, iD + k * n + t]
            vals += [-1.0, 1.0, -1.0]
            rhs.append(-(L[t] - P[t]) * C.DT); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    rows2, cols2, vals2, rhs2 = [], [], [], []
    for k in range(T):
        for t in range(n):
            g = k * n + t
            rows2.append(g); cols2.append(iS + g); vals2.append(1.0)
            if g > 0:
                rows2.append(g); cols2.append(iS + g - 1); vals2.append(-1.0)
            rows2.append(g); cols2.append(iC + g); vals2.append(-C.ETA_C)
            rows2.append(g); cols2.append(iD + g); vals2.append(1.0 / C.ETA_D)
            rhs2.append(0.0 if g > 0 else C.S_INIT)
    A_eq = coo_matrix((vals2, (rows2, cols2)), shape=(T * n, nv)).tocsr()
    bounds = [(0, None)] * (T * n) + [(0, C.E_MAX)] * (T * n) + [(0, C.E_MAX)] * (T * n) + \
             [(C.S_MIN, C.S_MAX)] * (T * n)
    t0 = time.time()
    res = linprog(obj, A_ub=A_ub, b_ub=np.array(rhs), A_eq=A_eq, b_eq=np.array(rhs2),
                  bounds=bounds, method="highs")
    dt = time.time() - t0
    return dict(cost=float(res.fun), n_var=int(nv), n_slot=int(T * n),
                seconds=float(dt), success=bool(res.success),
                base_cost=sum(C.baseline_cost(price, load.loc[d].to_numpy(float),
                                              pv.loc[d].to_numpy(float)) for d in win))


def main():
    out = dict(full_year=full_year_lp())
    out["decoupling_gap_vs_R1"] = None
    q2 = os.path.join(C.INTERMEDIATE_DIR, "q2_cmp_R1.json")
    if os.path.exists(q2):
        r1 = json.load(open(q2, encoding="utf-8"))["meta"]
        out["daily_decoupled_R1"] = r1
        out["decoupling_gap"] = r1["tot_cost"] - out["full_year"]["cost"]
        out["decoupling_gap_pct"] = (r1["tot_cost"] - out["full_year"]["cost"]) / r1["tot_cost"]
    with open(os.path.join(C.INTERMEDIATE_DIR, "exp_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print()
    print("全年单一 LP：%.6f 元（%.2f 万元），变量 %d，耗时 %.1f s"
          % (out["full_year"]["cost"], out["full_year"]["cost"] / 1e4,
             out["full_year"]["n_var"], out["full_year"]["seconds"]))
    if out.get("decoupling_gap") is not None:
        print("逐日解耦（完美信息 R1）比全年单一 LP 贵 %.2f 元（%.2f%%）"
              % (out["decoupling_gap"], out["decoupling_gap_pct"] * 100))


if __name__ == "__main__":
    main()
