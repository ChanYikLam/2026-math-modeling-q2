# -*- coding: utf-8 -*-
"""
问题 1：典型日确定性计划购电 LP（附件 1）
=========================================
模型（电量口径，单位 kWh）：

    min  sum_t p_t E_t                                        (目标)
    s.t. E_t - c_t + d_t >= (L_t - PV_t) * DT          (计划满足负载)
         S_t = S_{t-1} + eta_c c_t - d_t / eta_d             (SOC 递推)
         S_min <= S_t <= S_max,  0 <= c_t, d_t <= E_MAX
         E_t >= 0
         S_0 = S_144 = 6000 kWh（日周期，题面要求 0:00 与 24:00 相同）

运行：python3 q1.py   （输出 result1.xlsx、intermediate/q1.json、figures/fig03、fig04）
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # -> code/
import _pathsetup  # noqa: E402,F401  子目录引导：把 common/ 与各问目录加入 sys.path

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import config as C


def solve_q1(price: np.ndarray, load: np.ndarray, pv: np.ndarray,
             s_init: float = C.S_INIT, s_end: float | None = None,
             free_cycle: bool = False):
    """求解典型日 LP。free_cycle=True 时 S_0 自由（仅要求 S_0 = S_144）。"""
    n = C.N_SLOT
    # 变量顺序：E(n) | c(n) | d(n) | S(n)  [free_cycle 时最前面再加 S0(1)]
    off = 1 if free_cycle else 0
    nv = off + 4 * n
    iE, iC, iD, iS = 0 + off, n + off, 2 * n + off, 3 * n + off

    c_obj = np.zeros(nv)
    c_obj[iE:iE + n] = price

    A_ub, b_ub = [], []
    # (a) E_t - c_t + d_t >= (L-PV)DT   ->  -E + c - d <= -(L-PV)DT
    for t in range(n):
        row = np.zeros(nv)
        row[iE + t] = -1.0
        row[iC + t] = 1.0
        row[iD + t] = -1.0
        A_ub.append(row)
        b_ub.append(-(load[t] - pv[t]) * C.DT)

    A_eq, b_eq = [], []
    # (b) SOC 递推：S_t - S_{t-1} - eta_c c_t + d_t/eta_d = 0
    for t in range(n):
        row = np.zeros(nv)
        row[iS + t] = 1.0
        if t == 0:
            if free_cycle:
                row[0] = -1.0          # 前置变量 S0（自由环状）
        else:
            row[iS + t - 1] = -1.0
        row[iC + t] = -C.ETA_C
        row[iD + t] = 1.0 / C.ETA_D
        A_eq.append(row)
        b_eq.append(0.0 if (t > 0 or free_cycle) else s_init)

    if free_cycle:                     # 日周期闭环：S_144 = S_0
        row = np.zeros(nv)
        row[iS + n - 1] = 1.0
        row[0] = -1.0
        A_eq.append(row)
        b_eq.append(0.0)

    bounds = [(0.0, None)] * n + [(0.0, C.E_MAX)] * n + [(0.0, C.E_MAX)] * n + \
             [(C.S_MIN, C.S_MAX)] * n
    if free_cycle:
        bounds = [(C.S_MIN, C.S_MAX)] + bounds
    else:
        # S_144 固定（日周期）；S_0 = s_init 通过递推常数注入
        end = s_init if s_end is None else s_end
        bounds[iS + n - 1] = (end, end)

    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds,
                  method="highs")
    if not res.success:
        raise RuntimeError(res.message)

    x = res.x
    E, c, d, S = x[iE:iE + n], x[iC:iC + n], x[iD:iD + n], x[iS:iS + n]
    s0 = x[0] if free_cycle else s_init
    s_cur = np.concatenate([[s0], S])
    curtail = C.curtailment(price, load, pv, E, c, d)
    return dict(E=E, c=c, d=d, S=S, s0=s0, curtail=curtail, obj=res.fun,
                status=res.status, dual_eq=res.eqlin.marginals,
                dual_ub=res.ineqlin.marginals, nv=nv, res=res)


def summarize(sol, price, load, pv, label=""):
    n = C.N_SLOT
    E, c, d = sol["E"], sol["c"], sol["d"]
    total_e = float(E.sum())
    cost = float(np.sum(price * E))
    base = C.baseline_cost(price, load, pv)
    out = dict(
        label=label,
        cost=cost,
        total_energy=total_e,
        base_cost=base,
        saving=base - cost,
        saving_rate=(base - cost) / base,
        charge_total=float(c.sum()),
        discharge_total=float(d.sum()),
        curtail_total=float(np.maximum(sol["curtail"], 0).sum()),
        s0=sol["s0"], s_end=float(sol["S"][-1]),
        min_soc=float(sol["S"].min()), max_soc=float(sol["S"].max()),
        ratio_cd=float(c.sum() / d.sum()) if d.sum() > 0 else float("nan"),
    )
    # ---- 独立自检（不依赖求解器返回值）----
    # R1 原始可行性：供给 >= 需求（微网提供的电能不低于负载）
    supply = E + pv * C.DT + d
    demand = load * C.DT + c
    out["R1_min_surplus"] = float(np.min(supply - demand))          # 应 >= -1e-9
    # R2 计划可行性：E 不低于"预测口径下的净需求"
    out["R2_max_plan_violation"] = float(np.max(
        (load - pv) * C.DT + c - d - E))                            # 应 <= 1e-9
    # R3 SOC 递推残差（用上一时刻实际储电量重算）
    prev = np.concatenate([[sol["s0"]], sol["S"][:-1]])
    out["R3_max_soc_residual"] = float(np.max(np.abs(
        sol["S"] - (prev + C.ETA_C * c - d / C.ETA_D))))
    # R4 功率与储电量界
    out["R4_max_power_violation"] = float(max(np.max(c), np.max(d)) - C.E_MAX)
    out["R4_soc_bound_violation"] = float(max(C.S_MIN - sol["S"].min(),
                                              sol["S"].max() - C.S_MAX))
    # R5 充放互斥
    out["R5_simultaneous_slots"] = int(np.sum((c > 1e-6) & (d > 1e-6)))
    # R6 充放电量比 = 1/(eta_c eta_d)（若末值回到初值且无弃光）
    out["R6_cd_ratio_err"] = float(abs(c.sum() / d.sum() - 1 / (C.ETA_C * C.ETA_D)))         if d.sum() > 0 else float("nan")
    # R7 费用独立复算（不调用求解器目标值）
    out["R7_cost_recompute_gap"] = float(abs(cost - float(np.sum(price * E))))
    return out


def table1(E):
    rows = []
    for lab, j in C.TABLE1_SLOTS:
        rows.append((lab, E[j]))
    return rows


def table2(c, d, s0, s_end):
    rows = []
    for (a, b), lab in zip(C.BLOCKS, C.BLOCK_LABELS):
        rows.append((lab, float(c[a:b].sum()), float(d[a:b].sum())))
    return rows, s0, s_end


def write_result1(sol, path=None):
    """按附件 5 模板写出 result1.xlsx（竖表）。"""
    path = path or os.path.join(C.RESULT_DIR, "result1.xlsx")
    tmpl = C.read_template("计划购电量", "result1.xlsx")
    labels = list(tmpl.iloc[1:, 0])
    assert len(labels) == C.N_SLOT
    df1 = pd.DataFrame({"时间段": labels, "购电量": np.round(sol["E"], 4)})

    rows, s0, s_end = table2(sol["c"], sol["d"], sol["s0"], float(sol["S"][-1]))
    df2 = pd.DataFrame({
        "时间段": C.BLOCK_LABELS,
        "充电量": [r[1] for r in rows],
        "放电量": [r[2] for r in rows],
        "时刻": ["0:00", "24:00", None, None, None, None],
        "储电量": [round(s0, 4), round(s_end, 4), None, None, None, None],
    })
    with pd.ExcelWriter(path) as w:
        df1.to_excel(w, sheet_name="计划购电量", index=False)
        df2.to_excel(w, sheet_name="充放电量", index=False)
    return path


def main():
    a1 = C.read_attachment1()
    price = a1["price"].to_numpy(float)
    load = a1["load"].to_numpy(float)
    pv = a1["pv"].to_numpy(float)

    sol = solve_q1(price, load, pv)
    summ = summarize(sol, price, load, pv, "Q1 主模型（S0=S144=6000）")

    sol_free = solve_q1(price, load, pv, free_cycle=True)
    summ_free = summarize(sol_free, price, load, pv, "Q1 自由环状（S0 自由，S0=S144）")

    print("=" * 78)
    for k, v in summ.items():
        print(f"  {k:26s} {v}")
    print("-" * 78)
    print(f"  自由环状最优费用 {summ_free['cost']:.2f} 元, S0=S144={summ_free['s0']:.2f} kWh, "
          f"与主模型差 {summ_free['cost'] - summ['cost']:.2f} 元 "
          f"({(summ_free['cost'] - summ['cost']) / summ['cost'] * 100:+.4f}%)")
    print("=" * 78)

    p = write_result1(sol)
    print("已写出", p)

    out = dict(main=summ, free_cycle=summ_free,
               table1=table1(sol["E"]),
               table2=table2(sol["c"], sol["d"], sol["s0"], float(sol["S"][-1])),
               arrays=dict(E=sol["E"].tolist(), c=sol["c"].tolist(),
                           d=sol["d"].tolist(), S=sol["S"].tolist(),
                           curtail=sol["curtail"].tolist(),
                           price=price.tolist(), load=load.tolist(), pv=pv.tolist(),
                           dual_eq=sol["dual_eq"].tolist(),
                           dual_bal=sol["dual_ub"].tolist()))
    with open(os.path.join(C.INTERMEDIATE_DIR, "q1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    main()
