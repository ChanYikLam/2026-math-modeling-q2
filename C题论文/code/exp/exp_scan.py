# -*- coding: utf-8 -*-
"""
补充实验：Q1 容量-功率双向扫描、报童经验覆盖率、单参数灵敏度。
输出 intermediate/exp_scan.json / exp_newsboy.json / exp_sens.json
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
import lp_engine as LP
import q1 as Q1
import q2 as Q2


def q1_with(cap=None, pmax=None, eta_c=C.ETA_C, eta_d=C.ETA_D,
            s_min=None, s_max=None, s_init=C.S_INIT,
            price=None, load=None, pv=None):
    """在给定储能参数下重解 Q1（用于容量/功率/效率扫描）。

    price/load/pv 默认取附件 1；传入自定义数组时用于 Sobol 等全局灵敏度实验，
    否则改参数只会进入基线与目标的外围计算，形成口径错配。
    """
    a1 = C.read_attachment1()
    if price is None:
        price = a1["price"].to_numpy(float)
    if load is None:
        load = a1["load"].to_numpy(float)
    if pv is None:
        pv = a1["pv"].to_numpy(float)
    price = np.asarray(price, float).copy()
    load = np.asarray(load, float).copy()
    pv = np.asarray(pv, float).copy()
    n = C.N_SLOT
    e_max = (pmax * C.DT) if pmax is not None else C.E_MAX
    smin = C.S_MIN if s_min is None else s_min
    smax = C.S_MAX if s_max is None else s_max
    if cap is not None:                       # 保持 10%–90% DOD 同步移动上下界
        smin, smax = 0.10 * cap, 0.90 * cap
    nv = 4 * n
    iE, iC, iD, iS = 0, n, 2 * n, 3 * n
    obj = np.zeros(nv); obj[iE:iE + n] = price
    A_ub, b_ub = [], []
    for t in range(n):
        row = np.zeros(nv); row[iE + t] = -1; row[iC + t] = 1; row[iD + t] = -1
        A_ub.append(row); b_ub.append(-(load[t] - pv[t]) * C.DT)
    A_eq, b_eq = [], []
    for t in range(n):
        row = np.zeros(nv); row[iS + t] = 1
        if t > 0:
            row[iS + t - 1] = -1
        row[iC + t] = -eta_c; row[iD + t] = 1 / eta_d
        A_eq.append(row); b_eq.append(0.0 if t > 0 else s_init)
    bounds = [(0, None)] * n + [(0, e_max)] * n + [(0, e_max)] * n + [(smin, smax)] * n
    bounds[iS + n - 1] = (s_init, s_init)
    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq),
                  b_eq=np.array(b_eq), bounds=bounds, method="highs")
    return float(res.fun), res.x[iE:iE + n], res.x[iC:iC + n], res.x[iD:iD + n]


def capacity_scan():
    base = q1_with()[0]
    caps = [7200, 9600, 12000, 14400, 19200, 28800, 48000]
    out = []
    for cap in caps:
        c, E, cc, dd = q1_with(cap=cap)
        out.append(dict(cap=cap, cost=c, saving=base - c,
                        dcost=(base - c) / base * 100, charge=float(cc.sum()),
                        discharge=float(dd.sum())))
    powers = [1250, 2500, 5000, 7500, 10000]
    out2 = []
    for pm in powers:
        c, E, cc, dd = q1_with(pmax=pm)
        out2.append(dict(pmax=pm, cost=c, saving=base - c,
                         dcost=(base - c) / base * 100, charge=float(cc.sum())))
    # 单一自由度放开
    c_nocap, _, _, _ = q1_with(s_min=C.S_MIN, s_max=10 ** 7)   # 仅解除容量上限
    c_nopow, _, _, _ = q1_with(pmax=10 ** 6)                   # 仅解除功率上限
    return dict(base=base, cap_scan=out, power_scan=out2,
                no_capacity_limit=c_nocap, no_power_limit=c_nopow,
                gain_no_cap=(base - c_nocap) / base * 100,
                gain_no_pow=(base - c_nopow) / base * 100)


def eta_scan():
    out = []
    for ec, ed, name in [(0.9, 0.9, "单边各 0.9（往返 81%）"),
                         (0.9487, 0.9487, "往返 0.9 等效单边"),
                         (0.9, 1.0, "仅充电侧 0.9"),
                         (1.0, 0.9, "仅放电侧 0.9")]:
        c, *_ = q1_with(eta_c=ec, eta_d=ed)
        out.append(dict(eta_c=ec, eta_d=ed, name=name, cost=c,
                        saving=C.baseline_cost(C.read_attachment1()["price"].to_numpy(float),
                                               C.read_attachment1()["load"].to_numpy(float),
                                               C.read_attachment1()["pv"].to_numpy(float)) - c))
    return out


def newsboy_check(K=60):
    """报童经验覆盖率：计划口径下逐槽缺货频率与临界分位数理论的对比。"""
    j = json.load(open(os.path.join(C.INTERMEDIATE_DIR, "q2_main.json", encoding="utf-8")))
    G = np.vstack([np.array(r["g"]) for r in j["daily"]])
    E = np.vstack([np.array(r["E"]) for r in j["daily"]])
    c = np.vstack([np.array(r["c"]) for r in j["daily"]])
    d = np.vstack([np.array(r["d"]) for r in j["daily"]])
    panel = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D"))
    load = panel.L().loc[[pd.Timestamp(r["date"]) for r in j["daily"]]].to_numpy(float)
    pv = panel.PV().loc[[pd.Timestamp(r["date"]) for r in j["daily"]]].to_numpy(float)
    nd_act = (load - pv) * C.DT
    short = nd_act + c - d - E
    margin = E - np.maximum(nd_act * 0 + (load - pv) * C.DT, 0) * 0  # 占位
    # 计划边际：E 相对"预测口径净需求 + 充放电"的超出量
    fc = np.zeros_like(nd_act)
    for i, r in enumerate(j["daily"]):
        fc[i] = (load[i] - panel.pv_forecast(pd.Timestamp(r["date"]), 0)) * C.DT
    req = fc + c - d
    m = E - np.maximum(req, 0)
    return dict(short_slot_rate=float((G > 1e-6).mean()),
                short_energy=float(G.sum()),
                margin_total=float(np.maximum(m, 0).sum()),
                margin_per_slot=float(np.maximum(m, 0).mean()),
                short_q99=float(np.percentile(short, 99)),
                short_q80=float(np.percentile(short, 80)),
                short_q833=float(np.percentile(short, 83.33)),
                fill_rate=float(1 - G.sum() / (nd_act.sum())))


def sens_penalty():
    """紧急购电倍数 k 的灵敏度（用 Q1 日 + 单一场景集重算典型日口径）。"""
    out = []
    for k in [2, 3, 5, 8, 10]:
        C.EMERGENCY_MULT = float(k)
        # 在固定残差场景集上求典型日的最优安全边际
        a1 = C.read_attachment1()
        price = a1["price"].to_numpy(float)
        panel = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D"))
        R = Q2.build_residuals(panel, Q2.DATE_WINDOW)
        cen, w = LP.reduce_scenarios_stratified(R, K=60)
        nd_fcst = (a1["load"].to_numpy(float) - a1["pv"].to_numpy(float)) * C.DT
        sol = LP.solve_day(price, nd_fcst, nd_fcst[None, :] + cen, w, s_init=C.S_INIT)
        out.append(dict(k=k, margin=float(np.maximum(
            sol["E"] - np.maximum(nd_fcst + sol["c"] - sol["d"], 0), 0).sum()),
            exp_emg=float((sol["g"] * w[:, None]).sum()),
            obj=float(sol["obj"])))
    C.EMERGENCY_MULT = 5.0
    return out


def main():
    res = {}
    res["capacity"] = capacity_scan()
    res["eta"] = eta_scan()
    res["newsboy"] = newsboy_check()
    res["penalty"] = sens_penalty()
    with open(os.path.join(C.INTERMEDIATE_DIR, "exp_scan.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
