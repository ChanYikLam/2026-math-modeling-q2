# -*- coding: utf-8 -*-
"""问题 4 价格信息对照：负载条件化电价预测。

主模型在 0:00 使用联合情景的无条件均值 p̄_sc 作为首阶段电价预报。若当日负载曲线
已知，严格的首阶段系数应取 E[p_t | L_d]；本脚本用历史代表日的日负载水平对情景
重加权，构造该条件均值的近似，并沿真实 334 天轨迹回测，检验它是否改善实际费用。

方法：
  base       : 主模型，权重 w_k；
  knn20      : 取历史日负载总量最接近当日的 20 个代表日，等权；
  kernel0.5  : 高斯核，带宽 0.5 * std(历史日负载总量)；
  kernel1.0  : 高斯核，带宽 1.0 * std(历史日负载总量)。

输出：intermediate/exp_pricecond.json
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

import config as C
import q4 as Q4

K_SCEN = 64
SEED = 7
KNN_K = 20
KERNEL_BWS = (0.5, 1.0)


def _panel():
    return C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D"))


def _hist_load_total(panel, scen):
    """历史代表日的日负载总量（kWh）。"""
    dates = pd.to_datetime(scen["src"])
    return panel.L().loc[dates].sum(axis=1).to_numpy(float) * C.DT


def _conditional_weights(panel, scen, d, method):
    """返回第 d 天的情景重加权权重（未归一化前先乘历史权重）。"""
    w0 = np.asarray(scen["w"], float)
    ltot = _hist_load_total(panel, scen)
    now = float(panel.L().loc[d].sum() * C.DT)
    diff = ltot - now
    if method == "base":
        w = w0
    elif method == "knn20":
        idx = np.argsort(np.abs(diff))[:KNN_K]
        w = np.zeros_like(w0)
        w[idx] = 1.0 / KNN_K
    elif method.startswith("kernel"):
        bw = float(method.split("kernel")[1])
        h = bw * float(np.std(ltot, ddof=1))
        logw = -0.5 * (diff / h) ** 2 + np.log(np.maximum(w0, 1e-300))
        logw -= logw.max()
        w = np.exp(logw)
    else:
        raise ValueError(method)
    return w / w.sum()


def _ess(w):
    return float(1.0 / np.sum(np.asarray(w, float) ** 2))


def run_variant(panel, scen, method):
    """与 q4.run_q4_2 相同的 365 天滚动，唯一差别是逐日情景权重。"""
    dates = pd.date_range(C.WARMUP_START, C.WIN_END, freq="D")
    load = panel.L()
    pv_act = panel.PV()
    p4 = panel.P(volatile=True)
    recs = []
    s_init = C.S_INIT
    ess_list = []
    t0 = time.time()
    for i, d in enumerate(dates):
        L = load.loc[d].to_numpy(float)
        PVa = pv_act.loc[d].to_numpy(float)
        nd_act = (L - PVa) * C.DT
        fc = panel.pv_forecast(d, 0)
        nd_fcst = (L - fc) * C.DT
        nd_scen = nd_fcst[None, :] + scen["resid"]
        w = _conditional_weights(panel, scen, d, method)
        sol = Q4.solve_vol(scen["price"], nd_fcst, nd_scen, w,
                           s_init=s_init, s_from=0, term_floor=s_init)
        g = np.maximum(nd_act + sol["c"] - sol["d"] - sol["E"], 0.0)
        pact = p4.loc[d].to_numpy(float)
        if d >= pd.Timestamp(C.WIN_START):
            recs.append(dict(date=str(d.date()),
                             cost_buy=float(np.sum(pact * sol["E"])),
                             cost_emg=float(C.EMERGENCY_MULT * np.sum(pact * g)),
                             emg_qty=float(g.sum()),
                             base=float(C.baseline_cost(pact, L, PVa))))
            ess_list.append(_ess(w))
        s_init = float(sol["S"][-1])
        if (i + 1) % 120 == 0:
            print(f"  {method:10s} {i+1:3d}/365  {time.time()-t0:6.1f}s")
    total = float(sum(r["cost_buy"] + r["cost_emg"] for r in recs))
    buy = float(sum(r["cost_buy"] for r in recs))
    emg = float(sum(r["cost_emg"] for r in recs))
    base = float(sum(r["base"] for r in recs))
    return dict(
        method=method, days=len(recs),
        total=total, cost_buy=buy, cost_emg=emg,
        emg_qty=float(sum(r["emg_qty"] for r in recs)),
        base=base, saving=(base - total) / base,
        ess_mean=float(np.mean(ess_list)), ess_min=float(np.min(ess_list)),
        runtime_s=time.time() - t0,
    )


def main():
    panel = _panel()
    scen = Q4.build_joint_scenarios(panel, K=K_SCEN, mode="hist", seed=SEED)
    methods = ["base", "knn20"] + [f"kernel{bw}" for bw in KERNEL_BWS]
    out = {
        "meta": dict(
            script="exp_pricecond.py", problem="问题 4-2 价格信息对照",
            window=f"{C.WIN_START}..{C.WIN_END}", K_scen=K_SCEN,
            scenario_days=int(len(scen["w"])), knn_k=KNN_K,
            kernel_bws=list(KERNEL_BWS),
            method="以历史代表日的日负载总量对情景重加权，近似 E[p_t|L_d]",
            reproduce="cd C题论文/code && python3 exp_pricecond.py",
        ),
        "variants": [],
    }
    for method in methods:
        r = run_variant(panel, scen, method)
        out["variants"].append(r)
        print(f"  -> {method:10s} 总费用 {r['total']:,.2f} 元  "
              f"节省率 {r['saving']*100:.2f}%  紧急 {r['emg_qty']:,.1f} kWh  "
              f"ESS {r['ess_mean']:.1f}")
    with open(os.path.join(C.INTERMEDIATE_DIR, "exp_pricecond.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("已写出 intermediate/exp_pricecond.json")


if __name__ == "__main__":
    main()
