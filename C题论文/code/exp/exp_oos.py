# -*- coding: utf-8 -*-
"""
样本外回测：用 2025-02-01—06-30 的预报残差训练场景集，在 07-01—12-31 上回测
==========================================================================
目的：检验"按日总残差分层 + 层内 medoid"的场景集是否只是拟合了样本内残差。
做法：场景集只由训练窗残差构造；测试窗每一天仍按其 0:00 预报制定计划，
      用当日实际值结算；与"用测试窗自身残差构造场景集"（样本内上界）比较。

输出 intermediate/exp_oos.json
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

import config as C
import lp_engine as LP
import q2 as Q2

TRAIN = pd.date_range("2025-02-01", "2025-06-30", freq="D")
TEST = pd.date_range("2025-07-01", "2025-12-31", freq="D")
ALL = pd.date_range(C.WARMUP_START, C.WIN_END, freq="D")


def run(scen_days, test_days, K=60):
    panel = C.Panel(ALL)
    load = panel.L(); pv = panel.PV()
    price = C.read_attachment1()["price"].to_numpy(float)
    R = np.zeros((len(scen_days), C.N_SLOT))
    for i, d in enumerate(scen_days):
        R[i] = (panel.pv_forecast(d, 0) - pv.loc[d].to_numpy(float)) * C.DT
    cen, w = LP.reduce_scenarios_stratified(R, K=K)
    tot_plan = tot_emg = tot_emg_qty = 0.0
    base = 0.0
    soc = C.S_INIT
    for d in test_days:
        Ld = load.loc[d].to_numpy(float); Pv = pv.loc[d].to_numpy(float)
        nd_act = (Ld - Pv) * C.DT
        fc = panel.pv_forecast(d, 0)
        nd_fc = (Ld - fc) * C.DT
        sol = LP.solve_day(price, nd_fc, nd_fc[None, :] + cen, w, s_init=soc,
                           term_mode="ge")
        g, pc, ec = LP.realized_slot(sol, nd_act)
        tot_plan += pc; tot_emg += ec; tot_emg_qty += float(g.sum())
        base += C.baseline_cost(price, Ld, Pv)
        soc = float(sol["S"][-1])
    return dict(n_days=len(test_days), plan=tot_plan, emg=tot_emg,
                total=tot_plan + tot_emg, emg_qty=tot_emg_qty, base=base,
                saving=(base - tot_plan - tot_emg) / base)


def main():
    out = dict(train=[str(d.date()) for d in TRAIN], test=[str(d.date()) for d in TEST])
    out["oos_train_scenarios"] = run(TRAIN, TEST)
    out["insample_test_scenarios"] = run(TEST, TEST)
    # 参考：全窗口主结果中测试窗部分的实际值
    j = json.load(open(os.path.join(C.INTERMEDIATE_DIR, "q2_main.json", encoding="utf-8")))
    sel = [r for r in j["daily"] if pd.Timestamp(r["date"]) in set(TEST)]
    out["main_full_scenarios_test_window"] = dict(
        n_days=len(sel), plan=sum(r["plan_cost"] for r in sel),
        emg=sum(r["emg_cost"] for r in sel),
        total=sum(r["total_cost"] for r in sel),
        emg_qty=sum(r["emg_qty"] for r in sel),
        base=sum(r["base_cost"] for r in sel),
        saving=(sum(r["base_cost"] for r in sel) - sum(r["total_cost"] for r in sel))
        / sum(r["base_cost"] for r in sel))
    with open(os.path.join(C.INTERMEDIATE_DIR, "exp_oos.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    a, b, c = out["oos_train_scenarios"], out["insample_test_scenarios"], out["main_full_scenarios_test_window"]
    print()
    print("测试窗 %d 天：样本外场景集总费用 %.2f 元；样本内场景集 %.2f 元；全窗口主结果同窗 %.2f 元"
          % (a["n_days"], a["total"], b["total"], c["total"]))
    print("样本外相对样本内劣化 %.2f 元（%.3f%%）；相对主结果 %.2f 元（%.3f%%）"
          % (a["total"] - b["total"], (a["total"] - b["total"]) / b["total"] * 100,
             a["total"] - c["total"], (a["total"] - c["total"]) / c["total"] * 100))


if __name__ == "__main__":
    main()
