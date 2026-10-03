# -*- coding: utf-8 -*-
"""
问题 2：固定电价 + 实际波动 + 紧急购电惩罚的两阶段模型
======================================================
信息结构（R3 主方案，见假设 A9/R3）：
  计划依据 = 当天 0:00 发布的光伏预报（附件 3，0:00 行）+ 当日负载（题面未给负载预报，
             本模型按"负载曲线已知"处理，等价于负载预报无误差）
  结算依据 = 附件 2 当日实际负载与光伏
  缺口按 5 倍电价紧急购电

场景生成：对全部 334 天(决策窗口)的"预报残差" r_{d,t}=(PV^fcst-PV^act)*DT 做
         K-means 场景削减，得到 K 个代表残差与权重（三层保真的第 1、2 层）。

输出：results/result2.xlsx、intermediate/q2.json
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # -> code/
import _pathsetup  # noqa: E402,F401  子目录引导：把 common/ 与各问目录加入 sys.path

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config as C
import lp_engine as LP
import results_io as RIO

DATE_WINDOW = pd.date_range(C.WIN_START, C.WIN_END, freq="D")
DATE_ALL = pd.date_range(C.WARMUP_START, C.WIN_END, freq="D")


def build_residuals(panel, dates):
    """构造预报残差矩阵 r[d,t] = (PV^fcst_0:00 - PV^act) * DT，单位 kWh。"""
    pv_act = panel.PV()
    R = np.zeros((len(dates), C.N_SLOT))
    for i, d in enumerate(dates):
        fc = panel.pv_forecast(d, 0)
        R[i] = (fc - pv_act.loc[d].to_numpy(float)) * C.DT
    return R


def run_q2(K=30, term_mode="ge", verbose=True, info="R3", max_days=None, seed=0,
           scen="strat"):
    panel = C.Panel(DATE_ALL)
    load = panel.L()
    pv_act = panel.PV()
    price_fixed = C.read_attachment1()["price"].to_numpy(float)
    price_flat = panel.P(volatile=False)          # 每天相同
    a1 = C.read_attachment1()

    # 残差与场景削减（在决策窗口 334 天上做）
    R_all = build_residuals(panel, DATE_WINDOW)
    if scen == "zero":                      # 消融：不设安全边际（按预报精确购电）
        resid_cen = np.zeros((1, C.N_SLOT)); resid_w = np.array([1.0])
    elif scen == "strat":
        resid_cen, resid_w = LP.reduce_scenarios_stratified(R_all, K=K)
    else:
        resid_cen, resid_w = LP.reduce_scenarios(R_all, K=K, seed=seed)
    if verbose:
        print(f"[场景削减] K={len(resid_w)}  权重和={resid_w.sum():.4f}  "
              f"残差均值绝对值={np.abs(resid_cen).mean():.2f} kWh")

    dates = DATE_ALL if max_days is None else DATE_ALL[:max_days]
    recs, s_init = [], C.S_INIT
    t0 = time.time()
    for i, d in enumerate(dates):
        nd_act = (load.loc[d].to_numpy(float) - pv_act.loc[d].to_numpy(float)) * C.DT
        if info == "R3":
            pv_fc = panel.pv_forecast(d, 0)
            nd_fcst = (load.loc[d].to_numpy(float) - pv_fc) * C.DT
        elif info == "R1":                       # 完美信息（下界）
            nd_fcst = nd_act.copy()
        elif info == "R2":                       # 附件 1 典型日预报
            nd_fcst = (a1["load"].to_numpy(float) - a1["pv"].to_numpy(float)) * C.DT
        else:
            raise ValueError(info)

        nd_scen = nd_fcst[None, :] + resid_cen
        r = d in DATE_WINDOW
        tm = term_mode if r else "ge"
        sol = LP.solve_day(price_fixed, nd_fcst, nd_scen, resid_w,
                           s_init=s_init, term_mode=tm)
        g, pc, ec = LP.realized_slot(sol, nd_act)
        chk = LP.check_solution(sol, nd_fcst, nd_act)
        if r:
            recs.append(dict(
                date=d, E=sol["E"].copy(), c=sol["c"].copy(), d=sol["d"].copy(),
                S_end=float(sol["S"][-1]), s_init=s_init, g=g,
                plan_cost=pc, emg_cost=ec, total_cost=pc + ec,
                emg_qty=float(g.sum()), energy=float(sol["E"].sum()),
                base_cost=C.baseline_cost(price_fixed, load.loc[d].to_numpy(float),
                                          pv_act.loc[d].to_numpy(float)),
                sae_obj=float(sol["obj"]),
                max_viol=max(chk["R1_plan_violation"], chk["R2_power_violation"],
                             chk["R3_soc_bounds"], chk["R4_soc_recursion"])))
        s_init = float(sol["S"][-1])
        if verbose and (i + 1) % 30 == 0:
            print(f"  ... {i+1}/{len(dates)} 天  累计 {time.time()-t0:.1f}s  "
                  f"S={s_init:.0f}")
    return recs, dict(resid_cen=resid_cen, resid_w=resid_w), price_fixed


def to_frames(recs, price_fixed):
    """整理成 result2 表格所需结构。"""
    dates = [r["date"] for r in recs]
    E = np.vstack([r["E"] for r in recs])
    Cc = np.vstack([r["c"] for r in recs])
    Dd = np.vstack([r["d"] for r in recs])
    G = np.vstack([r["g"] for r in recs])
    S0 = np.array([r["s_init"] for r in recs])
    S1 = np.array([r["S_end"] for r in recs])
    return dates, E, Cc, Dd, G, S0, S1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--scen", default="strat", choices=["strat", "kmeans", "zero"])
    ap.add_argument("--term", default="ge", choices=["ge", "eq", "free"])
    ap.add_argument("--info", default="R3", choices=["R3", "R1", "R2"])
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    recs, scen, price_fixed = run_q2(K=args.K, term_mode=args.term, info=args.info,
                                     max_days=args.max_days, scen=args.scen)
    dates, E, Cc, Dd, G, S0, S1 = to_frames(recs, price_fixed)

    # ---- 写出 result2.xlsx（仅在主口径下写出，避免被对照实验覆盖）----
    if args.write:
        path = os.path.join(C.RESULT_DIR, "result2.xlsx")
        RIO.write_result(path, dates, E, Cc, Dd, S0, S1, price_fixed, g=G)
        print("已写出", path)

    tot_plan = sum(r["plan_cost"] for r in recs)
    tot_emg = sum(r["emg_cost"] for r in recs)
    tot_base = sum(r["base_cost"] for r in recs)
    tot_emg_qty = sum(r["emg_qty"] for r in recs)
    days_with_emg = int(np.sum(G.sum(axis=1) > 1e-6))
    print("=" * 78)
    print(f"  信息结构 {args.info} | 终端条件 {args.term} | 天数 {len(recs)}")
    print(f"  计划购电费        {tot_plan:14.2f} 元")
    print(f"  紧急购电费        {tot_emg:14.2f} 元")
    print(f"  总费用            {tot_plan+tot_emg:14.2f} 元")
    print(f"  无储能基线 C-01   {tot_base:14.2f} 元")
    print(f"  相对基线节省      {tot_base-(tot_plan+tot_emg):14.2f} 元 "
          f"({(tot_base-(tot_plan+tot_emg))/tot_base*100:.2f}%)")
    print(f"  紧急购电量        {tot_emg_qty:14.2f} kWh  出现在 {days_with_emg}/{len(recs)} 天")
    print(f"  最差单日残差      {max(r['max_viol'] for r in recs):.3e}")
    print("=" * 78)

    out = dict(meta=dict(K=args.K, term=args.term, info=args.info, tag=args.tag,
                         scen=args.scen,
                         n_days=len(recs), tot_plan=tot_plan, tot_emg=tot_emg,
                         tot_cost=tot_plan + tot_emg, tot_base=tot_base,
                         tot_emg_qty=tot_emg_qty, days_with_emg=days_with_emg),
               daily=[{k: (v.isoformat() if k == "date" else
                           (v.tolist() if isinstance(v, np.ndarray) else v))
                       for k, v in r.items()} for r in recs],
               scen=dict(cen=scen["resid_cen"].tolist(),
                         w=scen["resid_w"].tolist()))
    with open(os.path.join(C.INTERMEDIATE_DIR, f"q2_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    main()
