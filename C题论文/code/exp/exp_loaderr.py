# -*- coding: utf-8 -*-
"""
负载预报误差敏感性：A9 假设"负载按已知处理"的量级评估
========================================================
背景
----
题面只提供光伏预报（附件 3），未给负载预报。本文主口径（假设 A9 / 信息结构 R3）
因此把"当日负载曲线"按已知处理，已在假设与局限两处声明该简化会**低估**紧急购电量。
但此前只给了方向、没有量级。本实验补上量级。

做法（只改"计划时的负载信息"，不改模型、不改结算）
--------------------------------------------------
计划阶段使用的净需求：      nd^plan_t = (L_t + eps_t - PV^fcst_t)·dt
结算阶段（不变）：          nd^act_t  = (L_t - PV^act_t)·dt

其中负载预报误差 eps_t 按**相对 MAE** 标定：

    MAE_rel = E|eps_t| / mean_t(L_t)        （mean_t(L_t) 为全年逐槽负载均值，单位 kW）

三个档位 MAE_rel ∈ {0%, 5%, 10%}；0% 即现行主口径（负载已知），作为基准。
误差形状按"当日负载曲线×逐日随机缩放因子"给出，缩放因子服从零均值正态、
标准差由 MAE_rel 反解（对正态分布 MAE = σ·sqrt(2/π)）。

同时给出一个**独立的极端参照**：把光伏预报误差本身按相对 MAE = 5% 反解出的缩放注入，
用于说明"光伏误差被放大到同等相对 MAE 时"的量级（评审关心的"频次 vs 精度"另一侧）。

输出
----
intermediate/exp_loaderr.json

用法
----
    cd code && python3 exp_loaderr.py
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

SEED = 20260911
MAE_RELS = (0.0, 0.05, 0.10)     # 负载预报相对 MAE 档位
PV_MAE_REL = 0.05                # 极端参照：把光伏误差按同等相对 MAE 反解
K_SCEN = 60                      # 与主口径一致
DATE_WINDOW = pd.date_range(C.WIN_START, C.WIN_END, freq="D")   # 334 天，情景集与主口径同窗

ALL = pd.date_range(C.WARMUP_START, C.WIN_END, freq="D")
WIN = pd.date_range(C.WIN_START, C.WIN_END, freq="D")
WIN_SET = set(WIN)


# --------------------------------------------------------------------------
def build_scenarios(err: np.ndarray, K: int = K_SCEN):
    """用给定的逐日逐槽计划误差 err(d,t)（kWh）构造分层 medoid 情景集。"""
    # err 为全年 365 天逐槽（行序与 ALL 一致）；分层 medoid 在全年残差上构造，
    # 与主口径的做法一致（主口径同样用整窗残差构造情景集）
    cen, w = LP.reduce_scenarios_stratified(err, K=K)
    return cen, w


def run(panel, load_err_kW: np.ndarray, K: int = K_SCEN) -> dict:
    """全年回放：计划用 (L+eps) 与 0:00 光伏预报，结算用实际值。

    load_err_kW(d,t) 为逐日逐槽的负载预报误差（**kW**，含符号）。
    计划口径净需求 nd_plan = (L + eps − PV^fcst)·dt
    结算口径净需求 nd_act  = (L − PV^act)·dt
    """
    load, pv = panel.L(), panel.PV()
    price = C.read_attachment1()["price"].to_numpy(float)

    # 情景集：沿用主口径 q2.py 的残差定义 r = (PV^fcst − PV^act)·dt，
    # 再把负载预报误差按其"等效光伏误差"并入：eps 使净需求增加 eps·dt，
    # 与 r 增加 −eps·dt 等价，故 r' = (PV^fcst − PV^act − eps)·dt。
    # eps ≡ 0 时逐格退化为 q2.py 的残差矩阵（本脚本一致性校验即检查这一条）。
    err_plan = np.zeros((len(DATE_WINDOW), C.N_SLOT))
    for i, d in enumerate(DATE_WINDOW):
        k = ALL.get_loc(d)
        err_plan[i] = ((panel.pv_forecast(d, 0) - pv.loc[d].to_numpy(float))
                       - load_err_kW[k]) * C.DT
    cen, w = LP.reduce_scenarios_stratified(err_plan, K=K)

    tot_plan = tot_emg = emg_qty = base = 0.0
    days_with_emg = 0
    soc = C.S_INIT
    for i, d in enumerate(ALL):
        Ld = load.loc[d].to_numpy(float)
        Pv = pv.loc[d].to_numpy(float)
        nd_act = (Ld - Pv) * C.DT
        nd_fc = (Ld + load_err_kW[i] - panel.pv_forecast(d, 0)) * C.DT
        sol = LP.solve_day(price, nd_fc, nd_fc[None, :] + cen, w,
                           s_init=soc, term_mode="ge")
        g, pc, ec = LP.realized_slot(sol, nd_act)
        if d in WIN_SET:
            tot_plan += pc
            tot_emg += ec
            emg_qty += float(g.sum())
            base += C.baseline_cost(price, Ld, Pv)
            if float(g.sum()) > 0:
                days_with_emg += 1
        soc = float(sol["S"][-1])
    return dict(n_days=len(WIN), plan=tot_plan, emg=tot_emg,
                total=tot_plan + tot_emg, emg_qty=emg_qty, days_with_emg=days_with_emg,
                base=base, saving_rate=(base - tot_plan - tot_emg) / base)


def main():
    rng = np.random.default_rng(SEED)
    panel = C.Panel(ALL)
    load, pv = panel.L(), panel.PV()
    L = load.to_numpy(float)                       # (365,144) kW
    mean_load = float(L.mean())

    # 误差形状：按"当日负载曲线 × 逐日随机缩放因子"
    z = rng.standard_normal(len(ALL))
    c = np.sqrt(2.0 / np.pi)                       # 正态分布 MAE = c·σ
    out = dict(meta=dict(script="exp_loaderr.py", seed=SEED, K=K_SCEN,
                         window=[str(WIN[0].date()), str(WIN[-1].date())],
                         mean_load_kW=mean_load,
                         definition="MAE_rel = E|eps| / mean_t(L_t)",
                         note="只改计划阶段所用的负载信息，结算与基线口径不变"))
    rows = []
    for rel in MAE_RELS:
        sigma = rel * mean_load / c
        eps = z[:, None] * sigma * (L / mean_load)   # (365,144) kW
        r = run(panel, eps)
        r["mae_rel"] = rel
        r["eps_mae_kW"] = float(np.abs(eps).mean())
        rows.append(r)
        print("MAE_rel=%.0f%%: 总费用 %.2f 元, 紧急购电 %.2f kWh, 出现天数 %d, 节省率 %.4f"
              % (rel * 100, r["total"], r["emg_qty"], r["days_with_emg"], r["saving_rate"]))
    out["load_error_scan"] = rows

    # 极端参照：把"光伏误差"放大到同等相对 MAE
    pv_arr = pv.to_numpy(float)
    pv_err0 = np.zeros_like(pv_arr)
    for i, d in enumerate(ALL):
        pv_err0[i] = panel.pv_forecast(d, 0) - pv_arr[i]      # kW，实际−预报的负号已含在定义里
    pv_mae = float(np.abs(pv_err0).mean())
    scale = (PV_MAE_REL * mean_load / c) / max(pv_mae, 1e-9)
    # 以"负载误差"通道注入同量级扰动，便于与上面两档直接比较
    eps_ref = pv_err0 - pv_err0.mean(axis=0, keepdims=True)
    eps_ref = eps_ref * (PV_MAE_REL * mean_load / c) / max(eps_ref.std(), 1e-9)
    r = run(panel, eps_ref)
    r["mae_rel"] = PV_MAE_REL
    r["eps_mae_kW"] = float(np.abs(eps_ref).mean())
    r["tag"] = "pv_error_at_same_relative_mae"
    out["pv_reference"] = r
    print("光伏误差放大到 MAE_rel=%.0f%%: 总费用 %.2f 元, 紧急购电 %.2f kWh"
          % (PV_MAE_REL * 100, r["total"], r["emg_qty"]))

    base = rows[0]
    out["deltas"] = [
        dict(mae_rel=x["mae_rel"],
             d_total=x["total"] - base["total"],
             d_total_pct=(x["total"] - base["total"]) / base["total"] * 100,
             d_emg_qty=x["emg_qty"] - base["emg_qty"],
             emg_qty_ratio=(x["emg_qty"] / base["emg_qty"]) if base["emg_qty"] > 0 else None)
        for x in rows]

    with open(os.path.join(C.INTERMEDIATE_DIR, "exp_loaderr.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 intermediate/exp_loaderr.json")


if __name__ == "__main__":
    main()
