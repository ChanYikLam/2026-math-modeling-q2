# -*- coding: utf-8 -*-
"""
问题 4 对照实验：盒式不确定集的分布鲁棒（DRO）购电策略
======================================================
目的：为问题 4 的"分时骨架 + 随机残差情景式随机规划"主方案提供一个**分布鲁棒对照**，
      定量回答"保守度换最坏情形保障"值不值。

运行：cd C题论文/code && python3 exp_dro.py                 （全量 7 个 k，结果窗 334 天）
      cd C题论文/code && python3 exp_dro.py --quick         （k=0/1/2/3 且 92 天子窗，快速自检）
输出：intermediate/exp_dro.json

----------------------------------------------------------------------------
一、不确定集（盒式，逐槽笛卡尔积）
----------------------------------------------------------------------------
电价：    p_t  in  [ p^skel_t - k sigma_t ,  p^skel_t + k sigma_t ]
          p^skel = 附件 1 的分时骨架电价剖面（= 附件 4 逐时段均值，实测最大偏差 5.15e-5）；
          sigma_t  = 附件 4 电价在槽位 t 的**样本标准差**（全样本 365 天，ddof=1）。
净需求：  nd_t in  [ nd^fcst_t - k sigma^nd_t ,  nd^fcst_t + k sigma^nd_t ]
          nd^fcst_t = (L_t - PV^fcst_t) * DT（0:00 发布的附件 3 预报 + 当日负载，口径同问题 2/4）；
          sigma^nd_t = 预报残差 r_{d,t} = (PV^fcst_{d,t} - PV^act_{d,t}) * DT 的样本标准差
                       （= nd^act_t - nd^fcst_t，全样本 365 天，ddof=1）。
盒 Δ(k) = X_t [p^lo_t, p^hi_t] x X_t [nd^lo_t, nd^hi_t]：**各槽独立**（笛卡尔积，不含时序相关性），
k 同时是"电价带宽"和"安全边际"的倍数，k=0 时集合退化为单点（骨架价 + 预报净需求）。

----------------------------------------------------------------------------
二、为什么最坏情形一定在盒的**顶点** (p^hi, nd^hi) 取得（推导）
----------------------------------------------------------------------------
给定第一阶段决策 (E, c, d)，第二阶段（紧急购电）成本为
    Phi(p, nd) = sum_t [ p_t E_t + 5 p_t * max(0, nd_t + c_t - d_t - E_t) ] .
它对 (p, nd) **逐槽可分离**：Phi = sum_t phi_t(p_t, nd_t)，且每个 phi_t 对两个自变量**单调不减**：
    d phi_t / d p_t  = E_t + 5 max(0, nd_t + c_t - d_t - E_t) >= 0   (E_t >= 0)
    d phi_t / d nd_t = 5 p_t * 1{ nd_t + c_t - d_t - E_t > 0 }      >= 0   (p_t > 0)
单调 ⇒ 每个 phi_t 在矩形 [p^lo,p^hi] x [nd^lo,nd^hi] 上的最大值在右上端点取得；
可分离 ⇒ max_{Delta} Phi = sum_t max phi_t = sum_t phi_t(p^hi_t, nd^hi_t)。
两个上端点同时可达到（(p^hi, nd^hi) 本身就是盒 Δ 的一个顶点，盒是笛卡尔积，
不存在"取价格上端就必须取净需求下端"的互斥），故
    最坏情形恰好在该顶点取得，且价格与净需求的最坏取值相互独立、可同时发生。
（这也说明盒式模型的已知弱点：它把价格-净负荷的联合结构当成"同时最坏"，见 C4 相关性一节。）

于是鲁棒对应问题 (RC) 与 q4.solve_vol 同结构、同约束，只把"分情景期望"换成"盒内最坏情形"：
    (RC)  min_{E,c,d,z}  sum_t p^hi_t E_t + 5 sum_t p^hi_t z_t
          s.t.  z_t >= nd^hi_t + c_t - d_t - E_t ,  z_t >= 0            (最坏情形缺口 -> 紧急购电)
                E_t - c_t + d_t >= nd^fcst_t                            (计划满足预报口径需求)
                S_t = S_{t-1} + eta_c c_t - d_t / eta_d ,  S_min <= S_t <= S_max
                0 <= c_t, d_t <= E_MAX = P_MAX*DT ,   E_t >= 0
                S_144 >= S_0                                            (A6 终端不亏空)

**结构性推论（脚本输出中逐日核验）**：E_t 与 z_t 在同一槽位上的目标系数之比恒为 5:1
（同为 p^hi_t），故只要 z_t > 0 就存在"提前多买 1 kWh 省 5-1=4 元"的下降方向，
最优解必取 z_t = 0、E_t = max(0, nd^hi_t + c_t - d_t)：
    盒式 DRO 的解 = "按最坏情形参数做确定性优化"，逐槽安全边际恰为 k*sigma^nd_t（100% 覆盖最坏净需求）。
对照：情景式随机规划（问题 4 主方案）的边际由报童临界分位数 4/5 决定（约 80% 分位），
两者结构不同 —— 这正是本对照实验要量化的"保守度溢价"。
注意：主模型与 q2/q4 一致地**不对 E_t 设上限**（题面未给购电功率上限），
故 k 较大时计划购电量会被推到 nd^hi，单槽购电量可能超出现实并网容量；
脚本因此额外给出"E_t <= E_MAX"的 cap 变体作灵敏度（clear labeled，非主口径）。
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
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import config as C
import lp_engine as LP

# --------------------------------------------------------------------------
# 实验设置
# --------------------------------------------------------------------------
KS_DEFAULT = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
K_REF = 3.0                 # 参照不确定集 κ（用于"固定最坏情形"评估 W_fix）
K_REF2 = 2.0
ALPHA = 0.95                # CVaR 置信水平（C-11）
KS_CAP = [1.0, 3.0]         # E 上限灵敏度变体
E_CAP_SENS = 1.5 * C.E_MAX   # 1250 kWh/槽（≈7.5 MW）；E_MAX 本身不可行（单槽净需求峰值 1212.7 > 833.3）

DATE_DECIDE = pd.date_range(C.WARMUP_START, C.WIN_END, freq="D")   # 365 天（1 月预热）
DATE_SCORE = pd.date_range(C.WIN_START, C.WIN_END, freq="D")       # 334 天（结果窗）
SUB_START, SUB_END = "2025-03-01", "2025-05-31"                    # 92 天子窗（--quick）


# --------------------------------------------------------------------------
# 数据与不确定集参数
# --------------------------------------------------------------------------
def sigma_price(p4: pd.DataFrame) -> np.ndarray:
    """sigma_t：附件 4 电价在槽位 t 的样本标准差（全样本 365 天，ddof=1）。"""
    return p4.to_numpy(float).std(0, ddof=1)


def build_day_data(panel: "C.Panel", days):
    """预计算逐日：预报净需求、实际净需求、实际电价、无储能基线（所有 k 共用）。"""
    load = panel.L()
    pv = panel.PV()
    p4 = panel.P(volatile=True)
    n = len(days)
    NDF = np.zeros((n, C.N_SLOT))     # 预报口径净需求 (L - PV^fcst)*DT
    NDA = np.zeros((n, C.N_SLOT))     # 实际净需求 (L - PV^act)*DT
    PACT = np.zeros((n, C.N_SLOT))    # 附件 4 实际电价
    BASE = np.zeros(n)                # C-01 无储能基线购电费
    for i, d in enumerate(days):
        L = load.loc[d].to_numpy(float)
        PVa = pv.loc[d].to_numpy(float)
        fc = panel.pv_forecast(d, 0)
        NDF[i] = (L - fc) * C.DT
        NDA[i] = (L - PVa) * C.DT
        PACT[i] = p4.loc[d].to_numpy(float)
        BASE[i] = C.baseline_cost(PACT[i], L, PVa)
    RES = NDA - NDF                   # = (PV^fcst - PV^act)*DT，问题 4 的净需求预报残差
    SND = RES.std(0, ddof=1)          # sigma^nd_t
    return dict(days=list(days), NDF=NDF, NDA=NDA, PACT=PACT, BASE=BASE, RES=RES, SND=SND)


# --------------------------------------------------------------------------
# 盒式 DRO 的单日 LP（独立组装，不改 q4.py）
# --------------------------------------------------------------------------
def solve_dro_day(nd_fcst, nd_hi, p_hi, s_init, term_floor=None, e_cap=None):
    """(RC) 单日盒式鲁棒对应 LP。变量 [E(n), c(n), d(n), S(n), z(n)]，全部 kWh。
        z_t 为"最坏情形下的紧急购电量"，目标系数 5*p^hi_t。"""
    n = C.N_SLOT
    iE, iC, iD, iS, iZ = 0, n, 2 * n, 3 * n, 4 * n
    nv = 5 * n

    c_obj = np.zeros(nv)
    c_obj[iE:iE + n] = p_hi
    c_obj[iZ:iZ + n] = C.EMERGENCY_MULT * p_hi

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    # (1) 计划可行性：E - c + d >= nd^fcst   <=>  -E + c - d <= -nd^fcst
    for t in range(n):
        for a, b, v in ((r, iE + t, -1.0), (r, iC + t, 1.0), (r, iD + t, -1.0)):
            rows.append(a); cols.append(b); vals.append(v)
        rhs.append(-nd_fcst[t]); r += 1
    # (2) 最坏情形缺口：z >= nd^hi + c - d - E  <=>  -E + c - d - z <= -nd^hi
    for t in range(n):
        for a, b, v in ((r, iE + t, -1.0), (r, iC + t, 1.0), (r, iD + t, -1.0),
                        (r, iZ + t, -1.0)):
            rows.append(a); cols.append(b); vals.append(v)
        rhs.append(-nd_hi[t]); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    b_ub = np.array(rhs)

    # (3) SOC 递推
    rows2, cols2, vals2, rhs2 = [], [], [], []
    for t in range(n):
        for b, v in ((iS + t, 1.0), (iC + t, -C.ETA_C), (iD + t, 1.0 / C.ETA_D)):
            rows2.append(t); cols2.append(b); vals2.append(v)
        if t > 0:
            rows2.append(t); cols2.append(iS + t - 1); vals2.append(-1.0)
        rhs2.append(0.0 if t > 0 else s_init)
    A_eq = coo_matrix((vals2, (rows2, cols2)), shape=(n, nv)).tocsr()
    b_eq = np.array(rhs2)

    e_up = None if e_cap is None else float(e_cap)
    bounds = [(0.0, e_up)] * n + [(0.0, C.E_MAX)] * n + [(0.0, C.E_MAX)] * n + \
             [(C.S_MIN, C.S_MAX)] * n + [(0.0, None)] * n
    if term_floor is not None:      # A6：S_144 >= S_0
        bounds[iS + n - 1] = (max(float(term_floor), C.S_MIN), C.S_MAX)

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("DRO LP 不可行：" + res.message)
    x = res.x
    return dict(E=x[iE:iE + n].copy(), c=x[iC:iC + n].copy(), d=x[iD:iD + n].copy(),
                S=x[iS:iS + n].copy(), z=x[iZ:iZ + n].copy(), obj=float(res.fun),
                s_init=float(s_init), nd_fcst=nd_fcst, nd_hi=nd_hi, p_hi=p_hi)


# --------------------------------------------------------------------------
# 风险度量与自检
# --------------------------------------------------------------------------
def cvar_ru(x, alpha=ALPHA):
    """Rockafellar–Uryasev 估计量：CVaR = eta* + E[(X-eta*)^+]/(1-alpha)，eta*=样本 alpha 分位。"""
    x = np.asarray(x, float)
    eta = float(np.quantile(x, alpha))
    cvar = float(eta + np.mean(np.maximum(x - eta, 0.0)) / (1.0 - alpha))
    m = max(1, int(np.ceil((1.0 - alpha) * len(x))))       # 尾部均值（对照口径）
    tail = float(np.sort(x)[-m:].mean())
    return cvar, eta, tail, m


def audit_day(sol, nd_fcst, nd_hi):
    """与 lp_engine.check_solution 相同的方式 + DRO 专属项。violation 取正部（<=0 即合规）。"""
    n = C.N_SLOT
    E, c, d, S, z = sol["E"], sol["c"], sol["d"], sol["S"], sol["z"]
    rep = {}
    rep["R1_plan_violation"] = float(np.max(nd_fcst + c - d - E))
    rep["R2_power_violation"] = float(max(np.max(c), np.max(d)) - C.E_MAX)
    rep["R3_soc_bounds"] = float(max(C.S_MIN - S.min(), S.max() - C.S_MAX))
    prev = np.concatenate([[sol["s_init"]], S[:-1]])
    rep["R4_soc_recursion"] = float(np.max(np.abs(S - (prev + C.ETA_C * c - d / C.ETA_D))))
    rep["R5_simultaneous"] = int(np.sum((c > 1e-6) & (d > 1e-6)))
    rep["R6_E_nonneg"] = float(-min(0.0, float(E.min())))
    rep["R7_terminal"] = float(sol["s_init"] - S[-1])                       # <=0 合规
    rep["R8_z_consistency"] = float(np.max(np.abs(
        z - np.maximum(nd_hi + c - d - E, 0.0))))
    rep["R9_z_nonneg"] = float(-min(0.0, float(z.min())))
    rep["R10_worstcase_coverage"] = float(np.max(nd_hi + c - d - E))        # <=0 即 z 恒为 0
    # 直接复用 lp_engine.check_solution（同一套检查，独立于本次 LP 组装）
    for k_, v in LP.check_solution(sol, nd_fcst).items():
        rep["LP_" + k_] = float(v)
    return rep


def agg_audit(store, rep):
    for k_, v in rep.items():
        if k_ not in store:
            store[k_] = float(v)
        elif k_ in ("R5_simultaneous",):
            store[k_] = float(store[k_]) + float(v)
        else:
            store[k_] = max(float(store[k_]), float(v))
    return store


# --------------------------------------------------------------------------
# 滚动回测
# --------------------------------------------------------------------------
def roll(data, k, sp, skel, ks_ref=(K_REF2, K_REF), e_cap=None, audit=False):
    """在 365 天（含 1 月预热）上滚动：每天 0:00 用当日报制定计划，当日实际值结算。
       只有结果窗 334 天计入指标；跨日 SOC 连续递推；每日 S_144 >= S_0（A6）。"""
    NDF, NDA, PACT, BASE = data["NDF"], data["NDA"], data["PACT"], data["BASE"]
    SND, days = data["SND"], data["days"]
    score_idx = [i for i, d in enumerate(days) if d in set(DATE_SCORE)]

    p_hi = skel + k * sp
    p_hi_ref = {kk: skel + kk * sp for kk in ks_ref}
    nd_hi_all = NDF + k * SND[None, :]
    nd_hi_ref = {kk: NDF + kk * SND[None, :] for kk in ks_ref}

    s_init = C.S_INIT
    recs = []
    aud = {}
    t_solve = 0.0
    for i, d in enumerate(days):
        sol = solve_dro_day(NDF[i], nd_hi_all[i], p_hi, s_init, term_floor=s_init,
                            e_cap=e_cap)
        if audit:
            agg_audit(aud, audit_day(sol, NDF[i], nd_hi_all[i]))
        E, c, dd, S = sol["E"], sol["c"], sol["d"], sol["S"]
        g = np.maximum(NDA[i] + c - dd - E, 0.0)                 # 实际紧急购电量
        pact = PACT[i]
        cost_buy = float(np.sum(pact * E))
        cost_emg = float(C.EMERGENCY_MULT * np.sum(pact * g))
        surplus = float(np.sum(np.maximum(E + dd - c - NDA[i], 0.0)))   # C-06 超购/盈余处置
        w_own = float(np.sum(p_hi * E + C.EMERGENCY_MULT * p_hi * np.maximum(
            nd_hi_all[i] + c - dd - E, 0.0)))
        w_fix = {}
        for kk in ks_ref:
            w_fix[kk] = float(np.sum(p_hi_ref[kk] * E + C.EMERGENCY_MULT * p_hi_ref[kk]
                                     * np.maximum(nd_hi_ref[kk][i] + c - dd - E, 0.0)))
        if i in score_idx:
            recs.append(dict(date=d, plan_energy=float(E.sum()),
                             actual_buy=float(E.sum() + g.sum()), emg_qty=float(g.sum()),
                             surplus_qty=surplus, cost_buy=cost_buy, cost_emg=cost_emg,
                             total_cost=cost_buy + cost_emg, base_cost=float(BASE[i]),
                             W_own=w_own, W_fix=w_fix,
                             n_short=int(np.sum(g > 1e-6)),
                             c_sum=float(c.sum()), d_sum=float(dd.sum()),
                             pw_short=float(np.sum(pact[g > 1e-6])), pw_all=float(np.sum(pact)),
                             demand_after_batt=float(np.sum(np.maximum(NDA[i] + c - dd, 0.0))),
                             s_init=s_init, s_end=float(S[-1]), z_sum=float(sol["z"].sum()),
                             gap_hi=float(np.mean(E + dd - c - nd_hi_all[i]))))
        if i == score_idx[0]:
            s0_feb1 = s_init
        s_init = float(S[-1])
    return dict(k=k, recs=recs, audit=aud, s0_feb1=s0_feb1, s_end=s_init, t_solve=t_solve)


def summarize(roll_out, ks_ref=(K_REF2, K_REF), e_cap=None):
    recs = roll_out["recs"]
    tot = lambda key: float(sum(r[key] for r in recs))
    base = tot("base_cost")
    total = tot("total_cost")
    cost_daily = np.array([r["total_cost"] for r in recs], float)
    cvar, var, tail, m = cvar_ru(cost_daily)
    dem = tot("demand_after_batt")
    nshort = sum(r["n_short"] for r in recs)
    worst = int(np.argmax(cost_daily))
    out = dict(
        k=roll_out["k"],
        n_days=len(recs),
        total_cost=total, plan_cost=tot("cost_buy"), emg_cost=tot("cost_emg"),
        emg_qty=tot("emg_qty"), base_cost=base,
        saving=(base - total) / base,
        plan_energy=tot("plan_energy"), actual_buy=tot("actual_buy"),
        surplus_qty=tot("surplus_qty"),
        service_rate=1.0 - tot("emg_qty") / dem if dem > 0 else float("nan"),
        slot_shortage_freq=nshort / (len(recs) * C.N_SLOT),
        cvar95=cvar, var95=var, cvar95_tailmean=tail, n_tail=m,
        mean_daily_cost=float(cost_daily.mean()), worst_day_cost=float(cost_daily[worst]),
        worst_day=str(pd.Timestamp(recs[worst]["date"]).date()),
        mean_daily_emg_qty=tot("emg_qty") / len(recs),
        mean_daily_plan_energy=tot("plan_energy") / len(recs),
        charge_energy=tot("c_sum"), discharge_energy=tot("d_sum"),
        price_weighted_shortage_freq=tot("pw_short") / tot("pw_all"),
        W_own=tot("W_own"), z_total=tot("z_sum"),
        s0_feb1=roll_out["s0_feb1"], s_end=roll_out["s_end"],
    )
    for kk in ks_ref:
        out["W_fix_%g" % kk] = float(sum(r["W_fix"][kk] for r in recs))
    if e_cap is not None:
        out["e_cap"] = float(e_cap)
    return out


def daily_of(roll_out, ks_ref=(K_REF2, K_REF)):
    out = []
    for r in roll_out["recs"]:
        d = dict(date=str(pd.Timestamp(r["date"]).date()), plan_energy=r["plan_energy"],
                 actual_buy=r["actual_buy"], emg_qty=r["emg_qty"],
                 surplus_qty=r["surplus_qty"], cost_plan=r["cost_buy"],
                 cost_emg=r["cost_emg"], total_cost=r["total_cost"],
                 base_cost=r["base_cost"], W_own=r["W_own"], s_init=r["s_init"],
                 s_end=r["s_end"])
        for kk in ks_ref:
            d["W_fix_%g" % kk] = r["W_fix"][kk]
        out.append(d)
    return out


# --------------------------------------------------------------------------
# 残差/价格分布诊断（解释保守度—费用曲线的形状）
# --------------------------------------------------------------------------
def residual_diagnostics(RES, SND, sp, skel):
    """预报残差与电价的分布特征。

    要点：盒式 DRO 用"统一倍数 k"乘 σ^nd 代替"逐槽 4/5 分位数边际"，
    而 (a) 残差分布非正态、逐槽 q80/σ 不同，(b) 高价时段残差更大，
    故单一 k 必然存在跨槽误配（misallocation）—— 这是实证最优 k* 偏离
    逐槽报童值 q80/σ 的结构性原因。
    """
    mask = SND > 1e-9
    q = {p: np.quantile(RES, p, axis=0) for p in (0.1, 0.5, 0.8, 0.9, 0.95)}
    return dict(
        n_slots_sigma_nd_zero=int((~mask).sum()),
        n_slots=int(C.N_SLOT),
        bias_per_slot_mean_kwh=float(RES.mean(0)[mask].mean()),
        bias_over_sigma=float((RES.mean(0)[mask] / SND[mask]).mean()),
        q10_over_sigma=float((q[0.1][mask] / SND[mask]).mean()),
        q50_over_sigma=float((q[0.5][mask] / SND[mask]).mean()),
        q80_over_sigma=float((q[0.8][mask] / SND[mask]).mean()),
        q90_over_sigma=float((q[0.9][mask] / SND[mask]).mean()),
        q95_over_sigma=float((q[0.95][mask] / SND[mask]).mean()),
        newsvendor_k_equivalent=float((q[0.8][mask] / SND[mask]).mean()),
        pooled_mean_kwh=float(RES.mean()), pooled_sd_kwh=float(RES.std()),
        pooled_skew=float(((RES - RES.mean()) ** 3).mean() / RES.std() ** 3),
        frac_resid_positive=float((RES > 0).mean()),
        sigma_price_over_price=float((sp / skel).mean()),
        note="报童临界分位数 4/5 的逐槽等效边际倍数 = q80/σ^nd（上表均值）；"
             "盒式 DRO 以 k·σ^nd 统一加成、且以预报为中心（不含残差偏差），"
             "故 k 的最优值由电价加权的缺货频率=1/5 决定（见 results.price_weighted_shortage_freq），"
             "不等于逐槽 q80/σ^nd 的平均值。",
    )


# --------------------------------------------------------------------------
# 保守度—费用曲线
# --------------------------------------------------------------------------
def tradeoff(curve, k_ref=K_REF):
    key = "W_fix_%g" % k_ref
    c0 = curve[0]
    rows = []
    for c in curve:
        dW = c0[key] - c[key]
        dW_rel = dW / c0[key] * 100.0
        dR = c["total_cost"] - c0["total_cost"]
        dR_rel = dR / c0["total_cost"] * 100.0
        rows.append(dict(
            k=c["k"], dW_fix=dW, dW_fix_rel_pct=dW_rel,
            dR=dR, dR_rel_pct=dR_rel,
            yuan_per_1pct_W=(dR / dW_rel) if abs(dW_rel) > 1e-9 else None,
            pct_of_base_per_1pct_W=((dR / dW_rel) / c0["base_cost"] * 100.0)
            if abs(dW_rel) > 1e-9 else None,
        ))
    best = min(curve, key=lambda c: c["total_cost"])
    bestW = min(curve, key=lambda c: c[key])
    last = curve[-1]
    rowbest = [r for r in rows if abs(r["k"] - best["k"]) < 1e-12][0]
    rowlast = rows[-1]
    base = c0["base_cost"]
    prem_dR = rowlast["dR"] - rowbest["dR"]
    prem_dW = rowbest["dW_fix"] - rowlast["dW_fix"]
    prem_dW_rel = rowlast["dW_fix_rel_pct"] - rowbest["dW_fix_rel_pct"]
    prem = dict(
        k_from=best["k"], k_to=last["k"],
        dR=prem_dR, dR_rel_pct=prem_dR / c0["total_cost"] * 100.0,
        dR_pct_of_base=prem_dR / base * 100.0,
        dW_fix=prem_dW, dW_fix_rel_pct=prem_dW_rel,
        yuan_per_1pct_W=(prem_dR / prem_dW_rel) if abs(prem_dW_rel) > 1e-9 else None,
        pct_of_base_per_1pct_W=((prem_dR / prem_dW_rel) / base * 100.0)
        if abs(prem_dW_rel) > 1e-9 else None,
        note="从实证最优 k* 继续加大保守度到 k_max：最坏情形费用每降 1% 需要多付的回测(期望)费用",
    )
    pareto = dict(
        k_from=c0["k"], k_to=best["k"],
        dR=rowbest["dR"], dR_rel_pct=rowbest["dR_rel_pct"],
        dW_fix=rowbest["dW_fix"], dW_fix_rel_pct=rowbest["dW_fix_rel_pct"],
        note="k=0（零边际确定性计划）-> k*：最坏情形费用与实际费用同时改善（严格 Pareto 改善）",
    )
    kc = min(curve, key=lambda c: c["cvar95"])
    kw = min(curve, key=lambda c: c["worst_day_cost"])
    return dict(k_ref=k_ref, W_fix_k0=c0[key], rows=rows,
                k_best_realized=best["k"], k_best_realized_total=best["total_cost"],
                k_best_W=bestW["k"], k_best_W_value=bestW[key],
                k_best_cvar=kc["k"], k_best_cvar_value=kc["cvar95"],
                k_best_worstday=kw["k"], k_best_worstday_value=kw["worst_day_cost"],
                pareto=pareto, premium=prem)


def marginal_analysis(curve):
    """逐步加大保守度的边际账：多买的计划电量 vs 被替代的紧急购电量。"""
    rows = []
    for a, b in zip(curve[:-1], curve[1:]):
        dE = b["plan_energy"] - a["plan_energy"]
        dPC = b["plan_cost"] - a["plan_cost"]
        dG = a["emg_qty"] - b["emg_qty"]
        dEC = a["emg_cost"] - b["emg_cost"]
        up = (dPC / dE) if dE > 1e-9 else None
        ue = (dEC / dG) if dG > 1e-9 else None
        rows.append(dict(k_from=a["k"], k_to=b["k"],
                         d_plan_energy=dE, d_plan_cost=dPC, unit_plan_price=up,
                         d_emg_qty=dG, d_emg_cost=dEC, unit_emg_price=ue,
                         price_mismatch=(ue / up) if (up and ue) else None,
                         benefit_cost_ratio=(dEC / dPC) if dPC > 1e-9 else None,
                         net=dEC - dPC))
    cross = next((r for r in rows if r["benefit_cost_ratio"] is not None
                  and r["benefit_cost_ratio"] < 1.0), None)
    return dict(steps=rows, crossover_step=cross,
                note="边际判定：k→k+Δk 时，新增计划电量的单价（元/kWh，unit_plan_price）"
                     "对被替代的紧急购电量单价（含 5 倍，unit_emg_price）之比"
                     "benefit_cost_ratio = price_mismatch x (d_emg_qty/d_plan_energy)："
                     ">1 说明继续加大保守度仍省钱，<1 说明已开始亏（交叉点在 1.5->2.0 这一步）。"
                     "实测 price_mismatch≈6.2 = 5 x 1.25：边际电量买在 σ^nd 大的时段（单价≈0.70 元/kWh），"
                     "而被替代的紧急购电发生在均价更高的时段（单价≈4.4 元/kWh，其中 5 倍为紧急电价倍数）。"
                     "于是临界条件变为：边际替代率只需达到 1/6.2 ≈ 16% 即值得；"
                     "k=1.5 时的实际替代率（电价加权缺货频率）为 11.3%，k=1.0 时为 15.1%，"
                     "故实证最优 k* 落在 1.5 附近；"
                     "逐槽报童临界分位数 4/5 的等效边际 q80/σ^nd≈0.64 只在价格中性、逐槽可调时成立，"
                     "统一倍数 k 忽略了这一跨槽价格错配（misallocation）。")


def make_figure(out, path):
    """保守度—费用曲线（可选图件）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({                     # 与 plot_figures.py 同一中文字体栈
        "font.family": "sans-serif",
        "font.sans-serif": ["Songti SC", "Heiti SC", "PingFang SC", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
        "legend.fontsize": 7.4, "savefig.bbox": "tight",
    })
    ks = [c["k"] for c in out["curve"]["rows"]]
    res = [out["results"]["k=%.1f" % k] for k in ks]
    tot = [r["total_cost"] / 1e4 for r in res]
    plan = [r["plan_cost"] / 1e4 for r in res]
    emg = [r["emg_cost"] / 1e4 for r in res]
    wfix = [r["W_fix_%g" % K_REF] / 1e4 for r in res]
    Wown = [r["W_own"] / 1e4 for r in res]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax[0].plot(ks, tot, "o-", color="#c0392b", label="实际回测总费用")
    ax[0].plot(ks, plan, "s--", color="#2c6fb5", label="计划购电费用")
    ax[0].plot(ks, emg, "^--", color="#e08a1e", label="紧急购电费用")
    kb = out["curve"]["k_best_realized"]
    ax[0].axvline(kb, color="gray", ls=":", lw=1)
    ax[0].annotate("k*=%.1f" % kb, (kb, max(tot)), color="gray",
                   fontsize=8, ha="left", va="top")
    ax[0].set_xlabel(r"安全边际倍数 $k$（边际 $=k\sigma^{nd}_t$）")
    ax[0].set_ylabel("费用 / 万元")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=0.3)
    ax[1].plot(ks, wfix, "o-", color="#6c3483",
               label=r"固定最坏情形 $\kappa=3$ 下的费用")
    ax[1].plot(ks, Wown, "s--", color="#7f8c8d", label=r"自身盒 $k$ 内的最坏情形（DRO 目标）")
    ax[1].set_xlabel("安全边际倍数 $k$")
    ax[1].set_ylabel("最坏情形费用 / 万元")
    ax[1].legend(fontsize=7.5)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 主程序
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default=",".join(str(x) for x in KS_DEFAULT))
    ap.add_argument("--quick", action="store_true", help="92 天子窗 + 4 个 k")
    ap.add_argument("--no-cap", action="store_true", help="跳过 E<=E_MAX 灵敏度变体")
    ap.add_argument("--out", default=os.path.join(C.INTERMEDIATE_DIR, "exp_dro.json"))
    ap.add_argument("--fig", action="store_true", help="同时输出 figures/fig11_dro_conservatism.pdf")
    args = ap.parse_args()

    t0 = time.time()
    ks = [float(x) for x in args.ks.split(",")]
    if args.quick:
        ks = [0.0, 1.0, 2.0, 3.0]

    panel = C.Panel(DATE_DECIDE)
    p4 = panel.P(volatile=True)
    skel = C.read_attachment1()["price"].to_numpy(float)
    sp = sigma_price(p4)
    all_days = list(DATE_DECIDE)
    data = build_day_data(panel, all_days)
    snd = data["SND"]
    if args.quick:
        global DATE_SCORE
        DATE_SCORE = pd.date_range(SUB_START, SUB_END, freq="D")
        print("QUICK 模式：评分窗 %s..%s（%d 天）" % (SUB_START, SUB_END, len(DATE_SCORE)))

    base_total = float(data["BASE"][[i for i, d in enumerate(all_days)
                                     if d in set(DATE_SCORE)]].sum())
    print("=" * 78)
    print("DRO 对照实验：决策窗 %s..%s（%d 天，含 1 月预热），结果窗 %s..%s（%d 天）"
          % (C.WARMUP_START, C.WIN_END, len(all_days), DATE_SCORE[0].date(),
             DATE_SCORE[-1].date(), len(DATE_SCORE)))
    print("sigma^price: mean %.4f  [%.4f, %.4f] 元/kWh" % (sp.mean(), sp.min(), sp.max()))
    print("sigma^nd   : mean %.2f  [%.2f, %.2f] kWh" % (snd.mean(), snd.min(), snd.max()))
    print("无储能基线（结果窗）: %.2f 元" % base_total)
    print("=" * 78)

    results, dailies, audits = {}, {}, {}
    for k in ks:
        t1 = time.time()
        ro = roll(data, k, sp, skel, audit=(abs(k - 1.0) < 1e-12))
        sm = summarize(ro)
        sm["base_cost_meta"] = base_total
        results["k=%.1f" % k] = sm
        dailies["k=%.1f" % k] = daily_of(ro)
        if ro["audit"]:
            audits["k=%.1f" % k] = ro["audit"]
        print("k=%.1f 总费用 %12.2f 元 | 计划 %12.2f 紧急 %10.2f | 紧急量 %10.1f kWh | "
              "缺货频率 %5.2f%% | CVaR95 %9.2f (%.1fs)"
              % (k, sm["total_cost"], sm["plan_cost"], sm["emg_cost"], sm["emg_qty"],
                 sm["slot_shortage_freq"] * 100, sm["cvar95"], time.time() - t1))

    curve = [results["k=%.1f" % k] for k in ks]
    tro = tradeoff(curve)

    # ---- E 上限灵敏度（非主口径）----
    cap_res = {}
    if not args.no_cap:
        for k in KS_CAP:
            if k not in ks:
                continue
            ro = roll(data, k, sp, skel, e_cap=E_CAP_SENS)
            cap_res["k=%.1f" % k] = summarize(ro, e_cap=E_CAP_SENS)
            print("  [cap 变体 E<=%.2f kWh/槽] k=%.1f 总费用 %.2f 元（主口径 %.2f 元）"
                  % (E_CAP_SENS, k, cap_res["k=%.1f" % k]["total_cost"],
                     results["k=%.1f" % k]["total_cost"]))

    # ---- 与问题 4 主方案（情景式随机规划）对照 ----
    ref = {}
    try:
        with open(os.path.join(C.INTERMEDIATE_DIR, "q4_2.json"), encoding="utf-8") as f:
            j = json.load(open(os.path.join(C.INTERMEDIATE_DIR, "q4_2.json", encoding="utf-8")))
        m = j["meta"]
        ref = dict(source="intermediate/q4_2.json", model="情景式两阶段随机规划（K=60 联合情景）",
                   total_cost=m["total"], base_cost=m["base"], emg_qty=m["emg_qty"],
                   saving=(m["base"] - m["total"]) / m["base"],
                   comparable=bool(not args.quick and len(DATE_SCORE) == 334))
        best = min(curve, key=lambda c: c["total_cost"])
        ref["best_dro_k"] = best["k"]
        ref["best_dro_total"] = best["total_cost"]
        ref["diff_vs_best_dro"] = best["total_cost"] - m["total"]
        ref["diff_pct"] = (best["total_cost"] - m["total"]) / m["total"] * 100
        # 同计划购电量水平下的对照（主方案的边际"买得更准"）
        qt = 0.0
        for r in j["daily"]:
            qt += r["plan_energy"]
        near = min(curve, key=lambda c: abs(c["plan_energy"] - qt))
        ref["matched_plan_energy"] = dict(
            q4_2_plan_energy=qt, dro_k=near["k"], dro_plan_energy=near["plan_energy"],
            dro_emg_qty=near["emg_qty"], q4_2_emg_qty=m["emg_qty"],
            emg_qty_ratio=near["emg_qty"] / m["emg_qty"],
            note="计划购电量相当时的紧急购电量对比：主方案把边际买在更需要的槽位")
    except Exception as e:      # pragma: no cover
        ref = dict(error=str(e))

    c0 = curve[0]
    best = [c for c in curve if abs(c["k"] - tro["k_best_realized"]) < 1e-12][0]
    last = curve[-1]
    pa, pm = tro["pareto"], tro["premium"]
    sv = lambda c: (base_total - c["total_cost"]) / base_total * 100.0
    k1 = results.get("k=1.0", best)
    concl = (
        "盒式不确定集下安全边际为 k·σ^nd（k 同时是电价带宽倍数）：在 %d 天实际轨迹上回测，"
        "总费用随 k 先降后升 —— k=0（退化为按预报确定性购电、零边际、z≡0）为 %.2f 元（节省率 %.2f%%），"
        "k=1.0 为 %.2f 元（节省率 %.2f%%），实证最优 k*=%.1f 为 %.2f 元（节省率 %.2f%%，紧急购电 %.1f kWh），"
        "而 k=%.1f（100%% 覆盖 3σ 最坏净需求）回升到 %.2f 元（节省率 %.2f%%，紧急购电仅 %.1f kWh）；"
        "CVaR95 与最坏单日费用分别在 k=%.1f（%.2f 元/日）与 k=%.1f（%.2f 元）取最小。"
        "在 κ=%.1f 的固定最坏情形盒下，各计划的最坏情形费用随 k 单调下降："
        "由 k=0 的 %.2f 元降到 k=%.1f 的 %.2f 元；"
        "其中 k=0→k*=%.1f 是严格 Pareto 改善（最坏情形费用降 %.2f%%，实际费用同时降 %.2f 元）；"
        "再从 k*=%.1f 加大到 k=%.1f，最坏情形费用再降 %.2f%%（%.2f 元），实际费用则多付 %.2f 元（+%.2f%%，"
        "占无储能基线 %.3f%%），即**为把最坏情形费用再降低 1%%，需多付 %.2f 元期望（回测）费用**"
        "（约合基线的 %.3f%%/1%%）—— 这是盒式 DRO 的保守度溢价。"
        "故盒式 DRO 在 k≈%.1f 附近即可拿到几乎全部实证收益，k 继续加大只买到最坏情形保障而不省钱；"
        "其结构（z≡0、100%% 覆盖最坏净需求）与情景式随机规划的报童边际（80%% 分位）不同。"
        % (len(DATE_SCORE), c0["total_cost"], sv(c0), k1["total_cost"], sv(k1),
           best["k"], best["total_cost"], sv(best), best["emg_qty"],
           last["k"], last["total_cost"], sv(last), last["emg_qty"],
           tro["k_best_cvar"], tro["k_best_cvar_value"],
           tro["k_best_worstday"], tro["k_best_worstday_value"],
           K_REF, tro["W_fix_k0"], last["k"], last["W_fix_%g" % K_REF], best["k"],
           pa["dW_fix_rel_pct"], -pa["dR"],
           best["k"], last["k"], pm["dW_fix_rel_pct"], -pm["dW_fix"],
           pm["dR"], pm["dR_rel_pct"], pm["dR_pct_of_base"],
           (pm["yuan_per_1pct_W"] if pm["yuan_per_1pct_W"] is not None else float("nan")),
           (pm["pct_of_base_per_1pct_W"] if pm["pct_of_base_per_1pct_W"] is not None else float("nan")),
           best["k"])
    )
    if ref.get("comparable"):
        concl += ("与情景式随机规划（问题 4 主方案，intermediate/q4_2.json）相比，"
                  "实证最优的 DRO 计划（k*=%.1f）实测费用高 %.2f 元（%.2f%%）。"
                  % (best["k"], ref["diff_vs_best_dro"], ref["diff_pct"]))

    out = dict(
        meta=dict(
            script="code/exp_dro.py", created=time.strftime("%Y-%m-%d %H:%M:%S"),
            purpose="问题 4 主方案（情景式随机规划）的分布鲁棒（盒式不确定集）对照",
            n_solve=len(ks) * len(all_days) + (len([k for k in KS_CAP if k in ks]) if not args.no_cap else 0) * len(all_days),
            runtime_s=time.time() - t0,
            quick=bool(args.quick),
        ),
        config=dict(
            ks=ks, k_ref=[K_REF2, K_REF], alpha=ALPHA, quick=bool(args.quick),
            decide_window=[str(all_days[0].date()), str(all_days[-1].date())],
            decide_days=len(all_days),
            score_window=[str(DATE_SCORE[0].date()), str(DATE_SCORE[-1].date())],
            n_days=len(DATE_SCORE),
            warmup=[C.WARMUP_START, C.WIN_START],
            sigma_price_def="附件 4 电价在槽位 t 的样本标准差（全样本 365 天，ddof=1）",
            sigma_nd_def="0:00 预报残差 (PV^fcst-PV^act)*DT 在槽位 t 的样本标准差（全样本 365 天，ddof=1）",
            sigma_price_mean=float(sp.mean()), sigma_price_min=float(sp.min()),
            sigma_price_max=float(sp.max()), sigma_nd_mean=float(snd.mean()),
            sigma_nd_min=float(snd.min()), sigma_nd_max=float(snd.max()),
            sigma_price_mean_window=float(p4.loc[DATE_SCORE].to_numpy(float).std(0, ddof=1).mean()),
            sigma_nd_mean_window=float(data["RES"][[i for i, d in enumerate(all_days)
                                                    if d in set(DATE_SCORE)]].std(0, ddof=1).mean()),
            sigma_nd_warmup_only=float(data["RES"][[i for i, d in enumerate(all_days)
                                                    if d not in set(DATE_SCORE)]].std(0, ddof=1).mean()),
            unit="kWh / 元 / 元每kWh", DT=C.DT, E_MAX=C.E_MAX, S_INIT=C.S_INIT,
            EMERGENCY_MULT=C.EMERGENCY_MULT, terminal="S_144 >= S_0 (A6)",
            baseline_total=base_total,
            e_cap_variant=(None if args.no_cap else E_CAP_SENS),
        ),
        sigma_price=sp.tolist(), sigma_nd=snd.tolist(),
        diagnostics=residual_diagnostics(data["RES"], snd, sp, skel),
        results=results,
        curve=tro,
        marginal=marginal_analysis(curve),
        sensitivity_e_cap=cap_res,
        reference_q4_2=ref,
        audit=audits,
        daily=dailies,
        conclusion=concl,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    if args.fig:
        fp = os.path.join(C.FIGURE_DIR, "fig11_dro_conservatism.pdf")
        make_figure(out, fp)
        print("已写出图件", fp)
    print("=" * 78)
    print("已写出", args.out, "(%.1f KB, 总耗时 %.1fs)"
          % (os.path.getsize(args.out) / 1024, time.time() - t0))
    print(concl)
    print("=" * 78)
    print("自检（k=1.0，334 天最大值/违反计数）:", json.dumps(audits.get("k=1.0", {}), ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
