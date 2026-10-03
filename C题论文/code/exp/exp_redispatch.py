# -*- coding: utf-8 -*-
"""假设 A12 消融：实时重调度（电池能否对已实现值作出响应）
==========================================================
论文主模型把「充放电计划与计划购电量在 0:00 一次确定、实时不允许重调度」作为**隐含假设**，
本模块把它显式化为假设 A12 并做量化消融。

评审意见（待回应）：电池被冻结时，主模型在 334 天里仍需紧急购电 172 015.82 kWh、
并处置盈余 3 106 390.84 kWh，而电池在物理上本可对**已实现**净需求作出响应。

本模块的做法（保持 0:00 计划购电量 E^plan 原样执行、其购电费 Σ p_t E^plan_t 为沉没常数）：
    min  sum_t 5 * p_t * g_t
    s.t. E^plan_t - c_t + d_t              >= nd^fcst_t      （计划可行性行，主口径保留）
         g_t >= nd^act_t + c_t - d_t - E^plan_t,  g_t >= 0
         S_t = S_{t-1} + eta_c c_t - d_t/eta_d,  S_MIN <= S_t <= S_MAX
         0 <= c_t, d_t <= E_MAX,  S_0 = s_init（q2_main.json 当日值）,  S_143 >= S_0
         s_t = PV_t*DT + E^plan_t + d_t - L_t*DT - c_t >= 0    （盈余只能处置、不能凭空消失）
其中 nd^fcst_t = (L_t - PVhat^(0)_t) * DT，PVhat^(0) 由 code/q3.forecast_slots(panel, day, 0, arange(144))
给出（与 q2 主口径同一信息结构 R3）；nd^act_t = (L_t - PV_t) * DT 用附件 2 实际值。
每天一个 LP（144 槽 × 变量 c,d,S,g），HiGHS（scipy.optimize.linprog）。

**目标退化说明**：盈余约束 s_t >= 0 与 g_t >= 0 联立等价于「t 槽 g_t = 0」，
故只要 LP 可行，紧急购电量必为 0，主目标在最优面上完全退化：电池计划 (c,d) 不唯一，
处置盈余量随顶点漂移。为此除主口径（仅紧急购电费目标）外，另给两个**字典序**对照：
  min_disposal      第二层 min sum_t s_t（在紧急购电费不变的前提下最少处置盈余）；
  min_throughput    第二层 min sum_t (c_t + d_t)（最少动用电池，处置盈余最大）。
两者给出处置盈余的可达区间端点，避免把某个 LP 顶点的取值当成"模型结论"。

输出：intermediate/exp_redispatch.json
用法：python3 exp_redispatch.py [--max-days N] [--out exp_redispatch.json]
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
from scipy.sparse import coo_matrix, hstack, vstack

try:                                     # A3 精确对照用（同一 HiGHS 后端，仅多 144 个 0-1 变量）
    from scipy.optimize import Bounds, LinearConstraint, milp
    HAS_MILP = True
except ImportError:                      # pragma: no cover
    HAS_MILP = False

import config as C
import q3 as Q3

DAYS = pd.date_range(C.WIN_START, C.WIN_END, freq="D")     # 334 天结果窗口
PLAN_JSON = "q2_main.json"                                  # E^plan 与 s_init 的来源（主口径 K=60）

# 论文已报告的主模型（冻结调度）数值，用于脚本内自检
PAPER_EMG_KWH = 172015.81877383334
PAPER_EMG_COST = 627239.2384987464
PAPER_SURPLUS_KWH = 3106390.8433584627
TOL = 1e-6


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_plan(panel, price, max_days=None):
    """读取主口径（K=60, scen=strat, info=R3）的 0:00 计划，构造逐日 LP 输入。"""
    path = os.path.join(C.INTERMEDIATE_DIR, PLAN_JSON)
    j = json.load(open(path, encoding="utf-8"))
    rows = j["daily"] if max_days is None else j["daily"][:max_days]
    load = panel.L(); pv_act = panel.PV()
    recs = []
    for r in rows:
        d = pd.Timestamp(r["date"]).normalize()
        L = load.loc[d].to_numpy(float)
        PV = pv_act.loc[d].to_numpy(float)
        slots = np.arange(C.N_SLOT)
        pv_fc = Q3.forecast_slots(panel, d, 0, slots)        # 0:00 发布的 24 h 预报（kW）
        recs.append(dict(
            date=d, L=L, PV=PV,
            E=np.array(r["E"], float),
            c_plan=np.array(r["c"], float), d_plan=np.array(r["d"], float),
            g_plan=np.array(r["g"], float),
            s_init=float(r["s_init"]),
            nd_fcst=(L - pv_fc) * C.DT,                       # 计划口径净需求 kWh
            nd_act=(L - PV) * C.DT,                           # 实际净需求 kWh
            s_ub=PV * C.DT - L * C.DT,                        # s_t = s_ub[t] + E_t + d_t - c_t
        ))
    return recs, j["meta"]


# --------------------------------------------------------------------------
# 单日实时重调度 LP
# --------------------------------------------------------------------------
def _constraints(rec, use_plan_row=True, allow_deficit=False):
    """组装单日 LP 的 A_ub/b_ub/A_eq/b_eq/bounds（变量顺序 c,d,S,g）。"""
    n = C.N_SLOT
    E, nd_fcst, nd_act, s_ub = rec["E"], rec["nd_fcst"], rec["nd_act"], rec["s_ub"]
    iC, iD, iS, iG = 0, n, 2 * n, 3 * n
    nv = 4 * n

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    # (1) 计划可行性行：E_t - c_t + d_t >= nd^fcst_t  ->  c_t - d_t <= E_t - nd^fcst_t
    if use_plan_row:
        for t in range(n):
            rows += [r, r]; cols += [iC + t, iD + t]; vals += [1.0, -1.0]
            rhs.append(E[t] - nd_fcst[t]); r += 1
    # (2) 紧急购电：g_t >= nd^act_t + c_t - d_t - E_t  ->  c_t - d_t - g_t <= E_t - nd^act_t
    for t in range(n):
        rows += [r, r, r]; cols += [iC + t, iD + t, iG + t]; vals += [1.0, -1.0, -1.0]
        rhs.append(E[t] - nd_act[t]); r += 1
    # (3) 盈余处置/消纳：s_t = s_ub_t + E_t + d_t - c_t >= 0  ->  c_t - d_t <= s_ub_t + E_t
    if not allow_deficit:
        for t in range(n):
            rows += [r, r]; cols += [iC + t, iD + t]; vals += [1.0, -1.0]
            rhs.append(s_ub[t] + E[t]); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    b_ub = np.array(rhs)

    # SOC 递推等式
    rows2, cols2, vals2, rhs2 = [], [], [], []
    for t in range(n):
        rows2 += [t, t, t]; cols2 += [iS + t, iC + t, iD + t]
        vals2 += [1.0, -C.ETA_C, 1.0 / C.ETA_D]
        if t > 0:
            rows2.append(t); cols2.append(iS + t - 1); vals2.append(-1.0)
        rhs2.append(0.0 if t > 0 else rec["s_init"])
    A_eq = coo_matrix((vals2, (rows2, cols2)), shape=(n, nv)).tocsr()
    b_eq = np.array(rhs2)

    bounds = [(0.0, C.E_MAX)] * n + [(0.0, C.E_MAX)] * n + \
             [(C.S_MIN, C.S_MAX)] * n + [(0.0, None)] * n
    # A6 终端：S_143 >= S_0（S_0 = 当日 s_init）
    bounds[iS + n - 1] = (max(rec["s_init"], C.S_MIN), C.S_MAX)
    return A_ub, b_ub, A_eq, b_eq, bounds, nv


def _obj_emergency(price):
    n = C.N_SLOT
    c_obj = np.zeros(4 * n)
    c_obj[3 * n:4 * n] = C.EMERGENCY_MULT * price
    return c_obj


def _obj_tie(tie, price):
    """第二层字典序目标（最小化正数即最优）。"""
    n = C.N_SLOT
    c_obj = np.zeros(4 * n)
    if tie == "min_disposal":       # min sum_t s_t = const + sum_t (d_t - c_t)
        c_obj[0:n] = -1.0
        c_obj[n:2 * n] = 1.0
    elif tie == "min_throughput":   # min sum_t (c_t + d_t)
        c_obj[0:n] = 1.0
        c_obj[n:2 * n] = 1.0
    else:
        raise ValueError(tie)
    return c_obj


def _solve(A_ub, b_ub, A_eq, b_eq, bounds, c_obj):
    return linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")


def solve_day_redispatch(rec, price, use_plan_row=True, tie="none", allow_deficit=False):
    """单日 LP；返回解（含 c,d,S,g,处置盈余）或 None（不可行）。

    tie="none" 时只解主目标（紧急购电费）；否则按字典序追加第二层目标，
    并以 |sum 5 p g - z1| <= TOL 的行把第一层最优值固定住。
    """
    n = C.N_SLOT
    A_ub, b_ub, A_eq, b_eq, bounds, nv = _constraints(
        rec, use_plan_row=use_plan_row, allow_deficit=allow_deficit)
    res = _solve(A_ub, b_ub, A_eq, b_eq, bounds, _obj_emergency(price))
    if not res.success:
        return None
    z1 = float(res.fun)
    if tie != "none":
        # 追加行：sum_t 5 p_t g_t <= z1 + TOL
        row_c = np.zeros(nv); row_c[3 * n:4 * n] = C.EMERGENCY_MULT * price
        A_ub = vstack([A_ub, coo_matrix(row_c.reshape(1, -1))]).tocsr()
        b_ub = np.concatenate([b_ub, [z1 + TOL]])
        res2 = _solve(A_ub, b_ub, A_eq, b_eq, bounds, _obj_tie(tie, price))
        if res2.success:
            res = res2
    x = res.x
    c = x[0:n]; d = x[n:2 * n]; S = x[2 * n:3 * n]; g = x[3 * n:4 * n]
    g = np.maximum(g, 0.0)
    surplus = rec["s_ub"] + rec["E"] + d - c      # = PV DT + E + d - L DT - c
    return dict(c=c, d=d, S=S, g=g, surplus=surplus,
                emg_kwh=float(g.sum()),
                emg_cost=float(C.EMERGENCY_MULT * np.sum(price * g)),
                surplus_kwh=float(np.maximum(surplus, 0.0).sum()),   # C-06 口径（逐槽截断）
                surplus_unclipped_kwh=float(surplus.sum()),
                deficit_kwh=float(np.maximum(-surplus, 0.0).sum()),
                c_kwh=float(c.sum()), d_kwh=float(d.sum()),
                x_kwh=float((d - c).sum()), s_end=float(S[-1]),
                n_mixed_slots=int(np.sum((c > 1e-6) & (d > 1e-6))),
                max_min_cd=float(np.minimum(c, d).max()),
                obj=float(res.fun), z1=z1, tie=tie)


def solve_day_a3(rec, price, use_plan_row=True, allow_deficit=False, tol=TOL):
    """A3 精确版单日 LP+0-1（无同时充放电）：c_t <= E_MAX z_t, d_t <= E_MAX (1 - z_t)。

    主模型 LP 对 c_t, d_t 无互斥约束，第二层 min sum_t s_t 会主动同时充放（违反 A3）以
    在 SOC 不动的前提下把盈余"烧"进往返损耗。这里用 144 个 0-1 变量（scipy.optimize.milp，
    同一 HiGHS 后端）给出 **A3 下真实可达** 的处置盈余下界；主口径仍全部用 linprog。
    返回结构与 solve_day_redispatch 一致；不可行返回 None。
    """
    if not HAS_MILP:
        return None
    n = C.N_SLOT
    A_ub, b_ub, A_eq, b_eq, bounds, nv = _constraints(
        rec, use_plan_row=use_plan_row, allow_deficit=allow_deficit)
    N = nv + n
    A_ub = hstack([A_ub, coo_matrix((A_ub.shape[0], n))]).tocsr()
    A_eq = hstack([A_eq, coo_matrix((n, n))]).tocsr()
    rows, cols, vals = [], [], []
    for t in range(n):                       # 行序与 rhs 必须严格对齐（0..n-1 充、n..2n-1 放）
        rows += [t, t]; cols += [t, nv + t]; vals += [1.0, -C.E_MAX]
        rows += [n + t, n + t]; cols += [n + t, nv + t]; vals += [1.0, C.E_MAX]
    A_ub = vstack([A_ub, coo_matrix((vals, (rows, cols)), shape=(2 * n, N))]).tocsr()
    b_ub = np.concatenate([b_ub, [0.0] * n + [C.E_MAX] * n])
    lo = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds] + [0.0] * n, float)
    hi = np.array([(np.inf if b[1] is None else b[1]) for b in bounds] + [1.0] * n, float)
    integ = np.zeros(N); integ[nv:] = 1.0
    # A3 下 c_t > 0 只能在盈余余量 s_ub_t + E_t > 0 的槽出现（否则第 3 行要求 c_t <= s_ub+E+d_t
    # 而 d_t = 0）；这些槽直接令 c_t = 0 并让 z_t 连续，可省掉约一半 0-1 变量。
    chargeable = (rec["s_ub"] + rec["E"]) > 0.0
    hi[:n] = np.where(chargeable, C.E_MAX, 0.0)
    integ[nv:] = chargeable.astype(float)
    cons = [LinearConstraint(A_ub, -np.inf, b_ub),
            LinearConstraint(A_eq, b_eq, b_eq)]
    bnd = Bounds(lo, hi)

    c1 = np.zeros(N); c1[3 * n:4 * n] = C.EMERGENCY_MULT * price
    # 第一层先用 linprog 求解（同一个最优紧急购电费；LP 最优解实测已满足 A3，
    # 故其最优值可直接作为 A3 版的 z1；若 MILP 因此不可行再回退到 MILP 求第一层）
    A_ub_lp = A_ub[:, :nv]; A_eq_lp = A_eq[:, :nv]
    r0 = linprog(c1[:nv], A_ub=A_ub_lp, b_ub=b_ub, A_eq=A_eq_lp, b_eq=b_eq,
                 bounds=[(lo[j], hi[j]) for j in range(nv)], method="highs")
    z1 = float(r0.fun) if r0.success else None
    if z1 is None:
        r1 = milp(c=c1, constraints=cons, integrality=integ, bounds=bnd)
        if not r1.success:
            return None
        z1 = float(r1.fun)
    # 第二层：min sum_t (d_t - c_t)，并用一行把第一层最优紧急购电费固定住
    obj_row = np.zeros(N); obj_row[0:n] = -1.0; obj_row[n:2 * n] = 1.0
    cost_row = np.zeros(N); cost_row[3 * n:4 * n] = C.EMERGENCY_MULT * price
    A2 = vstack([A_ub, coo_matrix(cost_row.reshape(1, -1)),
                 coo_matrix(obj_row.reshape(1, -1))]).tocsr()
    b2 = np.concatenate([b_ub, [z1 + tol], [np.inf]])
    r2 = milp(c=obj_row, constraints=[LinearConstraint(A2, -np.inf, b2),
                                      LinearConstraint(A_eq, b_eq, b_eq)],
              integrality=integ, bounds=bnd)
    if not r2.success:                      # LP 的 z1 在 A3 下不可达 -> 用 MILP 重求第一层
        r1 = milp(c=c1, constraints=cons, integrality=integ, bounds=bnd)
        if not r1.success:
            return None
        z1 = float(r1.fun)
        b2 = np.concatenate([b_ub, [z1 + tol], [np.inf]])
        r2 = milp(c=obj_row, constraints=[LinearConstraint(A2, -np.inf, b2),
                                          LinearConstraint(A_eq, b_eq, b_eq)],
                  integrality=integ, bounds=bnd)
        if not r2.success:
            return None
    res = r2 if r2.success else r1
    x = res.x
    c = x[0:n]; d = x[n:2 * n]; S = x[2 * n:3 * n]; g = np.maximum(x[3 * n:4 * n], 0.0)
    surplus = rec["s_ub"] + rec["E"] + d - c
    return dict(c=c, d=d, S=S, g=g, surplus=surplus,
                emg_kwh=float(g.sum()),
                emg_cost=float(C.EMERGENCY_MULT * np.sum(price * g)),
                surplus_kwh=float(np.maximum(surplus, 0.0).sum()),
                surplus_unclipped_kwh=float(surplus.sum()),
                deficit_kwh=float(np.maximum(-surplus, 0.0).sum()),
                c_kwh=float(c.sum()), d_kwh=float(d.sum()),
                x_kwh=float((d - c).sum()), s_end=float(S[-1]),
                n_mixed_slots=int(np.sum((c > 1e-6) & (d > 1e-6))),
                max_min_cd=float(np.minimum(c, d).max()),
                obj=float(res.fun), z1=z1, tie="min_disposal_A3")


def check_solution(rec, sol):
    """独立可行性自检（不依赖求解器返回值）。"""
    n = C.N_SLOT
    c, d, S, g = sol["c"], sol["d"], sol["S"], sol["g"]
    prev = np.concatenate([[rec["s_init"]], S[:-1]])
    surplus = rec["s_ub"] + rec["E"] + d - c
    return dict(
        R1_soc_recursion=float(np.max(np.abs(S - (prev + C.ETA_C * c - d / C.ETA_D)))),
        R2_soc_bounds=float(max(C.S_MIN - S.min(), S.max() - C.S_MAX)),
        R3_power_bounds=float(max(-c.min(), c.max() - C.E_MAX, -d.min(), d.max() - C.E_MAX)),
        R4_plan_row=float(np.max(rec["nd_fcst"] - (rec["E"] - c + d))),
        R5_emergency=float(np.max(np.abs(
            g - np.maximum(rec["nd_act"] + c - d - rec["E"], 0.0)))),
        R6_surplus_min=float(surplus.min()),
        R7_terminal=float(rec["s_init"] - S[-1]),
        R8_simultaneous=int(np.sum((c > 1e-6) & (d > 1e-6))),
    )


# --------------------------------------------------------------------------
# 冻结调度（论文主模型的隐含口径 A12）与逐日扫描
# --------------------------------------------------------------------------
def frozen_dispatch(recs, price):
    """按 0:00 解出的 (E^plan, c^plan, d^plan) 原样执行，缺口用紧急购电补齐。

    盈余处置量按 C-06 口径逐槽截断求和：sum_t max(PV_t DT + E_t + d_t - L_t DT - c_t, 0)。
    """
    emg = emg_cost = surplus = plan_energy = plan_cost = 0.0
    c_kwh = d_kwh = 0.0
    days_with_emg = 0
    per_day = []
    for rec in recs:
        g = np.maximum(rec["nd_act"] + rec["c_plan"] - rec["d_plan"] - rec["E"], 0.0)
        s = np.maximum(rec["s_ub"] + rec["E"] + rec["d_plan"] - rec["c_plan"], 0.0)
        gq = float(g.sum())
        emg += gq
        emg_cost += float(C.EMERGENCY_MULT * np.sum(price * g))
        surplus += float(s.sum())
        plan_energy += float(rec["E"].sum())
        plan_cost += float(np.sum(price * rec["E"]))
        c_kwh += float(rec["c_plan"].sum())
        d_kwh += float(rec["d_plan"].sum())
        days_with_emg += int(gq > 1e-6)
        per_day.append(dict(date=str(rec["date"].date()), emg_kwh=gq,
                            emg_cost_yuan=float(C.EMERGENCY_MULT * np.sum(price * g)),
                            surplus_kwh=float(s.sum())))
    return dict(emg_kwh=emg, emg_cost_yuan=emg_cost, surplus_kwh=surplus,
                plan_energy_kwh=plan_energy, plan_cost_yuan=plan_cost,
                c_kwh=c_kwh, d_kwh=d_kwh, x_kwh=d_kwh - c_kwh,
                days_with_emg=days_with_emg, n_days=len(recs),
                per_day=per_day)


def frozen_dispatch_cd_zero(recs, price):
    """纯计划口径（电池完全不动作 c=d=0）的对照：任务书给定的自检口径。"""
    emg = emg_cost = surplus = 0.0
    for rec in recs:
        g = np.maximum(rec["nd_act"] - rec["E"], 0.0)
        s = np.maximum(rec["s_ub"] + rec["E"], 0.0)
        emg += float(g.sum())
        emg_cost += float(C.EMERGENCY_MULT * np.sum(price * g))
        surplus += float(s.sum())
    return dict(emg_kwh=emg, emg_cost_yuan=emg_cost, surplus_kwh=surplus,
                n_days=len(recs))


def run_variant(recs, price, use_plan_row=True, tie="none", a3=False, verbose=True):
    """逐日求解一个变体，返回合计与逐日明细。

    a3=True 时用 solve_day_a3（0-1 变量强制 A3 无同时充放），否则用 linprog 的 LP。
    严格不可行（或 A3 版不可行）的日子按回退链处理并记入 relax_level：
      0 = 严格可行；1 = 放开盈余行（s_t >= 0，允许缺口由紧急购电补齐）；
      2 = 连带放开计划可行性行。
    """
    solve_one = solve_day_a3 if a3 else solve_day_redispatch
    tot = dict(emg_kwh=0.0, emg_cost_yuan=0.0, surplus_kwh=0.0,
               surplus_unclipped_kwh=0.0, deficit_kwh=0.0,
               c_kwh=0.0, d_kwh=0.0, x_kwh=0.0, plan_cost_yuan=0.0,
               n_mixed_slots=0, max_min_cd=0.0)
    per_day = []
    relax_levels = {0: 0, 1: 0, 2: 0}
    worst = {}
    t0 = time.time()
    for i, rec in enumerate(recs):
        kw = dict(use_plan_row=use_plan_row)
        if not a3:
            kw["tie"] = tie
        sol = solve_one(rec, price, **kw)
        level = 0
        if sol is None:
            sol = solve_one(rec, price, allow_deficit=True, **kw)
            level = 1
            if sol is None:
                kw["use_plan_row"] = False
                sol = solve_one(rec, price, allow_deficit=True, **kw)
                level = 2
            if sol is None:
                raise RuntimeError(f"LP 不可行（含全部松弛）：{rec['date'].date()}")
        relax_levels[level] += 1
        chk = check_solution(rec, sol)
        # 恒等式自检：sum_t s_t = sum_t (PV-L) DT + sum_t E + sum_t (d-c)
        worst["R9_balance_identity"] = max(
            worst.get("R9_balance_identity", 0.0),
            abs(float(rec["s_ub"].sum() + rec["E"].sum() + sol["x_kwh"])
                - sol["surplus_unclipped_kwh"]))
        for k, v in chk.items():
            if k == "R8_simultaneous":
                worst[k] = max(worst.get(k, 0), v)
            else:
                worst[k] = max(worst.get(k, -np.inf), v)
        for k in ("emg_kwh", "emg_cost", "surplus_kwh", "surplus_unclipped_kwh",
                  "deficit_kwh", "c_kwh", "d_kwh", "x_kwh"):
            tot[k if k != "emg_cost" else "emg_cost_yuan"] += sol[k]
        tot["n_mixed_slots"] += sol["n_mixed_slots"]
        tot["max_min_cd"] = max(tot["max_min_cd"], sol["max_min_cd"])
        tot["plan_cost_yuan"] += float(np.sum(price * rec["E"]))
        per_day.append(dict(date=str(rec["date"].date()), relax_level=level,
                            emg_kwh=sol["emg_kwh"], emg_cost_yuan=sol["emg_cost"],
                            surplus_kwh=sol["surplus_kwh"], c_kwh=sol["c_kwh"],
                            d_kwh=sol["d_kwh"], x_kwh=sol["x_kwh"], s_end=sol["s_end"]))
        if verbose and (i + 1) % 80 == 0:
            print(f"  ... {i+1}/{len(recs)} 天  {time.time()-t0:6.1f}s  "
                  f"紧急 {tot['emg_kwh']:.1f} kWh  盈余 {tot['surplus_kwh']:.0f} kWh")
    tot["days_with_emg"] = int(sum(1 for r in per_day if r["emg_kwh"] > 1e-6))
    tot["n_days"] = len(recs)
    tot["n_days_strict_feasible"] = relax_levels[0]
    tot["n_days_relax_surplus_row"] = relax_levels[1]
    tot["n_days_relax_both_rows"] = relax_levels[2]
    tot["max_residual"] = worst
    tot["runtime_s"] = time.time() - t0
    return tot, per_day


def reductions(frozen, var):
    """相对冻结调度的降幅（%）。"""
    out = {}
    for key, tag in (("emg_kwh", "emg_kwh"), ("emg_cost_yuan", "emg_cost_yuan"),
                     ("surplus_kwh", "surplus_kwh")):
        f = frozen[key]
        out[tag + "_drop_pct"] = float((f - var[key]) / f * 100.0) if f > 0 else 0.0
    out["emg_kwh_abs_drop"] = float(frozen["emg_kwh"] - var["emg_kwh"])
    out["emg_cost_yuan_abs_drop"] = float(frozen["emg_cost_yuan"] - var["emg_cost_yuan"])
    out["surplus_kwh_abs_drop"] = float(frozen["surplus_kwh"] - var["surplus_kwh"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None, help="仅跑前 N 天（冒烟测试）")
    ap.add_argument("--out", default="exp_redispatch.json")
    ap.add_argument("--a3", action="store_true",
                    help="额外跑两个 A3 精确（0-1 变量、scipy.optimize.milp）变体。默认关闭："
                         "放不开盈余行的日子（188/334 天）big-M 无法收紧，分支定界很慢"
                         "（实测单日约 19 s，全年约 10–60 分钟）。"
                         "主口径与全部默认变体只用 scipy.optimize.linprog")
    args = ap.parse_args()

    panel = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D"))
    price = C.read_attachment1()["price"].to_numpy(float)
    recs, plan_meta = load_plan(panel, price, max_days=args.max_days)
    print(f"[数据] {len(recs)} 天  {recs[0]['date'].date()} .. {recs[-1]['date'].date()}  "
          f"计划来源 {PLAN_JSON}（K={plan_meta['K']}, scen={plan_meta['scen']}, "
          f"info={plan_meta['info']}）")

    # ---- 自检：冻结调度必须复现论文已报告的三项 ---- #
    fro = frozen_dispatch(recs, price)
    cd0 = frozen_dispatch_cd_zero(recs, price)
    print(f"[自检] 冻结调度（执行 0:00 的 E,c,d + 缺口紧急购电）：")
    print(f"       紧急购电量 {fro['emg_kwh']:.4f} kWh（论文 {PAPER_EMG_KWH:.4f}）")
    print(f"       紧急购电费 {fro['emg_cost_yuan']:.4f} 元（论文 {PAPER_EMG_COST:.4f}）")
    print(f"       盈余处置量 {fro['surplus_kwh']:.4f} kWh（论文 {PAPER_SURPLUS_KWH:.4f}）")
    if args.max_days is None:
        assert abs(fro["emg_kwh"] - PAPER_EMG_KWH) < 1e-3, fro["emg_kwh"]
        assert abs(fro["emg_cost_yuan"] - PAPER_EMG_COST) < 1e-2, fro["emg_cost_yuan"]
        assert abs(fro["surplus_kwh"] - PAPER_SURPLUS_KWH) < 1e-3, fro["surplus_kwh"]
        print("       -> 与论文一致（assert 通过）")
    print(f"[对照] 电池完全不动作（c=d=0，仅 E^plan 供能）：紧急购电 {cd0['emg_kwh']:.2f} kWh / "
          f"{cd0['emg_cost_yuan']:.2f} 元，盈余 {cd0['surplus_kwh']:.2f} kWh")
    print("       注：该口径**不能**复现论文的 172 015.82 kWh —— 论文的冻结调度里电池按 0:00 计划"
          "放电，\n       故自检口径必须是「E^plan + 0:00 的 c,d 原样执行」。")

    # ---- 变体扫描 ---- #
    variants = {
        "redispatch": dict(
            use_plan_row=True, tie="none",
            role="任务书给定口径（目标仅 min sum_t 5 p_t g_t）。主目标退化，处置盈余随 "
                 "HiGHS 顶点；实测与 min_throughput 近乎一致"),
        "redispatch_min_throughput": dict(
            use_plan_row=True, tie="min_throughput",
            role="建议主报：A3 洁净（无同时充放）且最少动用电池的第二层字典序口径，"
                 "给出处置盈余的可达上端"),
        "redispatch_min_disposal": dict(
            use_plan_row=True, tie="min_disposal",
            role="**非物理** LP 松弛下界：第二层 min sum_t s_t 会主动使用同时充放"
                 "（违反 A3，靠往返损耗'消化'盈余），只作参照，不可写入结论；"
                 "A3 精确值见 redispatch_A3_min_disposal"),
        "redispatch_noRow": dict(
            use_plan_row=False, tie="none",
            role="对照：放开计划可行性行 E^plan_t - c_t + d_t >= nd^fcst_t，观察该行的影响"),
        "redispatch_noRow_min_throughput": dict(
            use_plan_row=False, tie="min_throughput",
            role="对照 + A3 洁净口径（放开计划可行性行）"),
        "redispatch_noRow_min_disposal": dict(
            use_plan_row=False, tie="min_disposal",
            role="对照的 LP 松弛下界（同前，违反 A3）"),
    }
    if args.a3:
        variants["redispatch_A3_min_disposal"] = dict(
            use_plan_row=True, tie="min_disposal", a3=True,
            role="A3 精确下界（0-1 变量强制 c_t d_t = 0，scipy.optimize.milp）："
                 "A3 下真实可达的最小处置盈余（靠电池往返损耗'消化'盈余）")
        variants["redispatch_noRow_A3_min_disposal"] = dict(
            use_plan_row=False, tie="min_disposal", a3=True,
            role="A3 精确下界（放开计划可行性行）")
    out = {}
    for name, kw in variants.items():
        print(f"[变体] {name}  use_plan_row={kw['use_plan_row']}  tie={kw['tie']}"
              + ("  solver=milp(A3)" if kw.get("a3") else "  solver=linprog"))
        tot, per_day = run_variant(recs, price, use_plan_row=kw["use_plan_row"],
                                   tie=kw["tie"], a3=kw.get("a3", False))
        tot["use_plan_row"] = kw["use_plan_row"]
        tot["tie_break"] = kw["tie"]
        tot["solver"] = "scipy.optimize.milp（HiGHS，A3 0-1）" if kw.get("a3") \
            else "scipy.optimize.linprog（HiGHS）"
        tot["role"] = kw["role"]
        tot["definition"] = ("E^plan 固定为 q2_main.json 的 0:00 计划购电量（已付费，沉没），"
                             "只用实际净需求重优化 c,d,S,g；目标 min sum_t 5 p_t g_t"
                             + ("；第二层字典序 " + kw["tie"] if kw["tie"] != "none" else "（仅主目标）"))
        tot["reductions_vs_frozen"] = reductions(fro, tot)
        out[name] = dict(total=tot, per_day=per_day)
        r = tot["reductions_vs_frozen"]
        print(f"       紧急 {tot['emg_kwh']:.2f} kWh / {tot['emg_cost_yuan']:.2f} 元 "
              f"（降幅 {r['emg_kwh_drop_pct']:.2f}% / {r['emg_cost_yuan_drop_pct']:.2f}%）")
        print(f"       盈余 {tot['surplus_kwh']:.2f} kWh "
              f"（{'降' if r['surplus_kwh_drop_pct'] >= 0 else '升'} "
              f"{abs(r['surplus_kwh_drop_pct']):.2f}%，负降幅=上升）"
              f"  严格可行 {tot['n_days_strict_feasible']}/{tot['n_days']} 天"
              f"（放开盈余行 {tot['n_days_relax_surplus_row']} 天，再放开计划行 "
              f"{tot['n_days_relax_both_rows']} 天）")
        print(f"       同时充放槽数 {tot['n_mixed_slots']}（max min(c,d)={tot['max_min_cd']:.1f} kWh）"
              f"  充电 {tot['c_kwh']:.0f} / 放电 {tot['d_kwh']:.0f} kWh  {tot['runtime_s']:.1f}s")

    headline = out["redispatch_min_throughput"]["total"]
    no_row = out["redispatch_noRow_min_throughput"]["total"]
    # 诊断：放开盈余行的日子里，实际净需求超过 0:00 计划购电量的部分（kWh）
    relaxed_dates = {r["date"] for r in out["redispatch_min_throughput"]["per_day"]
                     if r["relax_level"] > 0}
    relaxed_deficit = float(sum(np.maximum(r["nd_act"] - r["E"], 0.0).sum()
                                for r in recs if str(r["date"].date()) in relaxed_dates))
    power_lb = float(sum(np.maximum(r["nd_act"] - r["E"] - C.E_MAX, 0.0).sum() for r in recs))
    def_drop = (fro["surplus_kwh"] - headline["surplus_kwh"]) / fro["surplus_kwh"] * 100.0
    summary = dict(
        frozen=dict(emg_kwh=fro["emg_kwh"], emg_cost_yuan=fro["emg_cost_yuan"],
                    surplus_kwh=fro["surplus_kwh"], days_with_emg=fro["days_with_emg"],
                    c_kwh=fro["c_kwh"], d_kwh=fro["d_kwh"]),
        headline_variant="redispatch_min_throughput",
        headline=dict(emg_kwh=headline["emg_kwh"], emg_cost_yuan=headline["emg_cost_yuan"],
                      surplus_kwh=headline["surplus_kwh"],
                      days_with_emg=headline["days_with_emg"],
                      emg_kwh_drop_pct=headline["reductions_vs_frozen"]["emg_kwh_drop_pct"],
                      emg_cost_yuan_drop_pct=headline["reductions_vs_frozen"]["emg_cost_yuan_drop_pct"],
                      surplus_kwh_drop_pct=headline["reductions_vs_frozen"]["surplus_kwh_drop_pct"]),
        plan_row_effect=dict(
            emg_kwh_with_row=headline["emg_kwh"],
            emg_kwh_without_row=no_row["emg_kwh"],
            emg_kwh_diff=headline["emg_kwh"] - no_row["emg_kwh"],
            emg_kwh_diff_pct_of_frozen=(headline["emg_kwh"] - no_row["emg_kwh"])
            / fro["emg_kwh"] * 100.0,
            surplus_kwh_with_row=headline["surplus_kwh"],
            surplus_kwh_without_row=no_row["surplus_kwh"],
            strict_feasible_with_row=headline["n_days_strict_feasible"],
            strict_feasible_without_row=no_row["n_days_strict_feasible"],
            verdict="该行显著影响残余紧急购电：放开后 334 天中严格可行天数由 "
                    f"{headline['n_days_strict_feasible']} 升到 {no_row['n_days_strict_feasible']}，"
                    "残余紧急购电下降 "
                    f"{(headline['emg_kwh'] - no_row['emg_kwh']) / headline['emg_kwh'] * 100:.2f}%"),
        mechanism=dict(
            identity="sum_t s_t = sum_t (PV_t - L_t) DT + sum_t E^plan_t + sum_t (d_t - c_t)",
            soc_identity="sum_t d_t = eta_c eta_d sum_t c_t - eta_d (S_143 - S_0) "
                         "<= 0.81 sum_t c_t（因 A6: S_143 >= S_0）",
            implication="故 sum_t (d_t - c_t) <= -0.19 sum_t c_t <= 0：一天内电池**不能净供能**，"
                        "放电必须由同日充电补偿；这正是重调度后仍有残余紧急购电、"
                        "且处置盈余无法实质下降的结构性原因",
            a3_note="min_disposal 口径的降幅由**同时充放**（违反 A3）取得：主模型 LP 中 c_t, d_t "
                    "无互斥约束，"
                    f"该口径下同时充放槽数达 {out['redispatch_min_disposal']['total']['n_mixed_slots']} 个"
                    f"（max min(c,d)={out['redispatch_min_disposal']['total']['max_min_cd']:.1f} kWh），"
                    "故其值不可作为物理可达下界，只作 LP 松弛参照",
            a3_exact_note=(("A3 精确最小处置盈余 sum_t s_t："
                            f"{out['redispatch_A3_min_disposal']['total']['surplus_kwh']:.2f} kWh "
                            "（靠电池往返损耗'消化'盈余，每次循环损失 "
                            f"{1 - C.ETA_C * C.ETA_D:.0%}）；见 variants.redispatch_A3_min_disposal")
                           if args.a3 else
                           "本次未跑 A3 精确变体（默认只跑 linprog；用 --a3 追加 0-1 变体）")),
        cd_zero_diagnostic=dict(emg_kwh=cd0["emg_kwh"], emg_cost_yuan=cd0["emg_cost_yuan"],
                                surplus_kwh=cd0["surplus_kwh"]),
        diagnostics=dict(
            relaxed_days=int(len(relaxed_dates)),
            relaxed_days_deficit_kwh=relaxed_deficit,
            relaxed_days_emg_kwh=headline["emg_kwh"],
            relaxed_days_emg_share_of_deficit_pct=(
                headline["emg_kwh"] / relaxed_deficit * 100.0 if relaxed_deficit else 0.0),
            power_cap_lower_bound_kwh=power_lb,
            power_cap_note="解析下界 sum_t max(nd^act_t - E_t - E_MAX, 0) 只计单槽放电功率上限；"
                           "它远小于重调度后的残余紧急购电，说明残余不是功率问题，"
                           "而是日能量预算问题（A6 + 往返效率使电池一天内不能净供能）",
        ),
    )

    payload = {
        "meta": dict(
            script="exp_redispatch.py",
            assumption="A12 充放电计划在 0:00 一次确定、实时不允许重调度（本模块量化其影响）",
            caliber=f"结果窗口 {C.WIN_START}..{C.WIN_END}（{len(DAYS)} 天），"
                    f"电价=附件 1 固定价 p_t 元/kWh，紧急购电电价 {C.EMERGENCY_MULT} p_t",
            units=dict(emg_kwh="kWh（紧急购电量）",
                       emg_cost_yuan="元（紧急购电费 5 sum_t p_t g_t）",
                       surplus_kwh="kWh（盈余/弃光处置量 C-06：sum_t max(s_t,0)，"
                                   "s_t = PV_t DT + E_t + d_t - L_t DT - c_t）",
                       c_kwh="kWh（充电量）", d_kwh="kWh（放电量）",
                       x_kwh="kWh（净放电量 sum_t (d_t - c_t)）",
                       plan_cost_yuan="元（0:00 计划购电费 sum_t p_t E^plan_t，沉没常数）"),
            n_days=len(recs),
            plan_source=f"intermediate/{PLAN_JSON}: daily[i].E（E^plan）与 daily[i].s_init（S_0）",
            forecast="nd^fcst_t = (L_t - PVhat^(0)_t) DT，code/q3.forecast_slots(panel, day, 0, arange(144))",
            actual="nd^act_t = (L_t - PV_t) DT，附件 2 实际负载与实际光伏",
            solver="scipy.optimize.linprog(method='highs')，一天一个 LP（变量 c,d,S,g 各 144）",
            random="无随机数（seed 不适用）；结果由 HiGHS 确定性给出",
            objective="min sum_t 5 p_t g_t；计划购电费为沉没常数，不进目标",
            objective_note="盈余行 s_t>=0 与 g_t>=0、紧急购电行联立 ⟹ 严格可行时 g_t 恒为 0，"
                           "主目标在最优面上退化（c,d 不唯一、处置盈余随顶点漂移）；"
                           "不可行的日子放开盈余行，g 取「计划固定下不可再减」的最小紧急购电量。"
                           "故除主口径外另给 min_disposal / min_throughput 两个字典序对照，"
                           "用于界定处置盈余的可达区间；其中 min_disposal 会违反 A3（同时充放），"
                           "只作 LP 松弛参照",
            constraints=["E^plan_t - c_t + d_t >= nd^fcst_t（计划可行性行）",
                         "g_t >= nd^act_t + c_t - d_t - E^plan_t, g_t >= 0",
                         "S_t = S_{t-1} + eta_c c_t - d_t/eta_d, S_MIN <= S_t <= S_MAX",
                         "0 <= c_t, d_t <= E_MAX",
                         "S_0 = q2_main.json 当日 s_init, S_143 >= S_0（A6）",
                         "s_t = PV_t DT + E^plan_t + d_t - L_t DT - c_t >= 0"],
            constants=dict(DT=C.DT, ETA_C=C.ETA_C, ETA_D=C.ETA_D, E_MAX=C.E_MAX,
                          S_MIN=C.S_MIN, S_MAX=C.S_MAX, S_INIT=C.S_INIT,
                          EMERGENCY_MULT=C.EMERGENCY_MULT),
            residual_note="variants.*.max_residual.R6_surplus_min 在放开盈余行的日子里为负（该日缺口由"
                          "紧急购电补齐）；R4_plan_row 在 no_plan_row 变体中度量的是被放开那行的剩余松弛，"
                          "不是可行性违反；R8_simultaneous 只对 min_disposal 变体非零（LP 松弛伪解）",
            plan_json_meta=plan_meta,
            paper_reference=dict(emg_kwh=PAPER_EMG_KWH, emg_cost_yuan=PAPER_EMG_COST,
                                 surplus_kwh=PAPER_SURPLUS_KWH),
            reproduce="cd C题论文/code && python3 exp_redispatch.py",
        ),
        "frozen_dispatch": dict(
            definition="0:00 解出的 (E^plan, c^plan, d^plan) 原样执行，缺口用紧急购电补齐"
                       "（主模型在 A12 下的实际结算口径）",
            emg_kwh=fro["emg_kwh"], emg_cost_yuan=fro["emg_cost_yuan"],
            surplus_kwh=fro["surplus_kwh"], plan_energy_kwh=fro["plan_energy_kwh"],
            plan_cost_yuan=fro["plan_cost_yuan"], c_kwh=fro["c_kwh"], d_kwh=fro["d_kwh"],
            days_with_emg=fro["days_with_emg"], n_days=fro["n_days"],
            matches_paper=(abs(fro["emg_kwh"] - PAPER_EMG_KWH) < 1e-3
                           and abs(fro["emg_cost_yuan"] - PAPER_EMG_COST) < 1e-2
                           and abs(fro["surplus_kwh"] - PAPER_SURPLUS_KWH) < 1e-3),
        ),
        "frozen_dispatch_cd_zero": dict(
            definition="对照：电池完全不动作（c=d=0），只用 E^plan 供能",
            emg_kwh=cd0["emg_kwh"], emg_cost_yuan=cd0["emg_cost_yuan"],
            surplus_kwh=cd0["surplus_kwh"], n_days=cd0["n_days"],
            note="该口径不能复现论文的 172 015.82 kWh：论文的冻结调度包含 0:00 计划的放电量，"
                 "故自检口径是 E^plan + c^plan + d^plan",
        ),
        "summary": summary,
        "variants": {k: v["total"] for k, v in out.items()},
        "per_day": {k: v["per_day"] for k, v in out.items()},
    }
    path = os.path.join(C.INTERMEDIATE_DIR, args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"已写出 {path}")
    return payload


if __name__ == "__main__":
    main()
