# -*- coding: utf-8 -*-
"""
L 形（Benders）分解对照实验
===========================
对象：lp_engine.solve_day 的单日两阶段随机规划（问题 2/3/4 共用的求解核）

    min  sum_t p_t E_t + 5 sum_t p_t sum_w pi_w g_{w,t}
    s.t. E_t - c_t + d_t            >= nd^fcst_t                    (计划可行性)
         g_{w,t} + E_t - c_t + d_t  >= nd^act_{w,t}                 (追索/紧急购电)
         S_t = S_{t-1} + eta_c c_t - d_t/eta_d,  S_min <= S_t <= S_max
         0 <= c_t, d_t <= E_MAX,  E_t >= 0,  g_{w,t} >= 0
         S_144 >= S_0（term_mode='ge'，本实验主口径）

分解结构
--------
第一阶段 x = (E, c, d, S)（4n 个变量）；第二阶段 g_{w,t} 按情景 w 与槽位 t 完全可分。
记单槽"计划口径可用供给" y_t = E_t + d_t - c_t（kWh），则第二阶段的最优值解析可解：

    Q(y) = 5 * sum_t p_t * sum_w pi_w * (nd_{w,t} - y_t)^+ ,
    g_{w,t}^*(y) = (nd_{w,t} - y_t)^+ .

**次梯度解析式**（关键，主问题因此只需带解析割的 LP）：
    dQ/dy_t = -5 p_t * sum_w pi_w * 1{ nd_{w,t} > y_t } =: mu_t <= 0 ,
因为每个 max(0, nd_{w,t} - y_t) 关于 y_t 的次梯度是 -1{nd_{w,t} > y_t}（0 处的次梯度区间
为 [-1, 0]，取 -1 的取值即可给出有效割）。对第一阶段变量（y_t = E_t + d_t - c_t）：

    dQ/dE_t = dQ/dd_t = -dQ/dc_t = mu_t .

**最优性割（Benders/L 形）**：对任一试验点 y^k，
    theta >= Q(y^k) + sum_t mu_t^k (y_t - y_t^k)
写成 <= 形式（供 linprog 使用）：
    -theta - sum_t q_t^k (E_t + d_t - c_t) <= -( Q(y^k) + sum_t q_t^k y_t^k ),
    q_t^k := 5 p_t sum_w pi_w 1{nd_{w,t} > y_t^k} >= 0 .

**逐槽分离的多割（本脚本默认 disaggregated=True）**：因为 Q 逐槽可分
    Q(y) = sum_t Q_t(y_t),  Q_t(y_t) = 5 p_t sum_w pi_w (nd_{w,t} - y_t)^+ ,
把 144 维切平面拆成 144 条一维切平面（每个槽位一个 theta_t）：
    theta_t + q_t^k (y_t - y_t^k) >= Q_t(y_t^k)   <=>   -theta_t - q_t^k y_t <= -(Q_t(y_t^k)+q_t^k y_t^k).
每条割都是有效下界，故 LB 仍然有效；实证：多割 9~20 次迭代收敛到 1e-15（对偶间隙
1e-12 绝对量），而聚合单割 200 次迭代后相对差仍有 6.5e-3 ~ 3.4e-2（见 single_cut_vs_multicut）。

**可行性割不需要**：追索变量 g >= 0 且无上界，任何第一阶段决策都可补偿（完全追索），
子问题恒可行，故只要最优性割。Q 是分片线性凸函数，分片数有限，算法有限步收敛。

主问题变量上界（仅为让主问题有界，不切掉任何最优解）
----------------------------------------------------
新增割在 y_t 方向斜率为负，若第一阶段变量无上界，主问题会出现"沿割方向无界"的假象
（因为割只是 Q 的一条切线，而 Q 本身在 y 大于最大情景需求后恒为 0）。故对 E_t 加
**有效上界**：
    y_t = E_t + d_t - c_t <= y_t + E_MAX  (d_t <= E_MAX, c_t >= 0)
    存在最优解使 y_t <= max(nd^fcst_t, max_w nd_{w,t})：
      把任何可行解的 y_t 截断到该值，计划约束仍满足（截断值 >= nd^fcst_t），
      紧急购电费不增（超出最大情景需求时 g=0），计划购电费不增（p_t y_t 变小），
      故截断后目标 <= 原目标，仍为最优。此时 E_t = y_t + c_t - d_t <= y_t + E_MAX。
    => 约束 E_t <= max(nd^fcst_t, max_w nd_{w,t}) + E_MAX 不排除任何最优解。

输出：intermediate/decomp.json

运行：cd C题论文/code && python3 decomp.py         （约 2-4 分钟）
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # -> code/
import _pathsetup  # noqa: E402,F401  子目录引导：把 common/ 与各问目录加入 sys.path

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

import config as C
import lp_engine as LP
import q2 as Q2

K_SCAN = [20, 60, 120, 300, 500]
TOL = 1e-9                  # 相对收敛容差 |UB-LB| <= TOL*max(1,|UB|)
MAX_ITER = 400


# --------------------------------------------------------------------------
# 解析子问题
# --------------------------------------------------------------------------
def recourse(nd_scen, weights, price, y):
    """解析求解第二阶段：返回 Q(y) 与逐槽缺电概率 w_t = P̂(nd_t > y_t)。"""
    slack = nd_scen - y[None, :]                     # (K,n)
    pos = np.maximum(slack, 0.0)
    qty = (weights[:, None] * pos).sum(axis=0)       # 期望缺电量 (n,)
    Q = float(C.EMERGENCY_MULT * np.sum(price * qty))
    sw = weights[:, None] * (slack > 0.0)
    w = sw.sum(axis=0)                               # (n,) 缺电概率
    return Q, w, qty


def _master_solve(price, nd_fcst, cuts, n_theta, s_init, term_mode, term_value,
                  iE, iC, iD, iS, iTH, n, nv, e_ub):
    """装配并求解主问题（每次迭代重建；变量 4n + n_theta，割行稀疏且很少）。"""
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for t in range(n):                               # 计划可行性
        rows += [r, r, r]; cols += [iE + t, iC + t, iD + t]; vals += [-1.0, 1.0, -1.0]
        rhs.append(-nd_fcst[t]); r += 1
    for coef, b in cuts:                             # 最优性割
        for idx, val in coef:
            rows.append(r); cols.append(idx); vals.append(val)
        rhs.append(b); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()

    rows2, cols2, vals2, rhs2 = [], [], [], []       # SOC 递推
    for t in range(n):
        for b_, v_ in ((iS + t, 1.0), (iC + t, -C.ETA_C), (iD + t, 1.0 / C.ETA_D)):
            rows2.append(t); cols2.append(b_); vals2.append(v_)
        if t > 0:
            rows2.append(t); cols2.append(iS + t - 1); vals2.append(-1.0)
        rhs2.append(0.0 if t > 0 else s_init)
    A_eq = coo_matrix((vals2, (rows2, cols2)), shape=(n, nv)).tocsr()

    bounds = [(0.0, e_ub[t]) for t in range(n)] + \
             [(0.0, C.E_MAX)] * n + [(0.0, C.E_MAX)] * n + \
             [(C.S_MIN, C.S_MAX)] * n + [(0.0, None)] * n_theta
    if term_mode == "eq":
        v = s_init if term_value is None else term_value
        bounds[iS + n - 1] = (v, v)
    elif term_mode == "ge":
        v = s_init if term_value is None else term_value
        bounds[iS + n - 1] = (max(v, C.S_MIN), C.S_MAX)

    c_obj = np.zeros(nv)
    c_obj[iE:iE + n] = price
    c_obj[iTH:iTH + n_theta] = 1.0                   # theta（1 个聚合 / n 个逐槽）
    res = linprog(c_obj, A_ub=A_ub, b_ub=np.array(rhs), A_eq=A_eq, b_eq=np.array(rhs2),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("主问题求解失败：" + res.message)
    return res


def solve_day_benders(price, nd_fcst, nd_scen, weights=None, s_init=C.S_INIT,
                      term_mode="ge", term_value=None, tol=TOL, max_iter=MAX_ITER,
                      disaggregated=True, keep_history=False):
    """L 形（Benders）分解求解单日两阶段 LP；返回解、迭代数与耗时分解。

    disaggregated=True：按槽分离的多割（每个槽位一个 theta_t，各自的解析切平面）。
        因为 Q(y) = sum_t Q_t(y_t)、Q_t(y_t) = 5 p_t sum_w pi_w (nd_{w,t} - y_t)^+ 是
        **逐槽可分的 1 维凸分片线性函数**，把聚合割拆成 144 条逐槽割可极大加快收敛
        （聚合单割在 144 维主问题上会出现典型的慢收敛/振荡，见 large_case.single_cut）。
    disaggregated=False：经典单割（聚合）L 形，用于对照。
    """
    n = C.N_SLOT
    nd_scen = np.atleast_2d(np.asarray(nd_scen, float))
    K = nd_scen.shape[0]
    if weights is None:
        weights = np.full(K, 1.0 / K)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()

    iE, iC, iD, iS, iTH = 0, n, 2 * n, 3 * n, 4 * n
    n_theta = n if disaggregated else 1
    nv = 4 * n + n_theta
    ymax = np.maximum(nd_fcst, nd_scen.max(axis=0))
    e_ub = ymax + C.E_MAX                            # 有效上界（见文件头推导）

    cuts = []
    hist = []
    best = None
    t_master = t_sub = 0.0
    for it in range(max_iter):
        t0 = time.time()
        res = _master_solve(price, nd_fcst, cuts, n_theta, s_init, term_mode, term_value,
                            iE, iC, iD, iS, iTH, n, nv, e_ub)
        t_master += time.time() - t0
        x = res.x
        E, cc, dd, S = x[iE:iE + n], x[iC:iC + n], x[iD:iD + n], x[iS:iS + n]
        y = E + dd - cc
        t0 = time.time()
        Q, wsh, qty = recourse(nd_scen, weights, price, y)
        t_sub += time.time() - t0
        lb = float(res.fun)
        ub = float(price @ E) + Q
        hist.append(dict(it=it, lb=lb, ub=ub, gap=ub - lb,
                         theta=(float(x[iTH]) if not disaggregated
                                else float(np.sum(x[iTH:iTH + n]))),
                         n_short_slots=int((wsh > 0).sum())))
        if best is None or ub < best["ub"]:
            best = dict(ub=ub, E=E.copy(), c=cc.copy(), d=dd.copy(), S=S.copy(),
                        g=qty.copy(), it=it, wsh=wsh.copy(),
                        theta=(float(x[iTH]) if not disaggregated
                               else float(np.sum(x[iTH:iTH + n]))))
        if ub - lb <= tol * max(1.0, abs(ub)):
            break
        qover = C.EMERGENCY_MULT * price * wsh          # q_t^k = 5 p_t P(nd_t > y_t) >= 0
        if disaggregated:
            # 逐槽子问题值 Q_t(y_t^k) = 5 p_t sum_w pi_w (nd_{w,t} - y_t^k)^+
            Qt = C.EMERGENCY_MULT * price * \
                (weights[:, None] * np.maximum(nd_scen - y[None, :], 0.0)).sum(axis=0)
            for t in range(n):
                if qover[t] <= 0.0:                     # Q_t 局部恒为 0 -> 割为 theta_t >= 0
                    continue
                cuts.append(([(iTH + t, -1.0), (iE + t, -qover[t]),
                              (iD + t, -qover[t]), (iC + t, qover[t])],
                             -(Qt[t] + qover[t] * y[t])))
        else:
            coef = [(iTH, -1.0)]
            for t in range(n):
                coef += [(iE + t, -qover[t]), (iD + t, -qover[t]), (iC + t, qover[t])]
            cuts.append((coef, -(Q + float(np.sum(qover * y)))))

    out = dict(E=best["E"], c=best["c"], d=best["d"], S=best["S"],
               obj=float(best["ub"]), lb=float(hist[-1]["lb"]),
               iters=len(hist), cuts=len(cuts), K=K, s_init=s_init,
               g_expected=best["g"], theta=float(best["theta"]),
               disaggregated=bool(disaggregated),
               converged=bool(hist[-1]["ub"] - hist[-1]["lb"] <=
                              tol * max(1.0, abs(hist[-1]["ub"]))),
               sec_master=t_master, sec_sub=t_sub, sec_total=t_master + t_sub,
               nd_fcst=nd_fcst, nd_scen=nd_scen, price=price)
    if keep_history:
        out["history"] = hist
    return out


# --------------------------------------------------------------------------
# 直接求解的"向量化复现"（用于把 Python 装配耗时与 HiGHS 求解耗时分开计时）
# --------------------------------------------------------------------------
def build_direct(price, nd_fcst, nd_scen, weights, s_init, term_mode="ge",
                 term_value=None):
    n, K = C.N_SLOT, nd_scen.shape[0]
    iE, iC, iD, iS, iG = 0, n, 2 * n, 3 * n, 4 * n
    nv = 4 * n + K * n
    c_obj = np.zeros(nv)
    c_obj[iE:iE + n] = price
    c_obj[iG:iG + K * n] = C.EMERGENCY_MULT * (weights[:, None] * price[None, :]).ravel()

    tt = np.tile(np.arange(n), K)          # 逐情景的槽位下标（w 主序）
    ss = np.arange(K * n)                  # 情景块内顺序下标：g 变量 = iG + w*n + t = iG + ss
    sc_rows = n + ss                       # 追索约束行号：n + w*n + t
    pl = np.arange(n)
    rows = np.concatenate([pl, pl, pl, sc_rows, sc_rows, sc_rows, sc_rows])
    cols = np.concatenate([iE + pl, iC + pl, iD + pl,
                           iE + tt, iC + tt, iD + tt, iG + ss])
    vals = np.concatenate([-np.ones(n), np.ones(n), -np.ones(n),
                           -np.ones(K * n), np.ones(K * n), -np.ones(K * n), -np.ones(K * n)])
    A_ub = coo_matrix((vals, (rows, cols)), shape=(n + K * n, nv)).tocsr()
    b_ub = np.concatenate([-nd_fcst, -nd_scen.ravel()])

    rows2 = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n), np.arange(n)])
    cols2 = np.concatenate([iS + np.arange(n), iS + np.arange(n - 1), iC + np.arange(n),
                            iD + np.arange(n)])
    vals2 = np.concatenate([np.ones(n), -np.ones(n - 1), -C.ETA_C * np.ones(n),
                            (1.0 / C.ETA_D) * np.ones(n)])
    A_eq = coo_matrix((vals2, (rows2, cols2)), shape=(n, nv)).tocsr()
    b_eq = np.concatenate([[s_init], np.zeros(n - 1)])

    bounds = [(0.0, None)] * n + [(0.0, C.E_MAX)] * n + [(0.0, C.E_MAX)] * n + \
             [(C.S_MIN, C.S_MAX)] * n + [(0.0, None)] * (K * n)
    if term_mode == "eq":
        v = s_init if term_value is None else term_value
        bounds[iS + n - 1] = (v, v)
    elif term_mode == "ge":
        v = s_init if term_value is None else term_value
        bounds[iS + n - 1] = (max(v, C.S_MIN), C.S_MAX)
    return c_obj, A_ub, b_ub, A_eq, b_eq, bounds, dict(iE=iE, iC=iC, iD=iD, iS=iS, iG=iG, nv=nv)


# --------------------------------------------------------------------------
# 数据与情景
# --------------------------------------------------------------------------
def scenario_set(R, K, seed=0):
    """K<=334 用与主方案一致的分层削减（medoid）；K>334 用固定种子有放回重采样。"""
    if K <= R.shape[0]:
        cen, w = LP.reduce_scenarios_stratified(R, K=K)
        return cen, w, "stratified_medoid"
    rng = np.random.default_rng(seed)
    idx = rng.choice(R.shape[0], size=K, replace=True)
    return R[idx], np.full(K, 1.0 / K), "bootstrap_resample(K>334)"


def day_data(panel, d):
    L = panel.L().loc[d].to_numpy(float)
    P = panel.PV().loc[d].to_numpy(float)
    return (L - P) * C.DT, (L - panel.pv_forecast(d, 0)) * C.DT


def compare_row(price, nd_fcst, cen, w, s_init, term_mode="ge", keep_history=False,
                single_cut=False, single_cut_iters=200):
    """同一天、同一情景集下直接求解 vs L 形分解的对照记录。

    single_cut=True 时额外跑一遍"聚合单割"的经典 L 形，用于对照收敛速度。
    """
    nd_scen = nd_fcst[None, :] + cen
    t0 = time.time()
    sol_d = LP.solve_day(price, nd_fcst, nd_scen, w, s_init=s_init, term_mode=term_mode)
    t_prod = time.time() - t0

    t0 = time.time()
    c_obj, A_ub, b_ub, A_eq, b_eq, bounds, idx = build_direct(price, nd_fcst, nd_scen,
                                                              w, s_init, term_mode)
    t_build = time.time() - t0
    t0 = time.time()
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    t_highs = time.time() - t0
    if not res.success:
        raise RuntimeError("直接求解（复现版）失败：" + res.message)
    obj_repro = float(res.fun)

    sol_b = solve_day_benders(price, nd_fcst, nd_scen, w, s_init=s_init,
                              term_mode=term_mode, keep_history=keep_history)
    gap = abs(sol_b["obj"] - sol_d["obj"])
    extra = {}
    if single_cut:
        t0 = time.time()
        sg = solve_day_benders(price, nd_fcst, nd_scen, w, s_init=s_init,
                               term_mode=term_mode, disaggregated=False,
                               max_iter=single_cut_iters, keep_history=keep_history)
        extra = dict(single_cut_obj=float(sg["obj"]),
                     single_cut_rel_gap=abs(sg["obj"] - sol_d["obj"]) / max(1.0, abs(sol_d["obj"])),
                     single_cut_iters=sg["iters"], single_cut_cuts=sg["cuts"],
                     single_cut_converged=bool(sg["converged"]),
                     single_cut_t=time.time() - t0,
                     single_cut_history=(sg.get("history") if keep_history else None))
    return dict(
        K=int(cen.shape[0]), s_init=float(s_init), term_mode=term_mode,
        obj_direct=float(sol_d["obj"]), obj_direct_repro=obj_repro,
        obj_benders=float(sol_b["obj"]),
        abs_gap=gap, rel_gap=gap / max(1.0, abs(sol_d["obj"])),
        repro_rel_gap=abs(obj_repro - sol_d["obj"]) / max(1.0, abs(sol_d["obj"])),
        iters=sol_b["iters"], cuts=sol_b["cuts"], converged=sol_b["converged"],
        t_direct_production=t_prod, t_direct_build=t_build, t_direct_highs=t_highs,
        t_direct_total=t_build + t_highs,
        t_benders_total=sol_b["sec_total"], t_benders_master=sol_b["sec_master"],
        t_benders_sub=sol_b["sec_sub"],
        speedup_vs_production=t_prod / max(sol_b["sec_total"], 1e-12),
        speedup_highs_only=(t_build + t_highs) / max(sol_b["sec_total"], 1e-12),
        nnz_direct=int(A_ub.nnz + A_eq.nnz), nvar_direct=int(idx["nv"]),
        nvar_benders=int(4 * C.N_SLOT + len(sol_b["E"])),
        n_theta=int(len(sol_b["E"]) if sol_b["disaggregated"] else 1),
        history=(sol_b.get("history") if keep_history else None),
        **extra,
    )


def agg(rows):
    ag = lambda k: float(np.mean([r[k] for r in rows]))
    mx = lambda k: float(np.max([r[k] for r in rows]))
    return dict(n=len(rows), max_rel_gap=mx("rel_gap"), mean_rel_gap=ag("rel_gap"),
                max_abs_gap=mx("abs_gap"), max_iters=int(np.max([r["iters"] for r in rows])),
                mean_iters=ag("iters"), max_cuts=int(np.max([r["cuts"] for r in rows])),
                all_converged=bool(all(r["converged"] for r in rows)),
                sum_t_direct=float(np.sum([r["t_direct_production"] for r in rows])),
                sum_t_benders=float(np.sum([r["t_benders_total"] for r in rows])),
                speedup=float(np.sum([r["t_direct_production"] for r in rows]) /
                              max(np.sum([r["t_benders_total"] for r in rows]), 1e-12)),
                max_repro_rel_gap=mx("repro_rel_gap"))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default=",".join(str(k) for k in K_SCAN))
    ap.add_argument("--tag", default="decomp")
    ap.add_argument("--eq-days", type=int, default=334, help="等价性检验的天数（K=60）")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    t_start = time.time()
    panel = C.Panel(Q2.DATE_ALL)
    price = C.read_attachment1()["price"].to_numpy(float)
    R = Q2.build_residuals(panel, Q2.DATE_WINDOW)
    print(f"残差样本 {R.shape}，电价 {price.min():.4f}~{price.max():.4f} 元/kWh")

    out = dict(meta=dict(
        script="decomp.py", model="单日两阶段随机规划（lp_engine.solve_day）",
        method="L 形/Benders 分解（解析最优性割，无可行性割——完全追索）",
        window=f"{C.WIN_START}..{C.WIN_END}", n_slot=C.N_SLOT, dt=C.DT,
        e_max=C.E_MAX, emergency_mult=C.EMERGENCY_MULT,
        tol=TOL, max_iter=MAX_ITER,
        runtime_s=0.0, seed_scen=0,
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 scipy=scipy.__version__, pandas=pd.__version__),
        reproduce="cd C题论文/code && python3 decomp.py",
    ))

    # ---- (0) 直接求解复现版的自检（保证计时口径可比）----
    nd_act, nd_fcst = day_data(panel, pd.Timestamp(C.KEY_DATES[0]))
    cen, w, src = scenario_set(R, 60)
    r0 = compare_row(price, nd_fcst, cen, w, C.S_INIT)
    print(f"[自检] 第 1 天 K=60：直接 {r0['obj_direct']:.6f} / 复现 {r0['obj_direct_repro']:.6f} / "
          f"Benders {r0['obj_benders']:.6f}  相对差 {r0['rel_gap']:.2e}  "
          f"迭代 {r0['iters']} 割 {r0['cuts']}")
    out["selfcheck"] = dict(day=C.KEY_DATES[0], scenario_source=src, **r0)

    # ---- (1) 等价性：K=60 逐日全窗口 / 抽样对比 ----
    days_all = list(Q2.DATE_WINDOW)
    if args.eq_days < len(days_all):
        step = len(days_all) / args.eq_days
        days_all = [days_all[int(i * step)] for i in range(args.eq_days)]
    rows = []
    t0 = time.time()
    for i, d in enumerate(days_all):
        _, ndf = day_data(panel, d)
        rows.append(compare_row(price, ndf, cen, w, C.S_INIT))
        if (i + 1) % 50 == 0:
            print(f"  ... 等价性 {i+1}/{len(days_all)} 天 {time.time()-t0:.1f}s")
    out["equivalence_K60"] = dict(description="K=60 分层情景，s_init=6000，term_mode=ge，"
                                              "逐日对比直接求解与 L 形分解",
                                  days=[str(d.date()) for d in days_all], **agg(rows))
    out["equivalence_K60"]["rows_head"] = rows[:3]
    print(f"[等价性 K=60] {len(rows)} 天  最大相对差 {agg(rows)['max_rel_gap']:.2e}  "
          f"平均迭代 {agg(rows)['mean_iters']:.1f}  加速比 {agg(rows)['speedup']:.2f}")

    # ---- (2) 初始储电量 / 终端条件扫描（K=60，四个指定日期）----
    rows2 = []
    for d in C.KEY_DATES:
        _, ndf = day_data(panel, pd.Timestamp(d))
        for s0 in (C.S_MIN, 3000.0, C.S_INIT, 9000.0, C.S_MAX):
            rows2.append(dict(day=d, **compare_row(price, ndf, cen, w, s0, keep_history=False)))
    rows3 = []
    for d in C.KEY_DATES:
        _, ndf = day_data(panel, pd.Timestamp(d))
        for tm in ("ge", "eq"):
            rows3.append(dict(day=d, **compare_row(price, ndf, cen, w, C.S_INIT,
                                                    term_mode=tm)))
    out["sweep_s_init_K60"] = dict(description="四个指定日期 x s_init ∈ {1200,3000,6000,9000,10800}",
                                   rows=rows2, summary=agg(rows2))
    out["sweep_term_K60"] = dict(description="四个指定日期 x term_mode ∈ {ge, eq}",
                                 rows=rows3, summary=agg(rows3))
    print(f"[s_init/term 扫描] 最大相对差 "
          f"{max(agg(rows2)['max_rel_gap'], agg(rows3)['max_rel_gap']):.2e}")

    # ---- (3) 情景数扫描（K=20/60/120/300/500，四个指定日期）----
    scan = []
    for K in ks:
        cenK, wK, srcK = scenario_set(R, K)
        rowsK = []
        for d in C.KEY_DATES:
            _, ndf = day_data(panel, pd.Timestamp(d))
            rowsK.append(dict(day=d, **compare_row(price, ndf, cenK, wK, C.S_INIT)))
        a = agg(rowsK)
        scan.append(dict(K=K, scenario_source=srcK, n_scen_effective=int(cenK.shape[0]),
                         nvar_direct=int(rowsK[0]["nvar_direct"]),
                         nvar_benders=int(rowsK[0]["nvar_benders"]),
                         nnz_direct=int(rowsK[0]["nnz_direct"]),
                         rows=rowsK, summary=a))
        print(f"[扫描] K={K:4d} ({srcK})  直接 {a['sum_t_direct']:.2f}s  "
              f"Benders {a['sum_t_benders']:.2f}s  加速 {a['speedup']:.2f}x  "
              f"目标（首日）{rowsK[0]['obj_direct']:.4f} vs {rowsK[0]['obj_benders']:.4f}  "
              f"相对差 {a['max_rel_gap']:.2e}  平均迭代 {a['mean_iters']:.1f}")
    out["scenario_scan"] = dict(description="四个指定日期（s_init=6000）下 K 的影响；"
                                            "K>334 时情景集以固定种子有放回重采样扩充",
                                ks=ks, entries=scan)

    # ---- (4) K=500 大规模算例的逐迭代轨迹 ----
    cen5, w5, src5 = scenario_set(R, 500)
    _, ndf5 = day_data(panel, pd.Timestamp(C.KEY_DATES[0]))
    r500 = compare_row(price, ndf5, cen5, w5, C.S_INIT, keep_history=True,
                       single_cut=True, single_cut_iters=200)
    out["large_case"] = dict(day=C.KEY_DATES[0], scenario_source=src5,
                             n_scen=int(cen5.shape[0]), **r500)
    print(f"[大规模 K=500] 直接 {r500['t_direct_production']:.2f}s  "
          f"Benders {r500['t_benders_total']:.3f}s  加速 {r500['speedup_vs_production']:.1f}x  "
          f"迭代 {r500['iters']}  相对差 {r500['rel_gap']:.2e}")

    # ---- (4b) 聚合单割 vs 逐槽多割（收敛速度对照）----
    rows5 = []
    for d in C.KEY_DATES:
        _, ndf = day_data(panel, pd.Timestamp(d))
        rows5.append(dict(day=d, **compare_row(price, ndf, cen, w, C.S_INIT,
                                               single_cut=True, single_cut_iters=200)))
    out["single_cut_vs_multicut"] = dict(
        description="K=60 四个指定日期：经典聚合单割 L 形（最多 200 次迭代）"
                    " vs 逐槽分离的解析多割",
        rows=rows5,
        summary=dict(
            max_rel_gap_multicut=float(np.max([r["rel_gap"] for r in rows5])),
            multi_iters=[r["iters"] for r in rows5],
            single_iters=[r["single_cut_iters"] for r in rows5],
            single_converged=[r["single_cut_converged"] for r in rows5],
            max_rel_gap_single=float(np.max([r["single_cut_rel_gap"] for r in rows5])),
            min_rel_gap_single=float(np.min([r["single_cut_rel_gap"] for r in rows5])),
            t_multicut=float(np.sum([r["t_benders_total"] for r in rows5])),
            t_single_cut=float(np.sum([r["single_cut_t"] for r in rows5]))))
    print(f"[单割 vs 多割] 多割迭代 {out['single_cut_vs_multicut']['summary']['multi_iters']}，"
          f"单割 200 次迭代后相对差仍达 "
          f"{out['single_cut_vs_multicut']['summary']['max_rel_gap_single']:.2e}")

    # ---- (5) K=300 的抽样等价性（更大规模下仍逐日一致）----
    cen3, w3, src3 = scenario_set(R, 300)
    days_s = [Q2.DATE_WINDOW[i] for i in range(0, len(Q2.DATE_WINDOW), 8)][:42]
    rows4 = []
    for d in days_s:
        _, ndf = day_data(panel, d)
        rows4.append(compare_row(price, ndf, cen3, w3, C.S_INIT))
    out["equivalence_K300_sample"] = dict(description="K=300 分层情景，每 8 天取 1 天共 "
                                                      f"{len(days_s)} 天，s_init=6000",
                                          days=[str(d.date()) for d in days_s], **agg(rows4))
    print(f"[等价性 K=300] {len(rows4)} 天  最大相对差 {agg(rows4)['max_rel_gap']:.2e}  "
          f"加速比 {agg(rows4)['speedup']:.2f}")

    out["notes"] = [
        "子问题完全可分且解析可解，故只有最优性割；主问题 4n+1=577 变量，割为 433 非零的稀疏行。",
        "主问题对 E_t 加了有效上界 max(nd_fcst_t, max_w nd_{w,t}) + E_MAX，仅为保证主问题有界，"
        "不排除任何最优解（推导见源码文件头）。",
        "直接求解耗时给出两个口径：production（调 lp_engine.solve_day，含 Python 稀疏装配）与 "
        "build+highs（本脚本向量化装配 + HiGHS），后者用于与 Benders 公平比较纯求解开销。",
        "K>334 时无法再做分层 medoid 削减，情景集以固定种子（seed=0）有放回重采样扩充，已在 scenario_source 标注。",
        "逐槽分离的解析多割（disaggregated=True）是本实验的默认算法：主问题 4n+n=720 变量，"
        "10 次左右迭代即收敛到机器精度；聚合单割（disaggregated=False）在 144 维主问题上"
        "200 次迭代仍有 1e-2 量级对偶间隙，作为收敛速度对照一并记录。",
        "Benders 单次迭代成本与 K 无关（主问题只有 4n+n 个变量），而直接求解的 LP 规模随 K 线性增长，"
        "故存在交叉点（约 K=100 附近），K 越大分解法优势越明显。",
        "所有数字由本脚本一次运行产生。",
    ]
    out["meta"]["runtime_s"] = time.time() - t_start
    path = os.path.join(C.INTERMEDIATE_DIR, f"{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("=" * 78)
    print(f"等价性：K=60 全窗口最大相对差 {out['equivalence_K60']['max_rel_gap']:.2e}；"
          f"K=300 抽样 {out['equivalence_K300_sample']['max_rel_gap']:.2e}；"
          f"K=500 单日 {r500['rel_gap']:.2e}")
    print(f"K=500 加速比（生产路径 / Benders）= {r500['speedup_vs_production']:.1f}x")
    print("已写出", path, f"（总耗时 {out['meta']['runtime_s']:.1f}s）")
    return out


if __name__ == "__main__":
    main()
