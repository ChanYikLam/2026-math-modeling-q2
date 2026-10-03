# -*- coding: utf-8 -*-
"""
问题 4：波动电价下的购电策略
============================
(1) 电价过程四项检验 C1–C4（正态性/自相关/方差结构/与负载光伏的相关性）；
(2) 两层电价建模：分时骨架（附件 1 均值剖面）+ 随机残差（附件 4 − 骨架）；
(3) 联合情景：以"代表性历史日"为主（价格剖面与净需求残差取自同一天，
    天然保留 rho(p, L)），并以分块 Bootstrap 合成情景作条件对照；
(4) CVaR 风险—费用前沿 + 真实 334 天轨迹回测；
(5) 在波动电价下重算问题 2（result4-2.xlsx）与问题 3（result4-3.xlsx）。

注意：附件 4 只给实际电价、未给日前电价预报，故假设"0:00 可用的电价预报 = 分时骨架"，
     结算按附件 4 的实际电价。该假设在假设节显式声明（A10）。
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
from scipy import stats
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import config as C
import lp_engine as LP
import results_io as RIO
import q3 as Q3

DATE_WINDOW = pd.date_range(C.WIN_START, C.WIN_END, freq="D")
DATE_ALL = pd.date_range(C.WARMUP_START, C.WIN_END, freq="D")


# --------------------------------------------------------------------------
# 通用：电价场景下的单日 LP（计划/调整两用）
# --------------------------------------------------------------------------
def solve_vol(price_scen, nd_fcst, nd_scen, weights=None, s_init=C.S_INIT,
              s_from=0, plan=None, term_floor=None):
    """price_scen: (K,n) 各情景实际电价；plan 非空时按偏差定价口径。"""
    n = C.N_SLOT
    price_scen = np.atleast_2d(price_scen)
    K = price_scen.shape[0]
    if weights is None:
        weights = np.full(K, 1.0 / K)
    weights = np.asarray(weights, float); weights = weights / weights.sum()
    m = n - s_from
    use_plan = plan is not None and s_from > 0
    iE, iC, iD, iS, iG = 0, m, 2 * m, 3 * m, 4 * m
    iU = iG + K * m
    nv = 4 * m + K * m + (2 * m if use_plan else 0)

    pw = price_scen[:, s_from:]                     # (K,m)
    c_obj = np.zeros(nv)
    if use_plan:
        c_obj[iU:iU + m] = -C.BREACH_MULT * (weights[:, None] * pw).sum(0)
        c_obj[iU + m:iU + 2 * m] = C.EXCESS_MULT * (weights[:, None] * pw).sum(0)
    else:
        c_obj[iE:iE + m] = (weights[:, None] * pw).sum(0)
    c_obj[iG:iG + K * m] = C.EMERGENCY_MULT * (weights[:, None] * pw).ravel()

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for j in range(m):
        for a, b, c_ in ((r, iE + j, -1.0), (r, iC + j, 1.0), (r, iD + j, -1.0)):
            rows.append(a); cols.append(b); vals.append(c_)
        rhs.append(-nd_fcst[s_from + j]); r += 1
    for w in range(K):
        for j in range(m):
            for a, b, c_ in ((r, iE + j, -1.0), (r, iC + j, 1.0),
                             (r, iD + j, -1.0), (r, iG + w * m + j, -1.0)):
                rows.append(a); cols.append(b); vals.append(c_)
            rhs.append(-nd_scen[w, s_from + j]); r += 1
    if use_plan:
        for j in range(m):
            for a, b, c_ in ((r, iE + j, 1.0), (r, iU + m + j, -1.0),
                             (r, iU + j, 1.0)):
                rows.append(a); cols.append(b); vals.append(c_)
            rhs.append(plan[s_from + j]); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    b_ub = np.array(rhs)

    rows2, cols2, vals2, rhs2 = [], [], [], []
    for j in range(m):
        for b, c_ in ((iS + j, 1.0), (iC + j, -C.ETA_C), (iD + j, 1.0 / C.ETA_D)):
            rows2.append(j); cols2.append(b); vals2.append(c_)
        if j > 0:
            rows2.append(j); cols2.append(iS + j - 1); vals2.append(-1.0)
        rhs2.append(0.0 if j > 0 else s_init)
    A_eq = coo_matrix((vals2, (rows2, cols2)), shape=(m, nv)).tocsr()
    b_eq = np.array(rhs2)

    bounds = [(0.0, None)] * m + [(0.0, C.E_MAX)] * m + [(0.0, C.E_MAX)] * m + \
             [(C.S_MIN, C.S_MAX)] * m + [(0.0, None)] * (K * m) + \
             ([(0.0, None)] * (2 * m) if use_plan else [])
    if term_floor is not None:
        bounds[iS + m - 1] = (max(term_floor, C.S_MIN), C.S_MAX)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("volatile LP 不可行：" + res.message)
    x = res.x
    return dict(E=x[iE:iE + m].copy(), c=x[iC:iC + m].copy(),
                d=x[iD:iD + m].copy(), S=x[iS:iS + m].copy(),
                g=x[iG:iG + K * m].reshape(K, m).copy(), obj=res.fun, s_from=s_from)


# --------------------------------------------------------------------------
# 电价过程检验与两层建模
# --------------------------------------------------------------------------
def price_tests(panel):
    a1 = C.read_attachment1()
    p4 = panel.P(volatile=True).to_numpy(float)
    skel = a1["price"].to_numpy(float)
    resid = p4 - skel[None, :]
    load = panel.L().to_numpy(float); pv = panel.PV().to_numpy(float)
    out = {}
    out["skeleton_is_mean"] = float(np.max(np.abs(p4.mean(0) - skel)))
    out["price_mean"] = float(p4.mean()); out["price_sd"] = float(p4.std())
    out["price_min"] = float(p4.min()); out["price_max"] = float(p4.max())
    out["resid_sd_within"] = float(resid.std(0).mean())
    out["cv_within"] = float((resid.std(0) / skel).mean())
    # C1 正态性（标准化残差）
    z = resid / resid.std(0, keepdims=True)
    zz = z.ravel()
    jb = stats.jarque_bera(zz)
    out["C1_JB_stat"] = float(jb.statistic); out["C1_JB_p"] = float(jb.pvalue)
    out["C1_skew"] = float(stats.skew(zz)); out["C1_kurt"] = float(stats.kurtosis(zz))
    # C2 自相关（日内残差序列，滞后 1..6）
    acf = [float(np.mean([np.corrcoef(resid[i, :-k], resid[i, k:])[0, 1]
                          for i in range(len(resid))])) for k in range(1, 7)]
    out["C2_acf_lag1_6"] = acf
    # C2b 日间自相关（同一槽位跨日）
    acf_day = [float(np.mean([np.corrcoef(resid[:-k, t], resid[k:, t])[0, 1]
                              for t in range(C.N_SLOT)])) for k in range(1, 4)]
    out["C2b_daily_acf"] = acf_day
    # C3 方差结构：各槽位残差标准差（谷/峰时段对比）
    sd_slot = resid.std(0)
    hrs = np.array([t // 6 for t in range(C.N_SLOT)])
    out["C3_sd_night"] = float(sd_slot[hrs == 3].mean())     # 03:00-04:00 谷段
    out["C3_sd_peak"] = float(sd_slot[hrs == 19].mean())     # 19:00-20:00 峰段
    out["C3_sd_ratio"] = out["C3_sd_peak"] / out["C3_sd_night"]
    out["C3_sd_all"] = float(sd_slot.mean())
    # C4 相关性（池化 / 日聚合 / 去形状）
    net = load - pv
    out["C4_pool_pL"] = float(np.corrcoef(p4.ravel(), load.ravel())[0, 1])
    out["C4_pool_pNet"] = float(np.corrcoef(p4.ravel(), net.ravel())[0, 1])
    out["C4_pool_pPV"] = float(np.corrcoef(p4.ravel(), pv.ravel())[0, 1])
    out["C4_day_pL"] = float(np.corrcoef(p4.mean(1), load.mean(1))[0, 1])
    out["C4_day_pPV"] = float(np.corrcoef(p4.mean(1), pv.mean(1))[0, 1])
    dev_p = p4 - p4.mean(1, keepdims=True); dev_l = load - load.mean(1, keepdims=True)
    out["C4_deshape_pL"] = float(np.corrcoef(dev_p.ravel(), dev_l.ravel())[0, 1])
    return out, resid, skel


def build_joint_scenarios(panel, K=60, mode="hist", seed=7, block=36):
    """联合情景：mode='hist' 取代表性历史日；mode='boot' 分块 Bootstrap 合成。"""
    load = panel.L().to_numpy(float); pv = panel.PV().to_numpy(float)
    p4 = panel.P(volatile=True).to_numpy(float)
    idx = [list(DATE_ALL).index(d) for d in DATE_WINDOW]
    res_nd = np.zeros((len(idx), C.N_SLOT))
    net_nd = np.zeros((len(idx), C.N_SLOT))
    for i, k in enumerate(idx):
        fc = panel.pv_forecast(DATE_WINDOW[i], 0)
        res_nd[i] = (fc - pv[k]) * C.DT
        net_nd[i] = (load[k] - pv[k]) * C.DT
    if mode == "hist":
        # 二维分层：按"日净需求残差"与"日均电价"各分层（网格），层内取 medoid，
        # 权重 = 层样本占比 → 同时保真净需求残差与电价的边际分布，并保留联合结构。
        np_str = int(np.floor(np.sqrt(K)))
        nd_tot = res_nd.sum(1)
        p_avg = p4[idx].mean(1)
        q1 = np.quantile(nd_tot, np.linspace(0, 1, np_str + 1))
        q2 = np.quantile(p_avg, np.linspace(0, 1, np_str + 1))
        b1 = np.clip(np.digitize(nd_tot, q1[1:-1]), 0, np_str - 1)
        b2 = np.clip(np.digitize(p_avg, q2[1:-1]), 0, np_str - 1)
        cell = b1 * np_str + b2
        pos, w = [], []
        for cell_id in range(np_str * np_str):
            g = np.where(cell == cell_id)[0]
            if len(g) == 0:
                continue
            sub = res_nd[g]
            if len(g) == 1:
                pos.append(g[0])
            else:
                dist = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
                pos.append(g[int(np.argmin(dist.sum(1)))])
            w.append(len(g) / len(nd_tot))
        pos = np.array(pos); w = np.array(w)
        return dict(price=p4[idx][pos], resid=res_nd[pos], w=w,
                    net=net_nd[pos], src=np.array(DATE_WINDOW)[pos])
    rng = np.random.default_rng(seed)
    nblk = C.N_SLOT // block
    price_s, resid_s = [], []
    for _ in range(K):
        pr, rr = [], []
        for j in range(nblk):                     # 块按时刻对齐重采样，保住日内形状
            aa = int(rng.integers(0, len(idx)))
            k = idx[aa]; off = j * block
            pr.append(p4[k, off:off + block])
            fc = panel.pv_forecast(DATE_WINDOW[aa], 0)[off:off + block]
            rr.append((fc - pv[k, off:off + block]) * C.DT)
        price_s.append(np.concatenate(pr)); resid_s.append(np.concatenate(rr))
    return dict(price=np.array(price_s), resid=np.array(resid_s),
                w=np.full(K, 1.0 / K), net=None, src=None)


def fidelity(scen, panel, price_tests_out):
    """情景保真度：均值/分位数差、相关系数保持。"""
    p4 = panel.P(volatile=True).to_numpy(float)
    idx = [list(DATE_ALL).index(d) for d in DATE_WINDOW]
    p_true = p4[idx].ravel()
    p_s = scen["price"].ravel()
    qs = [1, 5, 25, 50, 75, 95, 99]
    qt = np.percentile(p_true, qs); qsi = np.percentile(p_s, qs)
    out = dict(mean_gap=float(abs(p_s.mean() - p_true.mean()) / p_true.mean()),
               max_q_gap=float(np.max(np.abs(qsi - qt) / np.abs(qt))),
               q_levels=qs, q_true=qt.tolist(), q_scen=qsi.tolist(),
               q_gap_rel=[float(abs(a - b) / abs(b)) for a, b in zip(qsi, qt)])
    load = panel.L().to_numpy(float)[idx]
    net = load - panel.PV().to_numpy(float)[idx]
    out["corr_true_pL"] = float(np.corrcoef(p_true, load.ravel())[0, 1])
    out["corr_true_pNet"] = float(np.corrcoef(p_true, net.ravel())[0, 1])
    if scen.get("net") is not None:
        w = scen["w"] / scen["w"].sum()
        rep = np.maximum((w * len(p_true)).round().astype(int), 0)
        ps, ns = [], []
        for k, rr in enumerate(rep):
            if rr == 0:
                continue
            ps.append(np.repeat(scen["price"][k][None, :], rr, 0))
            ns.append(np.repeat(scen["net"][k][None, :], rr, 0))
        ps = np.vstack(ps).ravel(); ns = np.vstack(ns).ravel()
        out["corr_scen_pNet"] = float(np.corrcoef(ps, ns)[0, 1])
    else:
        out["corr_scen_pNet"] = float("nan")
    return out

# --------------------------------------------------------------------------
# 波动电价下重算问题 2（Q4-2）
# --------------------------------------------------------------------------
def run_q4_2(scen, verbose=True, max_days=None, write=False, tag="q42"):
    panel = C.Panel(DATE_ALL)
    load = panel.L(); pv_act = panel.PV()
    p4 = panel.P(volatile=True)
    skel = C.read_attachment1()["price"].to_numpy(float)
    dates = DATE_ALL if max_days is None else DATE_ALL[:max_days]
    recs, s_init = [], C.S_INIT
    t0 = time.time()
    for i, d in enumerate(dates):
        L = load.loc[d].to_numpy(float); PVa = pv_act.loc[d].to_numpy(float)
        nd_act = (L - PVa) * C.DT
        fc = panel.pv_forecast(d, 0)
        nd_fcst = (L - fc) * C.DT
        nd_scen = nd_fcst[None, :] + scen["resid"]
        sol = solve_vol(scen["price"], nd_fcst, nd_scen, scen["w"], s_init=s_init,
                        s_from=0, term_floor=s_init)
        g = np.maximum(nd_act + sol["c"] - sol["d"] - sol["E"], 0.0)
        pact = p4.loc[d].to_numpy(float)
        buy = float(np.sum(pact * sol["E"]))
        emg = float(C.EMERGENCY_MULT * np.sum(pact * g))
        if d in DATE_WINDOW:
            recs.append(dict(date=d, E=sol["E"].copy(), c=sol["c"].copy(),
                             d=sol["d"].copy(), S=sol["S"].copy(), g=g,
                             s_init=s_init, s_end=float(sol["S"][-1]),
                             cost_buy=buy, cost_emg=emg, total_cost=buy + emg,
                             emg_qty=float(g.sum()), plan_energy=float(sol["E"].sum()),
                             base_cost=C.baseline_cost(pact, L, PVa),
                             base_cost_skel=C.baseline_cost(skel, L, PVa)))
        s_init = float(sol["S"][-1])
        if verbose and (i + 1) % 60 == 0:
            print(f"  ... {i+1}/{len(dates)} 天 {time.time()-t0:.1f}s")
    if write:
        dates_w = [r["date"] for r in recs]
        path = os.path.join(C.RESULT_DIR, "result4-2.xlsx")
        RIO.write_result(path, dates_w,
                         np.vstack([r["E"] for r in recs]),
                         np.vstack([r["c"] for r in recs]),
                         np.vstack([r["d"] for r in recs]),
                         np.array([r["s_init"] for r in recs]),
                         np.array([r["s_end"] for r in recs]),
                         skel, g=np.vstack([r["g"] for r in recs]),
                         price_mat=p4.loc[dates_w].to_numpy(float))
        print("已写出", path)
    return recs


# --------------------------------------------------------------------------
# 波动电价下重算问题 3（Q4-3）：滚动调整
# --------------------------------------------------------------------------
def run_q4_3(scen, epochs=(0, 6, 12, 18), K=60, verbose=True, max_days=None,
             write=False, perfect_price=False, tag="q43"):
    panel = C.Panel(DATE_ALL)
    load = panel.L(); pv_act = panel.PV()
    p4 = panel.P(volatile=True)
    skel = C.read_attachment1()["price"].to_numpy(float)
    # 各预报时刻的净需求残差场景（该时刻预报 vs 实际）
    scens = {}
    for h in epochs:
        s_m = C.EPOCH_SLOT[h]
        slots = np.arange(s_m, C.N_SLOT)
        cen, w = Q3.build_epoch_scenarios(panel, h, K=K, slots=slots)
        scens[h] = dict(slots=slots, cen=cen, w=w)
    dates = DATE_ALL if max_days is None else DATE_ALL[:max_days]
    recs, s_init = [], C.S_INIT
    t0 = time.time()
    for i, d in enumerate(dates):
        L = load.loc[d].to_numpy(float); PVa = pv_act.loc[d].to_numpy(float)
        nd_act = (L - PVa) * C.DT
        pact = p4.loc[d].to_numpy(float)
        ps = pact[None, :] if perfect_price else scen["price"]
        s_day = s_init
        E_plan = np.zeros(C.N_SLOT); E_adj = np.zeros(C.N_SLOT)
        cc = np.zeros(C.N_SLOT); dd = np.zeros(C.N_SLOT)
        soc = s_init
        for h in epochs:
            s_m = C.EPOCH_SLOT[h]; slots = scens[h]["slots"]
            if perfect_price and h > 0:
                price_sc = pact[None, :]
            else:
                price_sc = ps
            fc = Q3.forecast_slots(panel, d, h, slots)
            nd_fc = np.zeros(C.N_SLOT); nd_fc[slots] = (L[slots] - fc) * C.DT
            cen = scens[h]["cen"]
            nd_sc = np.zeros((cen.shape[0], C.N_SLOT))
            nd_sc[:, slots] = nd_fc[slots][None, :] + cen
            sol = solve_vol(price_sc, nd_fc, nd_sc, scens[h]["w"], s_init=soc,
                            s_from=s_m, plan=(E_plan if h > 0 else None),
                            term_floor=s_day)
            E = sol["E"]
            if h == 0:
                E_plan = E.copy()
            later = [x for x in epochs if x > h]
            nxt = C.EPOCH_SLOT[later[0]] if later else C.N_SLOT
            for j, t in enumerate(range(s_m, nxt)):
                E_adj[t] = E[j]; cc[t] = sol["c"][j]; dd[t] = sol["d"][j]
            soc = float(sol["S"][nxt - s_m - 1])
        g = np.maximum(nd_act + cc - dd - E_adj, 0.0)
        dev = E_adj - E_plan
        u = np.maximum(-dev, 0.0); v = np.maximum(dev, 0.0)
        cost_buy = float(np.sum(pact * np.minimum(E_plan, E_adj)))
        cost_breach = float(C.BREACH_MULT * np.sum(pact * u))
        cost_excess = float(C.EXCESS_MULT * np.sum(pact * v))
        cost_emg = float(C.EMERGENCY_MULT * np.sum(pact * g))
        if d in DATE_WINDOW:
            recs.append(dict(date=d, E_plan=E_plan.copy(), E_adj=E_adj.copy(),
                             c=cc.copy(), d=dd.copy(), g=g, s_init=s_day,
                             s_end=soc, cost_buy=cost_buy, cost_breach=cost_breach,
                             cost_excess=cost_excess, cost_emg=cost_emg,
                             total_cost=cost_buy + cost_breach + cost_excess + cost_emg,
                             emg_qty=float(g.sum()), plan_energy=float(E_plan.sum()),
                             adj_energy=float(E_adj.sum()), breach_qty=float(u.sum()),
                             excess_qty=float(v.sum()),
                             base_cost=C.baseline_cost(pact, L, PVa)))
        s_init = soc
        if verbose and (i + 1) % 60 == 0:
            print(f"  ... {i+1}/{len(dates)} 天 {time.time()-t0:.1f}s")
    if write:
        dates_w = [r["date"] for r in recs]
        path = os.path.join(C.RESULT_DIR, "result4-3.xlsx")
        RIO.write_result(path, dates_w,
                         np.vstack([r["E_plan"] for r in recs]),
                         np.vstack([r["c"] for r in recs]),
                         np.vstack([r["d"] for r in recs]),
                         np.array([r["s_init"] for r in recs]),
                         np.array([r["s_end"] for r in recs]),
                         skel, g=np.vstack([r["g"] for r in recs]),
                         E_adj=np.vstack([r["E_adj"] for r in recs]),
                         price_mat=p4.loc[dates_w].to_numpy(float))
        print("已写出", path)
    return recs


# --------------------------------------------------------------------------
# CVaR 风险—费用前沿（Rockafellar–Uryasev 线性化）
# --------------------------------------------------------------------------
def solve_cvar_day(price_scen, nd_fcst, nd_scen, weights, s_init, lam=0.0,
                   alpha=0.95, term_floor=None):
    n = C.N_SLOT
    K = price_scen.shape[0]
    weights = np.asarray(weights, float); weights = weights / weights.sum()
    iE, iC, iD, iS, iG, iZ, iEta = 0, n, 2 * n, 3 * n, 4 * n, 4 * n + K * n, 4 * n + K * n + K
    nv = iEta + 1
    coef = (1 - lam) * weights[:, None] * price_scen
    c_obj = np.zeros(nv)
    c_obj[iE:iE + n] = coef.sum(0)
    c_obj[iG:iG + K * n] = C.EMERGENCY_MULT * coef.ravel()
    c_obj[iZ:iZ + K] = lam * weights / (1 - alpha)
    c_obj[iEta] = lam
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for t in range(n):
        for a, b, c_ in ((r, iE + t, -1.0), (r, iC + t, 1.0), (r, iD + t, -1.0)):
            rows.append(a); cols.append(b); vals.append(c_)
        rhs.append(-nd_fcst[t]); r += 1
    for w in range(K):
        for t in range(n):
            for a, b, c_ in ((r, iE + t, -1.0), (r, iC + t, 1.0),
                             (r, iD + t, -1.0), (r, iG + w * n + t, -1.0)):
                rows.append(a); cols.append(b); vals.append(c_)
            rhs.append(-nd_scen[w, t]); r += 1
    for w in range(K):          # z_w + eta >= cost_w  ->  cost_w - z_w - eta <= 0
        for t in range(n):
            rows.append(r); cols.append(iE + t); vals.append(price_scen[w, t])
            rows.append(r); cols.append(iG + w * n + t); vals.append(C.EMERGENCY_MULT * price_scen[w, t])
        rows.append(r); cols.append(iZ + w); vals.append(-1.0)
        rows.append(r); cols.append(iEta); vals.append(-1.0)
        rhs.append(0.0); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    b_ub = np.array(rhs)
    rows2, cols2, vals2, rhs2 = [], [], [], []
    for t in range(n):
        for b, c_ in ((iS + t, 1.0), (iC + t, -C.ETA_C), (iD + t, 1.0 / C.ETA_D)):
            rows2.append(t); cols2.append(b); vals2.append(c_)
        if t > 0:
            rows2.append(t); cols2.append(iS + t - 1); vals2.append(-1.0)
        rhs2.append(0.0 if t > 0 else s_init)
    A_eq = coo_matrix((vals2, (rows2, cols2)), shape=(n, nv)).tocsr()
    b_eq = np.array(rhs2)
    bounds = [(0.0, None)] * n + [(0.0, C.E_MAX)] * n + [(0.0, C.E_MAX)] * n + \
             [(C.S_MIN, C.S_MAX)] * n + [(0.0, None)] * (K * n) + \
             [(0.0, None)] * K + [(None, None)]
    if term_floor is not None:
        bounds[iS + n - 1] = (max(term_floor, C.S_MIN), C.S_MAX)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("CVaR LP 不可行：" + res.message)
    x = res.x
    return dict(E=x[iE:iE + n], c=x[iC:iC + n], d=x[iD:iD + n], S=x[iS:iS + n],
                g=x[iG:iG + K * n].reshape(K, n), eta=float(x[iEta]),
                obj=res.fun, z=x[iZ:iZ + K])


def weighted_cvar(cost, weights, alpha=0.95):
    """加权 Rockafellar–Uryasev 形式（与 solve_cvar_day 的 (eta, z) 线性化同口径）：

        CVaR_alpha(C) = eta + 1/(1-alpha) * sum_w w_w * max(0, C_w - eta)

    eta 取加权 alpha 分位（加权经验分布下该泛函的最小化点）。工况 w_omega 是分层
    medoid 情景集的**层样本占比**（非均匀，约 0.0030–0.0299），因此正部必须用同一组
    权重加权，不能用 1/|Omega| 的等权平均——否则报告值与优化器目标
    （c_obj[iZ] = lam*weights/(1-alpha)、c_obj[iEta] = lam）不同口径。
    返回 (CVaR, eta)。
    """
    cost = np.asarray(cost, float)
    w = np.asarray(weights, float)
    w = w / w.sum()
    order = np.argsort(cost, kind="stable")
    c_s, w_s = cost[order], w[order]
    cum = np.cumsum(w_s)
    i = int(np.searchsorted(cum, alpha, side="left"))
    eta = float(c_s[min(i, len(c_s) - 1)])
    return float(eta + np.sum(w * np.maximum(cost - eta, 0.0)) / (1.0 - alpha)), eta


# --------------------------------------------------------------------------
# 主程序
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tests",
                    choices=["tests", "q42", "q43", "cvar", "all"])
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--scen", default="hist", choices=["hist", "boot"])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--lams", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--step", type=int, default=6, help="CVaR 前沿子样本步长")
    args = ap.parse_args()

    panel = C.Panel(DATE_ALL)
    out = {}
    if args.mode in ("tests", "all"):
        pt, resid, skel = price_tests(panel)
        scen_h = build_joint_scenarios(panel, K=args.K, mode="hist")
        scen_b = build_joint_scenarios(panel, K=args.K, mode="boot")
        fid_h = fidelity(scen_h, panel, pt)
        fid_b = fidelity(scen_b, panel, pt)
        print("=" * 78)
        for k, v in pt.items():
            print("  %-22s %s" % (k, np.round(v, 3) if isinstance(v, list) else v))
        print("  保真度(历史代表日):", {k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in fid_h.items() if k not in ("q_true", "q_scen")})
        print("  保真度(分块Bootstrap):", {k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in fid_b.items() if k not in ("q_true", "q_scen")})
        print("=" * 78)
        out["price_tests"] = pt
        out["fidelity_hist"] = fid_h
        out["fidelity_boot"] = fid_b
        with open(os.path.join(C.INTERMEDIATE_DIR, "q4_price.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

    if args.mode in ("q42", "q43", "all"):
        scen = build_joint_scenarios(panel, K=args.K, mode=args.scen)
        if args.mode in ("q42", "all"):
            recs = run_q4_2(scen, max_days=args.max_days, write=args.write, tag="q42")
            tot = dict(cost_buy=sum(r["cost_buy"] for r in recs),
                       cost_emg=sum(r["cost_emg"] for r in recs),
                       total=sum(r["total_cost"] for r in recs),
                       base=sum(r["base_cost"] for r in recs),
                       emg_qty=sum(r["emg_qty"] for r in recs),
                       days_emg=int(sum(1 for r in recs if r["emg_qty"] > 1e-6)))
            print("Q4-2:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tot.items()})
            print("  节省率 %.2f%%" % ((tot["base"] - tot["total"]) / tot["base"] * 100))
            with open(os.path.join(C.INTERMEDIATE_DIR, "q4_2.json"), "w", encoding="utf-8") as f:
                json.dump(dict(meta=tot, daily=[{k: (v.isoformat() if k == "date" else
                                                     (v.tolist() if isinstance(v, np.ndarray) else v))
                                                for k, v in r.items()} for r in recs]),
                          f, ensure_ascii=False, indent=1)
        if args.mode in ("q43", "all"):
            recs3 = run_q4_3(scen, K=args.K, max_days=args.max_days, write=args.write)
            tot3 = {k: sum(r[k] for r in recs3) for k in
                    ("cost_buy", "cost_breach", "cost_excess", "cost_emg", "total_cost",
                     "base_cost", "emg_qty", "breach_qty", "excess_qty")}
            print("Q4-3:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tot3.items()})
            print("  节省率 %.2f%%" % ((tot3["base_cost"] - tot3["total_cost"]) / tot3["base_cost"] * 100))
            with open(os.path.join(C.INTERMEDIATE_DIR, "q4_3.json"), "w", encoding="utf-8") as f:
                json.dump(dict(meta=tot3, daily=[{k: (v.isoformat() if k == "date" else
                                                     (v.tolist() if isinstance(v, np.ndarray) else v))
                                                for k, v in r.items()} for r in recs3]),
                          f, ensure_ascii=False, indent=1)

    if args.mode in ("cvar", "all"):
        scen = build_joint_scenarios(panel, K=args.K, mode=args.scen)
        w_sc = np.asarray(scen["w"], float); w_sc = w_sc / w_sc.sum()   # 层样本占比，非均匀
        lams = [float(x) for x in args.lams.split(",")]
        sub = list(DATE_WINDOW)[::args.step]
        load = panel.L(); pv_act = panel.PV(); p4 = panel.P(volatile=True)
        res = []
        for lam in lams:
            exp_cost, real_cost, cvar_val, emg = 0.0, 0.0, 0.0, 0.0
            cvar_eq = 0.0                     # 等权旧口径，仅留作审计对照
            for d in sub:
                L = load.loc[d].to_numpy(float); PVa = pv_act.loc[d].to_numpy(float)
                nd_act = (L - PVa) * C.DT
                fc = panel.pv_forecast(d, 0)
                nd_fc = (L - fc) * C.DT
                nd_sc = nd_fc[None, :] + scen["resid"]
                sol = solve_cvar_day(scen["price"], nd_fc, nd_sc, scen["w"],
                                     C.S_INIT, lam=lam, alpha=0.95, term_floor=C.S_INIT)
                pact = p4.loc[d].to_numpy(float)
                g = np.maximum(nd_act + sol["c"] - sol["d"] - sol["E"], 0.0)
                real = float(np.sum(pact * (sol["E"] + C.EMERGENCY_MULT * g)))
                g_sc = np.maximum(nd_sc + sol["c"][None, :] - sol["d"][None, :]
                                  - sol["E"][None, :], 0)
                sc_cost = (scen["price"] * (sol["E"][None, :]
                                            + C.EMERGENCY_MULT * g_sc)).sum(1)
                exp_cost += float(scen["w"] @ sc_cost)
                real_cost += real; emg += float(g.sum())
                # 报告口径：与优化器同权重的加权 (eta, z) 形式
                cv, _eta = weighted_cvar(sc_cost, w_sc, 0.95)
                cvar_val += cv
                var_eq = np.quantile(sc_cost, 0.95)
                cvar_eq += float(var_eq + np.mean(np.maximum(sc_cost - var_eq, 0)) / 0.05)
            n = len(sub)
            res.append(dict(lam=lam, exp_cost=exp_cost / n, real_cost=real_cost / n,
                            cvar=cvar_val / n, emg_qty=emg / n,
                            cvar_equalweight=cvar_eq / n))
            print("  lam=%.1f  期望费用 %.2f  真实回测 %.2f  CVaR95 %.2f  紧急 %.1f kWh/日"
                  % (lam, exp_cost / n, real_cost / n, cvar_val / n, emg / n))
        with open(os.path.join(C.INTERMEDIATE_DIR, "q4_cvar.json"), "w", encoding="utf-8") as f:
            json.dump(dict(days=[str(pd.Timestamp(x).date()) for x in sub],
                           frontier=res,
                           note="cvar = 加权 Rockafellar–Uryasev (eq.25)，权重 = 分层 medoid 情景的层样本占比；"
                                "cvar_equalweight = 旧等权 1/|Omega| 口径，仅作审计对照"),
                      f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    main()
