# -*- coding: utf-8 -*-
"""
问题 3：多预报时刻下的滚动调整（多阶段 / MPC）
==============================================
信息结构：
  0:00 用附件 3 的 0:00 预报制定**全天**计划购电量 E^plan（144 槽，第一阶段承诺）；
  6:00/12:00/18:00 用当次预报对**其后至 24:00** 的剩余时段重新决策（后覆盖前），
  实际执行的是"最后覆盖该时段"的那次决策，记为调整购电量 E^adj。
  → 最终：槽 0-35 由 0:00 计划决定，36-71 由 6:00 决定，72-107 由 12:00 决定，
         108-143 由 18:00 决定（非预期性自然满足：每个决策只用该时刻及之前的预报）。

费用口径 C-06（偏差定价）：
  cost_t = p_t*min(E^plan,E^adj) + 0.5 p_t (E^plan-E^adj)^+ + 1.5 p_t (E^adj-E^plan)^+
  另加紧急购电 5 p_t g_t。第九章给出"计划费全额 + 差额惩罚"另一口径的对照。

VoI 实验：预报时刻数 n ∈ {1,2,4,∞}，同一场景集、同一实际轨迹回放。
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


def forecast_slots(panel, date, hour, slots):
    """取 date 日 hour 时刻发布的预报在给定槽位上的值（kW）。"""
    row = panel.pf3.loc[(pd.Timestamp(date).normalize(), hour)]
    hourly = np.array([row[f"lead{k}"] for k in range(1, 25)], dtype=float)
    h = np.array([t // 6 for t in slots])          # 槽位所属整点（0..23）
    lead = (h - hour) % 24                         # 该整点对应的预报提前期
    return hourly[lead]


def build_epoch_scenarios(panel, hour, K=40, slots=None, dates=DATE_WINDOW):
    """构造"hour 时刻预报"的残差场景集（仅覆盖 slots），按窗口总残差分层削减。"""
    pv_act = panel.PV()
    slots = np.arange(C.N_SLOT) if slots is None else np.asarray(slots)
    R = np.zeros((len(dates), len(slots)))
    for i, d in enumerate(dates):
        fc = forecast_slots(panel, d, hour, slots)
        R[i] = (fc - pv_act.loc[d].to_numpy(float)[slots]) * C.DT
    cen, w = LP.reduce_scenarios_stratified(R, K=K)
    return cen, w


def run_q3(epochs=(0, 6, 12, 18), K=40, term_mode="ge", verbose=True,
           perfect=False, max_days=None, tag="main", seed=0):
    panel = C.Panel(DATE_ALL)
    load = panel.L(); pv_act = panel.PV()
    price = C.read_attachment1()["price"].to_numpy(float)

    scens = {}
    for h in epochs:
        s_m = C.EPOCH_SLOT[h]
        slots = np.arange(s_m, C.N_SLOT)
        cen, w = build_epoch_scenarios(panel, h, K=K, slots=slots)
        scens[h] = dict(slots=slots, cen=cen, w=w)
        if verbose:
            print(f"[场景] 时刻 {h:>2}:00  槽位 {s_m}..143  代表场景 {len(w)}  "
                  f"残差均值绝对值 {np.abs(cen).mean():.2f} kWh")

    dates = DATE_ALL if max_days is None else DATE_ALL[:max_days]
    recs, s_init = [], C.S_INIT
    t0 = time.time()
    for i, d in enumerate(dates):
        L = load.loc[d].to_numpy(float)
        PVa = pv_act.loc[d].to_numpy(float)
        nd_act = (L - PVa) * C.DT
        s_day = s_init
        E_plan = np.zeros(C.N_SLOT); E_adj = np.zeros(C.N_SLOT)
        cc = np.zeros(C.N_SLOT); dd = np.zeros(C.N_SLOT); SS = np.zeros(C.N_SLOT)
        soc = s_init
        for h in epochs:
            s_m = C.EPOCH_SLOT[h]
            slots = scens[h]["slots"]
            if perfect and h > 0:
                nd_fc = nd_act.copy()
            else:
                fc = forecast_slots(panel, d, h, slots)
                nd_fc = np.zeros(C.N_SLOT)
                nd_fc[slots] = (L[slots] - fc) * C.DT
            cen, w = scens[h]["cen"], scens[h]["w"]
            nd_sc = np.zeros((cen.shape[0], C.N_SLOT))
            nd_sc[:, slots] = nd_fc[slots][None, :] + cen
            sol = LP.solve_epoch(price, nd_fc, nd_sc, w, s_init=soc, s_from=s_m,
                                 plan=(E_plan if h > 0 else None), term_floor=s_day)
            E = sol["E"]
            if h == 0:
                E_plan = E.copy()
            nxt = C.N_SLOT if h == np.max(epochs) else C.EPOCH_SLOT[
                [x for x in epochs if x > h][0]]
            for j, t in enumerate(range(s_m, nxt)):
                E_adj[t] = E[j]; cc[t] = sol["c"][j]; dd[t] = sol["d"][j]
                SS[t] = sol["S"][j]
            soc = float(SS[nxt - 1])
        # ---- 结算（用实际数据）----
        g = np.maximum(nd_act + cc - dd - E_adj, 0.0)
        dev = E_adj - E_plan
        u = np.maximum(-dev, 0.0); v = np.maximum(dev, 0.0)
        cost_buy = float(np.sum(price * (np.minimum(E_plan, E_adj) + 0.0)))
        cost_breach = float(np.sum(C.BREACH_MULT * price * u))
        cost_excess = float(np.sum(C.EXCESS_MULT * price * v))
        cost_emg = float(C.EMERGENCY_MULT * np.sum(price * g))
        is_win = d in DATE_WINDOW
        if is_win:
            recs.append(dict(
                date=d, E_plan=E_plan.copy(), E_adj=E_adj.copy(), c=cc.copy(),
                d=dd.copy(), S=SS.copy(), g=g, s_init=s_day, s_end=float(SS[-1]),
                cost_buy=cost_buy, cost_breach=cost_breach, cost_excess=cost_excess,
                cost_emg=cost_emg,
                total_cost=cost_buy + cost_breach + cost_excess + cost_emg,
                plan_energy=float(E_plan.sum()), adj_energy=float(E_adj.sum()),
                emg_qty=float(g.sum()),
                breach_qty=float(u.sum()), excess_qty=float(v.sum()),
                base_cost=C.baseline_cost(price, L, PVa),
                max_viol=float(max(abs(nd_act + cc - dd - E_adj).max(), 0.0))))
        s_init = soc
        if verbose and (i + 1) % 40 == 0:
            print(f"  ... {i+1}/{len(dates)} 天  {time.time()-t0:.1f}s")
    return recs, scens, price


def summary(recs, price, tag=""):
    tot = {k: sum(r[k] for r in recs) for k in
           ("cost_buy", "cost_breach", "cost_excess", "cost_emg", "total_cost",
            "plan_energy", "adj_energy", "emg_qty", "breach_qty", "excess_qty",
            "base_cost")}
    tot["n_days"] = len(recs)
    tot["days_with_emg"] = int(sum(1 for r in recs if r["emg_qty"] > 1e-6))
    tot["tag"] = tag
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="0,6,12,18")
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--perfect", action="store_true")
    ap.add_argument("--tag", default="n4")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    epochs = tuple(int(x) for x in args.epochs.split(","))
    recs, scens, price = run_q3(epochs=epochs, K=args.K, perfect=args.perfect,
                                tag=args.tag, max_days=args.max_days)
    tot = summary(recs, price, args.tag)
    for k in ("cost_buy", "cost_breach", "cost_excess", "cost_emg", "total_cost",
              "base_cost"):
        print(f"  {k:14s} {tot[k]:16.2f}")
    print(f"  总费用 {tot['total_cost']:.2f} 元 ({tot['total_cost']/1e4:.2f} 万元) "
          f"节省率 {(tot['base_cost']-tot['total_cost'])/tot['base_cost']*100:.2f}% "
          f"紧急 {tot['emg_qty']:.0f} kWh {tot['days_with_emg']}/{tot['n_days']} 天 "
          f"违约 {tot['breach_qty']:.0f} 超额 {tot['excess_qty']:.0f} kWh")
    out = dict(meta=tot, daily=[{k: (v.isoformat() if k == "date" else
                                     (v.tolist() if isinstance(v, np.ndarray) else v))
                                for k, v in r.items()} for r in recs])
    with open(os.path.join(C.INTERMEDIATE_DIR, f"q3_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if args.write and not args.perfect:
        dates = [r["date"] for r in recs]
        path = os.path.join(C.RESULT_DIR, "result3.xlsx")
        RIO.write_result(path, dates,
                         np.vstack([r["E_plan"] for r in recs]),
                         np.vstack([r["c"] for r in recs]),
                         np.vstack([r["d"] for r in recs]),
                         np.array([r["s_init"] for r in recs]),
                         np.array([r["s_end"] for r in recs]),
                         price, g=np.vstack([r["g"] for r in recs]),
                         E_adj=np.vstack([r["E_adj"] for r in recs]))
        print("已写出", path)
    return out


if __name__ == "__main__":
    main()
