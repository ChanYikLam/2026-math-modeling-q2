# -*- coding: utf-8 -*-
"""
v2 实验：电价—负荷—光伏相依结构（C 藤 Copula）与联合情景生成器对照
==================================================================
对应参考论文：
  · 宁楠《基于动态 C 藤 pair-Copula 的电力系统动态经济调度》
  · 崔丽敏等《基于核密度估计和 Copula 函数的风-光-荷联合场景生成》
  · 李翔 / 孙丰杰：聚类式场景削减

四部分：
  P1 三变量 C 藤拟合（日均价、日电量负载、日电量光伏）——族选择、Kendall tau、尾相依；
  P2 决策相关水平对（日均价 P，日总预报残差 R）的 KDE + Copula 拟合；
  P3 情景生成器对照实验（样本外）：历史分层 / 分块 Bootstrap / KDE+Copula / 按当日光伏规模缩放，
     统一在 2025-02-01..06-30 训练、2025-07-01..12-31 上按真实轨迹结算；
  P4 分时段 CVaR 风险画像（由 q2_main.json 与 q4_2.json 的逐日解复算）。

输出 intermediate/exp_copula.json
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
from scipy import stats

import config as C
import copula as CP
import lp_engine as LP
import q4 as Q4
import scen_copula as SC

DATE_ALL = SC.DATE_ALL
TRAIN = pd.date_range("2025-02-01", "2025-06-30", freq="D")
TEST = pd.date_range("2025-07-01", "2025-12-31", freq="D")
K_MAIN = 60


# --------------------------------------------------------------------------
# P1 三变量 C 藤
# --------------------------------------------------------------------------
def part1_vine(panel, dates):
    lv = SC.daily_levels(panel, dates)
    X = np.column_stack([lv["price"].mean(1),
                         lv["load"].sum(1) * C.DT,
                         lv["pv"].sum(1) * C.DT])
    vine = CP.CVine3(names=("price", "load", "pv"))
    rows = vine.fit(X, verbose=False)
    P_, L_, V_ = X[:, 0], X[:, 1], X[:, 2]
    plain = {
        "tau_price_load": float(stats.kendalltau(P_, L_).statistic),
        "tau_price_pv": float(stats.kendalltau(P_, V_).statistic),
        "tau_load_pv": float(stats.kendalltau(L_, V_).statistic),
        "pearson_price_load": float(np.corrcoef(P_, L_)[0, 1]),
        "pearson_price_pv": float(np.corrcoef(P_, V_)[0, 1]),
        "pearson_load_pv": float(np.corrcoef(L_, V_)[0, 1]),
        "n_days": int(len(P_)),
    }
    return dict(plain=plain, vine=[{k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                    for k, v in r.items()} for r in rows])


# --------------------------------------------------------------------------
# P2 水平对 (P, R) 的 KDE + Copula
# --------------------------------------------------------------------------
def part2_level_copula(panel, dates):
    lv = SC.daily_levels(panel, dates)
    cop = SC.LevelCopula()
    cop.fit(lv["P"], lv["R"])
    s = cop.summary()
    s["n_days"] = int(len(lv["P"]))
    s["tau_emp_PR"] = float(stats.kendalltau(lv["P"], lv["R"]).statistic)
    s["pearson_emp_PR"] = float(np.corrcoef(lv["P"], lv["R"])[0, 1])
    # 槽级：预报残差与当日电价残差形状的相关（去骨架后）
    skel = C.read_attachment1()["price"].to_numpy(float)
    pres = lv["price"] - skel[None, :]
    s["pearson_deshape_presid"] = float(np.corrcoef(
        (pres - pres.mean(1, keepdims=True)).ravel(),
        (lv["resid"] - lv["resid"].mean(1, keepdims=True)).ravel())[0, 1])
    s["pearson_level_presid"] = float(np.corrcoef(pres.mean(1), lv["R"])[0, 1])
    s["tau_level_presid"] = float(stats.kendalltau(pres.mean(1), lv["R"]).statistic)
    # 残差规模与光伏水平的关系（条件标度律）
    pvfc = np.vstack([panel.pv_forecast(d, 0) for d in dates]).sum(1) * C.DT
    s["absR_vs_pvfc_pearson"] = float(np.corrcoef(np.abs(lv["R"]), pvfc)[0, 1])
    s["relR_mean"] = float((lv["R"] / pvfc).mean())
    s["relR_sd"] = float((lv["R"] / pvfc).std(ddof=1))
    q = np.quantile(pvfc, np.linspace(0, 1, 6))
    band = np.clip(np.digitize(pvfc, q[1:-1]), 0, 4)
    s["relR_by_band"] = [dict(band=int(b), n=int((band == b).sum()),
                              pv_median=float(np.median(pvfc[band == b])),
                              absR_mean=float(np.abs(lv["R"][band == b]).mean()),
                              relR_mean=float((np.abs(lv["R"][band == b]) / pvfc[band == b]).mean()))
                         for b in range(5)]
    return s, cop


# --------------------------------------------------------------------------
# P3 情景生成器对照
# --------------------------------------------------------------------------
def hist_strat_scenarios(panel, train_dates, K=K_MAIN):
    """历史代表日（二维分层 + 层内 medoid），只在训练窗内取代表。"""
    p4 = panel.P(volatile=True)
    idx = [list(DATE_ALL).index(d) for d in train_dates]
    price = p4.to_numpy(float)[idx]
    resid = np.vstack([(panel.pv_forecast(d, 0) - panel.PV().loc[d].to_numpy(float)) * C.DT
                       for d in train_dates])
    np_str = int(np.floor(np.sqrt(K)))
    nd_tot = resid.sum(1); p_avg = price.mean(1)
    q1 = np.quantile(nd_tot, np.linspace(0, 1, np_str + 1))
    q2 = np.quantile(p_avg, np.linspace(0, 1, np_str + 1))
    b1 = np.clip(np.digitize(nd_tot, q1[1:-1]), 0, np_str - 1)
    b2 = np.clip(np.digitize(p_avg, q2[1:-1]), 0, np_str - 1)
    cell = b1 * np_str + b2
    pos, w = [], []
    for cid in range(np_str * np_str):
        g = np.where(cell == cid)[0]
        if len(g) == 0:
            continue
        sub = resid[g]
        if len(g) == 1:
            pos.append(g[0])
        else:
            dist = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
            pos.append(g[int(np.argmin(dist.sum(1)))])
        w.append(len(g) / len(nd_tot))
    pos = np.array(pos)
    return dict(price=price[pos], resid=resid[pos], w=np.array(w) / np.sum(w),
                kind="fixed")


def bootstrap_scenarios(panel, train_dates, K=K_MAIN, seed=13, block=36):
    p4 = panel.P(volatile=True)
    idx = [list(DATE_ALL).index(d) for d in train_dates]
    price = p4.to_numpy(float)[idx]
    resid = np.vstack([(panel.pv_forecast(d, 0) - panel.PV().loc[d].to_numpy(float)) * C.DT
                       for d in train_dates])
    rng = np.random.default_rng(seed)
    nb = C.N_SLOT // block
    ps, rs = [], []
    for _ in range(K):
        pr, rr = [], []
        for j in range(nb):
            a = int(rng.integers(0, len(idx))); off = j * block
            pr.append(price[a, off:off + block]); rr.append(resid[a, off:off + block])
        ps.append(np.concatenate(pr)); rs.append(np.concatenate(rr))
    return dict(price=np.array(ps), resid=np.array(rs),
                w=np.full(K, 1.0 / K), kind="fixed")


def eval_policy(scen, panel, test_dates, rel=None, verbose=False):
    """在测试窗上按真实轨迹结算（缺口按 5 倍实际电价紧急购电）。"""
    load = panel.L(); pv_act = panel.PV(); p4 = panel.P(volatile=True)
    s_init = C.S_INIT
    tot = emg = emg_q = base = 0.0
    dc, daily = [], []
    for d in test_dates:
        L = load.loc[d].to_numpy(float); PVa = pv_act.loc[d].to_numpy(float)
        nd_act = (L - PVa) * C.DT
        fc = panel.pv_forecast(d, 0); nd_fcst = (L - fc) * C.DT
        if scen["kind"] == "scaled":
            scale = fc.sum() * C.DT
            resid = rel["rel"] * scale; w = rel["w"]
        else:
            resid = scen["resid"]; w = scen["w"]
        sol = Q4.solve_vol(scen["price"], nd_fcst, nd_fcst[None, :] + resid, w,
                           s_init=s_init, s_from=0, term_floor=s_init)
        g = np.maximum(nd_act + sol["c"] - sol["d"] - sol["E"], 0.0)
        pact = p4.loc[d].to_numpy(float)
        c1 = float(np.sum(pact * sol["E"])); c2 = float(C.EMERGENCY_MULT * np.sum(pact * g))
        tot += c1 + c2; emg += c2; emg_q += float(g.sum())
        base += C.baseline_cost(pact, L, PVa)
        dc.append(c1 + c2)
        daily.append(dict(date=str(d.date()), cost=c1 + c2, emg_qty=float(g.sum()),
                          plan_energy=float(sol["E"].sum())))
        s_init = float(sol["S"][-1])
    dc = np.array(dc)
    q95 = float(np.quantile(dc, 0.95))
    return dict(n_days=len(test_dates), total_cost=tot, emg_cost=emg, emg_qty=emg_q,
                base_cost=base, saving=float((base - tot) / base),
                cvar95=float(dc[dc >= q95].mean()), worst_day=float(dc.max()),
                mean_day=float(dc.mean()), sd_day=float(dc.std(ddof=1)),
                daily=daily)


def part3_generators(panel):
    out = {}
    t0 = time.time()
    hist = hist_strat_scenarios(panel, TRAIN)
    out["fidelity_hist"] = SC.fidelity(hist, panel, TRAIN)
    out["hist_strat"] = eval_policy(hist, panel, TEST)
    out["hist_strat"].pop("daily")
    print("  hist_strat %.1fs 总费用 %.2f" % (time.time() - t0, out["hist_strat"]["total_cost"]))

    boot = bootstrap_scenarios(panel, TRAIN)
    out["fidelity_boot"] = SC.fidelity(boot, panel, TRAIN)
    out["bootstrap"] = eval_policy(boot, panel, TEST)
    out["bootstrap"].pop("daily")
    print("  bootstrap %.1fs 总费用 %.2f" % (time.time() - t0, out["bootstrap"]["total_cost"]))

    cop = SC.generate(panel, TRAIN, K=K_MAIN, n_draw=600, seed=7, k_nn=10, reduce="strat")
    out["fidelity_copula"] = SC.fidelity(cop, panel, TRAIN)
    out["copula"] = eval_policy(cop, panel, TEST)
    out["copula"].pop("daily")
    print("  copula %.1fs 总费用 %.2f" % (time.time() - t0, out["copula"]["total_cost"]))

    rel = SC.generate_relative(panel, TRAIN, K=len(hist["w"]))
    scaled = dict(price=hist["price"], resid=None, w=hist["w"], kind="scaled")
    out["scaled"] = eval_policy(scaled, panel, TEST, rel=rel)
    out["scaled"].pop("daily")
    out["rel_summary"] = dict(rel_mu=rel["rel_mu"], rel_sd=rel["rel_sd"])
    print("  scaled %.1fs 总费用 %.2f" % (time.time() - t0, out["scaled"]["total_cost"]))

    base = out["hist_strat"]["total_cost"]
    for k in ("bootstrap", "copula", "scaled"):
        out[k]["diff_vs_hist"] = out[k]["total_cost"] - base
        out[k]["diff_pct"] = (out[k]["total_cost"] / base - 1) * 100
    return out


# --------------------------------------------------------------------------
# P4 分时段 CVaR 风险画像
# --------------------------------------------------------------------------
def cvar(x, alpha=0.95):
    x = np.asarray(x, float)
    v = np.quantile(x, alpha)
    tail = x[x >= v]
    return float(tail.mean()) if len(tail) else float(v)


def part4_risk(alpha=0.95):
    out = dict(alpha=alpha)
    # (a) 问题 2（固定价）
    j2 = json.load(open(os.path.join(C.INTERMEDIATE_DIR, "q2_main.json", encoding="utf-8")))
    price2 = C.read_attachment1()["price"].to_numpy(float)
    E2 = np.vstack([r["E"] for r in j2["daily"]])
    G2 = np.vstack([r["g"] for r in j2["daily"]])
    cost2 = price2[None, :] * E2 + C.EMERGENCY_MULT * price2[None, :] * G2
    out["q2_slot_cvar"] = [cvar(cost2[:, t], alpha) for t in range(C.N_SLOT)]
    out["q2_slot_mean"] = cost2.mean(0).tolist()
    out["q2_slot_shortage_freq"] = (G2 > 1e-9).mean(0).tolist()
    out["q2_total"] = float(cost2.sum())
    hour = np.arange(C.N_SLOT) // 6
    out["q2_hour_cvar"] = [float(np.sum([out["q2_slot_cvar"][t] for t in range(C.N_SLOT)
                                         if hour[t] == h])) for h in range(24)]
    out["q2_hour_mean"] = [float(np.sum([out["q2_slot_mean"][t] for t in range(C.N_SLOT)
                                         if hour[t] == h])) for h in range(24)]
    out["q2_hour_shortage_freq"] = [float(np.mean([out["q2_slot_shortage_freq"][t]
                                                   for t in range(C.N_SLOT) if hour[t] == h]))
                                    for h in range(24)]
    # 风险集中度：CVaR 最高的 12 个槽位占总 CVaR 的比重
    sc = np.array(out["q2_slot_cvar"])
    o = np.argsort(-sc)
    out["q2_top12_slot_share"] = float(sc[o[:12]].sum() / sc.sum())
    out["q2_top12_slots"] = [C.slot_label(int(t)) for t in sorted(o[:12])]

    # (b) 问题 4-2（波动价）
    j4 = json.load(open(os.path.join(C.INTERMEDIATE_DIR, "q4_2.json", encoding="utf-8")))
    p4 = C.Panel(DATE_ALL).P(volatile=True)
    dates = [pd.Timestamp(r["date"]) for r in j4["daily"]]
    Pact = p4.loc[dates].to_numpy(float)
    E4 = np.vstack([r["E"] for r in j4["daily"]])
    G4 = np.vstack([r["g"] for r in j4["daily"]])
    cost4 = Pact * E4 + C.EMERGENCY_MULT * Pact * G4
    out["q4_slot_cvar"] = [cvar(cost4[:, t], alpha) for t in range(C.N_SLOT)]
    out["q4_slot_mean"] = cost4.mean(0).tolist()
    out["q4_slot_shortage_freq"] = (G4 > 1e-9).mean(0).tolist()
    out["q4_total"] = float(cost4.sum())
    out["q4_hour_cvar"] = [float(np.sum([out["q4_slot_cvar"][t] for t in range(C.N_SLOT)
                                         if hour[t] == h])) for h in range(24)]
    out["q4_hour_mean"] = [float(np.sum([out["q4_slot_mean"][t] for t in range(C.N_SLOT)
                                         if hour[t] == h])) for h in range(24)]
    out["q4_hour_shortage_freq"] = [float(np.mean([out["q4_slot_shortage_freq"][t]
                                                   for t in range(C.N_SLOT) if hour[t] == h]))
                                    for h in range(24)]
    sc4 = np.array(out["q4_slot_cvar"]); o4 = np.argsort(-sc4)
    out["q4_top12_slot_share"] = float(sc4[o4[:12]].sum() / sc4.sum())
    out["q4_top12_slots"] = [C.slot_label(int(t)) for t in sorted(o4[:12])]
    # 风险比：CVaR/均值（尾部放大倍数）
    out["q2_cvar_over_mean"] = float(np.sum(out["q2_slot_cvar"]) / np.sum(out["q2_slot_mean"]))
    out["q4_cvar_over_mean"] = float(np.sum(out["q4_slot_cvar"]) / np.sum(out["q4_slot_mean"]))
    return out


def main():
    import sys
    parts = sys.argv[1] if len(sys.argv) > 1 else "1234"
    panel = C.Panel(DATE_ALL)
    prev = {}
    jpath = os.path.join(C.INTERMEDIATE_DIR, "exp_copula.json")
    if os.path.exists(jpath):
        prev = json.load(open(jpath, encoding="utf-8"))
    p1 = prev.get("part1_vine", None); p1_train = prev.get("part1_vine_train", None)
    p2 = prev.get("part2_level", None); p3 = prev.get("part3_generators", None)
    p4 = prev.get("part4_risk", None)
    WINDOW = pd.date_range(C.WIN_START, C.WIN_END, freq="D")
    if "1" in parts:
        print("[P1] 三变量 C 藤拟合（结果窗口 %d 天 / 训练窗 %d 天）..." % (len(WINDOW), len(TRAIN)))
        p1 = part1_vine(panel, WINDOW)
        p1_train = part1_vine(panel, TRAIN)
    if "2" in parts:
        print("[P2] 水平对 KDE + Copula 拟合 ...")
        p2, _ = part2_level_copula(panel, TRAIN)
    if "3" in parts:
        print("[P3] 情景生成器对照（训练 %s..%s，测试 %s..%s）..."
              % (TRAIN[0].date(), TRAIN[-1].date(), TEST[0].date(), TEST[-1].date()))
        p3 = part3_generators(panel)
    if "4" in parts:
        print("[P4] 分时段 CVaR 风险画像 ...")
        p4 = part4_risk()
    out = dict(train=[str(d.date()) for d in TRAIN], test=[str(d.date()) for d in TEST],
               K=K_MAIN, part1_vine=p1, part1_vine_train=p1_train, part2_level=p2,
               part3_generators=p3, part4_risk=p4)
    path = os.path.join(C.INTERMEDIATE_DIR, "exp_copula.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("已写出", path)
    print(json.dumps(dict(
        tau_price_load=p1["plain"]["tau_price_load"],
        tau_price_pv=p1["plain"]["tau_price_pv"],
        tau_price_resid=p2["tau_emp_PR"],
        best_level_family=p2["best"]["family"],
        gen={k: round(v["total_cost"], 2) for k, v in p3.items() if isinstance(v, dict) and "total_cost" in v},
        q2_cvar_over_mean=round(p4["q2_cvar_over_mean"], 4),
        q4_cvar_over_mean=round(p4["q4_cvar_over_mean"], 4)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
