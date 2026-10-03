# -*- coding: utf-8 -*-
"""汇总全部论文数值 -> intermediate/paper_numbers.json 与 notes/数值台账.md（唯一数字源）。"""
from __future__ import annotations

import _pathsetup  # noqa: E402,F401  子目录引导：把 common/ 与各问目录加入 sys.path

import json
import os

import numpy as np
import pandas as pd

import config as C

INT = C.INTERMEDIATE_DIR
NOTES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")


def load(name):
    p = os.path.join(INT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def f(x, n=2):
    return f"{x:,.{n}f}"


def wan(x):
    return f"{x/1e4:,.2f}"


# 旧键名 -> 其真实窗口（台账中逐条括注，避免 §3.4 的“同一句话两个窗口”）
LEGACY_WINDOW = {
    "负载均值_kW": "旧键名＝_全年365天",
    "负载最小_kW": "旧键名＝_全年365天",
    "负载最大_kW": "旧键名＝_全年365天",
    "光伏均值_kW": "旧键名＝_全年365天",
    "光伏峰值_kW": "旧键名＝_全年365天",
    "负载日均电量_kWh": "旧键名＝_窗口334天",
    "光伏日均电量_kWh": "旧键名＝_窗口334天",
}


def main():
    q1 = load("q1.json")
    q2 = load("q2_main.json")
    q2k120 = load("q2_K120.json")
    q2r1 = load("q2_cmp_R1.json")
    q2r2 = load("q2_cmp_R2.json")
    q3 = {t: load(f"q3_{t}.json") for t in ("n1", "n2", "n4", "perf")}
    q4p = load("q4_price.json")
    q42 = load("q4_2.json")
    q43 = load("q4_3.json")
    scan = load("exp_scan.json")
    audit = load("audit.json")

    a1 = C.read_attachment1()
    panel = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D"))
    load_m = panel.L(); pv_m = panel.PV(); p4 = panel.P(volatile=True)
    win = pd.date_range(C.WIN_START, C.WIN_END, freq="D")

    N = {}
    N["数据"] = dict(
        # --- 均值类：两类窗口都显式写出（原键名把 365 天均值与 334 天日均电量混在同一句）---
        负载均值_kW_全年365天=float(load_m.to_numpy().mean()),
        负载均值_kW_窗口334天=float(load_m.loc[win].to_numpy().mean()),
        负载日均电量_kWh_窗口334天=float(load_m.loc[win].to_numpy().sum(1).mean() * C.DT),
        光伏均值_kW_全年365天=float(pv_m.to_numpy().mean()),
        光伏均值_kW_窗口334天=float(pv_m.loc[win].to_numpy().mean()),
        光伏日均电量_kWh_窗口334天=float(pv_m.loc[win].to_numpy().sum(1).mean() * C.DT),
        负载最小_kW_全年365天=float(load_m.to_numpy().min()),
        负载最大_kW_全年365天=float(load_m.to_numpy().max()),
        光伏峰值_kW_全年365天=float(pv_m.to_numpy().max()),
        固定电价均值=float(a1["price"].mean()), 固定电价最小=float(a1["price"].min()),
        固定电价最大=float(a1["price"].max()),
        波动电价均值=float(p4.to_numpy().mean()), 波动电价最小=float(p4.to_numpy().min()),
        波动电价最大=float(p4.to_numpy().max()),
        附件1为逐时段均值最大偏差=float(np.abs(a1["price"].to_numpy()
                                        - p4.to_numpy().mean(0)).max()),
        附件1负载为逐时段均值最大偏差=float(np.abs(a1["load"].to_numpy()
                                          - load_m.to_numpy().mean(0)).max()),
        附件1光伏为逐时段均值最大偏差=float(np.abs(a1["pv"].to_numpy()
                                          - pv_m.to_numpy().mean(0)).max()),
        # --- 旧键名：保留兼容，数值一字不改；台账中逐条标注其实际窗口（见 LEGACY_WINDOW）---
        负载均值_kW=float(load_m.to_numpy().mean()),
        负载最小_kW=float(load_m.to_numpy().min()),
        负载最大_kW=float(load_m.to_numpy().max()),
        负载日均电量_kWh=float(load_m.loc[win].to_numpy().sum(1).mean() * C.DT),
        光伏均值_kW=float(pv_m.to_numpy().mean()),
        光伏峰值_kW=float(pv_m.to_numpy().max()),
        光伏日均电量_kWh=float(pv_m.loc[win].to_numpy().sum(1).mean() * C.DT),
    )
    # 预报误差
    FC = np.vstack([panel.pv_forecast(d, 0) for d in panel.PV().index])
    err = FC - pv_m.to_numpy()
    hrs = np.array([t // 6 for t in range(C.N_SLOT)])
    day = hrs >= 5
    N["预报误差"] = dict(
        MAE_全时段_kW=float(np.abs(err).mean()), RMSE_全时段_kW=float(np.sqrt((err ** 2).mean())),
        bias_全时段_kW=float(err.mean()),
        MAE_白天_kW=float(np.abs(err[:, day]).mean()),
        RMSE_白天_kW=float(np.sqrt((err[:, day] ** 2).mean())),
        bias_白天_kW=float(err[:, day].mean()),
        逐整点bias_kW=[float(x) for x in err.reshape(len(err), 24, 6).mean(axis=(0, 2))],
    )
    if q1:
        m = q1["main"]
        N["Q1"] = dict(
            无储能基线元=m["base_cost"], 最优购电费元=m["cost"], 节省元=m["saving"],
            节省率=m["saving_rate"], 全天购电量kWh=m["total_energy"],
            充电量kWh=m["charge_total"], 放电量kWh=m["discharge_total"],
            充放比=m["ratio_cd"], 弃光kWh=m["curtail_total"],
            SOC最小=m["min_soc"], SOC最大=m["max_soc"], S0=m["s0"], S144=m["s_end"],
            自由环状费用元=q1["free_cycle"]["cost"], 自由环状S=q1["free_cycle"]["s0"],
            自由环状差元=q1["free_cycle"]["cost"] - m["cost"],
            表1=q1["table1"], 表2=q1["table2"],
            残差=dict(R1=m["R1_min_surplus"], R2=m["R2_max_plan_violation"],
                     R3=m["R3_max_soc_residual"], R5=m["R5_simultaneous_slots"],
                     R6=m["R6_cd_ratio_err"]),
        )
    if scan:
        N["容量功率扫描"] = scan["capacity"]
        N["效率口径"] = scan["eta"]
        N["报童经验"] = scan["newsboy"]
        N["惩罚倍数灵敏度"] = scan["penalty"]
    if q2:
        N["Q2"] = dict(meta=q2["meta"])
        d0 = q2["daily"]
        N["Q2"]["四个指定日期"] = [r for r in d0 if r["date"][:10] in C.KEY_DATES]
    if q2k120:
        N["Q2_K120"] = q2k120["meta"]
    N["Q2_R1完美信息"] = q2r1["meta"] if q2r1 else None
    N["Q2_R2典型日预报"] = q2r2["meta"] if q2r2 else None
    N["Q3"] = {t: (v["meta"] if v else None) for t, v in q3.items()}
    if q3["n4"]:
        N["Q3"]["四个指定日期"] = [r for r in q3["n4"]["daily"]
                                   if r["date"][:10] in C.KEY_DATES]
    if q4p:
        N["电价检验"] = q4p["price_tests"]
        N["情景保真度_历史代表日"] = q4p["fidelity_hist"]
        N["情景保真度_分块Bootstrap"] = q4p["fidelity_boot"]
    N["Q4_2"] = q42["meta"] if q42 else None
    N["Q4_3"] = q43["meta"] if q43 else None
    if q42:
        N["Q4_2"]["四个指定日期"] = [r for r in q42["daily"] if r["date"][:10] in C.KEY_DATES]
    if q43:
        N["Q4_3"]["四个指定日期"] = [r for r in q43["daily"] if r["date"][:10] in C.KEY_DATES]
    dual2 = load("q1_dual2.json")
    N["Q1对偶验证"] = dual2
    N["Q2无安全边际消融"] = load("q2_noMargin.json")["meta"] if load("q2_noMargin.json") else None
    N["Q2安全边际分解"] = load("q2_margin.json")
    N["非预期性置换检验"] = load("exp_na.json")
    N["Sobol"] = load("exp_sobol.json")
    N["基准实验"] = load("exp_benchmark.json")
    N["窗口聚合"] = load("window_aggregates.json")
    N["样本外回测"] = load("exp_oos.json")
    N["终端条件对照"] = {t: (load(f"q2_term_{t}.json") or {}).get("meta")
                    for t in ("ge", "eq", "free")}
    N["审计"] = audit
    cvar = load("q4_cvar.json")
    N["CVaR前沿"] = cvar["frontier"] if cvar else None
    # ---- v2 新增实验（相依结构/生成器对照/风险画像/机会约束/盒式鲁棒/L 形分解）----
    v2cop = load("exp_copula.json")
    v2chance = load("exp_chance.json")
    v2dro = load("exp_dro.json")
    v2dec = load("decomp.json")
    v2pc = load("exp_pricecond.json")
    N["v2相依结构与生成器对照"] = v2cop
    N["v2机会约束"] = v2chance
    N["v2盒式鲁棒"] = v2dro
    N["v2L形分解"] = v2dec
    N["v2负载条件化电价预测"] = v2pc

    # ---- 假设 A12 的消融：实时再调度（正文表 24 的 RT 行）----
    rtd = load("exp_redispatch.json") or {}
    _rt = (rtd.get("summary") or {}).get("headline")
    if _rt:
        _frozen = (rtd.get("summary") or {}).get("frozen") or {}
        _plan = ((rtd.get("meta") or {}).get("plan_json_meta") or {}).get("tot_plan")
        _row = dict(rtd.get("redispatch", {}).get("redispatch_min_throughput") or {})
        _row_effect = (rtd.get("summary") or {}).get("plan_row_effect") or {}
        _mech = (rtd.get("summary") or {}).get("mechanism") or {}
        rt_total = (_plan + _rt["emg_cost_yuan"]) if _plan is not None else None
        frozen_total = (_plan + _frozen.get("emg_cost_yuan", 0.0)) if _plan is not None else None
        base = ((load("q2_main.json") or {}).get("meta") or {}).get("tot_base")
        N["A12实时再调度消融"] = dict(
            口径="固定已结算的 E^plan（沉没），用当日实际净需求重优化 c,d,S,g；一天一个 LP",
            frozen=dict(emg_kwh=_frozen.get("emg_kwh"), emg_cost=_frozen.get("emg_cost_yuan"),
                        surplus_kwh=_frozen.get("surplus_kwh"),
                        days_with_emg=_frozen.get("days_with_emg")),
            redispatch=dict(emg_kwh=_rt.get("emg_kwh"), emg_cost=_rt.get("emg_cost_yuan"),
                           surplus_kwh=_rt.get("surplus_kwh"),
                           days_with_emg=_rt.get("days_with_emg")),
            降幅=dict(emg_kwh_pct=_rt.get("emg_kwh_drop_pct"),
                      emg_cost_pct=_rt.get("emg_cost_yuan_drop_pct"),
                      surplus_pct=_rt.get("surplus_kwh_drop_pct")),
            总费用=dict(计划项=_plan, frozen=frozen_total, redispatch=rt_total,
                        节省额=(frozen_total - rt_total) if (frozen_total and rt_total) else None,
                        相对基线节省率=((base - rt_total) / base * 100.0)
                        if (base and rt_total) else None),
            计划可行性行对照=dict(含行=_row_effect.get("emg_kwh_with_row"),
                                  放开行=_row_effect.get("emg_kwh_without_row"),
                                  结论=_row_effect.get("verdict")),
            机制=_mech.get("soc_identity"),
            边界="只放开调度、未联合重优化计划；故 63.49% 是该假设代价的下界",
        )
    N["非预期性置换检验"] = load("exp_na.json")
    N["非预期性置换检验_K40对照"] = load("exp_na_K40_backup.json")

    # ---- Sobol 一阶/总效应指数合计（正文 §9.3 与附录 K.3 引用的两个合计）----
    _sb = load("exp_sobol.json") or {}
    try:
        _c = _sb.get("cost") or {}
        if _c.get("S1"):
            _c1 = float(sum(_c["S1"]))
            _cT = float(sum(_c["ST"]))
            _s = _sb.get("saving") or {}
            _s1 = float(sum(_s.get("S1", []))) or None
            _sT = float(sum(_s.get("ST", []))) or None
            N["Sobol合计"] = dict(
                购电费=dict(S1合计=_c1, ST合计=_cT, 交互贡献_百分点=(_cT - _c1) * 100.0,
                            均值_元=_c.get("mean"), 方差=_c.get("var")),
                节省率=(dict(S1合计=_s1, ST合计=_sT, 交互贡献_百分点=(_sT - _s1) * 100.0)
                        if (_s1 and _sT) else None),
                参数名=_c.get("names"),
            )
    except Exception as exc:      # pragma: no cover - 台账不应因该可选块失败
        N["Sobol合计"] = dict(error=str(exc))

    with open(os.path.join(INT, "paper_numbers.json"), "w") as fh:
        json.dump(N, fh, ensure_ascii=False, indent=1, default=str)

    # ---- Markdown 台账（供写作引用，单位与口径写全）----
    L = ["# 论文数值台账（唯一数字源）", "",
         "> 由 code/paper_numbers.py 从各求解输出汇总；论文中每个数字都应能在本表找到。",
         "> 口径：电量 kWh，功率 kW，费用 元；固定价=附件 1 电价，波动价=附件 4；储能 12000 kWh /",
         "> 5000 kW / SOC 1200–10800 / 效率单边 0.9。",
         "> 两类窗口并存（不可直接混用）：`_全年365天` = 附件 2/4 全样本 2025-01-01 至 2025-12-31"
         "（365 天，含 1 月预热）；`_窗口334天` = 结果窗口 2025-02-01 至 2025-12-31（334 天）；"
         "四问求解与 results/*.xlsx 一律为 334 天窗口。", ""]
    L.append("## 一、数据与预报")
    L.append("")
    L.append("> 窗口提示：均值类指标同时给出 `_全年365天` 与 `_窗口334天` 两行；"
             "日均电量类指标为 `*_日均电量_kWh_窗口334天`（334 天窗口的日电量均值）。"
             "旧键名（无后缀）保留兼容，逐行括注其实际窗口。")
    L.append("")
    L.append("| 量 | 数值 |")
    L.append("|---|---|")
    for k, v in N["数据"].items():
        key = f"{k}（{LEGACY_WINDOW[k]}）" if k in LEGACY_WINDOW else k
        L.append(f"| {key} | {v:,.4f} |" if isinstance(v, float) else f"| {key} | {v} |")
    for k, v in N["预报误差"].items():
        L.append(f"| {k} | {np.round(v,2)} |")
    L.append("")
    if q1:
        m = q1["main"]
        L.append("## 二、问题 1（典型日）")
        L.append("")
        L.append("| 量 | 数值 |")
        L.append("|---|---|")
        for k, v in N["Q1"].items():
            if isinstance(v, float):
                L.append(f"| {k} | {v:,.4f} |")
            elif k == "残差":
                L.append(f"| 残差 | {v} |")
            elif k in ("表1", "表2"):
                L.append(f"| {k} | {v} |")
        L.append("")
        L.append("表 1 指定时段购电量（kWh）：" + "；".join(
            f"{a}={b:,.4f}" for a, b in q1["table1"]))
        L.append("")
        rows, s0, s24 = q1["table2"]
        L.append("表 2 充放电量（kWh）：" + "；".join(
            f"{a} 充{b:,.4f}/放{c:,.4f}" for a, b, c in rows) + f"；0:00={s0:,.4f}，24:00={s24:,.4f}")
        L.append("")
    L.append("## 三、容量/功率/效率与报童")
    L.append("")
    if scan:
        L.append("容量扫描（保持 10%–90% DOD）：")
        L.append("")
        L.append("| 容量 kWh | 最优费用 元 | 相对基线节省 元 | 节省率 % |")
        L.append("|---|---|---|---|")
        for r in scan["capacity"]["cap_scan"]:
            L.append(f"| {r['cap']:,} | {r['cost']:,.2f} | {r['saving']:,.2f} | {r['dcost']:.2f} |")
        L.append("")
        L.append(f"仅解除容量上限：{scan['capacity']['no_capacity_limit']:,.2f} 元（多省 "
                 f"{scan['capacity']['gain_no_cap']:.1f}%）；仅解除功率上限："
                 f"{scan['capacity']['no_power_limit']:,.2f} 元（多省 {scan['capacity']['gain_no_pow']:.1f}%）")
        L.append("")
        L.append("| 充放功率上限 kW | 最优费用 元 | 节省率 % |")
        L.append("|---|---|---|")
        for r in scan["capacity"]["power_scan"]:
            L.append(f"| {r['pmax']:,} | {r['cost']:,.2f} | {r['dcost']:.2f} |")
        L.append("")
        L.append("效率口径对照：")
        L.append("")
        L.append("| 口径 | 最优费用 元 | 节省 元 |")
        L.append("|---|---|---|")
        for r in scan["eta"]:
            L.append(f"| {r['name']} | {r['cost']:,.2f} | {r['saving']:,.2f} |")
        L.append("")
        nb = scan["newsboy"]
        L.append(f"报童经验覆盖率：逐槽缺货频率 {nb['short_slot_rate']*100:.2f}%，"
                 f"缺口电量 {nb['short_energy']:,.2f} kWh，"
                 f"电量满足率 {nb['fill_rate']*100:.2f}%，"
                 f"安全边际合计 {nb['margin_total']:,.2f} kWh（{nb['margin_per_slot']:.2f} kWh/槽）")
        L.append("")
        L.append("紧急购电倍数灵敏度（典型日）：" + "；".join(
            f"k={r['k']} 边际 {r['margin']:.0f} kWh/期望紧急 {r['exp_emg']:.0f} kWh"
            for r in scan["penalty"]))
        L.append("")
    L.append("## 四、四问主结果")
    L.append("")
    L.append("| 模型 | 总费用 元 | 万元 | 相对基线节省率 | 紧急购电量 kWh |")
    L.append("|---|---|---|---|---|")
    if q2:
        mm = q2["meta"]
        L.append(f"| 问题 2（固定价，两阶段） | {mm['tot_cost']:,.2f} | {wan(mm['tot_cost'])} | "
                 f"{(mm['tot_base']-mm['tot_cost'])/mm['tot_base']*100:.2f}% | {mm['tot_emg_qty']:,.2f} |")
    if q3["n4"]:
        mm = q3["n4"]["meta"]
        L.append(f"| 问题 3（固定价，四次滚动） | {mm['total_cost']:,.2f} | {wan(mm['total_cost'])} | "
                 f"{(mm['base_cost']-mm['total_cost'])/mm['base_cost']*100:.2f}% | {mm['emg_qty']:,.2f} |")
    if q42:
        mm = q42["meta"]
        L.append(f"| 问题 4-2（波动价，两阶段） | {mm['total']:,.2f} | {wan(mm['total'])} | "
                 f"{(mm['base']-mm['total'])/mm['base']*100:.2f}% | {mm['emg_qty']:,.2f} |")
    if q43:
        mm = q43["meta"]
        L.append(f"| 问题 4-3（波动价，四次滚动） | {mm['total_cost']:,.2f} | {wan(mm['total_cost'])} | "
                 f"{(mm['base_cost']-mm['total_cost'])/mm['base_cost']*100:.2f}% | {mm['emg_qty']:,.2f} |")
    L.append("")
    L.append(f"无储能基线：固定价 {q2['meta']['tot_base']:,.2f} 元；波动价 {q42['meta']['base']:,.2f} 元。")
    L.append("")
    L.append("VoI 实验（同一场景集、同一实际轨迹）：")
    L.append("")
    L.append("| 预报时刻数 | 总费用 元 | 紧急购电量 kWh |")
    L.append("|---|---|---|")
    for t, name in (("n1", "仅 0:00（1 次）"), ("n2", "0:00+12:00（2 次）"),
                    ("n4", "0:00/6:00/12:00/18:00（4 次）"), ("perf", "完全信息（上界）")):
        if q3[t]:
            L.append(f"| {name} | {q3[t]['meta']['total_cost']:,.2f} | {q3[t]['meta']['emg_qty']:,.2f} |")
    L.append("")
    L.append("## 五、v2 新增对照实验")
    L.append("")
    if v2cop:
        p1 = v2cop["part1_vine"]
        L.append("| C 藤 pair-Copula（334 个日水平观测） | 数值 |")
        L.append("|---|---|")
        for nm, key in (("tau(电价,负载)", "tau_price_load"), ("tau(电价,光伏)", "tau_price_pv"),
                        ("tau(负载,光伏)", "tau_load_pv"),
                        ("Pearson(电价,负载)", "pearson_price_load"),
                        ("Pearson(电价,光伏)", "pearson_price_pv")):
            L.append(f"| {nm} | {p1['plain'][key]:+.4f} |")
        for r in p1["vine"]:
            L.append(f"| {r['name']}：最优族 / tau / lambdaU / AIC | {r['best']} / "
                     f"{r['tau_emp']:+.4f} / {r['tail'][0]:.4f} / {r['aic']:.2f} |")
        p3 = v2cop["part3_generators"]
        L.append("")
        L.append("四种情景生成器的样本外对照（训练 2025-02-01 至 06-30，测试 07-01 至 12-31，184 天）：")
        L.append("")
        L.append("| 生成器 | 总费用 元 | 相对差 | 紧急购电量 kWh | 逐日 CVaR95 元 |")
        L.append("|---|---|---|---|---|")
        for k, nm in (("hist_strat", "历史代表日"), ("bootstrap", "分块 Bootstrap"),
                      ("copula", "KDE+Copula"), ("scaled", "按光伏规模缩放")):
            d = p3[k]
            L.append(f"| {nm} | {d['total_cost']:,.2f} | {d.get('diff_pct', 0.0):+.3f}% | "
                     f"{d['emg_qty']:,.2f} | {d['cvar95']:,.2f} |")
        p4 = v2cop["part4_risk"]
        L.append("")
        L.append(f"分时段 CVaR 风险画像（alpha={p4['alpha']}）：问题 2 逐槽 CVaR 之和/均值之和 "
                 f"{p4['q2_cvar_over_mean']:.4f}，前 12 槽占比 {p4['q2_top12_slot_share']:.4f}；"
                 f"问题 4-2 {p4['q4_cvar_over_mean']:.4f} / {p4['q4_top12_slot_share']:.4f}。")
        L.append("")
    if v2chance:
        L.append("机会约束对照（固定价 334 天）：")
        L.append("")
        L.append("| beta | 总费用 元 | 紧急购电量 kWh | 缺电槽位频率 | 逐日 CVaR95 元 |")
        L.append("|---|---|---|---|---|")
        for r in v2chance["chance"]:
            L.append(f"| {r['beta']:.2f} | {r['tot_cost']:,.2f} | {r['tot_emg_qty']:,.2f} | "
                     f"{r['shortfall_slot_freq']*100:.2f}% | {r['cvar95']:,.2f} |")
        sp = v2chance["reference_sp"]
        L.append(f"| 两阶段随机规划（主方案） | {sp['tot_cost']:,.2f} | {sp['tot_emg_qty']:,.2f} | "
                 f"{sp['shortfall_slot_freq']*100:.2f}% | {sp['cvar95']:,.2f} |")
        L.append("")
    if v2dro:
        L.append("盒式分布鲁棒对照（波动价 334 天回测）：")
        L.append("")
        L.append("| k | 回测总费用 元 | 节省率 | 紧急购电量 kWh | 逐日 CVaR95 元 |")
        L.append("|---|---|---|---|---|")
        res = v2dro["results"]
        ks = sorted(res, key=lambda s: float(s.split("=")[-1]))
        for k in ks:
            d = res[k]
            L.append(f"| {k.split('=')[-1]} | {d['total_cost']:,.2f} | {d['saving']*100:+.2f}% | "
                     f"{d['emg_qty']:,.2f} | {d['cvar95']:,.2f} |")
        L.append("")
    if v2dec:
        e = v2dec["equivalence_K60"]
        L.append(f"L 形（Benders）分解对拍：K=60 在 {e['n']} 天上最大相对间隙 {e['max_rel_gap']:.2e}，"
                 f"平均迭代 {e['mean_iters']:.2f} 次，全部收敛={e['all_converged']}；"
                 f"直接求解 {e['sum_t_direct']:.2f} s，分解 {e['sum_t_benders']:.2f} s。")
        L.append("")
        L.append("| 情景数 K | 直接求解耗时 s（含建模） | L 形分解耗时 s（含主/子问题） | 加速比 | 最大相对间隙 |")
        L.append("|---|---|---|---|---|")
        for ent in v2dec["scenario_scan"]["entries"]:
            rows = ent["rows"]
            td = sum(r.get("t_direct_total", r.get("t_direct_production", 0.0)) for r in rows)
            tb = sum(r.get("t_benders_total", 0.0) for r in rows)
            L.append(f"| {ent['K']} | {td:.3f} | {tb:.3f} | {td/tb:.2f}x | "
                     f"{max(r['rel_gap'] for r in rows):.2e} |")
        L.append("")
    with open(os.path.join(NOTES, "数值台账.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    return N


if __name__ == "__main__":
    main()
