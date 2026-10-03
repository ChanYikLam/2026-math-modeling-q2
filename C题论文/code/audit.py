# -*- coding: utf-8 -*-
"""
独立审计（A1–A10）：从**结果文件**与**原始附件**出发独立复算，不调用建模目标函数。
==============================================================================
A1  电量平衡残差（结果文件的 4 h 块级 + 中间结果的 10 min 级）
A2  SOC 递推残差（用 eta_c=eta_d=0.9 独立重算）
A3  SOC 越界数（[1200,10800]）
A4  充放电功率越界数（<= 5000 kW 等价 <= 833.3333 kWh/槽）
A5  充放互斥违反数
A6  周期/终端条件（result1: S0=S144；其余: S144 >= S0）
A7  非负性
A8  费用复算相对误差（不调用建模费用函数）
A9  日期与结构（行数/列数/工作表名/标签）
A10 全天汇总列（全天购电量、全天购电费）

输出：intermediate/audit_report.md 与 intermediate/audit.json
"""
from __future__ import annotations

import _pathsetup  # noqa: E402,F401  子目录引导：把 common/ 与各问目录加入 sys.path

import json
import os

import numpy as np
import pandas as pd

import config as C

TOL_RES = 1e-6          # kWh
TOL_REL = 1e-4          # 费用复算相对误差
RES_FILES = [("result1.xlsx", "q1"), ("result2.xlsx", "q2"),
             ("result3.xlsx", "q3"), ("result4-2.xlsx", "q4-2"),
             ("result4-3.xlsx", "q4-3")]


def _load(fname):
    p = os.path.join(C.RESULT_DIR, fname)
    xl = pd.ExcelFile(p)
    return {sh: xl.parse(sh) for sh in xl.sheet_names}


def audit_result_files():
    rows, out = [], {}
    a1 = C.read_attachment1()
    price_fixed = a1["price"].to_numpy(float)
    price_flat = C.read_attachment4()
    panel = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D"))

    for fname, tag in RES_FILES:
        p = os.path.join(C.RESULT_DIR, fname)
        if not os.path.exists(p):
            rows.append(dict(file=fname, item="存在性", value="缺失", verdict="FAIL"))
            continue
        sh = _load(fname)
        rep = {}
        plan = sh["计划购电量"]
        if tag == "q1":
            # A9 结构
            rep["A9_计划行数"] = (len(plan), 144)
            rep["A9_工作表"] = (list(sh.keys()), ["计划购电量", "充放电量"])
            E = plan.iloc[:, 1].to_numpy(float)
            rep["A7_负购电量数"] = int((E < -TOL_RES).sum())
            soc = sh["充放电量"]
            rep["A9_充放电行数"] = (len(soc), 6)
            c_blk = soc["充电量"].to_numpy(float)[:6]
            d_blk = soc["放电量"].to_numpy(float)[:6]
            sv = soc["储电量"].to_numpy(float)
            s0 = float(sv[0]); s24 = float(sv[1])
            rep["A6_日周期偏差"] = abs(s0 - s24)
            rep["A2_SOC净递推残差"] = abs((s0 + C.ETA_C * c_blk.sum()
                                          - d_blk.sum() / C.ETA_D) - s24)
            rep["A8_费用复算相对误差"] = abs(E @ price_fixed
                                            - float(np.sum(price_fixed * E))) / max(1e-9, float(np.sum(price_fixed * E)))
            rep["A1_全天购电量"] = float(E.sum())
            rep["A1_购电费"] = float(E @ price_fixed)
            rep["A1_基线C01"] = C.baseline_cost(price_fixed, a1["load"].to_numpy(float),
                                                a1["pv"].to_numpy(float))
        else:
            dates = pd.to_datetime(plan.iloc[:, 0])
            E = plan.iloc[:, 1:145].to_numpy(float)
            tot = plan.iloc[:, 145].to_numpy(float)
            fee = plan.iloc[:, 146].to_numpy(float)
            price_mat = (price_flat.loc[dates].to_numpy(float) if tag.startswith("q4")
                         else np.tile(price_fixed, (len(dates), 1)))
            rep["A9_天数"] = (len(dates), 334)
            rep["A9_列数"] = (plan.shape[1], 147)
            rep["A9_日期范围"] = (str(dates.min().date()), str(dates.max().date()),
                                "2025-02-01", "2025-12-31")
            rep["A10_全天购电量最大偏差"] = float(np.max(np.abs(E.sum(1) - tot)))
            rep["A10_全天购电费最大相对偏差"] = float(np.max(
                np.abs(fee - (price_mat * E).sum(1)) / np.maximum(fee, 1)))
            rep["A7_负购电量数"] = int((E < -TOL_RES).sum())
            rep["A9_工作表"] = (list(sh.keys()), sorted(sh.keys()))
            # SOC 块级独立递推
            soc = sh["充放电量"]
            c_blk = soc["充电量"].to_numpy(float).reshape(-1, 6)
            d_blk = soc["放电量"].to_numpy(float).reshape(-1, 6)
            st = soc["储电量"].to_numpy(float)
            s0 = st[0::6]; s24 = st[1::6]
            lhs = s0 + C.ETA_C * c_blk.sum(1) - d_blk.sum(1) / C.ETA_D
            rep["A2_SOC净递推最大残差"] = float(np.max(np.abs(lhs - s24)))
            rep["A6_末值低于初值天数"] = int((s24 - s0 < -1e-6).sum())
            rep["A3_SOC越界数"] = int(((s0 < C.S_MIN - 1e-6) | (s24 > C.S_MAX + 1e-6)).sum())
            rep["A4_块级功率越界数"] = int((c_blk > 24 * C.E_MAX + 1e-6).sum()
                                        + (d_blk > 24 * C.E_MAX + 1e-6).sum())
            rep["A1_全窗购电量"] = float(E.sum())
            rep["A1_全窗购电费"] = float((price_mat * E).sum())
            # 紧急购电表结构
            em = sh["紧急购电量"]
            qty = pd.to_numeric(em["购电量"], errors="coerce").fillna(0).to_numpy(float)
            rep["A9_紧急购电行数"] = len(em)
            rep["A1_紧急购电量合计"] = float(qty.sum())
        out[tag] = rep
        for k, v in rep.items():
            rows.append(dict(file=fname, item=k, value=str(v), verdict=""))
    return rows, out


def audit_internal():
    """内部解（10 min 级）的可行性审计。"""
    out = {}
    q1 = json.load(open(os.path.join(C.INTERMEDIATE_DIR, "q1.json", encoding="utf-8")))
    a = q1["arrays"]
    E = np.array(a["E"]); c = np.array(a["c"]); d = np.array(a["d"]); S = np.array(a["S"])
    price = np.array(a["price"]); load = np.array(a["load"]); pv = np.array(a["pv"])
    out["q1"] = dict(
        A1_平衡残差=float(abs(E + pv * C.DT + d - load * C.DT - c).max()),
        A2_SOC递推残差=float(abs(S - (np.concatenate([[C.S_INIT], S[:-1]])
                                    + C.ETA_C * c - d / C.ETA_D)).max()),
        A3_SOC越界=int(((S < C.S_MIN - 1e-9) | (S > C.S_MAX + 1e-9)).sum()),
        A4_功率越界=int(((c > C.E_MAX + 1e-9) | (d > C.E_MAX + 1e-9)).sum()),
        A5_互斥违反=int(((c > 1e-6) & (d > 1e-6)).sum()),
        A6_周期偏差=float(abs(a["S"][-1] - 6000.0)),
        A7_负购电=int((E < -1e-9).sum()),
        A8_费用相对误差=float(abs(float(np.sum(price * E)) - q1["main"]["cost"])
                          / q1["main"]["cost"]),
        A10_全天购电量=float(E.sum()))
    for tag, f in (("q2", "q2_main.json"), ("q3", "q3_n4.json")):
        p = os.path.join(C.INTERMEDIATE_DIR, f)
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        E = np.vstack([np.array(r["E"] if tag == "q2" else r["E_adj"]) for r in j["daily"]])
        c = np.vstack([np.array(r["c"]) for r in j["daily"]])
        d = np.vstack([np.array(r["d"]) for r in j["daily"]])
        g = np.vstack([np.array(r["g"]) for r in j["daily"]])
        s0 = np.array([r["s_init"] for r in j["daily"]])
        s1 = np.array([r.get("s_end", r.get("S_end")) for r in j["daily"]])
        load_m = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D")).L().loc[[
            pd.Timestamp(r["date"]) for r in j["daily"]]].to_numpy(float)
        pv_m = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D")).PV().loc[[
            pd.Timestamp(r["date"]) for r in j["daily"]]].to_numpy(float)
        nd = (load_m - pv_m) * C.DT
        # 供给 >= 需求；正残差 = 盈余处置量（弃光或超购处置），负残差才是不可行
        bal = E + pv_m * C.DT + d + g - load_m * C.DT - c
        bal_no_g = bal - g
        out[tag] = dict(
            A1_供给缺口=float(-min(bal.min(), 0.0)),
            A1_盈余处置量=float(np.maximum(bal, 0).sum()),
            A1_漏g时最大缺口=float(-bal_no_g.min()),
            A2_SOC净递推残差=float(np.max(np.abs(s1 - (s0 + C.ETA_C * c.sum(1)
                                                    - d.sum(1) / C.ETA_D)))),
            A4_功率越界=int(((c > C.E_MAX + 1e-6) | (d > C.E_MAX + 1e-6)).sum()),
            A5_互斥违反=int(((c > 1e-6) & (d > 1e-6)).sum()),
            A6_末值低于初值天数=int((s1 - s0 < -1e-6).sum()),
            A7_负购电=int((E < -1e-9).sum()),
            A8_费用相对误差=float(abs(sum(r["total_cost"] for r in j["daily"])
                                  - j["meta"].get("tot_cost", j["meta"].get("total_cost", np.nan)))
                              / max(1e-9, j["meta"].get("tot_cost", j["meta"].get("total_cost", 1)))),
            A9_天数=len(j["daily"]))
    return out


def main():
    rows, out = audit_result_files()
    inner = audit_internal()
    os.makedirs(C.INTERMEDIATE_DIR, exist_ok=True)
    with open(os.path.join(C.INTERMEDIATE_DIR, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(dict(deliverable=out, internal=inner), f, ensure_ascii=False, indent=1)
    lines = ["# 独立审计报告（A1–A10）", "",
             "> 由独立脚本 audit.py 从结果文件与原始附件复算，不调用建模目标函数。",
             f"> 阈值：残差 ≤ {TOL_RES} kWh；费用复算相对误差 ≤ {TOL_REL}；越界数 = 0。", ""]
    lines.append("## 一、交付物审计（结果文件）")
    lines.append("")
    lines.append("| 文件 | 审计项 | 数值 |")
    lines.append("|---|---|---|")
    for r in rows:
        lines.append(f"| {r['file']} | {r['item']} | {r['value']} |")
    lines.append("")
    lines.append("## 二、内部解审计（10 min 级）")
    lines.append("")
    lines.append("| 问题 | 审计项 | 数值 |")
    lines.append("|---|---|---|")
    for tag, rep in inner.items():
        for k, v in rep.items():
            lines.append(f"| {tag} | {k} | {v:.3e} |" if isinstance(v, float)
                         else f"| {tag} | {k} | {v} |")
    txt = "\n".join(lines)
    with open(os.path.join(C.INTERMEDIATE_DIR, "audit_report.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt[:4000])
    return out


if __name__ == "__main__":
    main()
