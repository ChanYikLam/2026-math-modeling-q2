# -*- coding: utf-8 -*-
"""
非预期性（non-anticipativity）置换检验
=====================================
做法：把某一次预报（6:00 / 12:00 / 18:00）在日与日之间**随机置换**后重新求解，
      检查"仅由该次预报控制的时段"是否变化，以及"早于该预报时刻的时段"是否**完全不变**。
- 若置换 6:00 预报后，槽 36-71 的决策发生变化、而槽 0-35 完全不变 → 该次预报只影响它之后的时段，
  说明实现中没有"用未来信息决策"的泄漏（非预期性成立）。
- 同时给出对照：置换 0:00 预报应使槽 0-35 也变化（证明检验本身有效力）。

口径：K = 60 场景（与论文主口径 run_all.py 的 --K 60 一致），仅置换一次预报、其余不变；
      对照 seed=1（置换用 numpy.random.default_rng(seed=1)）。
输出 intermediate/exp_na.json（默认；可用 --out 改文件名，例如 --K 40 --out exp_na_K40.json）
用法：python3 exp_na.py [--K 60] [--out exp_na.json]
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # -> code/
import _pathsetup  # noqa: E402,F401  子目录引导：把 common/ 与各问目录加入 sys.path

import argparse
import json
import os

import numpy as np
import pandas as pd

import config as C
import lp_engine as LP
import q3 as Q3

DAYS = pd.date_range("2025-02-01", "2025-02-12", freq="D")
EPOCHS = (0, 6, 12, 18)
CONTROL = {0: (0, 36), 6: (36, 72), 12: (72, 108), 18: (108, 144)}


def run_days(permuted_hour=None, seed=0, K=60):
    """运行 12 天；permuted_hour 不为 None 时，把该时刻的预报按日随机置换。

    K 为各预报时刻残差场景集的场景数，默认 60 = 论文主口径（run_all.py --K 60）。
    """
    panel = C.Panel(pd.date_range(C.WARMUP_START, C.WIN_END, freq="D"))
    load = panel.L(); pv_act = panel.PV()
    price = C.read_attachment1()["price"].to_numpy(float)
    scens = {}
    for h in EPOCHS:
        s_m = C.EPOCH_SLOT[h]
        slots = np.arange(s_m, C.N_SLOT)
        cen, w = Q3.build_epoch_scenarios(panel, h, K=K, slots=slots)
        scens[h] = dict(slots=slots, cen=cen, w=w)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(DAYS)) if permuted_hour is not None else None
    res = {}
    for i, d in enumerate(DAYS):
        L = load.loc[d].to_numpy(float)
        E_plan = np.zeros(C.N_SLOT); E_adj = np.zeros(C.N_SLOT)
        soc = C.S_INIT
        for h in EPOCHS:
            s_m = C.EPOCH_SLOT[h]; slots = scens[h]["slots"]
            src = d if (permuted_hour != h) else DAYS[perm[i]]
            fc = Q3.forecast_slots(panel, src, h, slots)
            nd_fc = np.zeros(C.N_SLOT); nd_fc[slots] = (L[slots] - fc) * C.DT
            cen = scens[h]["cen"]
            nd_sc = np.zeros((cen.shape[0], C.N_SLOT))
            nd_sc[:, slots] = nd_fc[slots][None, :] + cen
            sol = LP.solve_epoch(price, nd_fc, nd_sc, scens[h]["w"], s_init=soc,
                                 s_from=s_m, plan=(E_plan if h > 0 else None),
                                 term_floor=C.S_INIT)
            if h == 0:
                E_plan = sol["E"].copy()
            later = [x for x in EPOCHS if x > h]
            nxt = C.EPOCH_SLOT[later[0]] if later else C.N_SLOT
            for j, t in enumerate(range(s_m, nxt)):
                E_adj[t] = sol["E"][j]
            soc = float(sol["S"][nxt - s_m - 1])
        res[d] = E_adj.copy()
    return res


def main():
    ap = argparse.ArgumentParser(description="非预期性置换检验（默认 K=60，与论文主口径一致）")
    ap.add_argument("--K", type=int, default=60,
                    help="各预报时刻残差场景集的场景数，默认 60（论文主口径）；对照可用 40/120")
    ap.add_argument("--out", default="exp_na.json", help="输出文件名（位于 intermediate/）")
    args = ap.parse_args()
    base = run_days(None, K=args.K)
    out = {"baseline_days": len(base), "K": args.K}
    for h in EPOCHS:
        alt = run_days(permuted_hour=h, seed=1, K=args.K)
        diffs = {}
        for m, (a, b) in CONTROL.items():
            dmax = max(float(np.max(np.abs(alt[d][a:b] - base[d][a:b]))) for d in DAYS)
            diffs[f"槽{a}-{b}"] = dmax
        out[f"置换{h:02d}:00预报"] = diffs
    with open(os.path.join(C.INTERMEDIATE_DIR, args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print()
    print("判读：置换某时刻预报后，该时刻**之前**控制区的最大变化应为 0，")
    print("      而该时刻**之后**控制区的变化应显著大于 0（否则检验没有效力）。")


if __name__ == "__main__":
    main()
