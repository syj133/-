# -*- coding: utf-8 -*-
"""问题三敏感性分析：
1) 储能容量系数（0.25/0.5/1.0/1.5/2.0）下的成本最优与综合权衡策略；
2) 充放电效率偏移（±0.05）下的综合权衡策略。

输出：results/sensitivity_capacity.csv、results/sensitivity_efficiency.csv
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REGIONS, ROOT, build_params, load_baseline_series  # noqa: E402
from run_all import metrics_from_series  # noqa: E402
from storage_lp import compute_metrics, solve_region  # noqa: E402

RESULT_DIR = ROOT / "C题写作" / "问题3结果_我跑的" / "结果"
BALANCED_WEIGHTS = {"cost": 0.35, "carbon": 0.30, "peak": 0.15, "flat": 0.20}


def make_ref(params: dict, baseline_series: dict) -> dict:
    """按附件基线值构造 balanced 目标的归一化基准。"""
    ref = {}
    for reg in REGIONS:
        p = params[reg]
        bm = metrics_from_series(p, baseline_series[reg])
        ref[reg] = {
            "cost": bm["cost_cny"],
            "carbon": bm["carbon_tco2"],
            "peak": bm["peak_net_mw"],
            "flat": bm["net_mad_mw"] * p["T"],
            "weights": BALANCED_WEIGHTS,
        }
    return ref


def scale_capacity(params: dict, k: float) -> dict:
    """按容量系数 k 缩放储能容量、下限与初始 SOC（充放电功率不变）。"""
    out = {}
    for reg, p in params.items():
        q = copy.deepcopy(p)
        q["soc_max"] = p["soc_max"] * k
        q["soc_min"] = p["soc_min"] * k
        q["soc0"] = p["soc0"] * k
        out[reg] = q
    return out


def shift_efficiency(params: dict, delta: float) -> dict:
    out = {}
    for reg, p in params.items():
        q = copy.deepcopy(p)
        q["eta_c"] = p["eta_c"] + delta
        q["eta_d"] = p["eta_d"] + delta
        out[reg] = q
    return out


def aggregate(params: dict, outs: dict) -> dict:
    T = params["RegionA"]["T"]
    cost = carbon = charge = discharge = purchase = export = curtail = 0.0
    util_num = util_den = 0.0
    peak_sum = 0.0
    net_hour = np.zeros(T)
    for reg in REGIONS:
        p = params[reg]
        m = compute_metrics(p, outs[reg])
        cost += m["cost_cny"]
        carbon += m["carbon_tco2"]
        charge += m["charge_mwh"]
        discharge += m["discharge_mwh"]
        purchase += m["purchase_mwh"]
        export += m["export_mwh"]
        curtail += m["curtailment_mwh"]
        util_num += m["renewable_util"] * p["renewable"].sum()
        util_den += p["renewable"].sum()
        peak_sum += m["peak_net_mw"]
        net_hour += outs[reg]["gp"] - outs[reg]["gs"]
    return {
        "total_cost_cny": cost,
        "total_carbon_tco2": carbon,
        "sys_peak_net_mw": float(net_hour.max()),
        "sum_region_peak_mw": peak_sum,
        "sys_net_mad_mw": float(np.mean(np.abs(net_hour - net_hour.mean()))),
        "sys_net_std_mw": float(net_hour.std()),
        "total_renewable_util_pct": util_num / util_den * 100,
        "total_curtail_mwh": curtail,
        "total_charge_mwh": charge,
        "total_discharge_mwh": discharge,
        "total_purchase_mwh": purchase,
        "total_export_mwh": export,
    }


def run_capacity(params: dict) -> pd.DataFrame:
    rows = []
    ref = make_ref(params, load_baseline_series())
    for k in [0.25, 0.5, 1.0, 1.5, 2.0]:
        pk = scale_capacity(params, k)
        for obj in ["cost", "balanced"]:
            outs = {reg: solve_region(pk[reg], obj, True,
                                      ref=ref[reg] if obj == "balanced" else None,
                                      tiebreak_cost=(obj == "cost"))
                    for reg in REGIONS}
            agg = aggregate(pk, outs)
            rows.append({"capacity_mult": k, "objective": obj, **agg})
            print(f"capacity={k:.2f} obj={obj} cost={agg['total_cost_cny']/1e8:,.2f}亿元 "
                  f"peak={agg['sys_peak_net_mw']:.1f} MAD={agg['sys_net_mad_mw']:.3f}")
    return pd.DataFrame(rows)


def run_efficiency(params: dict) -> pd.DataFrame:
    rows = []
    ref = make_ref(params, load_baseline_series())
    for d in [-0.05, 0.0, 0.05]:
        pd_ = shift_efficiency(params, d)
        outs = {reg: solve_region(pd_[reg], "balanced", True, ref=ref[reg])
                for reg in REGIONS}
        agg = aggregate(pd_, outs)
        rows.append({"eta_delta": d, **agg})
        print(f"eta_delta={d:+.2f} cost={agg['total_cost_cny']/1e8:,.2f}亿元 "
              f"peak={agg['sys_peak_net_mw']:.1f} MAD={agg['sys_net_mad_mw']:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    params = build_params()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df_cap = run_capacity(params)
    df_cap.to_csv(RESULT_DIR / "sensitivity_capacity.csv", index=False, encoding="utf-8-sig")
    df_eff = run_efficiency(params)
    df_eff.to_csv(RESULT_DIR / "sensitivity_efficiency.csv", index=False, encoding="utf-8-sig")
    print("敏感性结果已保存至", RESULT_DIR)


if __name__ == "__main__":
    main()
