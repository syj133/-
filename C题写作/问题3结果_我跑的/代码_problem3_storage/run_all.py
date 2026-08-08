# -*- coding: utf-8 -*-
"""问题三主程序：运行基线、无储能优化与五种含储能优化策略，输出结果。

场景：
  baseline      附件基准运行状态（直接取自 region_time_data.xlsx）
  no_storage    仅优化购售电与新能源分配（储能固定初始 SOC、不充放电）
  cost_min      含储能，运行成本最优
  carbon_min    含储能，碳排放最优
  peak_min      含储能，区域峰值净购电最小
  flat_min      含储能，净购电波动（MAD）最小
  balanced      含储能，四目标加权归一化综合权衡
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REGIONS, build_params, load_baseline_series  # noqa: E402
from storage_lp import check_constraints, compute_metrics, solve_region  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "C题写作" / "问题3结果_我跑的" / "结果"

SCENARIOS = ["baseline", "no_storage", "cost_min", "carbon_min",
             "peak_min", "flat_min", "balanced"]
SCENARIO_LABELS = {
    "baseline": "附件基线",
    "no_storage": "优化-无储能",
    "cost_min": "优化-成本最优",
    "carbon_min": "优化-低碳最优",
    "peak_min": "优化-削峰最优",
    "flat_min": "优化-平抑波动",
    "balanced": "优化-综合权衡",
}


def metrics_from_series(p: dict, s: dict) -> dict:
    """由任意 (gp, gs, d, rc, gc, c, pdis, soc) 序列计算统一指标。"""
    net = s["gp"] - s["gs"]
    return {
        "cost_cny": float(np.sum(p["price"] * s["gp"] - p["sell_price"] * s["gs"])),
        "carbon_tco2": float(np.sum(p["carbon_intensity"] * s["gp"])),
        "peak_net_mw": float(net.max()),
        "net_std_mw": float(net.std()),
        "net_mad_mw": float(np.mean(np.abs(net - net.mean()))),
        "net_range_mw": float(net.max() - net.min()),
        "renewable_util": float((s["d"] + s["rc"] + s["gs"]).sum() / p["renewable"].sum()),
        "curtailment_mwh": float(s["c"].sum()),
        "charge_mwh": float((s["rc"] + s["gc"]).sum()),
        "discharge_mwh": float(s["pdis"].sum()),
        "purchase_mwh": float(s["gp"].sum()),
        "export_mwh": float(s["gs"].sum()),
        "final_soc_mwh": float(s["soc"][-1]),
    }


def run_region_scenarios(params: dict, baseline_series: dict) -> dict:
    """对每个区域求解各优化场景，返回 {region: {scenario: metrics}}。"""
    results: dict = {}
    for reg in REGIONS:
        p = params[reg]
        bs = baseline_series[reg]
        region_res = {"baseline": metrics_from_series(p, bs)}
        region_res["baseline_out"] = bs
        bm = region_res["baseline"]
        # 单目标最优值（供综合权衡归一化使用）
        ref: dict = {
            "cost": bm["cost_cny"],
            "carbon": bm["carbon_tco2"],
            "peak": bm["peak_net_mw"],
            "flat": bm["net_mad_mw"] * p["T"],
            "weights": {"cost": 0.35, "carbon": 0.30, "peak": 0.15, "flat": 0.20},
        }
        for obj in ["cost", "carbon", "peak", "flat"]:
            out = solve_region(p, objective=obj, with_storage=True, tiebreak_cost=True)
            m = compute_metrics(p, out)
            region_res[f"{obj}_min"] = m
            region_res[f"{obj}_min_out"] = out
        # 无储能
        out_ns = solve_region(p, objective="cost", with_storage=False)
        region_res["no_storage"] = compute_metrics(p, out_ns)
        region_res["no_storage_out"] = out_ns
        # 综合权衡
        out_bal = solve_region(p, objective="balanced", with_storage=True, ref=ref)
        region_res["balanced"] = compute_metrics(p, out_bal)
        region_res["balanced_out"] = out_bal
        results[reg] = region_res
    return results


def save_hourly(params: dict, results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sc in SCENARIOS:
        frames = []
        for reg in REGIONS:
            p = params[reg]
            if sc == "baseline":
                data = results[reg]["baseline_out"]
            else:
                data = results[reg][f"{sc}_out"]
            df = pd.DataFrame({
                "Hour": p["hours"],
                "Region": reg,
                "Load_MW": p["load"],
                "AvailableRenewable_MW": p["renewable"],
                "ElectricityPrice": p["price"],
                "SellPrice": p["sell_price"],
                "CarbonIntensity": p["carbon_intensity"],
                "GridPurchase_MW": data["gp"],
                "GridSell_MW": data["gs"],
                "DirectRenewable_MW": data["d"],
                "RenewableCharge_MW": data["rc"],
                "GridCharge_MW": data["gc"],
                "Curtailment_MW": data["c"],
                "Discharge_MW": data["pdis"],
                "SOC_MWh": data["soc"],
                "NetGridImport_MW": data["gp"] - data["gs"],
            })
            frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(
            out_dir / f"hourly_{sc}.csv", index=False, encoding="utf-8-sig")


def build_summaries(params: dict, results: dict, baseline_series: dict) -> tuple:
    rows_region = []
    rows_scenario = []
    for sc in SCENARIOS:
        agg = {k: 0.0 for k in
               ["cost_cny", "carbon_tco2", "curtailment_mwh", "charge_mwh",
                "discharge_mwh", "purchase_mwh", "export_mwh"]}
        util_num = util_den = 0.0
        peak_sum = 0.0
        net_hour = np.zeros(params["RegionA"]["T"])
        for reg in REGIONS:
            p = params[reg]
            m = results[reg][sc]
            for k in agg:
                agg[k] += m[k]
            util_num += m["renewable_util"] * p["renewable"].sum()
            util_den += p["renewable"].sum()
            peak_sum += m["peak_net_mw"]
            if sc == "baseline":
                bs = baseline_series[reg]
                net_hour += bs["gp"] - bs["gs"]
            else:
                out = results[reg][f"{sc}_out"]
                net_hour += out["gp"] - out["gs"]
            rows_region.append({
                "scenario": sc, "scenario_label": SCENARIO_LABELS[sc],
                "region": reg, **{k: round(v, 4) for k, v in m.items()},
            })
        sys_peak = float(net_hour.max())
        sys_mad = float(np.mean(np.abs(net_hour - net_hour.mean())))
        sys_std = float(net_hour.std())
        rows_scenario.append({
            "scenario": sc,
            "scenario_label": SCENARIO_LABELS[sc],
            "total_cost_cny": round(agg["cost_cny"], 2),
            "total_carbon_tco2": round(agg["carbon_tco2"], 4),
            "sys_peak_net_mw": round(sys_peak, 4),
            "sum_region_peak_mw": round(peak_sum, 4),
            "sys_net_mad_mw": round(sys_mad, 4),
            "sys_net_std_mw": round(sys_std, 4),
            "total_renewable_util_pct": round(util_num / util_den * 100, 4),
            "total_curtail_mwh": round(agg["curtailment_mwh"], 2),
            "total_charge_mwh": round(agg["charge_mwh"], 2),
            "total_discharge_mwh": round(agg["discharge_mwh"], 2),
            "total_purchase_mwh": round(agg["purchase_mwh"], 2),
            "total_export_mwh": round(agg["export_mwh"], 2),
        })
    df_region = pd.DataFrame(rows_region)
    df_scenario = pd.DataFrame(rows_scenario)
    return df_region, df_scenario


def build_constraint_check(params: dict, results: dict, baseline_series: dict) -> dict:
    checks: dict = {}
    for reg in REGIONS:
        p = params[reg]
        checks[reg] = {}
        for sc in SCENARIOS:
            if sc == "baseline":
                out = results[reg]["baseline_out"]
                checks[reg][sc] = {k: round(v, 6) for k, v in check_constraints(p, out).items()}
            else:
                checks[reg][sc] = {k: round(v, 6) for k, v in
                                   check_constraints(p, results[reg][f"{sc}_out"]).items()}
    return checks


def main() -> None:
    params = build_params()
    baseline_series = load_baseline_series()
    print("参数构建完成，区域数：", len(params), "，每小时时域：0-2406")
    results = run_region_scenarios(params, baseline_series)
    print("全部区域/场景求解完成")

    df_region, df_scenario = build_summaries(params, results, baseline_series)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_region.to_csv(OUT_DIR / "per_region_metrics.csv", index=False, encoding="utf-8-sig")
    df_scenario.to_csv(OUT_DIR / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    save_hourly(params, results, OUT_DIR)

    checks = build_constraint_check(params, results, baseline_series)
    (OUT_DIR / "constraints_check.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 场景汇总（全时域 0-2406） =====")
    print(df_scenario.to_string(index=False))


if __name__ == "__main__":
    main()
