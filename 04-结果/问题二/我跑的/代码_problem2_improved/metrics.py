"""Unified-caliber metrics, baseline comparison and constraint verification.

Power balance per region-hour (Problem 2, no storage, 附件1口径):
    GridPurchase + UsedRenewable = Total_Load
    UsedRenewable + GridSell + Curtailment = AvailableRenewable
    GridSell <= MaxGridExport(region)
    Cost = GridPurchase*Price - GridSell*SellPrice
    Carbon = GridPurchase*CarbonIntensity
    Utilization = (UsedRenewable + GridSell) / AvailableRenewable
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from common import REGIONS, TASK_END_EXCLUSIVE, existing_nonai_load, get_rt_value, overlaps


def compute_hourly_metrics(
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    min_hour: int,
    max_hour: int,
) -> pd.DataFrame:
    gpu_idx: pd.DataFrame = maps["gpu"]  # type: ignore[assignment]
    rt: pd.DataFrame = maps["rt"]  # type: ignore[assignment]
    export_limits: Dict[str, float] = maps["export_limits"]  # type: ignore[assignment]
    rows = []
    for hour in range(min_hour, min(max_hour + 1, TASK_END_EXCLUSIVE)):
        for region in REGIONS:
            nonai = existing_nonai_load(rt, region, hour)
            ai = ai_load.get((region, hour), 0.0)
            pue = float(gpu_idx.loc[region, "PUE"])
            it_load = nonai + ai
            total_load = it_load * pue

            available = get_rt_value(rt, region, hour, "AvailableRenewable_MW", 0.0)
            used = min(total_load, available)
            surplus = max(0.0, available - used)
            sell_limit = float(export_limits.get(region, 0.0))
            grid_sell = min(surplus, sell_limit)
            curtailment = surplus - grid_sell
            grid_purchase = max(0.0, total_load - used)

            price = get_rt_value(rt, region, hour, "ElectricityPrice_CNY_per_MWh", 0.0)
            sell_price = get_rt_value(rt, region, hour, "SellPrice_CNY_per_MWh", 0.0)
            carbon_intensity = get_rt_value(rt, region, hour, "CarbonIntensity_tCO2_per_MWh", 0.0)
            cost = grid_purchase * price - grid_sell * sell_price
            carbon = grid_purchase * carbon_intensity
            util = (used + grid_sell) / available if available > 0 else 0.0

            available_gpu = float(gpu_idx.loc[region, "Available_GPU"])
            gpu = gpu_used.get((region, hour), 0.0)
            rows.append(
                {
                    "Hour": hour,
                    "Region": region,
                    "GPU_Used": gpu,
                    "GPU_Utilization": gpu / available_gpu if available_gpu else 0.0,
                    "AI_IT_Load_MW": ai,
                    "NonAI_IT_Load_MW": nonai,
                    "IT_Load_MW": it_load,
                    "Total_Load_MW": total_load,
                    "AvailableRenewable_MW": available,
                    "UsedRenewable_MW": used,
                    "GridSell_MW": grid_sell,
                    "Curtailment_MW": curtailment,
                    "GridPurchase_MW": grid_purchase,
                    "NetImport_MW": grid_purchase - grid_sell,
                    "Cost_CNY": cost,
                    "CarbonEmission_tCO2": carbon,
                    "RenewableUtilization": util,
                }
            )
    return pd.DataFrame(rows)


def baseline_metrics(
    maps: Dict[str, object],
    min_hour: int,
    max_hour: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Baseline metrics computed from region_time_data over the same hours."""
    rt: pd.DataFrame = maps["rt"]  # type: ignore[assignment]
    rows = []
    for hour in range(min_hour, min(max_hour + 1, TASK_END_EXCLUSIVE)):
        for region in REGIONS:
            available = get_rt_value(rt, region, hour, "AvailableRenewable_MW", 0.0)
            used = get_rt_value(rt, region, hour, "UsedRenewable_MW", 0.0)
            ren_charge = get_rt_value(rt, region, hour, "RenewableCharge_MW", 0.0)
            grid_sell = get_rt_value(rt, region, hour, "GridSell_MW", 0.0)
            grid_purchase = get_rt_value(rt, region, hour, "GridPurchase_MW", 0.0)
            price = get_rt_value(rt, region, hour, "ElectricityPrice_CNY_per_MWh", 0.0)
            sell_price = get_rt_value(rt, region, hour, "SellPrice_CNY_per_MWh", 0.0)
            carbon_intensity = get_rt_value(rt, region, hour, "CarbonIntensity_tCO2_per_MWh", 0.0)
            total_load = get_rt_value(rt, region, hour, "Total_Load_MW", 0.0)
            curtailment = get_rt_value(rt, region, hour, "Curtailment_MW", 0.0)
            net_import = get_rt_value(rt, region, hour, "NetGridImport_MW", 0.0)
            rows.append(
                {
                    "Hour": hour,
                    "Region": region,
                    "Total_Load_MW": total_load,
                    "AvailableRenewable_MW": available,
                    "UsedRenewable_MW": used,
                    "RenewableCharge_MW": ren_charge,
                    "GridSell_MW": grid_sell,
                    "Curtailment_MW": curtailment,
                    "GridPurchase_MW": grid_purchase,
                    "NetImport_MW": net_import,
                    "Cost_CNY": grid_purchase * price - grid_sell * sell_price,
                    "CarbonEmission_tCO2": grid_purchase * carbon_intensity,
                }
            )
    df = pd.DataFrame(rows)
    avail = float(df["AvailableRenewable_MW"].sum()) if not df.empty else 0.0
    util = (
        float((df["UsedRenewable_MW"] + df["RenewableCharge_MW"] + df["GridSell_MW"]).sum()) / avail
        if avail > 0
        else 0.0
    )
    hourly_net = df.groupby("Hour")["NetImport_MW"].sum()
    hourly_pur = df.groupby("Hour")["GridPurchase_MW"].sum()
    summary = {
        "baseline_total_cost_cny": float(df["Cost_CNY"].sum()) if not df.empty else 0.0,
        "baseline_total_carbon_tco2": float(df["CarbonEmission_tCO2"].sum()) if not df.empty else 0.0,
        "baseline_renewable_utilization": util,
        "baseline_total_curtailment_mwh": float(df["Curtailment_MW"].sum()) if not df.empty else 0.0,
        "baseline_system_peak_grid_purchase_mw": float(hourly_pur.max()) if not hourly_pur.empty else 0.0,
        "baseline_system_peak_net_import_mw": float(hourly_net.max()) if not hourly_net.empty else 0.0,
        "baseline_region_peak_net_import_mw": {
            r: float(df.loc[df["Region"] == r, "NetImport_MW"].max())
            for r in REGIONS
            if not df.loc[df["Region"] == r].empty
        },
    }
    return df, summary


def verify_schedule(
    schedule: pd.DataFrame,
    maps: Dict[str, object],
) -> Dict[str, object]:
    """Check latency, timing and hourly capacity constraints of a schedule."""
    gpu_idx: pd.DataFrame = maps["gpu"]  # type: ignore[assignment]
    latency_map: Dict[Tuple[str, str], float] = maps["latency"]  # type: ignore[assignment]
    power_map: Dict[str, float] = maps["power"]  # type: ignore[assignment]
    rt: pd.DataFrame = maps["rt"]  # type: ignore[assignment]

    violations = {
        "start_before_arrival": 0,
        "finish_after_deadline": 0,
        "finish_at_or_after_2406": 0,
        "latency_exceeded": 0,
        "gpu_capacity": 0,
        "it_power_capacity": 0,
        "facility_power_capacity": 0,
        "task_duplicates": 0,
        "missing_region": 0,
    }
    gpu_used: Dict[Tuple[str, int], float] = {}
    ai_load: Dict[Tuple[str, int], float] = {}

    sched = schedule[schedule["Status"] == "scheduled"] if not schedule.empty else schedule
    seen: set = set()
    for _, row in sched.iterrows():
        tid = int(row["TaskID"])
        if tid in seen:
            violations["task_duplicates"] += 1
        seen.add(tid)
        if row["AssignedRegion"] is None or pd.isna(row["AssignedRegion"]):
            violations["missing_region"] += 1
            continue
        region = str(row["AssignedRegion"])
        start = float(row["StartHour"])
        finish = float(row["FinishHour"])
        if start + 1e-9 < float(row["ArrivalHour"]):
            violations["start_before_arrival"] += 1
        if finish > float(row["LatestFinishHour"]) + 1e-9:
            violations["finish_after_deadline"] += 1
        if finish > TASK_END_EXCLUSIVE + 1e-9:
            violations["finish_at_or_after_2406"] += 1
        if float(row["NetworkLatency_ms"]) > float(row["MaxLatency_ms"]) + 1e-9:
            violations["latency_exceeded"] += 1

        gpu_demand = float(row["GPU_Demand"])
        ai_full = gpu_demand * float(power_map[row["TaskType"]])
        for hour, frac in overlaps(int(start), float(row["DurationHours"])):
            gpu_used[(region, hour)] = gpu_used.get((region, hour), 0.0) + gpu_demand * frac
            ai_load[(region, hour)] = ai_load.get((region, hour), 0.0) + ai_full * frac

    for (region, hour), gpu in gpu_used.items():
        if gpu > float(gpu_idx.loc[region, "Available_GPU"]) + 1e-6:
            violations["gpu_capacity"] += 1
        nonai = existing_nonai_load(rt, region, hour)
        it = nonai + ai_load[(region, hour)]
        if it > float(gpu_idx.loc[region, "Max_IT_Power_MW"]) + 1e-6:
            violations["it_power_capacity"] += 1
        if it * float(gpu_idx.loc[region, "PUE"]) > float(gpu_idx.loc[region, "Max_Facility_Power_MW"]) + 1e-6:
            violations["facility_power_capacity"] += 1

    violations["total"] = sum(violations.values())
    return violations


def compute_summary(
    schedule: pd.DataFrame,
    hourly: pd.DataFrame,
    solver_used: str,
    args,
    weights,
    baseline: Optional[Dict[str, float]] = None,
    verification: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    scheduled = schedule["Status"].eq("scheduled") if not schedule.empty else pd.Series(dtype=bool)
    total_available = float(hourly["AvailableRenewable_MW"].sum()) if not hourly.empty else 0.0
    total_used = float(hourly["UsedRenewable_MW"].sum()) if not hourly.empty else 0.0
    total_sell = float(hourly["GridSell_MW"].sum()) if not hourly.empty else 0.0
    hourly_purchase = hourly.groupby("Hour")["GridPurchase_MW"].sum() if not hourly.empty else pd.Series(dtype=float)
    hourly_net = hourly.groupby("Hour")["NetImport_MW"].sum() if not hourly.empty else pd.Series(dtype=float)

    latency_by_type: Dict[str, float] = {}
    if scheduled.any():
        for ttype, grp in schedule.loc[scheduled].groupby("TaskType"):
            latency_by_type[str(ttype)] = float(grp["NetworkLatency_ms"].mean())

    load_std = float(hourly["Total_Load_MW"].std()) if not hourly.empty else 0.0
    load_mean = float(hourly["Total_Load_MW"].mean()) if not hourly.empty else 0.0
    load_peak = float(hourly["Total_Load_MW"].max()) if not hourly.empty else 0.0

    region_peak_net = {
        r: float(hourly.loc[hourly["Region"] == r, "NetImport_MW"].max())
        for r in REGIONS
        if not hourly.loc[hourly["Region"] == r].empty
    }

    summary: Dict[str, object] = {
        "solver_used": solver_used,
        "arrival_hour_start": getattr(args, "hour_start", None),
        "arrival_hour_end": getattr(args, "hour_end", None),
        "metric_hour_start": int(hourly["Hour"].min()) if not hourly.empty else None,
        "metric_hour_end": int(hourly["Hour"].max()) if not hourly.empty else None,
        "step_hours": getattr(args, "step_hours", None),
        "lookahead_hours": getattr(args, "lookahead_hours", None),
        "max_candidates_per_task": getattr(args, "max_candidates_per_task", None),
        "task_count": int(len(schedule)),
        "scheduled_count": int(scheduled.sum()) if not schedule.empty else 0,
        "unassigned_count": int((~scheduled).sum()) if not schedule.empty else 0,
        "on_time_rate": float(scheduled.mean()) if len(schedule) else 0.0,
        "total_cost_cny": float(hourly["Cost_CNY"].sum()) if not hourly.empty else 0.0,
        "total_carbon_tco2": float(hourly["CarbonEmission_tCO2"].sum()) if not hourly.empty else 0.0,
        "renewable_utilization": (total_used + total_sell) / total_available if total_available else 0.0,
        "renewable_direct_use_mwh": total_used,
        "renewable_sell_mwh": total_sell,
        "total_curtailment_mwh": float(hourly["Curtailment_MW"].sum()) if not hourly.empty else 0.0,
        "system_peak_grid_purchase_mw": float(hourly_purchase.max()) if not hourly_purchase.empty else 0.0,
        "system_peak_net_import_mw": float(hourly_net.max()) if not hourly_net.empty else 0.0,
        "region_peak_net_import_mw": region_peak_net,
        "average_latency_ms": float(schedule.loc[scheduled, "NetworkLatency_ms"].mean()) if scheduled.any() else None,
        "latency_by_type_ms": latency_by_type,
        "load_fluctuation_std_mw": load_std,
        "load_peak_to_average": load_peak / load_mean if load_mean else 0.0,
        "avg_gpu_utilization": float(hourly["GPU_Utilization"].mean()) if not hourly.empty else 0.0,
        "weights": {
            "cost": weights.cost,
            "carbon": weights.carbon,
            "latency": weights.latency,
            "renewable": weights.renewable,
            "congestion": weights.congestion,
            "unassigned": weights.unassigned,
        },
        "verification": verification or {},
    }
    if baseline:
        b = baseline
        summary["baseline"] = b
        for key, bkey in [
            ("cost", "baseline_total_cost_cny"),
            ("carbon", "baseline_total_carbon_tco2"),
            ("utilization", "baseline_renewable_utilization"),
            ("curtailment", "baseline_total_curtailment_mwh"),
            ("peak_net_import", "baseline_system_peak_net_import_mw"),
        ]:
            base_val = float(b.get(bkey, 0.0) or 0.0)
            ours = float(summary["total_cost_cny"] if key == "cost" else (
                summary["total_carbon_tco2"] if key == "carbon" else (
                    summary["renewable_utilization"] if key == "utilization" else (
                        summary["total_curtailment_mwh"] if key == "curtailment" else summary["system_peak_net_import_mw"]
                    )
                )
            ))
            summary[f"delta_{key}_vs_baseline"] = ours - base_val
            summary[f"delta_{key}_pct_vs_baseline"] = (
                (ours - base_val) / abs(base_val) if base_val else None
            )
    return summary
