"""
Problem 2: weighted single-objective rolling-window scheduling.

This script implements a practical framework for the contest problem:
1. Read workload, GPU, latency, power-mapping, and region-time data.
2. Generate feasible region/start-time candidates for each task.
3. Solve each rolling window as a weighted single-objective assignment model.
4. Update GPU and AI IT load states, then recompute cost/carbon/renewable metrics.

Solver modes:
- auto: use PuLP if installed; otherwise use greedy fallback.
- pulp: rolling MILP with PuLP. Install with: pip install pulp
- greedy: candidate scoring fallback that requires only pandas/numpy/openpyxl.

Example:
python problem2_rolling_milp.py --solver greedy --hour-start 2376 --hour-end 2399
python problem2_rolling_milp.py --solver pulp --step-hours 24 --lookahead-hours 72
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
TASK_END_EXCLUSIVE = 2406


@dataclass(frozen=True)
class Weights:
    cost: float
    carbon: float
    latency: float
    renewable: float
    congestion: float
    unassigned: float


@dataclass
class Candidate:
    task_id: int
    region: str
    start: int
    finish: float
    latency_ms: float
    score: float
    cost_cny: float
    carbon_tco2: float
    renewable_mwh: float
    congestion: float
    overlaps: List[Tuple[int, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\35485\AppData\Local\Temp"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/problem2_results"))
    parser.add_argument("--solver", choices=["auto", "pulp", "greedy"], default="auto")
    parser.add_argument("--hour-start", type=int, default=0)
    parser.add_argument("--hour-end", type=int, default=2399, help="Last arrival hour included.")
    parser.add_argument("--step-hours", type=int, default=24)
    parser.add_argument("--lookahead-hours", type=int, default=72)
    parser.add_argument("--max-candidates-per-task", type=int, default=24)
    parser.add_argument("--max-tasks", type=int, default=None, help="Debug limit after hour filtering.")
    parser.add_argument("--mip-time-limit-sec", type=int, default=120)
    parser.add_argument("--cost-weight", type=float, default=1.0)
    parser.add_argument("--carbon-weight", type=float, default=500.0)
    parser.add_argument("--latency-weight", type=float, default=2.0)
    parser.add_argument("--renewable-weight", type=float, default=100.0)
    parser.add_argument("--congestion-weight", type=float, default=100.0)
    parser.add_argument("--unassigned-penalty", type=float, default=1e9)
    return parser.parse_args()


def find_file(data_dir: Path, prefix: str, suffix: str = ".xlsx") -> Path:
    matches = sorted(data_dir.glob(f"{prefix}*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {prefix}*{suffix} under {data_dir}")
    return matches[0]


def load_inputs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workload = pd.read_excel(find_file(data_dir, "workload_trace"), sheet_name="Sheet1")
    gpu = pd.read_excel(find_file(data_dir, "GPU_information"), sheet_name="GPU中心基础情况")
    latency = pd.read_excel(find_file(data_dir, "network_latency"), sheet_name="network_latency")
    power = pd.read_excel(find_file(data_dir, "power_mapping"), sheet_name="任务功率映射")
    region_time = pd.read_excel(find_file(data_dir, "region_time_data"), sheet_name="region_time_data")
    return workload, gpu, latency, power, region_time


def build_maps(
    gpu: pd.DataFrame,
    latency: pd.DataFrame,
    power: pd.DataFrame,
    region_time: pd.DataFrame,
) -> Dict[str, object]:
    gpu_idx = gpu.set_index("Region")
    latency_map = {
        (row.FromRegion, row.ToRegion): float(row.NetworkLatency_ms)
        for row in latency.itertuples(index=False)
    }
    power_map = {
        row.TaskType: float(row.GPU_Power_MW_per_EquivalentGPU)
        for row in power.itertuples(index=False)
    }
    rt = region_time.set_index(["Region", "Hour"]).sort_index()
    return {
        "gpu": gpu_idx,
        "latency": latency_map,
        "power": power_map,
        "rt": rt,
    }


def prepare_workload(
    workload: pd.DataFrame,
    hour_start: int,
    hour_end: int,
    max_tasks: Optional[int],
) -> pd.DataFrame:
    cols = [
        "TaskID",
        "TaskType",
        "ArrivalHour",
        "GPU_Demand",
        "EstimatedDuration_min",
        "DelaySensitivity",
        "SourceRegion",
        "MaxLatency_ms",
        "LatestFinishHour",
    ]
    df = workload.loc[(workload["ArrivalHour"] >= hour_start) & (workload["ArrivalHour"] <= hour_end), cols].copy()
    df = df.sort_values(["ArrivalHour", "TaskID"]).reset_index(drop=True)
    if max_tasks is not None:
        df = df.head(max_tasks).copy()
    df["DurationHours"] = df["EstimatedDuration_min"] / 60.0
    df["DiscreteDuration"] = np.ceil(df["DurationHours"]).astype(int)
    df["LatestStartHour"] = np.floor(
        np.minimum(df["LatestFinishHour"], TASK_END_EXCLUSIVE) - df["DurationHours"]
    ).astype(int)
    return df


def overlaps(start: int, duration_hours: float) -> List[Tuple[int, float]]:
    end = start + duration_hours
    first = int(math.floor(start))
    last = int(math.ceil(end))
    out: List[Tuple[int, float]] = []
    for hour in range(first, last):
        if hour < 0 or hour >= TASK_END_EXCLUSIVE:
            continue
        ov = max(0.0, min(hour + 1.0, end) - max(float(hour), float(start)))
        if ov > 1e-9:
            out.append((hour, ov))
    return out


def get_rt_value(rt: pd.DataFrame, region: str, hour: int, column: str, default: float = 0.0) -> float:
    try:
        return float(rt.loc[(region, hour), column])
    except KeyError:
        return default


def existing_nonai_load(rt: pd.DataFrame, region: str, hour: int) -> float:
    return get_rt_value(rt, region, hour, "NonAI_IT_Load_MW", 0.0)


def candidate_score(
    task: pd.Series,
    region: str,
    start: int,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
) -> Optional[Candidate]:
    gpu_idx: pd.DataFrame = maps["gpu"]  # type: ignore[assignment]
    latency_map: Dict[Tuple[str, str], float] = maps["latency"]  # type: ignore[assignment]
    power_map: Dict[str, float] = maps["power"]  # type: ignore[assignment]
    rt: pd.DataFrame = maps["rt"]  # type: ignore[assignment]

    latency_ms = latency_map.get((task.SourceRegion, region), math.inf)
    if latency_ms > float(task.MaxLatency_ms):
        return None

    ov = overlaps(int(start), float(task.DurationHours))
    if not ov:
        return None

    gpu_demand = float(task.GPU_Demand)
    unit_power = float(power_map[task.TaskType])
    task_ai_mw_full = gpu_demand * unit_power
    pue = float(gpu_idx.loc[region, "PUE"])

    cost = 0.0
    carbon = 0.0
    renewable_used = 0.0
    max_capacity_ratio = 0.0

    for hour, frac in ov:
        current_gpu = gpu_used.get((region, hour), 0.0)
        available_gpu = float(gpu_idx.loc[region, "Available_GPU"])
        if current_gpu + gpu_demand * frac > available_gpu + 1e-9:
            return None

        current_ai = ai_load.get((region, hour), 0.0)
        nonai = existing_nonai_load(rt, region, hour)
        add_ai_mw = task_ai_mw_full * frac
        it_load = nonai + current_ai + add_ai_mw
        total_load = it_load * pue
        if it_load > float(gpu_idx.loc[region, "Max_IT_Power_MW"]) + 1e-9:
            return None
        if total_load > float(gpu_idx.loc[region, "Max_Facility_Power_MW"]) + 1e-9:
            return None

        price = get_rt_value(rt, region, hour, "ElectricityPrice_CNY_per_MWh", 0.0)
        carbon_intensity = get_rt_value(rt, region, hour, "CarbonIntensity_tCO2_per_MWh", 0.0)
        available_renewable = get_rt_value(rt, region, hour, "AvailableRenewable_MW", 0.0)

        base_total_load = (nonai + current_ai) * pue
        renewable_surplus = max(0.0, available_renewable - base_total_load)
        add_facility_mw = add_ai_mw * pue
        add_renewable = min(add_facility_mw, renewable_surplus)
        add_grid = max(0.0, add_facility_mw - add_renewable)

        cost += add_grid * price
        carbon += add_grid * carbon_intensity
        renewable_used += add_renewable
        max_capacity_ratio = max(max_capacity_ratio, (current_gpu + gpu_demand * frac) / available_gpu)

    score = (
        weights.cost * cost
        + weights.carbon * carbon
        + weights.latency * latency_ms
        - weights.renewable * renewable_used
        + weights.congestion * max_capacity_ratio
    )
    return Candidate(
        task_id=int(task.TaskID),
        region=region,
        start=int(start),
        finish=float(start) + float(task.DurationHours),
        latency_ms=float(latency_ms),
        score=float(score),
        cost_cny=float(cost),
        carbon_tco2=float(carbon),
        renewable_mwh=float(renewable_used),
        congestion=float(max_capacity_ratio),
        overlaps=ov,
    )


def generate_candidates(
    task: pd.Series,
    current_hour: int,
    lookahead_hours: int,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
    max_candidates: int,
) -> List[Candidate]:
    is_realtime = task.TaskType == "RealTimeInference" or str(task.DelaySensitivity).lower() == "high"
    earliest = max(int(task.ArrivalHour), current_hour)
    latest_start = int(min(task.LatestStartHour, current_hour + lookahead_hours - 1))
    if is_realtime:
        starts = [int(task.ArrivalHour)] if int(task.ArrivalHour) >= current_hour else []
    else:
        starts = list(range(earliest, latest_start + 1))
    if not starts:
        return []

    candidates: List[Candidate] = []
    for region in REGIONS:
        for start in starts:
            cand = candidate_score(task, region, start, maps, gpu_used, ai_load, weights)
            if cand is not None:
                candidates.append(cand)

    candidates.sort(key=lambda c: c.score)
    return candidates[:max_candidates]


def apply_candidate(
    task: pd.Series,
    cand: Candidate,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
) -> None:
    power_map: Dict[str, float] = maps["power"]  # type: ignore[assignment]
    gpu_demand = float(task.GPU_Demand)
    task_ai_mw_full = gpu_demand * float(power_map[task.TaskType])
    for hour, frac in cand.overlaps:
        gpu_used[(cand.region, hour)] = gpu_used.get((cand.region, hour), 0.0) + gpu_demand * frac
        ai_load[(cand.region, hour)] = ai_load.get((cand.region, hour), 0.0) + task_ai_mw_full * frac


def solve_window_greedy(
    window_tasks: pd.DataFrame,
    current_hour: int,
    lookahead_hours: int,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
    max_candidates: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, task in window_tasks.sort_values(["ArrivalHour", "LatestFinishHour", "TaskID"]).iterrows():
        candidates = generate_candidates(
            task, current_hour, lookahead_hours, maps, gpu_used, ai_load, weights, max_candidates
        )
        if not candidates:
            rows.append(schedule_row(task, None, "unassigned"))
            continue
        best = candidates[0]
        apply_candidate(task, best, maps, gpu_used, ai_load)
        rows.append(schedule_row(task, best, "scheduled"))
    return rows


def solve_window_pulp(
    window_tasks: pd.DataFrame,
    current_hour: int,
    lookahead_hours: int,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
    max_candidates: int,
    time_limit_sec: int,
) -> List[Dict[str, object]]:
    try:
        import pulp  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PuLP is not installed. Use --solver greedy or install pulp.") from exc

    gpu_idx: pd.DataFrame = maps["gpu"]  # type: ignore[assignment]
    power_map: Dict[str, float] = maps["power"]  # type: ignore[assignment]
    rt: pd.DataFrame = maps["rt"]  # type: ignore[assignment]

    task_records = {int(row.TaskID): row for _, row in window_tasks.iterrows()}
    candidate_by_task: Dict[int, List[Candidate]] = {}
    for _, task in window_tasks.iterrows():
        candidate_by_task[int(task.TaskID)] = generate_candidates(
            task, current_hour, lookahead_hours, maps, gpu_used, ai_load, weights, max_candidates
        )

    model = pulp.LpProblem(f"rolling_window_{current_hour}", pulp.LpMinimize)
    x: Dict[Tuple[int, int], object] = {}
    unassigned: Dict[int, object] = {}

    objective_terms = []
    for task_id, cands in candidate_by_task.items():
        for k, cand in enumerate(cands):
            var = pulp.LpVariable(f"x_{task_id}_{k}", lowBound=0, upBound=1, cat="Binary")
            x[(task_id, k)] = var
            objective_terms.append(cand.score * var)
        u = pulp.LpVariable(f"unassigned_{task_id}", lowBound=0, upBound=1, cat="Binary")
        unassigned[task_id] = u
        task = task_records[task_id]
        objective_terms.append(weights.unassigned * float(task.GPU_Demand) * float(task.DurationHours) * u)
        model += pulp.lpSum([x[(task_id, k)] for k in range(len(cands))] + [u]) == 1

    model += pulp.lpSum(objective_terms)

    affected_hours = range(current_hour, min(TASK_END_EXCLUSIVE, current_hour + lookahead_hours + 24))
    for region in REGIONS:
        for hour in affected_hours:
            gpu_terms = []
            ai_terms = []
            for task_id, cands in candidate_by_task.items():
                task = task_records[task_id]
                gpu_demand = float(task.GPU_Demand)
                task_ai_full = gpu_demand * float(power_map[task.TaskType])
                for k, cand in enumerate(cands):
                    frac = next((ov for h, ov in cand.overlaps if h == hour and cand.region == region), 0.0)
                    if frac > 0:
                        gpu_terms.append(gpu_demand * frac * x[(task_id, k)])
                        ai_terms.append(task_ai_full * frac * x[(task_id, k)])

            available_gpu = float(gpu_idx.loc[region, "Available_GPU"])
            model += pulp.lpSum(gpu_terms) + gpu_used.get((region, hour), 0.0) <= available_gpu

            nonai = existing_nonai_load(rt, region, hour)
            max_it = float(gpu_idx.loc[region, "Max_IT_Power_MW"])
            pue = float(gpu_idx.loc[region, "PUE"])
            max_facility = float(gpu_idx.loc[region, "Max_Facility_Power_MW"])
            total_ai = pulp.lpSum(ai_terms) + ai_load.get((region, hour), 0.0)
            model += nonai + total_ai <= max_it
            model += (nonai + total_ai) * pue <= max_facility

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    model.solve(solver)

    rows: List[Dict[str, object]] = []
    for task_id, task in task_records.items():
        chosen: Optional[Candidate] = None
        for k, cand in enumerate(candidate_by_task[task_id]):
            if pulp.value(x[(task_id, k)]) is not None and pulp.value(x[(task_id, k)]) > 0.5:
                chosen = cand
                break
        if chosen is None:
            rows.append(schedule_row(task, None, "unassigned"))
        else:
            apply_candidate(task, chosen, maps, gpu_used, ai_load)
            rows.append(schedule_row(task, chosen, "scheduled"))
    return rows


def schedule_row(task: pd.Series, cand: Optional[Candidate], status: str) -> Dict[str, object]:
    base = {
        "TaskID": int(task.TaskID),
        "TaskType": task.TaskType,
        "ArrivalHour": int(task.ArrivalHour),
        "SourceRegion": task.SourceRegion,
        "GPU_Demand": float(task.GPU_Demand),
        "DurationHours": float(task.DurationHours),
        "LatestFinishHour": float(task.LatestFinishHour),
        "Status": status,
    }
    if cand is None:
        base.update(
            {
                "AssignedRegion": None,
                "StartHour": None,
                "FinishHour": None,
                "NetworkLatency_ms": None,
                "ObjectiveScore": None,
                "EstimatedCost_CNY": None,
                "EstimatedCarbon_tCO2": None,
                "EstimatedRenewableUsed_MWh": None,
            }
        )
    else:
        base.update(
            {
                "AssignedRegion": cand.region,
                "StartHour": cand.start,
                "FinishHour": cand.finish,
                "NetworkLatency_ms": cand.latency_ms,
                "ObjectiveScore": cand.score,
                "EstimatedCost_CNY": cand.cost_cny,
                "EstimatedCarbon_tCO2": cand.carbon_tco2,
                "EstimatedRenewableUsed_MWh": cand.renewable_mwh,
            }
        )
    return base


def run_rolling_schedule(
    tasks: pd.DataFrame,
    maps: Dict[str, object],
    solver: str,
    step_hours: int,
    lookahead_hours: int,
    weights: Weights,
    max_candidates: int,
    mip_time_limit_sec: int,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], float], Dict[Tuple[str, int], float], str]:
    try:
        import pulp  # noqa: F401
        pulp_available = True
    except ImportError:
        pulp_available = False

    chosen_solver = "pulp" if solver == "auto" and pulp_available else solver
    if chosen_solver == "auto":
        chosen_solver = "greedy"
    if chosen_solver == "pulp" and not pulp_available:
        raise RuntimeError("PuLP is not installed. Use --solver greedy or install pulp.")

    gpu_used: Dict[Tuple[str, int], float] = {}
    ai_load: Dict[Tuple[str, int], float] = {}
    rows: List[Dict[str, object]] = []

    min_hour = int(tasks["ArrivalHour"].min()) if not tasks.empty else 0
    max_hour = int(tasks["ArrivalHour"].max()) if not tasks.empty else 0
    for current in range(min_hour, max_hour + 1, step_hours):
        step_end = current + step_hours - 1
        window_tasks = tasks.loc[(tasks["ArrivalHour"] >= current) & (tasks["ArrivalHour"] <= step_end)].copy()
        if window_tasks.empty:
            continue
        print(f"[window {current}-{step_end}] tasks={len(window_tasks)} solver={chosen_solver}")
        if chosen_solver == "pulp":
            rows.extend(
                solve_window_pulp(
                    window_tasks,
                    current,
                    lookahead_hours,
                    maps,
                    gpu_used,
                    ai_load,
                    weights,
                    max_candidates,
                    mip_time_limit_sec,
                )
            )
        else:
            rows.extend(
                solve_window_greedy(
                    window_tasks,
                    current,
                    lookahead_hours,
                    maps,
                    gpu_used,
                    ai_load,
                    weights,
                    max_candidates,
                )
            )
    return pd.DataFrame(rows), gpu_used, ai_load, chosen_solver


def compute_hourly_metrics(
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    min_hour: int,
    max_hour: int,
) -> pd.DataFrame:
    gpu_idx: pd.DataFrame = maps["gpu"]  # type: ignore[assignment]
    rt: pd.DataFrame = maps["rt"]  # type: ignore[assignment]
    rows = []
    for hour in range(min_hour, min(max_hour + 1, TASK_END_EXCLUSIVE)):
        for region in REGIONS:
            nonai = existing_nonai_load(rt, region, hour)
            ai = ai_load.get((region, hour), 0.0)
            pue = float(gpu_idx.loc[region, "PUE"])
            total_load = (nonai + ai) * pue
            available_renewable = get_rt_value(rt, region, hour, "AvailableRenewable_MW", 0.0)
            used_renewable = min(total_load, available_renewable)
            grid_purchase = max(0.0, total_load - used_renewable)
            curtailment = max(0.0, available_renewable - used_renewable)
            price = get_rt_value(rt, region, hour, "ElectricityPrice_CNY_per_MWh", 0.0)
            carbon_intensity = get_rt_value(rt, region, hour, "CarbonIntensity_tCO2_per_MWh", 0.0)
            cost = grid_purchase * price
            carbon = grid_purchase * carbon_intensity
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
                    "IT_Load_MW": nonai + ai,
                    "Total_Load_MW": total_load,
                    "AvailableRenewable_MW": available_renewable,
                    "UsedRenewable_MW": used_renewable,
                    "GridPurchase_MW": grid_purchase,
                    "Curtailment_MW": curtailment,
                    "Cost_CNY": cost,
                    "CarbonEmission_tCO2": carbon,
                }
            )
    return pd.DataFrame(rows)


def compute_summary(schedule: pd.DataFrame, hourly: pd.DataFrame, solver_used: str, args: argparse.Namespace) -> Dict[str, object]:
    scheduled = schedule["Status"].eq("scheduled") if not schedule.empty else pd.Series(dtype=bool)
    total_available_renewable = float(hourly["AvailableRenewable_MW"].sum()) if not hourly.empty else 0.0
    total_used_renewable = float(hourly["UsedRenewable_MW"].sum()) if not hourly.empty else 0.0
    net_grid = hourly.groupby("Hour")["GridPurchase_MW"].sum() if not hourly.empty else pd.Series(dtype=float)
    return {
        "solver_used": solver_used,
        "arrival_hour_start": args.hour_start,
        "arrival_hour_end": args.hour_end,
        "step_hours": args.step_hours,
        "lookahead_hours": args.lookahead_hours,
        "max_candidates_per_task": args.max_candidates_per_task,
        "task_count": int(len(schedule)),
        "scheduled_count": int(scheduled.sum()) if not schedule.empty else 0,
        "unassigned_count": int((~scheduled).sum()) if not schedule.empty else 0,
        "on_time_rate": float(scheduled.mean()) if len(schedule) else 0.0,
        "total_cost_cny": float(hourly["Cost_CNY"].sum()) if not hourly.empty else 0.0,
        "total_carbon_tco2": float(hourly["CarbonEmission_tCO2"].sum()) if not hourly.empty else 0.0,
        "renewable_utilization": total_used_renewable / total_available_renewable if total_available_renewable else 0.0,
        "total_curtailment_mwh": float(hourly["Curtailment_MW"].sum()) if not hourly.empty else 0.0,
        "system_peak_grid_purchase_mw": float(net_grid.max()) if not net_grid.empty else 0.0,
        "average_latency_ms": float(schedule.loc[scheduled, "NetworkLatency_ms"].mean()) if scheduled.any() else None,
        "weights": {
            "cost": args.cost_weight,
            "carbon": args.carbon_weight,
            "latency": args.latency_weight,
            "renewable": args.renewable_weight,
            "congestion": args.congestion_weight,
            "unassigned": args.unassigned_penalty,
        },
    }


def main() -> None:
    args = parse_args()
    weights = Weights(
        cost=args.cost_weight,
        carbon=args.carbon_weight,
        latency=args.latency_weight,
        renewable=args.renewable_weight,
        congestion=args.congestion_weight,
        unassigned=args.unassigned_penalty,
    )

    workload, gpu, latency, power, region_time = load_inputs(args.data_dir)
    tasks = prepare_workload(workload, args.hour_start, args.hour_end, args.max_tasks)
    maps = build_maps(gpu, latency, power, region_time)

    if tasks.empty:
        raise ValueError("No tasks found after filtering.")

    schedule, gpu_used, ai_load, solver_used = run_rolling_schedule(
        tasks,
        maps,
        args.solver,
        args.step_hours,
        args.lookahead_hours,
        weights,
        args.max_candidates_per_task,
        args.mip_time_limit_sec,
    )
    min_metric_hour = int(min(args.hour_start, schedule["StartHour"].dropna().min() if schedule["StartHour"].notna().any() else args.hour_start))
    max_finish = schedule["FinishHour"].dropna().max() if schedule["FinishHour"].notna().any() else args.hour_end
    max_metric_hour = int(min(TASK_END_EXCLUSIVE - 1, math.ceil(max(max_finish, args.hour_end))))
    hourly = compute_hourly_metrics(maps, gpu_used, ai_load, min_metric_hour, max_metric_hour)
    summary = compute_summary(schedule, hourly, solver_used, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = args.output_dir / "problem2_task_schedule.csv"
    hourly_path = args.output_dir / "problem2_region_hour_metrics.csv"
    summary_path = args.output_dir / "problem2_summary.json"
    schedule.to_csv(schedule_path, index=False, encoding="utf-8-sig")
    hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {schedule_path}")
    print(f"saved: {hourly_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
