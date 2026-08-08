"""Improved rolling-window scheduler for Problem 2.

Design (best-effort, aims at zero unassigned tasks):
1. Real-time tasks are scheduled first over the whole horizon (fixed start,
   latency-constrained regions), so flexible tasks can never block them.
2. Flexible tasks are scheduled in rolling windows, ordered by deadline slack.
3. A repair phase re-tries every unassigned task with feasibility-first weights
   over its full legal start range.
4. If still blocked, a bounded displacement search moves one or two flexible
   blockers to alternative slots and then places the task.

Optional solver: scipy.optimize.milp (HiGHS) per rolling window, intended for
small horizons (e.g. the final 24 hours); falls back to greedy+repair when a
window cannot be solved within the time limit.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common import (
    REGIONS,
    TASK_END_EXCLUSIVE,
    Candidate,
    Weights,
    existing_nonai_load,
    get_rt_value,
    overlaps,
    schedule_row,
)

CLOSURE_START_PENALTY = 1e6  # soft penalty for starting in the 2400-2405 closure window


# ---------------------------------------------------------------- candidates

def candidate_score(
    task: pd.Series,
    region: str,
    start: int,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
) -> Optional[Candidate]:
    """Feasibility + marginal cost/carbon/renewable/congestion score of one slot."""
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
    if int(start) >= 2400:
        score += CLOSURE_START_PENALTY
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
    earliest: int,
    latest: int,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
    max_candidates: int = 0,
) -> List[Candidate]:
    """Enumerate all legal (region, start) slots; keep top-K by score if K > 0."""
    candidates: List[Candidate] = []
    for region in REGIONS:
        for start in range(earliest, latest + 1):
            cand = candidate_score(task, region, start, maps, gpu_used, ai_load, weights)
            if cand is not None:
                candidates.append(cand)
    candidates.sort(key=lambda c: c.score)
    if max_candidates and len(candidates) > max_candidates:
        return candidates[:max_candidates]
    return candidates


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


def remove_candidate(
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
        key = (cand.region, hour)
        gpu_used[key] = gpu_used.get(key, 0.0) - gpu_demand * frac
        ai_load[key] = ai_load.get(key, 0.0) - task_ai_mw_full * frac
        if gpu_used[key] < 1e-9:
            gpu_used.pop(key, None)
        if ai_load[key] < 1e-9:
            ai_load.pop(key, None)


# ------------------------------------------------------------ rolling greedy

def schedule_real_time_first(
    tasks: pd.DataFrame,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
) -> Dict[int, Dict[str, object]]:
    rows_by_id: Dict[int, Dict[str, object]] = {}
    rt = tasks[tasks["IsRealtime"]].sort_values(["ArrivalHour", "TaskID"])
    for _, task in rt.iterrows():
        cands = generate_candidates(
            task, int(task.ArrivalHour), int(task.ArrivalHour), maps, gpu_used, ai_load, weights, 0
        )
        if cands:
            best = cands[0]
            apply_candidate(task, best, maps, gpu_used, ai_load)
            rows_by_id[int(task.TaskID)] = schedule_row(task, best, "scheduled")
        else:
            rows_by_id[int(task.TaskID)] = schedule_row(task, None, "unassigned")
    return rows_by_id


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
    # flexible tasks only; real-time handled globally. Deadline-tightest first.
    ordered = window_tasks.sort_values(["LatestStartHour", "ArrivalHour", "TaskID"])
    for _, task in ordered.iterrows():
        earliest = max(int(task.ArrivalHour), current_hour)
        latest = int(min(task.LatestStartHour, current_hour + lookahead_hours - 1))
        if latest < earliest:
            rows.append(schedule_row(task, None, "unassigned"))
            continue
        cands = generate_candidates(
            task, earliest, latest, maps, gpu_used, ai_load, weights, max_candidates
        )
        if not cands:
            rows.append(schedule_row(task, None, "unassigned"))
            continue
        best = cands[0]
        apply_candidate(task, best, maps, gpu_used, ai_load)
        rows.append(schedule_row(task, best, "scheduled"))
    return rows


def solve_window_milp_scipy(
    window_tasks: pd.DataFrame,
    current_hour: int,
    lookahead_hours: int,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
    max_candidates: int,
    time_limit_sec: int,
) -> Optional[List[Dict[str, object]]]:
    """Time-indexed assignment MILP for one window using scipy/HiGHS."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        return None

    gpu_idx: pd.DataFrame = maps["gpu"]  # type: ignore[assignment]
    power_map: Dict[str, float] = maps["power"]  # type: ignore[assignment]
    rt: pd.DataFrame = maps["rt"]  # type: ignore[assignment]

    tasks_by_id = {int(row.TaskID): row for _, row in window_tasks.iterrows()}
    cands_by_task: Dict[int, List[Candidate]] = {}
    for _, task in window_tasks.iterrows():
        earliest = max(int(task.ArrivalHour), current_hour)
        latest = int(min(task.LatestStartHour, current_hour + lookahead_hours - 1))
        cands_by_task[int(task.TaskID)] = (
            generate_candidates(task, earliest, latest, maps, gpu_used, ai_load, weights, max_candidates)
            if latest >= earliest
            else []
        )

    var_index: Dict[Tuple[int, int], int] = {}
    n_vars = 0
    for tid, cands in cands_by_task.items():
        for k in range(len(cands)):
            var_index[(tid, k)] = n_vars
            n_vars += 1
        var_index[(tid, "u")] = n_vars  # unassigned indicator
        n_vars += 1

    c = np.zeros(n_vars)
    for tid, cands in cands_by_task.items():
        for k, cand in enumerate(cands):
            c[var_index[(tid, k)]] = cand.score
        task = tasks_by_id[tid]
        c[var_index[(tid, "u")]] = (
            weights.unassigned * float(task.GPU_Demand) * float(task.DurationHours)
        )

    rows_c: List[Tuple[int, int, float]] = []
    rhs_lo: List[float] = []
    rhs_hi: List[float] = []
    n_con = 0

    # one task -> exactly one candidate or unassigned
    for tid, cands in cands_by_task.items():
        for k in range(len(cands)):
            rows_c.append((n_con, var_index[(tid, k)], 1.0))
        rows_c.append((n_con, var_index[(tid, "u")], 1.0))
        rhs_lo.append(1.0)
        rhs_hi.append(1.0)
        n_con += 1

    affected_hours = range(current_hour, min(TASK_END_EXCLUSIVE, current_hour + lookahead_hours + 24))
    for region in REGIONS:
        for hour in affected_hours:
            gpu_terms: List[Tuple[int, float]] = []
            ai_terms: List[Tuple[int, float]] = []
            for tid, cands in cands_by_task.items():
                task = tasks_by_id[tid]
                gpu_demand = float(task.GPU_Demand)
                ai_full = gpu_demand * float(power_map[task.TaskType])
                for k, cand in enumerate(cands):
                    frac = next((ov for h, ov in cand.overlaps if h == hour and cand.region == region), 0.0)
                    if frac > 0:
                        vi = var_index[(tid, k)]
                        gpu_terms.append((vi, gpu_demand * frac))
                        ai_terms.append((vi, ai_full * frac))

            available_gpu = float(gpu_idx.loc[region, "Available_GPU"])
            for vi, coeff in gpu_terms:
                rows_c.append((n_con, vi, coeff))
            rhs_lo.append(0.0)
            rhs_hi.append(max(0.0, available_gpu - gpu_used.get((region, hour), 0.0)))
            n_con += 1

            nonai = existing_nonai_load(rt, region, hour)
            max_it = float(gpu_idx.loc[region, "Max_IT_Power_MW"])
            pue = float(gpu_idx.loc[region, "PUE"])
            max_facility = float(gpu_idx.loc[region, "Max_Facility_Power_MW"])
            base_ai = ai_load.get((region, hour), 0.0)
            for vi, coeff in ai_terms:
                rows_c.append((n_con, vi, coeff))
            rhs_lo.append(0.0)
            rhs_hi.append(max(0.0, max_it - nonai - base_ai))
            n_con += 1
            for vi, coeff in ai_terms:
                rows_c.append((n_con, vi, coeff * pue))
            rhs_lo.append(0.0)
            rhs_hi.append(max(0.0, max_facility - (nonai + base_ai) * pue))
            n_con += 1

    A = lil_matrix((n_con, n_vars))
    for (r, col, val) in rows_c:
        A[r, col] += val
    constraints = LinearConstraint(A.tocsr(), np.array(rhs_lo), np.array(rhs_hi))
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))
    integrality = np.ones(n_vars)

    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit_sec, "disp": False},
    )
    if res.x is None:
        return None

    rows: List[Dict[str, object]] = []
    for tid, cands in cands_by_task.items():
        task = tasks_by_id[tid]
        chosen: Optional[Candidate] = None
        for k, cand in enumerate(cands):
            if res.x[var_index[(tid, k)]] > 0.5:
                chosen = cand
                break
        if chosen is None:
            rows.append(schedule_row(task, None, "unassigned"))
        else:
            apply_candidate(task, chosen, maps, gpu_used, ai_load)
            rows.append(schedule_row(task, chosen, "scheduled"))
    return rows


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
    gpu_used: Dict[Tuple[str, int], float] = {}
    ai_load: Dict[Tuple[str, int], float] = {}
    chosen_solver = "greedy"

    # Phase 0: real-time tasks first over the whole horizon.
    rt_rows = schedule_real_time_first(tasks, maps, gpu_used, ai_load, weights)
    rows: List[Dict[str, object]] = list(rt_rows.values())

    # Phase 1: flexible tasks in rolling windows.
    flex = tasks[~tasks["IsRealtime"]]
    if not flex.empty:
        min_hour = int(flex["ArrivalHour"].min())
        max_hour = int(flex["ArrivalHour"].max())
        for current in range(min_hour, max_hour + 1, step_hours):
            step_end = current + step_hours - 1
            window_tasks = flex.loc[(flex["ArrivalHour"] >= current) & (flex["ArrivalHour"] <= step_end)].copy()
            if window_tasks.empty:
                continue
            if solver == "scipy-milp":
                window_rows = solve_window_milp_scipy(
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
                if window_rows is not None:
                    chosen_solver = "scipy-milp"
                    rows.extend(window_rows)
                    continue
                chosen_solver = "greedy"
            rows.extend(
                solve_window_greedy(
                    window_tasks, current, lookahead_hours, maps, gpu_used, ai_load, weights, max_candidates
                )
            )

    schedule_df = pd.DataFrame(rows)
    if schedule_df.empty:
        schedule_df = pd.DataFrame(columns=["TaskID", "TaskType", "ArrivalHour", "SourceRegion",
                                            "GPU_Demand", "DurationHours", "LatestFinishHour", "Status",
                                            "AssignedRegion", "StartHour", "FinishHour", "NetworkLatency_ms"])
    return schedule_df, gpu_used, ai_load, chosen_solver


# ------------------------------------------------------------------ repair

FEASIBILITY_WEIGHTS = Weights(
    cost=0.2, carbon=0.2, latency=1.0, renewable=0.0, congestion=0.2, unassigned=1e12
)


def _probe(
    task: pd.Series,
    region: str,
    start: int,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
) -> Tuple[bool, List[Tuple[int, str]]]:
    """Return (feasible, failing (hour, reason)) without allocating anything."""
    gpu_idx: pd.DataFrame = maps["gpu"]  # type: ignore[assignment]
    latency_map: Dict[Tuple[str, str], float] = maps["latency"]  # type: ignore[assignment]
    rt: pd.DataFrame = maps["rt"]  # type: ignore[assignment]
    latency_ms = latency_map.get((task.SourceRegion, region), math.inf)
    if latency_ms > float(task.MaxLatency_ms):
        return False, [(-1, "latency")]
    ov = overlaps(int(start), float(task.DurationHours))
    if not ov:
        return False, [(-1, "horizon")]
    fails: List[Tuple[int, str]] = []
    for hour, frac in ov:
        gpu_demand = float(task.GPU_Demand)
        if gpu_used.get((region, hour), 0.0) + gpu_demand * frac > float(gpu_idx.loc[region, "Available_GPU"]) + 1e-9:
            fails.append((hour, "gpu"))
        nonai = existing_nonai_load(rt, region, hour)
        ai_full = gpu_demand * float(maps["power"][task.TaskType])  # type: ignore[index]
        it_load = nonai + ai_load.get((region, hour), 0.0) + ai_full * frac
        if it_load > float(gpu_idx.loc[region, "Max_IT_Power_MW"]) + 1e-9:
            fails.append((hour, "it"))
        if it_load * float(gpu_idx.loc[region, "PUE"]) > float(gpu_idx.loc[region, "Max_Facility_Power_MW"]) + 1e-9:
            fails.append((hour, "facility"))
    return (not fails), fails


def _blockers_in_slots(
    failing_hours: List[Tuple[int, str]],
    region: str,
    rows_by_id: Dict[int, Dict[str, object]],
    max_blockers: int = 2,
) -> List[Dict[str, object]]:
    hours = {h for h, _ in failing_hours if h >= 0}
    blockers: List[Dict[str, object]] = []
    seen: set = set()
    for row in rows_by_id.values():
        if row["Status"] != "scheduled":
            continue
        if row["TaskType"] == "RealTimeInference":
            continue  # never displace real-time tasks
        if row["AssignedRegion"] != region:
            continue
        start = int(row["StartHour"])
        end = float(row["FinishHour"])
        if any(h + 1 > start and h < end for h in hours):
            tid = int(row["TaskID"])
            if tid not in seen:
                seen.add(tid)
                blockers.append(row)
                if len(blockers) >= max_blockers:
                    break
    return blockers


def _try_place(
    task: pd.Series,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    weights: Weights,
    latest_cap: Optional[int] = None,
) -> Optional[Candidate]:
    latest = int(task.LatestStartHour) if latest_cap is None else int(min(task.LatestStartHour, latest_cap))
    earliest = int(task.ArrivalHour)
    if latest < earliest:
        return None
    # Chunked search: near-arrival window first, then near-deadline window,
    # so tasks with huge slack stay bounded (<= ~144 starts x 6 regions).
    ranges = [(earliest, min(latest, earliest + 71))]
    if latest > earliest + 72:
        ranges.append((max(earliest, latest - 71), latest))
    best: Optional[Candidate] = None
    for lo, hi in ranges:
        cands = generate_candidates(task, lo, hi, maps, gpu_used, ai_load, weights, 0)
        if cands:
            cand = cands[0]
            if best is None or cand.score < best.score:
                best = cand
            break  # first chunk with a feasible slot is good enough for repair
    if best is None:
        return None
    apply_candidate(task, best, maps, gpu_used, ai_load)
    return best


def _try_displace(
    task: pd.Series,
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    rows_by_id: Dict[int, Dict[str, object]],
    tasks_by_id: Dict[int, pd.Series],
) -> bool:
    """Try to free a slot by moving at most two flexible blockers."""
    # candidate slots ordered by latency then start (feasibility-first)
    slots: List[Tuple[str, int]] = []
    for region in REGIONS:
        latency_map: Dict[Tuple[str, str], float] = maps["latency"]  # type: ignore[assignment]
        if latency_map.get((task.SourceRegion, region), math.inf) > float(task.MaxLatency_ms):
            continue
        for start in range(int(task.ArrivalHour), int(task.LatestStartHour) + 1):
            slots.append((region, start))
    slots.sort(key=lambda s: (float(maps["latency"][(task.SourceRegion, s[0])]), s[1]))  # type: ignore[index]

    for region, start in slots[:40]:
        ok, fails = _probe(task, region, start, maps, gpu_used, ai_load)
        if ok:
            best = _try_place(task, maps, gpu_used, ai_load, FEASIBILITY_WEIGHTS)
            if best is not None:
                rows_by_id[int(task.TaskID)] = schedule_row(task, best, "scheduled")
                return True
            continue
        blockers = _blockers_in_slots(fails, region, rows_by_id, max_blockers=2)
        if not blockers:
            continue
        for blocker in blockers[:2]:
            btid = int(blocker["TaskID"])
            blocker_task = tasks_by_id.get(btid)
            if blocker_task is None:
                continue
            old_start = int(blocker["StartHour"])
            old_region = blocker["AssignedRegion"]
            old_cand = Candidate(
                task_id=btid,
                region=old_region,
                start=old_start,
                finish=float(blocker["FinishHour"]),
                latency_ms=float(blocker["NetworkLatency_ms"]),
                score=0.0,
                cost_cny=0.0,
                carbon_tco2=0.0,
                renewable_mwh=0.0,
                congestion=0.0,
                overlaps=overlaps(old_start, float(blocker_task["DurationHours"])),
            )
            remove_candidate(blocker_task, old_cand, maps, gpu_used, ai_load)
            # alternative placement: any legal slot except the target (region,start)
            new_cand: Optional[Candidate] = None
            alt_cands = generate_candidates(
                blocker_task,
                int(blocker_task["ArrivalHour"]),
                int(blocker_task["LatestStartHour"]),
                maps,
                gpu_used,
                ai_load,
                FEASIBILITY_WEIGHTS,
                0,
            )
            for cand in alt_cands:
                if cand.region == region and cand.start == start:
                    continue
                new_cand = cand
                break
            if new_cand is None:
                # restore blocker
                apply_candidate(blocker_task, old_cand, maps, gpu_used, ai_load)
                continue
            apply_candidate(blocker_task, new_cand, maps, gpu_used, ai_load)
            rows_by_id[btid] = schedule_row(blocker_task, new_cand, "scheduled")
            # try to place the task now
            ok2, fails2 = _probe(task, region, start, maps, gpu_used, ai_load)
            if ok2:
                best = _try_place(task, maps, gpu_used, ai_load, FEASIBILITY_WEIGHTS)
                if best is not None:
                    rows_by_id[int(task.TaskID)] = schedule_row(task, best, "scheduled")
                    return True
            # undo blocker move and try next blocker
            remove_candidate(blocker_task, new_cand, maps, gpu_used, ai_load)
            apply_candidate(blocker_task, old_cand, maps, gpu_used, ai_load)
            rows_by_id[btid] = schedule_row(blocker_task, old_cand, "scheduled")
    return False


def repair_unassigned(
    schedule: pd.DataFrame,
    tasks_by_id: Dict[int, pd.Series],
    maps: Dict[str, object],
    gpu_used: Dict[Tuple[str, int], float],
    ai_load: Dict[Tuple[str, int], float],
    max_displace: int = 2000,
) -> Tuple[pd.DataFrame, int, int]:
    """Re-try unassigned tasks; optionally displace flexible blockers."""
    rows_by_id: Dict[int, Dict[str, object]] = {
        int(r["TaskID"]): dict(r) for _, r in schedule.iterrows()
    }
    unassigned_ids = [
        int(r["TaskID"]) for _, r in schedule.iterrows() if r["Status"] != "scheduled"
    ]
    unassigned_tasks = [
        tasks_by_id[tid] for tid in unassigned_ids if tid in tasks_by_id
    ]
    unassigned_tasks.sort(
        key=lambda t: (0 if t["IsRealtime"] else 1, int(t["LatestStartHour"]), int(t["ArrivalHour"]), int(t["TaskID"]))
    )
    placed_plain = 0
    placed_displace = 0
    remaining: List[pd.Series] = []
    for task in unassigned_tasks:
        best = _try_place(task, maps, gpu_used, ai_load, FEASIBILITY_WEIGHTS)
        if best is not None:
            rows_by_id[int(task.TaskID)] = schedule_row(task, best, "scheduled")
            placed_plain += 1
            continue
        if placed_displace < max_displace and _try_displace(
            task, maps, gpu_used, ai_load, rows_by_id, tasks_by_id
        ):
            placed_displace += 1
            continue
        remaining.append(task)

    rows = list(rows_by_id.values())
    rows.sort(key=lambda r: int(r["TaskID"]))
    return pd.DataFrame(rows), placed_plain, placed_displace


def run_improved_schedule(
    tasks: pd.DataFrame,
    maps: Dict[str, object],
    solver: str = "greedy",
    step_hours: int = 6,
    lookahead_hours: int = 24,
    weights: Optional[Weights] = None,
    max_candidates: int = 24,
    mip_time_limit_sec: int = 120,
    repair: bool = True,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], float], Dict[Tuple[str, int], float], str]:
    if weights is None:
        weights = Weights(cost=1.0, carbon=500.0, latency=2.0, renewable=100.0, congestion=100.0, unassigned=1e9)
    schedule, gpu_used, ai_load, solver_used = run_rolling_schedule(
        tasks, maps, solver, step_hours, lookahead_hours, weights, max_candidates, mip_time_limit_sec
    )
    placed_plain = 0
    placed_displace = 0
    if repair:
        tasks_by_id = {int(row.TaskID): row for _, row in tasks.iterrows()}
        schedule, placed_plain, placed_displace = repair_unassigned(
            schedule, tasks_by_id, maps, gpu_used, ai_load
        )
    schedule.attrs["repair"] = (placed_plain, placed_displace)
    return schedule, gpu_used, ai_load, solver_used
