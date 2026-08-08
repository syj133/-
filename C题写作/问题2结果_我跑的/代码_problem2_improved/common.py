"""Shared data loading and helpers for the improved Problem-2 pipeline.

This module keeps the exact unified conventions of 附件1:
- GPU capacity is enforced hourly as GPU-h (fractional overlaps allowed).
- AI IT load = sum(GPU_Demand * Overlap * GPU_Power(TaskType)).
- Facility load = IT load * PUE; both IT and facility caps are enforced.
- Tasks are non-preemptive and must finish before hour 2406.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
TASK_END_EXCLUSIVE = 2406  # tasks must finish before this hour; hour 2406 is settlement only


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


def find_file(data_dir: Path, prefix: str, suffix: str = ".xlsx") -> Path:
    matches = sorted(data_dir.glob(f"{prefix}*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {prefix}*{suffix} under {data_dir}")
    return matches[0]


def load_inputs(data_dir: Path):
    """Load the six data tables. Sheet 0 is used everywhere for robustness."""
    workload = pd.read_excel(find_file(data_dir, "workload_trace"), sheet_name=0)
    gpu = pd.read_excel(find_file(data_dir, "GPU_information"), sheet_name=0)
    latency = pd.read_excel(find_file(data_dir, "network_latency"), sheet_name=0)
    power = pd.read_excel(find_file(data_dir, "power_mapping"), sheet_name=0)
    region_time = pd.read_excel(find_file(data_dir, "region_time_data"), sheet_name=0)
    storage = pd.read_excel(find_file(data_dir, "storage_information"), sheet_name=0)
    return workload, gpu, latency, power, region_time, storage


def prepare_workload(
    workload: pd.DataFrame,
    hour_start: int,
    hour_end: int,
    max_tasks: Optional[int] = None,
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
    df = workload.loc[
        (workload["ArrivalHour"] >= hour_start) & (workload["ArrivalHour"] <= hour_end), cols
    ].copy()
    df = df.sort_values(["ArrivalHour", "TaskID"]).reset_index(drop=True)
    if max_tasks is not None:
        df = df.head(max_tasks).copy()
    df["DurationHours"] = df["EstimatedDuration_min"] / 60.0
    df["DiscreteDuration"] = np.ceil(df["DurationHours"]).astype(int)
    df["LatestStartHour"] = np.floor(
        np.minimum(df["LatestFinishHour"], TASK_END_EXCLUSIVE) - df["DurationHours"]
    ).astype(int)
    df["IsRealtime"] = df["TaskType"].eq("RealTimeInference") | df["DelaySensitivity"].astype(str).str.lower().eq("high")
    return df


def build_maps(
    gpu: pd.DataFrame,
    latency: pd.DataFrame,
    power: pd.DataFrame,
    region_time: pd.DataFrame,
    storage: pd.DataFrame,
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
    export_limits = storage.set_index("Region")["MaxGridExport_MW"].to_dict()
    return {
        "gpu": gpu_idx,
        "latency": latency_map,
        "power": power_map,
        "rt": rt,
        "export_limits": export_limits,
    }


def overlaps(start: int, duration_hours: float) -> List[Tuple[int, float]]:
    """Fractional overlap of a task with each integer hour, capped at hour 2405."""
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


def schedule_row(task: pd.Series, cand: Optional[Candidate], status: str) -> Dict[str, object]:
    base = {
        "TaskID": int(task.TaskID),
        "TaskType": task.TaskType,
        "ArrivalHour": int(task.ArrivalHour),
        "SourceRegion": task.SourceRegion,
        "MaxLatency_ms": float(task.MaxLatency_ms),
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
