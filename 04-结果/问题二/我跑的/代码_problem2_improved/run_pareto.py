"""Parallel Pareto runner for the improved Problem-2 pipeline.

Each weighted strategy is solved in its own process (identical to a serial
run), then the non-dominated front is extracted. Output layout:
    output_dir/all_strategy_summary.csv
    output_dir/pareto_front.csv
    output_dir/strategy_notes.json
    output_dir/detail_<strategy>/problem2_task_schedule.csv
    output_dir/detail_<strategy>/problem2_region_hour_metrics.csv
    output_dir/detail_<strategy>/problem2_summary.json
    output_dir/detail_<strategy>/baseline_region_hour_metrics.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd

from common import REGIONS, Weights
from metrics import baseline_metrics, compute_hourly_metrics, compute_summary, verify_schedule
from scheduler import run_improved_schedule


def strategy_weights() -> List[tuple]:
    """Named multi-objective weight settings (same spirit as teammate's set)."""
    W = Weights
    return [
        ("balanced", W(cost=1.0, carbon=500.0, latency=2.0, renewable=100.0, congestion=100.0, unassigned=1e9),
         "综合权衡：成本、碳排、时延、新能源和拥塞均衡考虑。"),
        ("cost_first", W(cost=3.0, carbon=150.0, latency=1.0, renewable=30.0, congestion=100.0, unassigned=1e9),
         "成本优先：更倾向低电价时段和区域。"),
        ("carbon_first", W(cost=0.7, carbon=1600.0, latency=1.0, renewable=80.0, congestion=100.0, unassigned=1e9),
         "低碳优先：更倾向低碳强度区域和时段。"),
        ("renewable_first", W(cost=0.6, carbon=500.0, latency=1.0, renewable=450.0, congestion=80.0, unassigned=1e9),
         "新能源消纳优先：更倾向新能源富余区域和时段。"),
        ("latency_first", W(cost=0.6, carbon=250.0, latency=12.0, renewable=30.0, congestion=80.0, unassigned=1e9),
         "服务质量优先：更倾向低网络时延、本地或近区执行。"),
        ("carbon_renewable", W(cost=0.6, carbon=1200.0, latency=1.5, renewable=350.0, congestion=80.0, unassigned=1e9),
         "低碳与新能源协同：同时重视低碳强度和新能源消纳。"),
        ("cost_carbon", W(cost=2.0, carbon=1000.0, latency=1.0, renewable=60.0, congestion=120.0, unassigned=1e9),
         "成本与碳排协同：适合比较经济性和减排性。"),
        ("load_balanced", W(cost=0.7, carbon=450.0, latency=1.5, renewable=80.0, congestion=600.0, unassigned=1e9),
         "负载均衡优先：更强惩罚接近容量上限的候选方案。"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solver", choices=["greedy", "scipy-milp"], default="greedy")
    parser.add_argument("--hour-start", type=int, default=0)
    parser.add_argument("--hour-end", type=int, default=2399)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--lookahead-hours", type=int, default=24)
    parser.add_argument("--max-candidates-per-task", type=int, default=24)
    parser.add_argument("--mip-time-limit-sec", type=int, default=120)
    parser.add_argument("--max-strategies", type=int, default=None)
    parser.add_argument("--only-strategy", type=str, default=None,
                        help="Run only this strategy by name (overrides max-strategies).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--save-detail", choices=["all", "none"], default="all")
    return parser.parse_args()


def _metric_window(args, schedule: pd.DataFrame) -> tuple:
    min_hour = int(args.hour_start)
    if not schedule.empty and schedule["StartHour"].notna().any():
        min_hour = int(min(min_hour, schedule["StartHour"].dropna().min()))
    max_finish = args.hour_end
    if not schedule.empty and schedule["FinishHour"].notna().any():
        max_finish = max(max_finish, float(schedule["FinishHour"].dropna().max()))
    max_hour = int(min(2405, math.ceil(max_finish)))
    return min_hour, max_hour


def worker_strategy(payload: dict) -> dict:
    name = payload["name"]
    note = payload["note"]
    weights = Weights(**payload["weights"])
    args = SimpleNamespace(**payload["args"])
    try:
        import importlib.util
        from common import load_inputs, prepare_workload, build_maps

        # modules are imported relative to this file; add its dir to sys.path
        here = Path(__file__).resolve().parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        from common import load_inputs as _li
        workload, gpu, latency, power, region_time, storage = _li(Path(args.data_dir))
        tasks = prepare_workload(workload, args.hour_start, args.hour_end, args.max_tasks)
        maps = build_maps(gpu, latency, power, region_time, storage)

        schedule, gpu_used, ai_load, solver_used = run_improved_schedule(
            tasks,
            maps,
            solver=args.solver,
            step_hours=args.step_hours,
            lookahead_hours=args.lookahead_hours,
            weights=weights,
            max_candidates=args.max_candidates_per_task,
            mip_time_limit_sec=args.mip_time_limit_sec,
            repair=not args.no_repair,
        )
        min_hour, max_hour = _metric_window(args, schedule)
        hourly = compute_hourly_metrics(maps, gpu_used, ai_load, min_hour, max_hour)
        _, baseline = baseline_metrics(maps, min_hour, max_hour)
        verification = verify_schedule(schedule, maps)
        summary = compute_summary(
            schedule, hourly, solver_used, args, weights, baseline=baseline, verification=verification
        )
        summary["strategy"] = name
        summary["strategy_note"] = note
        repair_counts = getattr(schedule, "attrs", {}).get("repair", (0, 0))
        summary["repair_plain"] = int(repair_counts[0])
        summary["repair_displace"] = int(repair_counts[1])
        summary["solver_used"] = solver_used

        if payload["save_detail"] == "all":
            sdir = Path(payload["output_dir"]) / f"detail_{name}"
            sdir.mkdir(parents=True, exist_ok=True)
            schedule.to_csv(sdir / "problem2_task_schedule.csv", index=False, encoding="utf-8-sig")
            hourly.to_csv(sdir / "problem2_region_hour_metrics.csv", index=False, encoding="utf-8-sig")
            (sdir / "problem2_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            bdf = baseline_metrics(maps, min_hour, max_hour)[0]
            bdf.to_csv(sdir / "baseline_region_hour_metrics.csv", index=False, encoding="utf-8-sig")
        return {"name": name, "note": note, "ok": True, "summary": summary}
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"name": name, "ok": False, "error": repr(exc), "traceback": traceback.format_exc()}


def pareto_front(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated rows. Minimize cost/carbon/latency/peak/unassigned/curtailment,
    maximize utilization and on-time rate."""
    df = summary_df.copy()
    df["_avg_latency"] = df["average_latency_ms"].fillna(1e12)
    df["_neg_util"] = -df["renewable_utilization"]
    df["_neg_ontime"] = -df["on_time_rate"]
    objectives = [
        "unassigned_count",
        "total_cost_cny",
        "total_carbon_tco2",
        "_avg_latency",
        "system_peak_net_import_mw",
        "total_curtailment_mwh",
        "_neg_util",
        "_neg_ontime",
    ]
    values = df[objectives].to_numpy(dtype=float)
    dominated = [False] * len(df)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            no_worse = (values[j] <= values[i] + 1e-9).all()
            strictly_better = (values[j] < values[i] - 1e-9).any()
            if no_worse and strictly_better:
                dominated[i] = True
                break
    front = df.loc[[not x for x in dominated]].drop(
        columns=["_avg_latency", "_neg_util", "_neg_ontime"]
    )
    return front.sort_values(["unassigned_count", "total_cost_cny", "total_carbon_tco2"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strategies = strategy_weights()
    if args.only_strategy is not None:
        strategies = [s for s in strategies if s[0] == args.only_strategy]
        if not strategies:
            raise SystemExit(f"Unknown strategy: {args.only_strategy}")
    if args.max_strategies is not None:
        strategies = strategies[: args.max_strategies]

    base_args = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "solver": args.solver,
        "hour_start": args.hour_start,
        "hour_end": args.hour_end,
        "max_tasks": args.max_tasks,
        "step_hours": args.step_hours,
        "lookahead_hours": args.lookahead_hours,
        "max_candidates_per_task": args.max_candidates_per_task,
        "mip_time_limit_sec": args.mip_time_limit_sec,
        "no_repair": args.no_repair,
    }
    payloads = []
    for name, weights, note in strategies:
        payloads.append(
            {
                "name": name,
                "note": note,
                "weights": {
                    "cost": weights.cost,
                    "carbon": weights.carbon,
                    "latency": weights.latency,
                    "renewable": weights.renewable,
                    "congestion": weights.congestion,
                    "unassigned": weights.unassigned,
                },
                "args": base_args,
                "save_detail": args.save_detail,
                "output_dir": str(args.output_dir),
            }
        )

    n_workers = min(args.workers, len(payloads))
    print(f"workers={n_workers} strategies={len(payloads)} horizon=[{args.hour_start},{args.hour_end}] "
          f"solver={args.solver} repair={not args.no_repair}")
    results: List[dict] = []
    with mp.Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(worker_strategy, payloads):
            results.append(res)
            if res["ok"]:
                print(f"[done] {res['name']}")
            else:
                print(f"[FAILED] {res['name']}: {res['error']}")

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    if not ok:
        for r in failed:
            print(r["traceback"])
        raise SystemExit("all strategies failed")
    if failed:
        print("WARNING failed strategies:", [r["name"] for r in failed])

    by_name = {r["name"]: r["summary"] for r in ok}
    all_rows = [by_name[n] for n, _, _ in strategies if n in by_name]
    summary_df = pd.DataFrame(all_rows)
    front_df = pareto_front(summary_df)

    summary_path = args.output_dir / "all_strategy_summary.csv"
    front_path = args.output_dir / "pareto_front.csv"
    notes_path = args.output_dir / "strategy_notes.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    front_df.to_csv(front_path, index=False, encoding="utf-8-sig")
    strategy_notes = [
        {"strategy": r["name"], "note": r["note"], "weights": r["summary"]["weights"]} for r in ok
    ]
    notes_path.write_text(json.dumps(strategy_notes, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPareto front:")
    display_cols = [
        "strategy", "task_count", "unassigned_count", "total_cost_cny", "total_carbon_tco2",
        "renewable_utilization", "average_latency_ms", "system_peak_net_import_mw",
        "system_peak_grid_purchase_mw",
    ]
    print(front_df[display_cols].to_string(index=False))
    print(f"\nsaved: {summary_path}")
    print(f"saved: {front_path}")
    print(f"saved: {notes_path}")
    if args.save_detail == "all":
        print("saved detail folders: " + ", ".join("detail_" + r["name"] for r in ok))


if __name__ == "__main__":
    main()
