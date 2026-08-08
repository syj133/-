"""Analysis and charts for the improved Problem-2 Pareto results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import REGIONS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, required=True, help="full-horizon result dir")
    parser.add_argument("--last24-dir", type=Path, default=None, help="last-24h result dir (optional)")
    parser.add_argument("--out-dir", type=Path, default=Path("charts"))
    parser.add_argument("--strategies", nargs="*", default=None, help="strategies for detail charts")
    return parser.parse_args()


def load_summary(result_dir: Path) -> pd.DataFrame:
    return pd.read_csv(result_dir / "all_strategy_summary.csv")


def pareto_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    pairs = [
        ("total_cost_cny", "total_carbon_tco2", "成本 (元)", "碳排 (tCO2)"),
        ("total_cost_cny", "average_latency_ms", "成本 (元)", "平均时延 (ms)"),
        ("total_carbon_tco2", "average_latency_ms", "碳排 (tCO2)", "平均时延 (ms)"),
        ("renewable_utilization", "total_cost_cny", "新能源利用率", "成本 (元)"),
    ]
    for ax, (x, y, xlab, ylab) in zip(axes.flat, pairs):
        for _, row in df.iterrows():
            ax.scatter(row[x], row[y], s=90, alpha=0.85)
            ax.annotate(row["strategy"], (row[x], row[y]), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    fig.suptitle("全时域 Problem 2 策略帕累托权衡", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_scatter.png", dpi=150)
    plt.close(fig)


def baseline_comparison(df: pd.DataFrame, baseline: Dict[str, float], out_dir: Path) -> None:
    metrics = [
        ("total_cost_cny", "baseline_total_cost_cny", "电费成本"),
        ("total_carbon_tco2", "baseline_total_carbon_tco2", "碳排放"),
        ("total_curtailment_mwh", "baseline_total_curtailment_mwh", "弃电"),
        ("system_peak_net_import_mw", "baseline_system_peak_net_import_mw", "系统峰值净购电"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, bkey, label) in zip(axes.flat, metrics):
        base_val = float(baseline.get(bkey, 0.0) or 0.0)
        values = df[key].astype(float)
        rel = values / base_val * 100 if base_val else values
        bars = ax.bar(df["strategy"], rel, color="#4472C4")
        ax.axhline(100, color="red", linestyle="--", linewidth=1.2, label="基线=100%")
        ax.set_ylabel(f"{label} (相对基线 %)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    fig.suptitle("各策略 vs 附件基线（相对 100%）", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_comparison.png", dpi=150)
    plt.close(fig)


def migration_stats(result_dir: Path, strategies: List[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    for name in strategies:
        path = result_dir / f"detail_{name}" / "problem2_task_schedule.csv"
        if not path.exists():
            continue
        sched = pd.read_csv(path)
        sched = sched[sched["Status"] == "scheduled"]
        total = len(sched)
        if total == 0:
            continue
        cross = sched["SourceRegion"] != sched["AssignedRegion"]
        to_west = sched["AssignedRegion"].isin(["RegionD", "RegionE", "RegionF"])
        by_type = sched.groupby("TaskType").apply(
            lambda g: pd.Series(
                {
                    "count": len(g),
                    "migrated": int((g["SourceRegion"] != g["AssignedRegion"]).sum()),
                    "to_west": int(g["AssignedRegion"].isin(["RegionD", "RegionE", "RegionF"]).sum()),
                }
            ),
            include_groups=False,
        )
        rows.append(
            {
                "strategy": name,
                "scheduled": total,
                "migrated_count": int(cross.sum()),
                "migrated_pct": float(cross.mean()),
                "to_west_count": int(to_west.sum()),
                "to_west_pct": float(to_west.mean()),
                "avg_latency_ms": float(sched["NetworkLatency_ms"].mean()),
                "avg_latency_rt_ms": float(sched.loc[sched["TaskType"] == "RealTimeInference", "NetworkLatency_ms"].mean()),
                "avg_latency_batch_ms": float(sched.loc[sched["TaskType"] == "BatchInference", "NetworkLatency_ms"].mean()),
                "avg_latency_train_ms": float(sched.loc[sched["TaskType"] == "AITraining", "NetworkLatency_ms"].mean()),
                "training_to_west_pct": float(
                    sched.loc[sched["TaskType"] == "AITraining", "AssignedRegion"]
                    .isin(["RegionD", "RegionE", "RegionF"])
                    .mean()
                ),
            }
        )
        # detailed crosstab
        ct = pd.crosstab(sched["SourceRegion"], sched["AssignedRegion"])
        ct = ct.reindex(index=REGIONS, columns=REGIONS, fill_value=0)
        ct.to_csv(out_dir / f"migration_matrix_{name}.csv", encoding="utf-8-sig")
    stat = pd.DataFrame(rows)
    if not stat.empty:
        stat.to_csv(out_dir / "migration_summary.csv", index=False, encoding="utf-8-sig")
    return stat


def load_curves(result_dir: Path, strategies: List[str], out_dir: Path) -> None:
    for name in strategies[:3]:
        path = result_dir / f"detail_{name}" / "problem2_region_hour_metrics.csv"
        if not path.exists():
            continue
        hourly = pd.read_csv(path)
        daily = hourly.groupby(["Region", hourly["Hour"] // 24])[["GPU_Utilization", "AI_IT_Load_MW", "Total_Load_MW"]].mean().reset_index()
        fig, axes = plt.subplots(3, 1, figsize=(13, 12))
        for ax, col, lab in zip(
            axes,
            ["GPU_Utilization", "AI_IT_Load_MW", "Total_Load_MW"],
            ["GPU 利用率（日均）", "AI IT 负荷（日均, MW）", "设施总负荷（日均, MW）"],
        ):
            for region in REGIONS:
                sub = daily[daily["Region"] == region]
                ax.plot(sub["Hour"], sub[col], label=region, linewidth=1.2)
            ax.set_ylabel(lab)
            ax.set_xlabel("运行日（第0天起）")
            ax.grid(alpha=0.3)
            ax.legend(ncol=3, fontsize=8)
        fig.suptitle(f"策略 {name}：逐日区域曲线", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / f"load_curves_{name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = load_summary(args.full_dir)
    baseline_path = args.full_dir / "detail_balanced" / "problem2_summary.json"
    baseline: Dict[str, float] = {}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8")).get("baseline", {})

    pareto_scatter(full, args.out_dir)
    if baseline:
        baseline_comparison(full, baseline, args.out_dir)

    strategies = args.strategies or list(full["strategy"])
    stats = migration_stats(args.full_dir, strategies, args.out_dir)
    if not stats.empty:
        print(stats.to_string(index=False))
        stats.to_csv(args.out_dir / "migration_summary.csv", index=False, encoding="utf-8-sig")

    load_curves(args.full_dir, strategies, args.out_dir)

    if args.last24_dir is not None:
        last24 = load_summary(args.last24_dir)
        last24.to_csv(args.out_dir / "last24_all_strategy_summary.csv", index=False, encoding="utf-8-sig")
        front24 = pd.read_csv(args.last24_dir / "pareto_front.csv")
        front24.to_csv(args.out_dir / "last24_pareto_front.csv", index=False, encoding="utf-8-sig")

    print("charts saved to:", args.out_dir)
    for p in sorted(args.out_dir.glob("*.png")):
        print(p)
    for p in sorted(args.out_dir.glob("*.csv")):
        print(p)


if __name__ == "__main__":
    main()
