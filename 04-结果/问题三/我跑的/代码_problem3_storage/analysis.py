# -*- coding: utf-8 -*-
"""问题三结果可视化：SOC 曲线、净购电曲线、充放电策略与多场景对比图。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REGIONS, ROOT  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = ROOT / "C题写作" / "问题3结果_我跑的" / "结果"
FIG_DIR = ROOT / "C题写作" / "问题3结果_我跑的" / "图表与分析"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SC_LABEL = {
    "baseline": "附件基线",
    "no_storage": "无储能优化",
    "cost_min": "成本最优",
    "carbon_min": "低碳最优",
    "peak_min": "削峰最优",
    "flat_min": "平抑波动",
    "balanced": "综合权衡",
}


def read_hourly(sc: str) -> pd.DataFrame:
    return pd.read_csv(RESULT_DIR / f"hourly_{sc}.csv")


def plot_soc_curves() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    base = read_hourly("baseline")
    cost = read_hourly("cost_min")
    flat = read_hourly("flat_min")
    for ax, reg in zip(axes.ravel(), REGIONS):
        for df, lab, ls in [(base, "附件基线", "--"),
                            (cost, "成本最优", "-"),
                            (flat, "平抑波动", "-.")]:
            s = df[df["Region"] == reg]
            ax.plot(s["Hour"], s["SOC_MWh"], ls, lw=1.2, label=lab)
        ax.set_title(reg)
        ax.set_ylabel("SOC (MWh)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("小时")
    fig.suptitle("储能 SOC 曲线：基线 vs 优化策略（0–2406 小时）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "soc_curves.png", dpi=150)
    plt.close(fig)


def plot_net_import() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
    for ax, reg in zip(axes, ["RegionA", "RegionD", "RegionF"]):
        for sc, ls in [("baseline", "--"), ("no_storage", ":"),
                       ("cost_min", "-"), ("flat_min", "-.")]:
            df = read_hourly(sc)
            s = df[df["Region"] == reg]
            ax.plot(s["Hour"], s["NetGridImport_MW"], ls, lw=1.0,
                    label=SC_LABEL[sc])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{reg} 净购电功率")
        ax.set_ylabel("MW")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.set_xlabel("小时")
    fig.suptitle("净购电功率曲线（负值表示向外送电）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "net_import_curves.png", dpi=150)
    plt.close(fig)


def plot_charge_discharge() -> None:
    df = read_hourly("cost_min")
    reg = "RegionF"
    s = df[(df["Region"] == reg) & (df["Hour"] <= 720)]
    h = s["Hour"]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.fill_between(h, s["RenewableCharge_MW"], label="新能源充电", alpha=0.65)
    ax.fill_between(h, s["RenewableCharge_MW"],
                    s["RenewableCharge_MW"] + s["GridCharge_MW"],
                    label="电网充电", alpha=0.65)
    ax.fill_between(h, -s["Discharge_MW"], 0, label="储能放电", alpha=0.65,
                    color="tab:red")
    ax.plot(h, s["Load_MW"], color="k", lw=1.0, label="设施负荷")
    ax.set_title(f"{reg} 成本最优策略的储能充放电与负荷（前 721 小时）")
    ax.set_xlabel("小时")
    ax.set_ylabel("MW")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "charge_discharge.png", dpi=150)
    plt.close(fig)


def plot_comparison() -> None:
    summ = pd.read_csv(RESULT_DIR / "scenario_summary.csv")
    scenarios = ["baseline", "no_storage", "cost_min", "peak_min", "flat_min", "balanced"]
    labels = [SC_LABEL[s] for s in scenarios]
    metrics = [
        ("total_cost_cny", "运行成本（亿元）", lambda v: v / 1e8),
        ("total_carbon_tco2", "碳排放（tCO2）", lambda v: v),
        ("sys_peak_net_mw", "系统峰值净购电（MW）", lambda v: v),
        ("sys_net_mad_mw", "净购电波动 MAD（MW）", lambda v: v),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    colors = ["#888888", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for ax, (col, title, fmt) in zip(axes.ravel(), metrics):
        vals = [fmt(float(summ[summ["scenario"] == s][col].iloc[0])) for s in scenarios]
        bars = ax.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:,.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    fig.suptitle("各场景关键指标对比（全时域 0–2406）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "comparison_bars.png", dpi=150)
    plt.close(fig)


def plot_improvement() -> None:
    summ = pd.read_csv(RESULT_DIR / "scenario_summary.csv")
    base = summ[summ["scenario"] == "baseline"].iloc[0]
    scenarios = ["no_storage", "cost_min", "peak_min", "flat_min", "balanced"]
    labels = [SC_LABEL[s] for s in scenarios]
    rows = [
        ("total_cost_cny", "运行成本"),
        ("total_carbon_tco2", "碳排放"),
        ("sys_peak_net_mw", "峰值净购电"),
        ("sys_net_mad_mw", "负荷波动 MAD"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = 0.19
    x = np.arange(len(labels))
    for i, (col, name) in enumerate(rows):
        bv = float(base[col])
        vals = [(float(summ[summ["scenario"] == s][col].iloc[0]) - bv) / abs(bv) * 100
                for s in scenarios]
        ax.bar(x + (i - 1.5) * width, vals, width, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("相对基线变化（%）")
    ax.set_title("优化策略相对附件基线的指标变化（负值=改善）")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "improvement_vs_baseline.png", dpi=150)
    plt.close(fig)


def plot_tradeoff() -> None:
    summ = pd.read_csv(RESULT_DIR / "scenario_summary.csv")
    scenarios = ["baseline", "no_storage", "cost_min", "peak_min", "flat_min", "balanced"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (xc, yc, xt, yt) in zip(
            axes,
            [("sys_net_mad_mw", "total_cost_cny", "净购电波动 MAD（MW）", "运行成本（亿元）"),
             ("sys_peak_net_mw", "total_cost_cny", "峰值净购电（MW）", "运行成本（亿元）")]):
        xs = [float(summ[summ["scenario"] == s][xc].iloc[0]) for s in scenarios]
        ys = [float(summ[summ["scenario"] == s][yc].iloc[0]) / 1e8 for s in scenarios]
        for xi, yi, s in zip(xs, ys, scenarios):
            ax.scatter(xi, yi, s=70, label=SC_LABEL[s], zorder=3)
            ax.annotate(SC_LABEL[s], (xi, yi), textcoords="offset points",
                        xytext=(6, 5), fontsize=8)
        ax.set_xlabel(xt)
        ax.set_ylabel(yt)
        ax.grid(alpha=0.3)
    fig.suptitle("多目标权衡：成本 vs 波动 / 峰值", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "tradeoff_scatter.png", dpi=150)
    plt.close(fig)


def plot_capacity_sensitivity() -> None:
    df = pd.read_csv(RESULT_DIR / "sensitivity_capacity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    cols = [
        ("total_cost_cny", "运行成本（亿元）", lambda v: v / 1e8),
        ("sys_peak_net_mw", "系统峰值净购电（MW）", lambda v: v),
        ("sys_net_mad_mw", "净购电波动 MAD（MW）", lambda v: v),
    ]
    for ax, (col, title, fmt) in zip(axes, cols):
        for obj, ls, mk in [("cost", "-", "o"), ("balanced", "--", "s")]:
            s = df[df["objective"] == obj]
            ax.plot(s["capacity_mult"], [fmt(v) for v in s[col]], ls,
                    marker=mk, label=SC_LABEL["cost_min"] if obj == "cost" else SC_LABEL["balanced"])
        ax.set_title(title)
        ax.set_xlabel("储能容量系数（1=附件容量）")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("储能容量敏感性：成本 / 峰值 / 波动随容量变化", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "sensitivity_capacity.png", dpi=150)
    plt.close(fig)


def plot_efficiency_sensitivity() -> None:
    df = pd.read_csv(RESULT_DIR / "sensitivity_efficiency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (col, title, fmt) in zip(
            axes,
            [("sys_peak_net_mw", "系统峰值净购电（MW）", lambda v: v),
             ("sys_net_mad_mw", "净购电波动 MAD（MW）", lambda v: v)]):
        ax.plot(df["eta_delta"], [fmt(v) for v in df[col]], "-o")
        ax.set_title(title)
        ax.set_xlabel("充放电效率偏移 Δη")
        ax.grid(alpha=0.3)
    fig.suptitle("充放电效率敏感性（综合权衡策略）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "sensitivity_efficiency.png", dpi=150)
    plt.close(fig)


def plot_pareto_exact() -> None:
    df = pd.read_csv(RESULT_DIR / "pareto_exact.csv")
    gra = pd.read_csv(RESULT_DIR / "gra_selection.csv")
    df = df.drop_duplicates(subset="epsilon_frac", keep="last")
    gra = gra.drop_duplicates(subset="epsilon_frac", keep="last")
    best_ew = gra.loc[gra["grey_rank"].idxmin()]
    best_w = gra.loc[gra["grey_rank_w"].idxmin()]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    # 左：成本 vs MAD
    ax = axes[0]
    ax.plot(df["sys_net_mad_mw"], df["total_cost_cny"] / 1e8, "-o", color="#4C72B0",
            lw=1.8, ms=5, label="精确 Pareto 前沿（ε-约束）")
    ax.axhline(0, color="k", lw=0.7)
    for _, r in df.iterrows():
        ax.annotate(f"ε={r['epsilon_frac']:.2g}", (r["sys_net_mad_mw"], r["total_cost_cny"] / 1e8),
                    textcoords="offset points", xytext=(4, -2), fontsize=7, alpha=0.8)
    for b, mk, lab, c in [(best_ew, "D", "GRA等权最优", "#C44E52"),
                          (best_w, "s", "GRA加权最优", "#55A868")]:
        ax.scatter(b["sys_net_mad_mw"], b["total_cost_cny"] / 1e8, marker=mk, s=90,
                   color=c, zorder=4, label=lab)
    ax.set_xlabel("净购电波动 MAD（MW）")
    ax.set_ylabel("运行成本（亿元）")
    ax.set_title("成本—波动精确 Pareto 前沿")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    # 右：成本 vs 峰值
    ax = axes[1]
    ax.plot(-df["sys_peak_net_mw"], df["total_cost_cny"] / 1e8, "-o", color="#DD8452",
            lw=1.8, ms=5, label="精确 Pareto 前沿")
    for _, r in df.iterrows():
        ax.annotate(f"ε={r['epsilon_frac']:.2g}", (-r["sys_peak_net_mw"], r["total_cost_cny"] / 1e8),
                    textcoords="offset points", xytext=(4, -2), fontsize=7, alpha=0.8)
    ax.set_xlabel("外送深度（-峰值净购电，MW）")
    ax.set_ylabel("运行成本（亿元）")
    ax.set_title("成本—峰值净购电关系")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle("精确 Pareto 前沿与灰色关联决策（ε-约束法）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "pareto_front_exact.png", dpi=150)
    plt.close(fig)


def plot_nsga2_comparison() -> None:
    cmp = pd.read_csv(RESULT_DIR / "nsga2_comparison.csv")
    mine = cmp[cmp["method"] == "本文精确Pareto"]
    theirs = cmp[cmp["method"] == "同学NSGA-II"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    # 左：成本 vs 总变异波动
    ax = axes[0]
    ax.scatter(mine["sys_net_tv_mw"], mine["total_cost_cny"] / 1e8, s=55, color="#4C72B0",
               zorder=3, label="本文精确Pareto")
    ax.scatter(theirs["sys_net_tv_mw"], theirs["total_cost_cny"] / 1e8, s=40,
               color="#C44E52", marker="x", zorder=3, label="同学NSGA-II")
    ax.set_xlabel("净购电波动-总变异（MW）")
    ax.set_ylabel("运行成本（亿元）")
    ax.set_title("成本—波动：精确前沿 vs NSGA-II")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    # 右：成本 vs 峰值
    ax = axes[1]
    ax.scatter(-mine["sys_peak_net_mw"], mine["total_cost_cny"] / 1e8, s=55, color="#4C72B0",
               zorder=3, label="本文精确Pareto")
    ax.scatter(-theirs["sys_peak_net_mw"], theirs["total_cost_cny"] / 1e8, s=40,
               color="#C44E52", marker="x", zorder=3, label="同学NSGA-II")
    ax.set_xlabel("外送深度（-峰值净购电，MW）")
    ax.set_ylabel("运行成本（亿元）")
    ax.set_title("成本—峰值：精确前沿 vs NSGA-II")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle("精确 MILP 前沿与 NSGA-II 启发式前沿对比", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "nsga2_comparison.png", dpi=150)
    plt.close(fig)


def plot_gra_selection() -> None:
    gra = pd.read_csv(RESULT_DIR / "gra_selection.csv").drop_duplicates(
        subset="epsilon_frac", keep="last")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(gra))
    ax.plot(x, gra["grey_grade"], "-o", label="GRA 关联度（等权）")
    ax.plot(x, gra["grey_grade_w"], "-s", label="GRA 关联度（加权）")
    ax.plot(x, gra["topsis_score"], "-^", label="TOPSIS 贴近度（加权）")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.2g}" for e in gra["epsilon_frac"]], rotation=30, fontsize=8)
    ax.set_xlabel("波动上限比例 ε")
    ax.set_ylabel("关联度 / 贴近度")
    ax.set_title("精确前沿上的灰色关联与 TOPSIS 评价")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gra_selection.png", dpi=150)
    plt.close(fig)


def main() -> None:
    plot_soc_curves()
    plot_net_import()
    plot_charge_discharge()
    plot_comparison()
    plot_improvement()
    plot_tradeoff()
    plot_capacity_sensitivity()
    plot_efficiency_sensitivity()
    plot_pareto_exact()
    plot_nsga2_comparison()
    plot_gra_selection()
    print("图表已生成：", FIG_DIR)


if __name__ == "__main__":
    main()
