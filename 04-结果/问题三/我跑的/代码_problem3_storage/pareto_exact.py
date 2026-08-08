# -*- coding: utf-8 -*-
"""结合方案：精确 Pareto 前沿（ε-约束法）+ 灰色关联/TOPSIS 决策 + NSGA-II 对照。

思路：
1. 用精确 MILP（HiGHS）对“成本 vs 波动”做 ε-约束扫描：逐级收紧净购电波动
   （Σ|net-mean|）上限、最小化运行成本，得到精确 Pareto 前沿（每个点全局最优），
   替代同学的 NSGA-II 近似前沿；
2. 引入灰色关联分析（GRA）与 TOPSIS，从精确前沿中选取综合最优解
   （同学方案的决策层，此处改用加权口径）；
3. 读取同学 NSGA-II 结果作对照，输出“精确前沿 vs 启发式前沿”对比数据。

输出：results/pareto_exact.csv、results/gra_selection.csv、
      results/pareto_gra_best.json、results/hourly_gra_best.csv、
      results/nsga2_reference/（同学结果副本）、results/nsga2_comparison.csv
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REGIONS, ROOT, build_params, load_baseline_series  # noqa: E402
from sensitivity import aggregate, make_ref  # noqa: E402
from storage_lp import check_constraints, solve_region  # noqa: E402

RESULT_DIR = ROOT / "C题写作" / "问题3结果_我跑的" / "结果"
CLASSMATE_DIR = Path.home() / "AppData" / "Local" / "Temp" / "huashu_classmate" \
    / "数学建模c题" / "problem3_nsga_gra"
# 波动上限比例：1.0 = 成本最优时的波动，0 = 完全平抑（逐级收紧）
EPS_FRACTIONS = [1.0, 0.75, 0.5, 0.3, 0.15, 0.07, 0.03, 0.01, 0.001, 0.0]
GRA_WEIGHTS = {"cost": 0.35, "carbon": 0.30, "peak": 0.15, "mad": 0.20}


def aggregate_from_hourly(df: pd.DataFrame) -> dict:
    net = df.groupby("Hour")["NetGridImport_MW"].sum().sort_index()
    return {
        "total_cost_cny": float((df["GridPurchase_MW"] * df["ElectricityPrice"]
                                 - df["GridSell_MW"] * df["SellPrice"]).sum()),
        "total_carbon_tco2": float((df["GridPurchase_MW"] * df["CarbonIntensity"]).sum()),
        "sys_peak_net_mw": float(net.max()),
        "sys_net_mad_mw": float(np.abs(net - net.mean()).mean()),
        "sys_net_std_mw": float(net.std()),
        "sys_net_tv_mw": float(np.abs(net.diff().dropna()).sum()),
        "total_renewable_util_pct": float(
            ((df["DirectRenewable_MW"] + df["RenewableCharge_MW"] + df["GridSell_MW"]).sum()
             / df["AvailableRenewable_MW"].sum()) * 100),
        "total_curtail_mwh": float(df["Curtailment_MW"].sum()),
        "total_charge_mwh": float((df["RenewableCharge_MW"] + df["GridCharge_MW"]).sum()),
        "total_discharge_mwh": float(df["Discharge_MW"].sum()),
        "total_purchase_mwh": float(df["GridPurchase_MW"].sum()),
        "total_export_mwh": float(df["GridSell_MW"].sum()),
    }


def region_dev_sum(df: pd.DataFrame, region: str) -> float:
    s = df[df["Region"] == region]
    net = s["NetGridImport_MW"].values
    return float(np.abs(net - net.mean()).sum())


def run_exact_pareto(params: dict) -> pd.DataFrame:
    costmin = pd.read_csv(RESULT_DIR / "hourly_cost_min.csv")
    dev0 = {reg: region_dev_sum(costmin, reg) for reg in REGIONS}
    rows = []
    fallback_used = []
    for frac in EPS_FRACTIONS:
        outs = {}
        for reg in REGIONS:
            # 成本目标下 LP 松弛即为整数可行（同时充放电不会出现在最优解），
            # 先快速解 LP，再逐点校验；若出现同时充放电则回退到 MILP。
            out = solve_region(params[reg], "cost", True,
                               flat_cap=frac * dev0[reg], enforce_exclusive=False)
            ck = check_constraints(params[reg], out)
            if ck["simultaneous_ch_dis"] > 1e-6:
                out = solve_region(params[reg], "cost", True,
                                   flat_cap=frac * dev0[reg], enforce_exclusive=True)
                fallback_used.append((frac, reg))
            outs[reg] = out
        agg = {k: float(v) for k, v in aggregate(params, outs).items()}
        agg["sys_net_tv_mw"] = aggregate_from_hourly(
            _outs_to_df(params, outs))["sys_net_tv_mw"]
        rows.append({"epsilon_frac": frac, **agg})
        print(f"epsilon={frac:.3f} cost={agg['total_cost_cny']/1e8:,.2f}亿元 "
              f"peak={agg['sys_peak_net_mw']:.1f} MAD={agg['sys_net_mad_mw']:.4f}")
    if fallback_used:
        print("回退 MILP 的点：", fallback_used)
    # 追加字典序“平抑波动”端点（MAD 严格为 0，成本二级最优）
    agg = aggregate_from_hourly(pd.read_csv(RESULT_DIR / "hourly_flat_min.csv"))
    rows.append({"epsilon_frac": 0.0, **agg})
    return pd.DataFrame(rows)


def _outs_to_df(params: dict, outs: dict) -> pd.DataFrame:
    frames = []
    for reg in REGIONS:
        p = params[reg]
        out = outs[reg]
        frames.append(pd.DataFrame({
            "Hour": p["hours"], "Region": reg,
            "GridPurchase_MW": out["gp"], "GridSell_MW": out["gs"],
            "DirectRenewable_MW": out["d"], "RenewableCharge_MW": out["rc"],
            "GridCharge_MW": out["gc"], "Curtailment_MW": out["c"],
            "Discharge_MW": out["pdis"], "SOC_MWh": out["soc"],
            "ElectricityPrice": p["price"], "SellPrice": p["sell_price"],
            "CarbonIntensity": p["carbon_intensity"],
            "AvailableRenewable_MW": p["renewable"],
        }))
    df = pd.concat(frames, ignore_index=True)
    df["NetGridImport_MW"] = df["GridPurchase_MW"] - df["GridSell_MW"]
    return df


def grey_relational_analysis(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """GRA：以相对基线改善率为效益型指标，等权与加权口径同时输出。"""
    crit = {
        "total_cost_cny": baseline["total_cost_cny"],
        "total_carbon_tco2": baseline["total_carbon_tco2"],
        "sys_peak_net_mw": baseline["sys_peak_net_mw"],
        "sys_net_mad_mw": baseline["sys_net_mad_mw"],
    }
    out = df.copy()
    for col, base in crit.items():
        out[f"{col}_reduction"] = (base - out[col]) / max(abs(base), 1e-9)
    red_cols = [f"{c}_reduction" for c in crit]
    X = out[red_cols].to_numpy()
    x0 = X.max(axis=0)
    d = np.abs(x0[None, :] - X)
    dmin, dmax = d.min(), d.max()
    rho = 0.5
    gamma = (dmin + rho * dmax) / (d + rho * dmax)
    out["grey_grade"] = gamma.mean(axis=1)
    out["grey_rank"] = out["grey_grade"].rank(ascending=False, method="min").astype(int)
    w = np.array([GRA_WEIGHTS["cost"], GRA_WEIGHTS["carbon"],
                  GRA_WEIGHTS["peak"], GRA_WEIGHTS["mad"]])
    out["grey_grade_w"] = (gamma * w).sum(axis=1) / w.sum()
    out["grey_rank_w"] = out["grey_grade_w"].rank(ascending=False, method="min").astype(int)

    norm = X / np.sqrt((X ** 2).sum(axis=0))
    ideal, anti = norm.max(axis=0), norm.min(axis=0)
    sd = np.sqrt((w * (norm - ideal) ** 2).sum(axis=1))
    sda = np.sqrt((w * (norm - anti) ** 2).sum(axis=1))
    out["topsis_score"] = sda / (sd + sda)
    out["topsis_rank"] = out["topsis_score"].rank(ascending=False, method="min").astype(int)
    return out


def copy_nsga2_reference() -> None:
    dest = RESULT_DIR / "nsga2_reference"
    dest.mkdir(parents=True, exist_ok=True)
    if CLASSMATE_DIR.exists():
        for f in ["problem3_pareto_front.csv", "problem3_grey_relation.csv",
                  "problem3_best_summary.json", "problem3_best_storage_schedule.csv"]:
            src = CLASSMATE_DIR / f
            if src.exists():
                shutil.copy2(src, dest / f)
        print("已复制同学 NSGA-II 参考结果")
    else:
        print("未找到同学结果目录，跳过 NSGA-II 对照")


def build_nsga2_comparison(df_exact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_exact.iterrows():
        rows.append({"method": "本文精确Pareto", "epsilon_frac": r["epsilon_frac"],
                     "total_cost_cny": r["total_cost_cny"],
                     "sys_peak_net_mw": r["sys_peak_net_mw"],
                     "sys_net_tv_mw": r["sys_net_tv_mw"],
                     "sys_net_mad_mw": r["sys_net_mad_mw"],
                     "total_renewable_util_pct": r["total_renewable_util_pct"]})
    ref = RESULT_DIR / "nsga2_reference" / "problem3_pareto_front.csv"
    if ref.exists():
        ns = pd.read_csv(ref)
        for _, r in ns.iterrows():
            rows.append({"method": "同学NSGA-II", "epsilon_frac": np.nan,
                         "total_cost_cny": r["total_cost_cny"],
                         "sys_peak_net_mw": r["system_peak_net_import_mw"],
                         "sys_net_tv_mw": r["load_fluctuation_mw"],
                         "sys_net_mad_mw": np.nan,
                         "total_renewable_util_pct": r["renewable_utilization"] * 100})
    return pd.DataFrame(rows)


def save_gra_best(params: dict, best_row: pd.Series) -> None:
    costmin = pd.read_csv(RESULT_DIR / "hourly_cost_min.csv")
    dev0 = {reg: region_dev_sum(costmin, reg) for reg in REGIONS}
    frac = float(best_row["epsilon_frac"])
    if frac <= 1e-12:
        df = pd.read_csv(RESULT_DIR / "hourly_flat_min.csv")
    else:
        outs = {}
        for reg in REGIONS:
            out = solve_region(params[reg], "cost", True,
                               flat_cap=frac * dev0[reg], enforce_exclusive=False)
            if check_constraints(params[reg], out)["simultaneous_ch_dis"] > 1e-6:
                out = solve_region(params[reg], "cost", True,
                                   flat_cap=frac * dev0[reg], enforce_exclusive=True)
            outs[reg] = out
        df = _outs_to_df(params, outs)
    df.to_csv(RESULT_DIR / "hourly_gra_best.csv", index=False, encoding="utf-8-sig")
    payload = {"epsilon_frac": frac}
    for k, v in best_row.items():
        if k not in ("epsilon_frac",) and not k.startswith("grey") \
                and k not in ("topsis_score", "topsis_rank", "grey_rank"):
            payload[k] = float(v)
    (RESULT_DIR / "pareto_gra_best.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GRA 加权综合最优：epsilon_frac={frac:.4f}，逐时方案已保存")


def main() -> None:
    params = build_params()
    baseline = aggregate_from_hourly(pd.read_csv(RESULT_DIR / "hourly_baseline.csv"))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 精确 Pareto 前沿扫描（ε-约束：成本最小 + 波动上限） ===")
    df_exact = run_exact_pareto(params)
    df_exact.to_csv(RESULT_DIR / "pareto_exact.csv", index=False, encoding="utf-8-sig")

    print("=== 灰色关联分析 + TOPSIS 决策 ===")
    df_gra = grey_relational_analysis(df_exact, baseline)
    df_gra.to_csv(RESULT_DIR / "gra_selection.csv", index=False, encoding="utf-8-sig")
    best = df_gra.loc[df_gra["grey_rank_w"].idxmin()]
    print(df_gra[["epsilon_frac", "total_cost_cny", "sys_peak_net_mw", "sys_net_mad_mw",
                  "grey_grade_w", "grey_rank_w", "topsis_rank"]].to_string(index=False))
    save_gra_best(params, best)

    print("=== NSGA-II 对照 ===")
    copy_nsga2_reference()
    cmp = build_nsga2_comparison(df_exact)
    cmp.to_csv(RESULT_DIR / "nsga2_comparison.csv", index=False, encoding="utf-8-sig")
    print("对照数据已保存：", RESULT_DIR / "nsga2_comparison.csv")


if __name__ == "__main__":
    main()
