# -*- coding: utf-8 -*-
"""数据读取与参数准备（问题三：储能协同优化）。

统一口径（与附件1 一致）：
- 给定 IT 负荷 = Baseline_AI_IT_Load_MW + NonAI_IT_Load_MW；
- 设施负荷 Total_Load = 给定 IT 负荷 × PUE；
- 碳排放 = 购电功率 × 碳强度 × 1h；
- 新能源利用率 = (直接消纳 + 新能源充电 + 新能源外送) / 可用新能源累计；
- 储能 SOC 递推：SOC(t) = SOC(t-1) + ηc·ChargePower(t) - DischargePower(t)/ηd；
- 优化结束时 SOC(2406) ≥ InitialSOC。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
MAIN_END = 2405  # 主时域 0-2399 + 收尾 2400-2405，计入成本/碳排放结算
TERMINAL = 2406  # 终端状态结算小时（SOC(2406) ≥ InitialSOC）


def find_data_dir(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else ROOT
    cands = sorted(p for p in root.rglob("*") if p.is_dir() and p.name == "附件数据")
    if not cands:
        raise FileNotFoundError(f"未在 {root} 下找到附件数据目录")
    return cands[0]


def load_problem3_data(data_dir: Path | None = None):
    data_dir = Path(data_dir) if data_dir is not None else find_data_dir()
    gpu = pd.read_excel(data_dir / "GPU_information.xlsx", sheet_name=0)
    storage = pd.read_excel(data_dir / "storage_information.xlsx", sheet_name=0)
    rt = pd.read_excel(data_dir / "region_time_data.xlsx", sheet_name=0)
    return gpu, storage, rt


def build_params(data_dir: Path | None = None) -> dict:
    """为六个区域分别构建问题三的逐时参数。"""
    gpu, storage, rt = load_problem3_data(data_dir)
    pue = gpu.set_index("Region")["PUE"].to_dict()
    stor = storage.set_index("Region")
    hours = np.array(sorted(rt["Hour"].unique()), dtype=int)

    params: dict = {}
    for reg in REGIONS:
        sub = rt[rt["Region"] == reg].set_index("Hour").sort_index()
        load = (sub["Baseline_AI_IT_Load_MW"] + sub["NonAI_IT_Load_MW"]) * pue[reg]
        params[reg] = {
            "region": reg,
            "hours": hours,
            "T": len(hours),
            "load": load.values.astype(float),
            "renewable": sub["AvailableRenewable_MW"].values.astype(float),
            "price": sub["ElectricityPrice_CNY_per_MWh"].values.astype(float),
            "sell_price": sub["SellPrice_CNY_per_MWh"].values.astype(float),
            "carbon_intensity": sub["CarbonIntensity_tCO2_per_MWh"].values.astype(float),
            "soc0": float(stor.loc[reg, "InitialSOC_MWh"]),
            "soc_min": float(stor.loc[reg, "MinSOC_MWh"]),
            "soc_max": float(stor.loc[reg, "StorageCapacity_MWh"]),
            "pch_max": float(stor.loc[reg, "MaxChargePower_MW"]),
            "pdis_max": float(stor.loc[reg, "MaxDischargePower_MW"]),
            "eta_c": float(stor.loc[reg, "ChargeEfficiency"]),
            "eta_d": float(stor.loc[reg, "DischargeEfficiency"]),
            "import_max": float(stor.loc[reg, "MaxGridImport_MW"]),
            "export_max": float(stor.loc[reg, "MaxGridExport_MW"]),
            "pue": float(pue[reg]),
        }
    return params


def load_baseline_series(data_dir: Path | None = None) -> dict:
    """读取附件基准运行状态（逐区域逐小时），用于对比。"""
    gpu, _storage, rt = load_problem3_data(data_dir)
    pue = gpu.set_index("Region")["PUE"].to_dict()
    out: dict = {}
    for reg in REGIONS:
        sub = rt[rt["Region"] == reg].set_index("Hour").sort_index()
        out[reg] = {
            "load": ((sub["Baseline_AI_IT_Load_MW"] + sub["NonAI_IT_Load_MW"]) * pue[reg]).values.astype(float),
            "renewable": sub["AvailableRenewable_MW"].values.astype(float),
            "price": sub["ElectricityPrice_CNY_per_MWh"].values.astype(float),
            "sell_price": sub["SellPrice_CNY_per_MWh"].values.astype(float),
            "carbon_intensity": sub["CarbonIntensity_tCO2_per_MWh"].values.astype(float),
            "gp": sub["GridPurchase_MW"].values.astype(float),
            "gs": sub["GridSell_MW"].values.astype(float),
            "d": sub["UsedRenewable_MW"].values.astype(float),
            "rc": sub["RenewableCharge_MW"].values.astype(float),
            "gc": sub["GridCharge_MW"].values.astype(float),
            "c": sub["Curtailment_MW"].values.astype(float),
            "pdis": sub["DischargePower_MW"].values.astype(float),
            "soc": sub["SOC_MWh"].values.astype(float),
        }
    return out
