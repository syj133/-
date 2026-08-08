# -*- coding: utf-8 -*-
"""结果复核：读取结果包逐时 CSV，按附件1统一口径逐项校验约束。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REGIONS, ROOT, build_params  # noqa: E402

RESULT_DIR = ROOT / "C题写作" / "问题3结果_我跑的" / "结果"
SCENARIOS = ["baseline", "no_storage", "cost_min", "carbon_min",
             "peak_min", "flat_min", "balanced"]


def main() -> None:
    params = build_params()
    ok_all = True
    for sc in SCENARIOS:
        df = pd.read_csv(RESULT_DIR / f"hourly_{sc}.csv")
        worst = {
            "energy_balance": 0.0, "soc_rec": 0.0, "sim_ch_dis": 0.0,
            "gp_over_import": 0.0, "gs_over_export": 0.0,
            "ch_over": 0.0, "dis_over": 0.0, "soc_viol": 0.0,
        }
        final_ok = True
        for reg in REGIONS:
            p = params[reg]
            s = df[df["Region"] == reg].sort_values("Hour")
            gp = s["GridPurchase_MW"].values
            gs = s["GridSell_MW"].values
            d = s["DirectRenewable_MW"].values
            rc = s["RenewableCharge_MW"].values
            gc = s["GridCharge_MW"].values
            c = s["Curtailment_MW"].values
            pdis = s["Discharge_MW"].values
            soc = s["SOC_MWh"].values
            L, R = p["load"], p["renewable"]
            eb = gp + R + pdis - (L + (rc + gc) + gs + c)
            worst["energy_balance"] = max(worst["energy_balance"], np.abs(eb).max())
            rec = np.empty(p["T"])
            rec[0] = soc[0] - p["eta_c"] * (rc[0] + gc[0]) + pdis[0] / p["eta_d"] - p["soc0"]
            for t in range(1, p["T"]):
                rec[t] = soc[t] - soc[t - 1] - p["eta_c"] * (rc[t] + gc[t]) + pdis[t] / p["eta_d"]
            worst["soc_rec"] = max(worst["soc_rec"], np.abs(rec).max())
            worst["sim_ch_dis"] = max(worst["sim_ch_dis"], np.minimum(rc + gc, pdis).max())
            worst["gp_over_import"] = max(worst["gp_over_import"], gp.max() - p["import_max"])
            worst["gs_over_export"] = max(worst["gs_over_export"], gs.max() - p["export_max"])
            worst["ch_over"] = max(worst["ch_over"], (rc + gc).max() - p["pch_max"])
            worst["dis_over"] = max(worst["dis_over"], pdis.max() - p["pdis_max"])
            worst["soc_viol"] = max(worst["soc_viol"],
                                   max(p["soc_min"] - soc.min(), soc.max() - p["soc_max"]))
            if soc[-1] < p["soc0"] - 1e-6:
                final_ok = False
        bad = {k: round(v, 6) for k, v in worst.items() if v > 1e-4}
        status = "通过" if (not bad and final_ok) else "异常"
        if status == "异常":
            ok_all = False
        print(f"{sc:12s} {status}  max_residuals={bad or '<=1e-4'}  终态SOC达标={final_ok}")
    print("\n全部场景校验：", "通过" if ok_all else "存在异常，请检查")


if __name__ == "__main__":
    main()
