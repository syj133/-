# -*- coding: utf-8 -*-
"""问题三：储能协同优化模型（各区域独立求解，MILP）。

决策变量（每区域、每时段 t = 0..2406）：
  gp_t  电网购电功率（含储能电网充电 gc_t）
  gs_t  新能源外送/售电功率
  d_t   新能源直接消纳功率
  rc_t  新能源充电功率
  gc_t  电网充电功率
  c_t   弃风弃光功率
  p_t   储能放电功率
  SOC_t 时段末储能荷电状态
  y_t   0-1 变量：1=充电时段、0=放电时段（禁止同时充放电）
  z / m / dev_t  削峰与平抑波动辅助变量

约束：
  ① 新能源分配平衡：d + rc + gs + c = R
  ② 功率平衡：gp + d + p = L + gc
  ③ SOC 递推：SOC_t = SOC_{t-1} + ηc·(rc+gc) - p/ηd，SOC(-1)=InitialSOC
  ④ SOC 上下限与终态约束：MinSOC ≤ SOC_t ≤ Capacity，SOC(2406) ≥ InitialSOC
  ⑤ 储能功率约束：rc+gc ≤ MaxChargePower，p ≤ MaxDischargePower，
     且 rc+gc ≤ MaxChargePower·y，p ≤ MaxDischargePower·(1-y)（互斥）
  ⑥ 购售电边界：gp ≤ MaxGridImport，gs ≤ MaxGridExport

目标（多目标，加权归一化与逐目标最优）：
  f1 运行成本 = Σ(price·gp - sell·gs)
  f2 碳排放 = Σ(carbon·gp)
  f3 区域峰值净购电 = max_t(gp - gs)
  f4 负荷波动（净购电平均绝对偏差） = Σ|gp - gs - mean|
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


def _assemble(p: dict, with_storage: bool, objective: str,
              ref: dict | None = None):
    T = p["T"]
    base = 8 * T
    i_z, i_m = base, base + 1
    i_y = base + 2 + T  # 0-1 互斥变量起点

    def i_gp(t): return t
    def i_gs(t): return T + t
    def i_d(t): return 2 * T + t
    def i_rc(t): return 3 * T + t
    def i_gc(t): return 4 * T + t
    def i_c(t): return 5 * T + t
    def i_p(t): return 6 * T + t
    def i_soc(t): return 7 * T + t
    def i_dev(t): return base + 2 + t
    n_vars = base + 2 + T + (T if with_storage else 0)

    eta_c, eta_d = p["eta_c"], p["eta_d"]
    pch, pdis = p["pch_max"], p["pdis_max"]

    # 等式约束：每一行显式携带一个 rhs，杜绝错位
    eq_entries = []
    for t in range(T):
        eq_entries.append(
            (t, [(i_d(t), 1.0), (i_rc(t), 1.0), (i_gs(t), 1.0), (i_c(t), 1.0)],
             p["renewable"][t]))
    for t in range(T):
        eq_entries.append(
            (T + t, [(i_gp(t), 1.0), (i_d(t), 1.0), (i_p(t), 1.0), (i_gc(t), -1.0)],
             p["load"][t]))
    for t in range(T):
        soc_items = [(i_soc(t), 1.0),
                     (i_rc(t), -eta_c), (i_gc(t), -eta_c), (i_p(t), 1.0 / eta_d)]
        if t > 0:
            soc_items.append((i_soc(t - 1), -1.0))
        eq_entries.append((2 * T + t, soc_items, p["soc0"] if t == 0 else 0.0))
    eq_entries.append(
        (3 * T,
         [(i_m, float(T))] + [(i_gp(t), -1.0) for t in range(T)]
         + [(i_gs(t), 1.0) for t in range(T)],
         0.0))

    # 不等式约束
    ub_entries = []
    for t in range(T):
        ub_entries.append((t, [(i_rc(t), 1.0), (i_gc(t), 1.0)], pch))
        ub_entries.append((T + t, [(i_p(t), 1.0)], pdis))
        ub_entries.append((2 * T + t, [(i_gp(t), 1.0)], p["import_max"]))
        ub_entries.append((3 * T + t, [(i_gs(t), 1.0)], p["export_max"]))
    ub_entries.append((4 * T, [(i_soc(T - 1), -1.0)], -p["soc0"]))
    for t in range(T):
        ub_entries.append(
            (4 * T + 1 + t, [(i_gp(t), 1.0), (i_gs(t), -1.0), (i_z, -1.0)], 0.0))
    for t in range(T):
        ub_entries.append(
            (5 * T + 1 + t,
             [(i_gp(t), 1.0), (i_gs(t), -1.0), (i_m, -1.0), (i_dev(t), -1.0)], 0.0))
        ub_entries.append(
            (6 * T + 1 + t,
             [(i_gp(t), -1.0), (i_gs(t), 1.0), (i_m, 1.0), (i_dev(t), -1.0)], 0.0))
    if with_storage:
        for t in range(T):
            ub_entries.append(
                (7 * T + 1 + t,
                 [(i_rc(t), 1.0), (i_gc(t), 1.0), (i_y + t, -pch)], 0.0))
            ub_entries.append(
                (8 * T + 1 + t, [(i_p(t), 1.0), (i_y + t, pdis)], pdis))

    def flatten(entries):
        rows, cols, vals, rhs = [], [], [], []
        for row, items, r in entries:
            for col, val in items:
                rows.append(row)
                cols.append(col)
                vals.append(val)
            rhs.append(r)
        return rows, cols, vals, rhs

    eq_rows, eq_cols, eq_vals, eq_rhs = flatten(eq_entries)
    ub_rows, ub_cols, ub_vals, ub_rhs = flatten(ub_entries)
    ub_row_ids = [row for row, _items, _r in ub_entries]

    # 关键：求解器按“行号顺序”比较 A 的行与 rhs，因此必须把 rhs 重排为行号顺序。
    # （追加顺序与行号顺序不同：行是交错追加的。）
    ub_n_rows = len(ub_row_ids)
    row_rhs = {int(r): float(rv) for r, rv in zip(ub_row_ids, ub_rhs)}
    ub_rhs_sorted = np.array([row_rhs[r] for r in range(ub_n_rows)])

    # 目标函数
    c = np.zeros(n_vars)
    if objective == "cost":
        for t in range(T):
            c[i_gp(t)] = p["price"][t]
            c[i_gs(t)] = -p["sell_price"][t]
    elif objective == "carbon":
        for t in range(T):
            c[i_gp(t)] = p["carbon_intensity"][t]
    elif objective == "peak":
        c[i_z] = 1.0
    elif objective == "flat":
        for t in range(T):
            c[i_dev(t)] = 1.0
    elif objective == "balanced" and ref is not None:
        w = ref.get("weights", {"cost": 0.35, "carbon": 0.30, "peak": 0.15, "flat": 0.20})
        for key, base_val in ref.items():
            if key == "weights" or base_val is None:
                continue
            nrm = abs(base_val) if abs(base_val) > 1e-9 else 1e-9
            ww = w[key]
            if key == "cost":
                for t in range(T):
                    c[i_gp(t)] += ww * p["price"][t] / nrm
                    c[i_gs(t)] += ww * (-p["sell_price"][t]) / nrm
            elif key == "carbon":
                for t in range(T):
                    c[i_gp(t)] += ww * p["carbon_intensity"][t] / nrm
            elif key == "peak":
                c[i_z] += ww / nrm
            elif key == "flat":
                for t in range(T):
                    c[i_dev(t)] += ww / nrm
    else:
        raise ValueError(f"未知目标: {objective}")

    # 边界
    lo = np.zeros(n_vars)
    hi = np.full(n_vars, np.inf)
    for t in range(T):
        lo[i_soc(t)] = p["soc_min"]
        hi[i_soc(t)] = p["soc_max"]
    if not with_storage:
        for t in range(T):
            hi[i_rc(t)] = hi[i_gc(t)] = hi[i_p(t)] = 0.0
            lo[i_soc(t)] = hi[i_soc(t)] = p["soc0"]
    lo[i_z] = -np.inf
    hi[i_z] = np.inf
    lo[i_m] = -np.inf
    hi[i_m] = np.inf
    if with_storage:
        for t in range(T):
            lo[i_y + t] = 0.0
            hi[i_y + t] = 1.0
    bounds = list(zip(lo, hi))

    A_eq = coo_matrix((eq_vals, (eq_rows, eq_cols)),
                      shape=(len(eq_rhs), n_vars)).tocsr()
    A_ub = coo_matrix((ub_vals, (ub_rows, ub_cols)),
                      shape=(len(ub_rhs), n_vars)).tocsr()
    return {
        "c": c, "A_eq": A_eq, "b_eq": np.array(eq_rhs),
        "A_ub": A_ub, "b_ub": ub_rhs_sorted, "bounds": bounds,
        "ub_rows": ub_rows, "ub_rhs": ub_rhs_sorted,
        "ub_row_ids": ub_row_ids,
        "T": T, "base": base, "n_vars": n_vars,
        "i_gp": i_gp, "i_gs": i_gs, "i_d": i_d, "i_rc": i_rc,
        "i_gc": i_gc, "i_c": i_c, "i_p": i_p, "i_soc": i_soc, "i_y": i_y,
    }


def _solve(prob: dict, with_storage: bool, integrality: np.ndarray,
           extra_rows=None):
    """在 prob 基础上（可选追加若干行上界约束）求解 LP/MILP。

    extra_rows: [(A_row, rhs), ...]，每项表示 A_row·x ≤ rhs。
    """
    A_ub, b_ub = prob["A_ub"], prob["b_ub"]
    if extra_rows:
        rows = [r for r, _ in extra_rows]
        rhs = [rv for _, rv in extra_rows]
        A_ub = sp.vstack([A_ub] + rows).tocsr()
        b_ub = np.append(b_ub, rhs)
    if with_storage:
        res = milp(
            prob["c"],
            integrality=integrality,
            bounds=Bounds(np.array([b[0] for b in prob["bounds"]]),
                          np.array([b[1] for b in prob["bounds"]])),
            constraints=(
                LinearConstraint(A_ub, -np.inf, b_ub),
                LinearConstraint(prob["A_eq"], prob["b_eq"], prob["b_eq"]),
            ),
            options={"time_limit": 600, "mip_rel_gap": 0.0},
        )
    else:
        res = linprog(
            prob["c"], A_ub=A_ub, b_ub=b_ub,
            A_eq=prob["A_eq"], b_eq=prob["b_eq"], bounds=prob["bounds"],
            method="highs",
        )
    return res


def solve_region(p: dict, objective: str = "cost", with_storage: bool = True,
                 ref: dict | None = None, tiebreak_cost: bool = False,
                 flat_cap: float | None = None, enforce_exclusive: bool = True,
                 verbose: bool = False) -> dict:
    """求解单个区域的储能优化问题，返回结果与指标。"""
    prob = _assemble(p, with_storage, objective, ref)
    integrality = np.zeros(prob["n_vars"], dtype=int)
    if with_storage and enforce_exclusive:
        for t in range(prob["T"]):
            integrality[prob["i_y"] + t] = 1
    use_milp = with_storage and enforce_exclusive
    extra_rows = []
    if flat_cap is not None:
        T = prob["T"]
        dev_cols = [prob["base"] + 2 + t for t in range(T)]
        row = coo_matrix((np.ones(T, dtype=float),
                          (np.zeros(T, dtype=int), dev_cols)),
                         shape=(1, prob["n_vars"]))
        extra_rows.append((row, float(flat_cap)))
    res = _solve(prob, use_milp, integrality, extra_rows=extra_rows)
    if not res.success:
        raise RuntimeError(f"{p['region']} {objective} 求解失败: {res.message}")
    primary_obj = float(res.fun)

    # 两阶段（字典序）：保持主目标最优的前提下最小化运行成本，
    # 避免主目标存在多个最优解时策略随意。
    if tiebreak_cost and objective != "cost":
        prob2 = _assemble(p, with_storage, "cost", ref)
        c1 = prob["c"]
        nz = np.nonzero(c1)[0]
        row = coo_matrix((c1[nz], (np.zeros(len(nz), dtype=int), nz)),
                         shape=(1, len(c1)))
        tol = 1e-6 * max(1.0, abs(primary_obj))
        extra_rows2 = list(extra_rows) + [(row, primary_obj + tol)]
        res2 = _solve(prob2, use_milp, integrality, extra_rows=extra_rows2)
        if res2.success:
            res = res2

    T = prob["T"]
    x = res.x
    out = {
        "gp": np.array([x[prob["i_gp"](t)] for t in range(T)]),
        "gs": np.array([x[prob["i_gs"](t)] for t in range(T)]),
        "d": np.array([x[prob["i_d"](t)] for t in range(T)]),
        "rc": np.array([x[prob["i_rc"](t)] for t in range(T)]),
        "gc": np.array([x[prob["i_gc"](t)] for t in range(T)]),
        "c": np.array([x[prob["i_c"](t)] for t in range(T)]),
        "pdis": np.array([x[prob["i_p"](t)] for t in range(T)]),
        "soc": np.array([x[prob["i_soc"](t)] for t in range(T)]),
        "obj": primary_obj,
        "status": res.status,
    }
    return out


def compute_metrics(p: dict, out: dict) -> dict:
    """按附件1 统一口径计算评价指标。"""
    T = p["T"]
    gp, gs = out["gp"], out["gs"]
    net = gp - gs
    ch = out["rc"] + out["gc"]
    cost = float(np.sum(p["price"] * gp - p["sell_price"] * gs))
    carbon = float(np.sum(p["carbon_intensity"] * gp))
    peak = float(net.max())
    mad = float(np.mean(np.abs(net - net.mean())))
    std = float(net.std())
    rng = float(net.max() - net.min())
    util = float((out["d"] + out["rc"] + gs).sum() / p["renewable"].sum())
    curtail = float(out["c"].sum())
    charge = float(ch.sum())
    discharge = float(out["pdis"].sum())
    purchase = float(gp.sum())
    export = float(gs.sum())
    return {
        "cost_cny": cost,
        "carbon_tco2": carbon,
        "peak_net_mw": peak,
        "net_std_mw": std,
        "net_mad_mw": mad,
        "net_range_mw": rng,
        "renewable_util": util,
        "curtailment_mwh": curtail,
        "charge_mwh": charge,
        "discharge_mwh": discharge,
        "purchase_mwh": purchase,
        "export_mwh": export,
        "final_soc_mwh": float(out["soc"][-1]),
    }


def check_constraints(p: dict, out: dict) -> dict:
    """逐项校验约束满足情况，返回最大残差。"""
    T = p["T"]
    gp, gs, d, rc, gc, c = out["gp"], out["gs"], out["d"], out["rc"], out["gc"], out["c"]
    pdis, soc = out["pdis"], out["soc"]
    res = {}
    res["renewable_balance"] = float(np.max(np.abs(d + rc + gs + c - p["renewable"])))
    res["power_balance"] = float(np.max(np.abs(gp + d + pdis - p["load"] - gc)))
    soc_rec = np.empty(T)
    soc_rec[0] = soc[0] - p["eta_c"] * (rc[0] + gc[0]) + pdis[0] / p["eta_d"] - p["soc0"]
    for t in range(1, T):
        soc_rec[t] = soc[t] - soc[t - 1] - p["eta_c"] * (rc[t] + gc[t]) + pdis[t] / p["eta_d"]
    res["soc_recursion"] = float(np.max(np.abs(soc_rec)))
    res["soc_lower"] = float(soc.min() - p["soc_min"])
    res["soc_upper"] = float(p["soc_max"] - soc.max())
    res["final_soc"] = float(soc[-1] - p["soc0"])
    res["charge_limit"] = float(np.max(rc + gc) - p["pch_max"])
    res["discharge_limit"] = float(pdis.max() - p["pdis_max"])
    res["import_limit"] = float(gp.max() - p["import_max"])
    res["export_limit"] = float(gs.max() - p["export_max"])
    res["simultaneous_ch_dis"] = float(np.max(np.minimum(rc + gc, pdis)))
    res["nonneg"] = float(min(gp.min(), gs.min(), d.min(), rc.min(), gc.min(), c.min(), pdis.min()))
    return res
