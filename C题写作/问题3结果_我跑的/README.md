# 问题三结果包（储能协同优化）

## 场景

- `baseline`：附件基线（region_time_data.xlsx 原始运行状态）
- `no_storage`：无储能，仅优化购售电与新能源分配
- `cost_min` / `carbon_min` / `peak_min` / `flat_min`：含储能，分别以运行成本、碳排放、峰值净购电、净购电波动（MAD）为目标的字典序最优
- `balanced`：含储能，四目标按基线归一化加权（0.35/0.30/0.15/0.20）

## 文件说明

```
代码_problem3_storage/
  common.py      数据读取与参数构建（口径与附件1一致）
  storage_lp.py  储能协同优化 MILP 模型（约束、目标、指标、校验）
  run_all.py     主程序：求解全部场景并输出结果
  sensitivity.py 敏感性分析（储能容量、充放电效率）
  verify.py      结果复核（统一功率平衡/SOC递推/边界/互斥逐项校验）
  analysis.py    图表生成
结果/
  scenario_summary.csv      场景汇总（全时域 0–2406）
  per_region_metrics.csv    分区域分场景指标
  hourly_<scenario>.csv     逐区域逐小时决策明细（购售电/充放电/SOC 等）
  constraints_check.json    约束逐项校验
  sensitivity_capacity.csv   储能容量敏感性（0.25/0.5/1.0/1.5/2.0 × 成本/综合权衡）
  sensitivity_efficiency.csv 充放电效率敏感性（Δη = -0.05/0/+0.05，综合权衡）
图表与分析/                  8 张结果图（SOC、净购电、充放电、对比、权衡、敏感性）
```

## 关键结果（全时域 0–2406）

| 指标 | 基线 | 成本最优 | 削峰最优 | 平抑波动 |
| --- | --- | --- | --- | --- |
| 运行成本（万元） | 180 166 | -45 922 | -45 922 | -42 223 |
| 碳排放（tCO2） | 2 045 367 | 0 | 0 | 0 |
| 峰值净购电（MW） | 1 907.6 | -413.5 | -568.4 | -568.4 |
| 净购电波动 MAD（MW） | 507.9 | 0.51 | 0.51 | 0.00 |
| 新能源利用率 | 32.9% | 69.5% | 69.5% | 68.4% |

运行方法：
1. `python 代码_problem3_storage/run_all.py`（约 3–5 分钟）求解全部场景；
2. `python 代码_problem3_storage/sensitivity.py`（约 20–25 分钟）敏感性分析；
3. `python 代码_problem3_storage/verify.py` 复核约束；
4. `python 代码_problem3_storage/analysis.py` 生成图表。

说明：基线数据本身存在不一致（RegionE SOC 递推残差约 1 MWh；D/E/F 终态 SOC 低于初始值），复核脚本会将其标为“异常”，优化场景应全部“通过”。
