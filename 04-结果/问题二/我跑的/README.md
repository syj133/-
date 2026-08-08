# 华数杯 C 题 Problem 2 改进版结果包

## 包内结构

```
代码_problem2_improved/      改进版 Q2 完整代码（可复现）
结果_全时域0-2399/           全时域 8 策略结果（论文主结果）
   all_strategy_summary.csv  8 策略汇总指标
   pareto_front.csv          帕累托前沿
   strategy_notes.json       各策略权重与含义
   detail_<策略名>/          每个策略的明细
      problem2_task_schedule.csv        任务级调度表（5 万行）
      problem2_region_hour_metrics.csv  逐时区域指标
      baseline_region_hour_metrics.csv  附件基线逐时指标（同口径）
      problem2_summary.json             汇总 + 基线对比 + 约束校验
结果_最后24h/               第 2376-2399 小时窗口结果（同上结构）
图表与分析/                 帕累托图、基线对比图、负荷曲线、迁移统计
```

## 核心结论（全时域，8 策略均 50000/50000 任务排上）

- 0 未调度、0 约束违规、100% 按时完成（GPU/IT/设施功率/时延/时限均已校验）。
- 新能源利用率 ~68-69%（基线 33%）；弃电较基线降 53.6%；系统峰值净购电较基线降约 91%。
- 电费为负（-4.2 ~ -4.4 亿元）来自统一口径下 D/E/F 富余新能源外送收益，写论文时建议把购电成本与外送收益分开列示。
- 策略定位：latency_first 平均时延 5.1ms（几乎不迁移）；carbon_renewable 碳排最低（129.6 tCO2）且峰值净购电最低（107.6 MW）；load_balanced 西部承接比例最高（50.9%）。

## 如何复现

```powershell
cd 代码_problem2_improved
python run_pareto.py --data-dir "赛题附件数据目录" `
    --output-dir "输出目录" --solver greedy `
    --hour-start 0 --hour-end 2399 --workers 8 --save-detail all
```

依赖：Python 3.10+，pandas、numpy、scipy、matplotlib（本机 matplotlib 需 packaging 包）。

## 口径说明

- 电力结算按附件 1 统一口径：新能源优先消纳 → 富余外送（≤ MaxGridExport）→ 弃电；碳排=购电×碳强度；电费=购电×电价−外送×售电价；新能源利用率=(消纳+外送)/可用。
- 2400-2405 为收尾结算时段，任务开工加了软惩罚，仅在必要时占用。
- 本包为 Q2 任务调度层结果，不含 Q3 储能模型。
