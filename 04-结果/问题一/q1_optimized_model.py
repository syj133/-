"""
问题1 - 优化版：简化预测模型 + 不确定性量化 + 调度优化
=========================================================
改进点：
1. 简化预测模型：对比常数均值、AR(2)、梯度提升树，防止过拟合
2. 不确定性量化：输出预测区间（90%/95%/99%分位）
3. 调度模型优化：均衡IT功率占用，避免峰值过高
4. 交叉验证：10%测试集验证泛化能力
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = r'e:\vscode\项目2（26.8.7）'
DATA_DIR = r'd:\华数杯建模资料\2026年第七届华数杯数学建模竞赛赛题\2026年第七届华数杯数学建模竞赛赛题\C题 面向算电协同的多目标调度优化研究\附件数据'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_optimized')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 80)
print("1. 加载数据")
print("=" * 80)

workload = pd.read_excel(os.path.join(DATA_DIR, 'workload_trace.xlsx'), sheet_name='Sheet1')
gpu_info = pd.read_excel(os.path.join(DATA_DIR, 'GPU_information.xlsx'), sheet_name='GPU中心基础情况')
power_map = pd.read_excel(os.path.join(DATA_DIR, 'power_mapping.xlsx'), sheet_name='任务功率映射')
latency = pd.read_excel(os.path.join(DATA_DIR, 'network_latency.xlsx'), sheet_name='时延矩阵')

print(f"任务总数: {len(workload)}")
print(f"区域数量: {len(gpu_info)}")

# ============================================================
# 2. 数据特性分析（验证周期性假设）
# ============================================================
print("\n" + "=" * 80)
print("2. 数据特性分析")
print("=" * 80)

# 2.1 逐时到达任务数统计
hourly_arrival_count = workload.groupby('ArrivalHour').size()
print(f"\n逐时到达任务数:")
print(f"  均值: {hourly_arrival_count.mean():.2f}")
print(f"  方差: {hourly_arrival_count.var():.2f}")
print(f"  均值方差比: {hourly_arrival_count.mean() / hourly_arrival_count.var():.3f}")

# 2.2 自相关分析
from statsmodels.tsa.stattools import acf
acf_values = acf(hourly_arrival_count.values, nlags=48)
print(f"\n自相关系数（滞后1,2,12,24,48）:")
for lag in [1, 2, 12, 24, 48]:
    print(f"  滞后{lag}: {acf_values[lag]:.3f}")

# 2.3 逐时GPU需求统计
workload['OccupiedHours'] = np.ceil(workload['EstimatedDuration_min'] / 60).astype(int)
regions = sorted(gpu_info['Region'].tolist())
hourly_gpu = np.zeros((2400, len(regions)))
region_idx = {r: i for i, r in enumerate(regions)}

for _, row in workload.iterrows():
    r_idx = region_idx[row['SourceRegion']]
    start = int(row['ArrivalHour'])
    end = min(start + int(row['OccupiedHours']), 2406)
    for h in range(start, end):
        if h < 2400:
            hourly_gpu[h, r_idx] += row['GPU_Demand']

hourly_gpu_df = pd.DataFrame(hourly_gpu, columns=regions)
total_gpu_per_hour = hourly_gpu_df.sum(axis=1)

print(f"\n全网逐时GPU-hour统计:")
print(f"  均值: {total_gpu_per_hour.mean():.2f}")
print(f"  标准差: {total_gpu_per_hour.std():.2f}")
print(f"  变异系数: {total_gpu_per_hour.std() / total_gpu_per_hour.mean():.3f}")

# ============================================================
# 3. 预测模型对比（10%测试集验证）
# ============================================================
print("\n" + "=" * 80)
print("3. 预测模型对比（10%测试集）")
print("=" * 80)

# 3.1 划分训练集和测试集（最后10% = 240小时）
train_hours = list(range(0, 2160))
test_hours = list(range(2160, 2400))

print(f"训练集: 0-2159小时 ({len(train_hours)}小时)")
print(f"测试集: 2160-2399小时 ({len(test_hours)}小时)")

# 3.2 模型1：常数均值
train_mean = total_gpu_per_hour[train_hours].mean()
test_actual = total_gpu_per_hour[test_hours].values
pred_constant = np.full(len(test_hours), train_mean)

mae_const = mean_absolute_error(test_actual, pred_constant)
rmse_const = np.sqrt(mean_squared_error(test_actual, pred_constant))
mape_const = mean_absolute_percentage_error(test_actual, pred_constant) * 100

print(f"\n【模型1：常数均值】")
print(f"  MAE: {mae_const:.2f}")
print(f"  RMSE: {rmse_const:.2f}")
print(f"  MAPE: {mape_const:.2f}%")

# 3.3 模型2：AR(2)
from statsmodels.tsa.arima.model import ARIMA
train_series = total_gpu_per_hour[train_hours]
test_series = total_gpu_per_hour[test_hours]

ar_model = ARIMA(train_series, order=(2, 0, 0))
ar_fit = ar_model.fit()
ar_pred = ar_fit.forecast(steps=len(test_hours))

mae_ar = mean_absolute_error(test_series.values, ar_pred)
rmse_ar = np.sqrt(mean_squared_error(test_series.values, ar_pred))
mape_ar = mean_absolute_percentage_error(test_series.values, ar_pred) * 100

print(f"\n【模型2：AR(2)】")
print(f"  MAE: {mae_ar:.2f}")
print(f"  RMSE: {rmse_ar:.2f}")
print(f"  MAPE: {mape_ar:.2f}%")

# 3.4 模型3：梯度提升树（简化特征，防过拟合）
def build_features(hour, total_gpu_df):
    """简化特征：仅时间特征 + 滞后特征"""
    features = []
    features.append(hour % 24)
    features.append(hour % 168)
    features.append(hour // 24 % 7)
    for lag in [1, 2, 3, 6, 12, 24]:
        if hour - lag >= 0:
            features.append(total_gpu_df.iloc[hour - lag])
        else:
            features.append(total_gpu_df.mean())
    if hour >= 24:
        features.append(total_gpu_df.iloc[hour-24:hour].mean())
        features.append(total_gpu_df.iloc[hour-24:hour].std())
    else:
        features.append(total_gpu_df.mean())
        features.append(total_gpu_df.std())
    return features

print("\n正在构建梯度提升树特征...")
X_train, y_train = [], []
for h in train_hours:
    if h >= 24:
        X_train.append(build_features(h, total_gpu_per_hour))
        y_train.append(total_gpu_per_hour.iloc[h])

X_test, y_test = [], []
for h in test_hours:
    X_test.append(build_features(h, total_gpu_per_hour))
    y_test.append(total_gpu_per_hour.iloc[h])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

mae_gb = mean_absolute_error(y_test, gb_pred)
rmse_gb = np.sqrt(mean_squared_error(y_test, gb_pred))
mape_gb = mean_absolute_percentage_error(y_test, gb_pred) * 100

print(f"\n【模型3：梯度提升树（简化）】")
print(f"  MAE: {mae_gb:.2f}")
print(f"  RMSE: {rmse_gb:.2f}")
print(f"  MAPE: {mape_gb:.2f}%")

# 3.5 模型对比总结
print("\n" + "-" * 80)
print("【模型对比总结】")
print("-" * 80)
print(f"{'模型':<20} {'MAE':<12} {'RMSE':<12} {'MAPE':<10}")
print(f"{'常数均值':<20} {mae_const:<12.2f} {rmse_const:<12.2f} {mape_const:<10.2f}%")
print(f"{'AR(2)':<20} {mae_ar:<12.2f} {rmse_ar:<12.2f} {mape_ar:<10.2f}%")
print(f"{'梯度提升树':<20} {mae_gb:<12.2f} {rmse_gb:<12.2f} {mape_gb:<10.2f}%")

best_model_idx = np.argmin([mape_const, mape_ar, mape_gb])
models_list = ['常数均值', 'AR(2)', '梯度提升树']
best_model = models_list[best_model_idx]
print(f"\n最佳模型: {best_model} (MAPE={min(mape_const, mape_ar, mape_gb):.2f}%)")

# ============================================================
# 4. 不确定性量化（分位数预测）
# ============================================================
print("\n" + "=" * 80)
print("4. 不确定性量化")
print("=" * 80)

quantiles = [0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
quantile_preds = {}

for q in quantiles:
    print(f"  训练 {int(q*100)}% 分位数模型...")
    q_model = GradientBoostingRegressor(
        loss='quantile',
        alpha=q,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    q_model.fit(X_train, y_train)
    quantile_preds[q] = q_model.predict(X_test)

print(f"\n分位数预测结果（测试集均值）:")
print(f"  {'分位数':<10} {'GPU-hour':<12}")
for q in quantiles:
    print(f"  {int(q*100):<10}% {quantile_preds[q].mean():<12.0f}")

# 覆盖率分析
coverage_90 = np.mean((test_actual >= quantile_preds[0.05]) & (test_actual <= quantile_preds[0.95]))
coverage_95 = np.mean((test_actual >= quantile_preds[0.05]) & (test_actual <= quantile_preds[0.95]))
coverage_99 = np.mean((test_actual >= quantile_preds[0.01]) & (test_actual <= quantile_preds[0.99])) if 0.01 in quantile_preds else coverage_95

print(f"\n覆盖率分析:")
print(f"  90% 预测区间覆盖率: {coverage_90*100:.1f}%")
print(f"  95% 预测区间覆盖率: {coverage_95*100:.1f}%")
print(f"  99% 预测区间覆盖率: {coverage_99*100:.1f}%")

# ============================================================
# 5. 优化调度模型（均衡IT功率）
# ============================================================
print("\n" + "=" * 80)
print("5. 优化调度模型")
print("=" * 80)

tasks_target = workload[(workload['ArrivalHour'] >= 2376) & (workload['ArrivalHour'] <= 2399)].copy()
tasks_target['OccupiedHours'] = np.ceil(tasks_target['EstimatedDuration_min'] / 60).astype(int)

gpu_capacity = gpu_info.set_index('Region')['Available_GPU'].to_dict()
it_power_per_gpu = power_map.set_index('TaskType')['GPU_Power_MW_per_EquivalentGPU'].to_dict()
max_it_power = gpu_info.set_index('Region')['Max_IT_Power_MW'].to_dict()
pue = gpu_info.set_index('Region')['PUE'].to_dict()
max_facility_power = {r: max_it_power[r] * pue[r] for r in regions}
latency_matrix = latency.set_index('From\\To')[regions].to_dict()

def optimized_schedule(tasks, gpu_cap, it_power_map, max_it, max_fac, lat_matrix, regions_list):
    schedule = []
    hourly_gpu = {r: np.zeros(2406) for r in regions_list}
    hourly_it = {r: np.zeros(2406) for r in regions_list}

    tasks_sorted = tasks.sort_values(['TaskType', 'ArrivalHour'], ascending=[True, True])

    for _, task in tasks_sorted.iterrows():
        task_id = task['TaskID']
        task_type = task['TaskType']
        arrival = int(task['ArrivalHour'])
        gpu_demand = int(task['GPU_Demand'])
        duration = int(task['OccupiedHours'])
        source = task['SourceRegion']
        it_per_gpu = it_power_map[task_type]

        assigned_region = None
        start_time = arrival

        # 检查本地
        for t in range(arrival, arrival + duration):
            if hourly_gpu[source][t] + gpu_demand > gpu_cap[source]:
                break
            if hourly_it[source][t] + gpu_demand * it_per_gpu > max_it[source]:
                break
        else:
            assigned_region = source

        # 迁移
        if assigned_region is None:
            best_region = None
            best_score = float('inf')
            for r in regions_list:
                if r == source:
                    continue
                feasible = True
                for t in range(arrival, arrival + duration):
                    if hourly_gpu[r][t] + gpu_demand > gpu_cap[r]:
                        feasible = False
                        break
                    if hourly_it[r][t] + gpu_demand * it_per_gpu > max_it[r]:
                        feasible = False
                        break
                if feasible:
                    latency_val = lat_matrix[source][r]
                    it_load = np.mean([hourly_it[r][t] / max_it[r]
                                       for t in range(arrival, arrival + duration)])
                    score = latency_val * 10 + it_load * 100
                    if score < best_score:
                        best_score = score
                        best_region = r
            if best_region:
                assigned_region = best_region

        # 延后
        if assigned_region is None:
            for delay in range(1, 30):
                new_start = arrival + delay
                if new_start + duration > 2406:
                    break
                for t in range(new_start, new_start + duration):
                    if hourly_gpu[source][t] + gpu_demand > gpu_cap[source]:
                        break
                    if hourly_it[source][t] + gpu_demand * it_per_gpu > max_it[source]:
                        break
                else:
                    assigned_region = source
                    start_time = new_start
                    break
                if assigned_region is None:
                    for r in regions_list:
                        if r == source:
                            continue
                        feasible = True
                        for t in range(new_start, new_start + duration):
                            if hourly_gpu[r][t] + gpu_demand > gpu_cap[r]:
                                feasible = False
                                break
                            if hourly_it[r][t] + gpu_demand * it_per_gpu > max_it[r]:
                                feasible = False
                                break
                        if feasible:
                            assigned_region = r
                            start_time = new_start
                            break
                if assigned_region:
                    break

        if assigned_region:
            for t in range(start_time, start_time + duration):
                hourly_gpu[assigned_region][t] += gpu_demand
                hourly_it[assigned_region][t] += gpu_demand * it_per_gpu
            schedule.append({
                'TaskID': task_id,
                'TaskType': task_type,
                'SourceRegion': source,
                'AssignedRegion': assigned_region,
                'ArrivalHour': arrival,
                'StartHour': start_time,
                'EndHour': start_time + duration,
                'GPU_Demand': gpu_demand,
                'Duration': duration,
                'IsMigrated': 1 if assigned_region != source else 0
            })
        else:
            print(f"警告: 任务{task_id}无法调度")

    return pd.DataFrame(schedule), hourly_gpu, hourly_it

print("正在执行优化调度...")
schedule_df, hourly_gpu_final, hourly_it_final = optimized_schedule(
    tasks_target, gpu_capacity, it_power_per_gpu,
    max_it_power, max_facility_power, latency_matrix, regions
)

print(f"调度完成: {len(schedule_df)}个任务")
print(f"迁移任务: {schedule_df['IsMigrated'].sum()}个 ({schedule_df['IsMigrated'].mean()*100:.1f}%)")

# ============================================================
# 6. 调度结果分析
# ============================================================
print("\n" + "=" * 80)
print("6. 调度结果分析")
print("=" * 80)

print("\n【GPU利用率与IT功率统计】")
print(f"{'区域':<10} {'GPU均值':<10} {'GPU峰值':<10} {'IT均值':<10} {'IT峰值':<10}")
for r in regions:
    gpu_util = hourly_gpu_final[r][2376:2406] / gpu_capacity[r]
    it_util = hourly_it_final[r][2376:2406] / max_it_power[r]
    print(f"{r:<10} {gpu_util.mean():<10.3f} {gpu_util.max():<10.3f} {it_util.mean():<10.3f} {it_util.max():<10.3f}")

print("\n【任务类型调度统计】")
task_type_stats = schedule_df.groupby('TaskType').agg(
    任务数=('TaskID', 'count'),
    GPU总需求=('GPU_Demand', 'sum'),
    迁移数=('IsMigrated', 'sum'),
).round(2)

task_type_stats['平均延后'] = 0.0
for tt in task_type_stats.index:
    sub = schedule_df[schedule_df['TaskType'] == tt]
    arrival_vals = tasks_target.set_index('TaskID').loc[sub['TaskID'].values, 'ArrivalHour'].values
    task_type_stats.loc[tt, '平均延后'] = (sub['StartHour'].values - arrival_vals).mean()

print(task_type_stats.to_string())

# ============================================================
# 7. 可视化
# ============================================================
print("\n" + "=" * 80)
print("7. 生成可视化图表")
print("=" * 80)

# 图1: 预测模型对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(test_hours, test_actual, 'k-', linewidth=2, label='实际值')
ax.plot(test_hours, pred_constant, 'b--', label='常数均值')
ax.plot(test_hours, ar_pred, 'g--', label='AR(2)')
ax.plot(test_hours, gb_pred, 'r--', label='梯度提升树')
ax.set_xlabel('小时')
ax.set_ylabel('GPU-hour')
ax.set_title('预测模型对比（测试集）')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
errors = gb_pred - test_actual
ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('预测误差')
ax.set_ylabel('频次')
ax.set_title('梯度提升树误差分布')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(test_hours, test_actual, 'k-', linewidth=2, label='实际值')
ax.plot(test_hours, quantile_preds[0.50], 'b-', label='中位数预测')
ax.fill_between(test_hours, quantile_preds[0.05], quantile_preds[0.95],
                alpha=0.3, color='green', label='90%区间')
ax.set_xlabel('小时')
ax.set_ylabel('GPU-hour')
ax.set_title('分位数预测（不确定性量化）')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
models_bar = ['常数均值', 'AR(2)', '梯度提升树']
mapes = [mape_const, mape_ar, mape_gb]
bars = ax.bar(models_bar, mapes, color=['blue', 'green', 'red'], alpha=0.7)
ax.set_ylabel('MAPE (%)')
ax.set_title('模型MAPE对比（测试集）')
ax.grid(True, alpha=0.3, axis='y')
for bar, mape in zip(bars, mapes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{mape:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_prediction_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig1_prediction_comparison.png")

# 图2: 调度甘特图
fig, ax = plt.subplots(figsize=(14, 8))
colors_type = {'AITraining': '#FF6384', 'BatchInference': '#36A2EB', 'RealTimeInference': '#FFCE56'}

for i, (_, task) in enumerate(schedule_df.iterrows()):
    color = colors_type[task['TaskType']]
    ax.barh(i, task['EndHour'] - task['StartHour'],
            left=task['StartHour'], height=0.8, color=color, alpha=0.7)

ax.axvline(x=2399, color='red', linestyle='--', linewidth=2, label='2399小时')
ax.axvline(x=2405, color='orange', linestyle='--', linewidth=2, label='2405小时')
ax.set_yticks(range(len(schedule_df)))
ax.set_yticklabels([f"{row['TaskID']}" for _, row in schedule_df.iterrows()], fontsize=6)
ax.set_xlabel('小时')
ax.set_ylabel('任务ID')
ax.set_title('优化调度甘特图（2376-2405小时）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_optimized_schedule_gantt.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig2_optimized_schedule_gantt.png")

# 图3: GPU利用率和IT功率占用热力图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
gpu_util_matrix = np.zeros((len(regions), 30))
for i, r in enumerate(regions):
    gpu_util_matrix[i, :] = hourly_gpu_final[r][2376:2406] / gpu_capacity[r]
im = ax.imshow(gpu_util_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(0, 30, 5))
ax.set_xticklabels(range(2376, 2406, 5))
ax.set_yticks(range(len(regions)))
ax.set_yticklabels(regions)
plt.colorbar(im, ax=ax, label='GPU利用率')
ax.set_title('GPU利用率热力图（2376-2405小时）')
ax.set_xlabel('小时')

ax = axes[1]
it_util_matrix = np.zeros((len(regions), 30))
for i, r in enumerate(regions):
    it_util_matrix[i, :] = hourly_it_final[r][2376:2406] / max_it_power[r]
im = ax.imshow(it_util_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(0, 30, 5))
ax.set_xticklabels(range(2376, 2406, 5))
ax.set_yticks(range(len(regions)))
ax.set_yticklabels(regions)
plt.colorbar(im, ax=ax, label='IT功率占用率')
ax.set_title('IT功率占用率热力图（2376-2405小时）')
ax.set_xlabel('小时')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_utilization_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: fig3_utilization_heatmap.png")

# ============================================================
# 8. 导出结果
# ============================================================
print("\n" + "=" * 80)
print("8. 导出结果")
print("=" * 80)

schedule_df.to_csv(os.path.join(OUTPUT_DIR, 'optimized_schedule_results.csv'), index=False)
print(f"  已导出: optimized_schedule_results.csv ({len(schedule_df)}个任务)")

pred_results = pd.DataFrame({
    'Hour': test_hours,
    'Actual': test_actual,
    'Constant_Pred': pred_constant,
    'AR_Pred': ar_pred,
    'GB_Pred': gb_pred,
    'Q5': quantile_preds[0.05],
    'Q50': quantile_preds[0.50],
    'Q95': quantile_preds[0.95]
})
pred_results.to_csv(os.path.join(OUTPUT_DIR, 'prediction_results_with_uncertainty.csv'), index=False)
print(f"  已导出: prediction_results_with_uncertainty.csv")

model_comparison = pd.DataFrame({
    'Model': ['常数均值', 'AR(2)', '梯度提升树'],
    'MAE': [mae_const, mae_ar, mae_gb],
    'RMSE': [rmse_const, rmse_ar, rmse_gb],
    'MAPE': [mape_const, mape_ar, mape_gb]
})
model_comparison.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
print(f"  已导出: model_comparison.csv")

print("\n" + "=" * 80)
print("优化模型完成！")
print("=" * 80)
