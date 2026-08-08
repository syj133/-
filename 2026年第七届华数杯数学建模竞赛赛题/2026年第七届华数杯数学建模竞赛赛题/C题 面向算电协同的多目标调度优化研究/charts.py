"""
C题 面向算电协同的多目标调度优化研究 —— 数据可视化
Generates 5 publication-quality charts in a single HTML file.

Charts:
  1. 各区域电价/碳强度/新能源出力 2400h 时序图
  2. 任务到达量的时间分布（区分三类任务）
  3. 各区域 GPU 资源分布饼图
  4. 区域间时延热力图
  5. 任务类型×区域交叉统计
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Reference palette (dataviz skill validated default) ──────────────────
# Categorical slots (fixed order, never cycled)
CAT = {
    1: "#2a78d6",  # blue
    2: "#eb6834",  # orange
    3: "#1baf7a",  # aqua
    4: "#eda100",  # yellow
    5: "#e87ba4",  # magenta
    6: "#008300",  # green
}

# Sequential blue ramp (100 → 700)
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# Status
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# Chrome
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
MUTED = "#898781"
SECONDARY = "#52514e"
PRIMARY = "#0b0b0b"

# Region → categorical slot (color follows entity, never rank)
REGION_COLORS = {
    "RegionA": CAT[1],
    "RegionB": CAT[2],
    "RegionC": CAT[3],
    "RegionD": CAT[4],
    "RegionE": CAT[5],
    "RegionF": CAT[6],
}
REGION_ORDER = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
TASK_TYPE_ORDER = ["AITraining", "BatchInference", "RealTimeInference"]
TASK_LABELS = {"AITraining": "AI训练", "BatchInference": "批量推理", "RealTimeInference": "实时推理"}
TASK_COLORS = {"AITraining": CAT[1], "BatchInference": CAT[3], "RealTimeInference": CAT[2]}
REGION_LABELS = {r: r.replace("Region", "区域") for r in REGION_ORDER}

DATA_DIR = Path("附件数据")

# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════

rt = pd.read_excel(DATA_DIR / "region_time_data.xlsx", sheet_name="region_time_data")
workload = pd.read_excel(DATA_DIR / "workload_trace.xlsx", sheet_name="Sheet1")
gpu = pd.read_excel(DATA_DIR / "GPU_information.xlsx", sheet_name=0)  # first sheet
latency_long = pd.read_excel(DATA_DIR / "network_latency.xlsx", sheet_name="network_latency")
# Pivot latency matrix
latency_matrix = latency_long.pivot(index="FromRegion", columns="ToRegion", values="NetworkLatency_ms")
latency_matrix = latency_matrix.loc[REGION_ORDER, REGION_ORDER]

# ═══════════════════════════════════════════════════════════════════════════
# Common layout template
# ═══════════════════════════════════════════════════════════════════════════

def apply_layout(fig, title, x_title=None, y_title=None, height=600):
    """Apply consistent layout to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=PRIMARY, family="system-ui, sans-serif"),
                   x=0, xref="paper"),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, sans-serif", color=SECONDARY, size=12),
        legend=dict(font=dict(color=SECONDARY), orientation="h", yanchor="top", y=-0.12,
                    xanchor="center", x=0.5, traceorder="normal"),
        margin=dict(l=60, r=30, t=50, b=60),
        height=height,
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor=GRIDLINE, griddash="solid", gridwidth=0.5,
        zerolinecolor=BASELINE, zerolinewidth=1,
        linecolor=BASELINE, linewidth=1,
        title=dict(text=x_title, font=dict(color=SECONDARY)) if x_title else None,
    )
    fig.update_yaxes(
        gridcolor=GRIDLINE, griddash="solid", gridwidth=0.5,
        zerolinecolor=BASELINE, zerolinewidth=1,
        linecolor=BASELINE, linewidth=1,
        title=dict(text=y_title, font=dict(color=SECONDARY)) if y_title else None,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1: 2400h Time Series — Electricity Price / Carbon Intensity / Renewable
# ═══════════════════════════════════════════════════════════════════════════

def chart1_timeseries():
    """Three-panel time series: 电价, 碳强度, 新能源出力 across 6 regions."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("电价 (元/MWh)", "碳强度 (tCO₂/MWh)", "可用新能源出力 (MW)"),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    metrics = [
        ("ElectricityPrice_CNY_per_MWh", "电价 (元/MWh)"),
        ("CarbonIntensity_tCO2_per_MWh", "碳强度 (tCO₂/MWh)"),
        ("AvailableRenewable_MW", "可用新能源出力 (MW)"),
    ]

    for row, (col, ylabel) in enumerate(metrics, start=1):
        for region in REGION_ORDER:
            df_r = rt[rt["Region"] == region]
            show_legend = (row == 1)  # legend only on top panel
            fig.add_trace(
                go.Scatter(
                    x=df_r["Hour"], y=df_r[col],
                    mode="lines", name=REGION_LABELS[region],
                    line=dict(color=REGION_COLORS[region], width=1.2),
                    legendgroup=region, showlegend=show_legend,
                    hovertemplate=f"{REGION_LABELS[region]}: %{{y:,.2f}}<extra></extra>",
                ),
                row=row, col=1,
            )
        fig.update_yaxes(title_text=ylabel, row=row, col=1)

    fig.update_xaxes(title_text="小时 (0–2399)", row=3, col=1)

    apply_layout(fig, "各区域 2400h 电价、碳强度与新能源出力时序",
                 x_title="小时", height=900)
    fig.update_layout(hovermode="x unified")
    # Adjust subplot title font
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=SECONDARY, family="system-ui, sans-serif")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2: Task Arrival Distribution — (a) full 2400h area, (b) hour-of-day bars
# ═══════════════════════════════════════════════════════════════════════════

def chart2_task_arrival():
    """Two-panel: full 2400h timeline as area chart + 24h daily pattern as grouped bars."""
    arrivals = workload.groupby(["ArrivalHour", "TaskType"]).size().reset_index(name="Count")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("全 2400h 任务到达时序", "24h 日内到达模式"),
        vertical_spacing=0.14,
        row_heights=[0.55, 0.45],
    )

    # Panel A: full 2400h area chart (stacked)
    for task_type in TASK_TYPE_ORDER:
        df_t = arrivals[arrivals["TaskType"] == task_type].sort_values("ArrivalHour")
        fig.add_trace(
            go.Scatter(
                x=df_t["ArrivalHour"], y=df_t["Count"],
                mode="lines", name=TASK_LABELS[task_type],
                line=dict(color=TASK_COLORS[task_type], width=1.2),
                stackgroup="one",  # stacked area
                legendgroup=task_type, showlegend=True,
                hovertemplate=f"{TASK_LABELS[task_type]}: %{{y}} 个任务<extra></extra>",
            ),
            row=1, col=1,
        )

    # Panel B: hour-of-day aggregation (0–23)
    arrivals["HourOfDay"] = arrivals["ArrivalHour"] % 24
    daily = arrivals.groupby(["HourOfDay", "TaskType"])["Count"].sum().reset_index()
    for task_type in TASK_TYPE_ORDER:
        df_t = daily[daily["TaskType"] == task_type].sort_values("HourOfDay")
        fig.add_trace(
            go.Bar(
                x=df_t["HourOfDay"], y=df_t["Count"],
                name=TASK_LABELS[task_type],
                marker=dict(color=TASK_COLORS[task_type], line=dict(width=0)),
                legendgroup=task_type, showlegend=False,
                hovertemplate=f"{TASK_LABELS[task_type]}: %{{y}} 个任务<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_xaxes(title_text="小时 (0–2399)", row=1, col=1)
    fig.update_xaxes(title_text="日内小时 (0–23)", row=2, col=1)
    fig.update_yaxes(title_text="任务到达数量", row=1, col=1)
    fig.update_yaxes(title_text="任务到达数量", row=2, col=1)

    apply_layout(fig, "任务到达量的时间分布（按任务类型）",
                 x_title=None, y_title=None, height=750)
    fig.update_layout(hovermode="x unified")
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=SECONDARY, family="system-ui, sans-serif")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3: GPU Resource Distribution Pie
# ═══════════════════════════════════════════════════════════════════════════

def chart3_gpu_pie():
    """Pie chart of Total GPU across 6 regions."""
    gpu_sorted = gpu.sort_values("Total_GPU", ascending=False)
    labels = gpu_sorted["Region"].map(REGION_LABELS)
    values = gpu_sorted["Total_GPU"]
    colors = [REGION_COLORS[r] for r in gpu_sorted["Region"]]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color=SURFACE, width=2)),
        textinfo="label+percent", textfont=dict(color=PRIMARY, family="system-ui, sans-serif", size=13),
        hovertemplate="%{label}: %{value:,} GPU<br>占比: %{percent}<extra></extra>",
        sort=False,
    ))

    apply_layout(fig, "各区域 GPU 资源分布", height=550)
    fig.update_layout(showlegend=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4: Inter-Region Latency Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def chart4_latency_heatmap():
    """Heatmap of network latency between regions."""
    labels = [r.replace("Region", "区域") for r in REGION_ORDER]
    values = latency_matrix.values

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=values, x=labels, y=labels,
        colorscale=[
            [0.0, SEQ_BLUE[0]],   # lightest
            [0.15, SEQ_BLUE[3]],
            [0.3, SEQ_BLUE[5]],
            [0.5, SEQ_BLUE[7]],
            [0.7, SEQ_BLUE[9]],
            [0.85, SEQ_BLUE[11]],
            [1.0, SEQ_BLUE[12]],  # darkest
        ],
        text=values, texttemplate="%{text} ms",
        textfont=dict(color=PRIMARY, size=13, family="system-ui, sans-serif"),
        hovertemplate="%{y} → %{x}: %{z} ms<extra></extra>",
        colorbar=dict(
            title=dict(text="时延 (ms)", font=dict(color=SECONDARY)),
            tickfont=dict(color=SECONDARY),
            outlinewidth=0,
        ),
        xgap=2, ygap=2,
        zmin=0, zmax=values.max(),
    ))

    apply_layout(fig, "区域间网络时延热力图", height=520)
    fig.update_xaxes(side="bottom", title=None)
    fig.update_yaxes(title=None)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Chart 5: Task Type × Region Cross-Tabulation
# ═══════════════════════════════════════════════════════════════════════════

def chart5_cross_tab():
    """Grouped bar chart: TaskType × SourceRegion."""
    cross = workload.groupby(["SourceRegion", "TaskType"]).size().reset_index(name="Count")
    cross["RegionLabel"] = cross["SourceRegion"].map(REGION_LABELS)
    cross["TaskLabel"] = cross["TaskType"].map(TASK_LABELS)
    # Ensure consistent ordering
    cross["RegionLabel"] = pd.Categorical(
        cross["RegionLabel"], categories=[REGION_LABELS[r] for r in REGION_ORDER], ordered=True
    )
    cross["TaskLabel"] = pd.Categorical(
        cross["TaskLabel"], categories=[TASK_LABELS[t] for t in TASK_TYPE_ORDER], ordered=True
    )
    cross = cross.sort_values(["RegionLabel", "TaskLabel"])

    fig = go.Figure()
    for task_type in TASK_TYPE_ORDER:
        df_t = cross[cross["TaskType"] == task_type]
        fig.add_trace(go.Bar(
            x=df_t["RegionLabel"], y=df_t["Count"],
            name=TASK_LABELS[task_type],
            marker=dict(color=TASK_COLORS[task_type], line=dict(width=0)),
            hovertemplate=f"{TASK_LABELS[task_type]} @ %{{x}}: %{{y}} 个任务<extra></extra>",
            text=df_t["Count"], textposition="outside",
            textfont=dict(color=SECONDARY, size=10, family="system-ui, sans-serif"),
        ))

    fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.08)
    apply_layout(fig, "任务类型 × 源区域交叉统计",
                 x_title="源区域", y_title="任务数量")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Assemble & export
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Generating charts...")

    fig1 = chart1_timeseries()
    fig2 = chart2_task_arrival()
    fig3 = chart3_gpu_pie()
    fig4 = chart4_latency_heatmap()
    fig5 = chart5_cross_tab()

    # Combine into single HTML
    html_parts = []
    for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5], start=1):
        # Each chart as a full-width div
        chart_html = fig.to_html(
            include_plotlyjs=(i == 1),  # only include plotly.js once
            full_html=False,
            div_id=f"chart{i}",
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoom2d"],
                "displaylogo": False,
                "toImageButtonOptions": {"format": "png", "filename": f"chart{i}"},
            },
        )
        html_parts.append(f'<div class="chart-container" id="chart{i}-container">{chart_html}</div>')

    # Full page wrapper
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>C题 算电协同 — 数据可视化</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: #f9f9f7;
    color: #0b0b0b;
    padding: 24px 32px;
    max-width: 1200px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 600;
    color: #0b0b0b;
    margin-bottom: 4px;
  }}
  h2 {{
    font-size: 14px;
    font-weight: 400;
    color: #52514e;
    margin-bottom: 28px;
  }}
  .chart-container {{
    background: #fcfcfb;
    border: 1px solid rgba(11,11,11,0.10);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 28px;
    page-break-inside: avoid;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0d0d0d; color: #ffffff; }}
    h1 {{ color: #ffffff; }}
    h2 {{ color: #c3c2b7; }}
    .chart-container {{ background: #1a1a19; border-color: rgba(255,255,255,0.10); }}
  }}
  @media print {{
    body {{ background: white; padding: 12px; }}
    .chart-container {{ break-inside: avoid; border: none; padding: 8px 0; }}
  }}
</style>
</head>
<body>
<h1>C题 面向算电协同的多目标调度优化研究</h1>
<h2>数据概览可视化 · 2026年第七届华数杯</h2>
{''.join(html_parts)}
</body>
</html>"""

    out_path = Path("charts.html")
    out_path.write_text(full_html, encoding="utf-8")
    print(f"Saved to: {out_path.resolve()}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")
    print("Done!")


if __name__ == "__main__":
    main()
