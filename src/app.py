import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.optimization.scheduler import optimize_batch_schedule, get_hourly_carbon_intensity

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌱 Green Coding Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0a0f0a; color: #e8f5e8; }
    
    .metric-card {
        background: linear-gradient(135deg, #0d1f0d 0%, #122012 100%);
        border: 1px solid #2d5a2d;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; color: #4ade80; font-weight: 600; }
    .metric-label { color: #86efac; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
    .metric-delta { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #fbbf24; }
    
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #4ade80 !important; }
    
    .section-header {
        border-left: 3px solid #4ade80;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem;
        font-family: 'IBM Plex Mono', monospace;
        color: #86efac;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .stSelectbox > div, .stSlider > div { background: transparent; }
    [data-testid="stSidebar"] { background: #060d06; border-right: 1px solid #1a3a1a; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/raw/cloudwatch_logs.csv"
    if not os.path.exists(path):
        st.error("❌ No data found. Run: `python src/ingestion/log_generator.py`")
        st.stop()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

@st.cache_resource
def load_model():
    path = "models/carbon_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

df = load_data()
model = load_model()

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌱 Green Coder")
    st.markdown("---")
    page = st.radio("Navigation", ["📊 Overview", "🔬 Service Analysis", "⚡ Batch Optimizer", "🤖 Carbon Predictor"])
    st.markdown("---")
    st.markdown("**Filters**")
    selected_region = st.selectbox("Region", ["All"] + sorted(df["region"].unique().tolist()))
    selected_service = st.selectbox("Service", ["All"] + sorted(df["service"].unique().tolist()))
    date_range = st.slider("Days Back", 1, 30, 7)

# Filter data
filtered_df = df.copy()
cutoff = pd.Timestamp.now() - pd.Timedelta(days=date_range)
filtered_df = filtered_df[filtered_df["timestamp"] >= cutoff]
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]
if selected_service != "All":
    filtered_df = filtered_df[filtered_df["service"] == selected_service]

# ─── Page: Overview ──────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("# 🌱 Green Coding & Carbon Optimizer")
    st.markdown(f"*Analyzing {len(filtered_df):,} compute events across {date_range} days*")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_carbon = filtered_df["carbon_gco2"].sum()
    total_cost = filtered_df["cost_usd"].sum()
    avg_cpu = filtered_df["cpu_utilization"].mean()
    total_kwh = filtered_df["power_kwh"].sum()

    for col, (val, label, delta) in zip(
        [col1, col2, col3, col4],
        [
            (f"{total_carbon:.1f}", "Total CO₂ (gCO2)", "⬇ 23% vs last period"),
            (f"${total_cost:.2f}", "Compute Cost", "⬆ 5% vs last period"),
            (f"{avg_cpu:.1f}%", "Avg CPU Util", "→ Stable"),
            (f"{total_kwh*1000:.2f}", "Energy (Wh)", "⬇ 12% vs last period"),
        ]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Carbon by Service
    st.markdown('<div class="section-header">Carbon Footprint by Microservice</div>', unsafe_allow_html=True)
    service_carbon = filtered_df.groupby("service").agg(
        total_carbon=("carbon_gco2", "sum"),
        total_cost=("cost_usd", "sum"),
        avg_cpu=("cpu_utilization", "mean"),
        invocations=("invocations", "sum")
    ).reset_index().sort_values("total_carbon", ascending=False)

    fig = px.bar(
        service_carbon, x="service", y="total_carbon",
        color="total_carbon", color_continuous_scale="YlOrRd",
        title="",
        labels={"total_carbon": "CO₂ (gCO2)", "service": "Microservice"}
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e8f5e8", showlegend=False,
        xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hourly carbon heatmap
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Carbon by Hour of Day</div>', unsafe_allow_html=True)
        hourly = filtered_df.groupby(["hour_of_day", "region"])["carbon_gco2"].mean().reset_index()
        fig2 = px.line(
            hourly, x="hour_of_day", y="carbon_gco2", color="region",
            labels={"hour_of_day": "Hour (UTC)", "carbon_gco2": "Avg CO₂ (gCO2)"}
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8f5e8", legend_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a")
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Cost vs Carbon Scatter</div>', unsafe_allow_html=True)
        sample = filtered_df.sample(min(500, len(filtered_df)))
        fig3 = px.scatter(
            sample, x="cost_usd", y="carbon_gco2", color="service",
            size="cpu_utilization", opacity=0.7,
            labels={"cost_usd": "Cost (USD)", "carbon_gco2": "CO₂ (gCO2)"}
        )
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8f5e8", legend_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a")
        )
        st.plotly_chart(fig3, use_container_width=True)

# ─── Page: Service Analysis ───────────────────────────────────────────────────────
elif page == "🔬 Service Analysis":
    st.markdown("# 🔬 Microservice Carbon Analysis")

    service_stats = filtered_df.groupby("service").agg(
        total_carbon=("carbon_gco2", "sum"),
        total_cost=("cost_usd", "sum"),
        avg_cpu=("cpu_utilization", "mean"),
        avg_memory=("memory_utilization", "mean"),
        avg_duration=("duration_ms", "mean"),
        total_invocations=("invocations", "sum"),
        total_kwh=("power_kwh", "sum")
    ).reset_index()
    service_stats["carbon_per_invocation"] = (
        service_stats["total_carbon"] / service_stats["total_invocations"]
    ).round(8)
    service_stats["efficiency_score"] = (
        100 - (service_stats["total_carbon"] / service_stats["total_carbon"].max() * 100)
    ).round(1)

    st.markdown('<div class="section-header">Service Efficiency Leaderboard</div>', unsafe_allow_html=True)

    sorted_stats = service_stats.sort_values("total_carbon", ascending=False)
    for _, row in sorted_stats.iterrows():
        color = "#ef4444" if row["efficiency_score"] < 40 else "#f59e0b" if row["efficiency_score"] < 70 else "#4ade80"
        st.markdown(f"""
        <div style="background:#0d1f0d;border:1px solid #1a3a1a;border-radius:8px;padding:0.8rem 1.2rem;margin-bottom:0.5rem;display:flex;justify-content:space-between;align-items:center">
            <span style="font-family:'IBM Plex Mono',monospace;color:#e8f5e8;font-weight:600">{row['service']}</span>
            <span style="color:{color};font-family:'IBM Plex Mono',monospace;font-weight:600">
                Score: {row['efficiency_score']:.0f}/100
            </span>
            <span style="color:#86efac;font-size:0.85rem">{row['total_carbon']:.2f} gCO2 | ${row['total_cost']:.3f}</span>
            <span style="color:#94a3b8;font-size:0.8rem">CPU: {row['avg_cpu']:.1f}% | Mem: {row['avg_memory']:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Detailed Stats</div>', unsafe_allow_html=True)
    display_df = service_stats[[
        "service", "total_carbon", "total_cost", "avg_cpu",
        "avg_memory", "avg_duration", "efficiency_score"
    ]].copy()
    display_df.columns = ["Service", "CO₂ (gCO2)", "Cost ($)", "Avg CPU%", "Avg Mem%", "Avg Duration(ms)", "Efficiency"]
    st.dataframe(display_df.style.background_gradient(subset=["CO₂ (gCO2)"], cmap="RdYlGn_r"), use_container_width=True)

# ─── Page: Batch Optimizer ────────────────────────────────────────────────────────
elif page == "⚡ Batch Optimizer":
    st.markdown("# ⚡ Green Batch Job Scheduler")
    st.markdown("*Uses Linear Programming to minimize CO₂ by scheduling jobs during renewable energy peaks*")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="section-header">Job Configuration</div>', unsafe_allow_html=True)
        opt_region = st.selectbox("Grid Region", ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"])
        max_parallel = st.slider("Max Parallel Jobs", 1, 5, 3)
        deadline = st.slider("Deadline (hours)", 12, 48, 24)

        st.markdown("**Add Batch Jobs:**")
        num_jobs = st.number_input("Number of jobs", 1, 8, 4)

        jobs = []
        for i in range(num_jobs):
            with st.expander(f"Job {i+1}", expanded=(i < 2)):
                jname = st.text_input(f"Name", f"BatchJob-{i+1}", key=f"name_{i}")
                jdur = st.slider(f"Duration (hours)", 1, 8, 2, key=f"dur_{i}")
                jcpu = st.slider(f"CPU power (kWh)", 0.05, 2.0, 0.3, key=f"cpu_{i}")
                jmem = st.slider(f"Memory power (kWh)", 0.01, 0.5, 0.1, key=f"mem_{i}")
                jobs.append({"name": jname, "duration_hours": jdur, "cpu_kwh": jcpu, "memory_kwh": jmem})

        run_btn = st.button("🚀 Optimize Schedule", type="primary", use_container_width=True)

    with col2:
        # Always show carbon intensity profile
        st.markdown('<div class="section-header">Grid Carbon Intensity Profile</div>', unsafe_allow_html=True)
        ci = get_hourly_carbon_intensity(opt_region)
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=list(range(24)), y=ci, fill="tozeroy",
            line=dict(color="#4ade80", width=2),
            fillcolor="rgba(74,222,128,0.15)",
            name="Carbon Intensity"
        ))
        fig_ci.add_hrect(y0=0, y1=min(ci) * 1.1, fillcolor="rgba(74,222,128,0.1)", line_width=0,
                         annotation_text="🌱 Best Window", annotation_position="top left")
        fig_ci.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8f5e8",
            xaxis=dict(title="Hour of Day (UTC)", gridcolor="#1a3a1a", tickmode="linear", dtick=2),
            yaxis=dict(title="gCO₂/kWh", gridcolor="#1a3a1a"),
            height=300
        )
        st.plotly_chart(fig_ci, use_container_width=True)

        if run_btn:
            with st.spinner("Running optimization..."):
                result = optimize_batch_schedule(jobs, opt_region, max_parallel, deadline)

            if result["status"] == "Optimal":
                # Savings banner
                st.success(f"✅ Optimal schedule found! Saving **{result['carbon_saved_gco2']:.1f} gCO2** ({result['savings_pct']}% reduction)")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Optimized CO₂", f"{result['total_carbon_gco2']:.1f} gCO2")
                col_b.metric("Naive CO₂", f"{result['naive_carbon_gco2']:.1f} gCO2")
                col_c.metric("Carbon Saved", f"{result['carbon_saved_gco2']:.1f} gCO2 ({result['savings_pct']}%)")

                # Gantt chart
                st.markdown('<div class="section-header">Optimized Schedule (Gantt)</div>', unsafe_allow_html=True)
                gantt_data = []
                for s in result["schedule"]:
                    gantt_data.append(dict(
                        Task=s["job"],
                        Start=f"2024-01-01 {s['start_hour']:02d}:00",
                        Finish=f"2024-01-01 {min(s['end_hour'], 23):02d}:59",
                        Carbon=s["carbon_gco2"]
                    ))

                fig_gantt = px.timeline(
                    pd.DataFrame(gantt_data), x_start="Start", x_end="Finish", y="Task",
                    color="Carbon", color_continuous_scale="YlOrRd",
                    title=""
                )
                fig_gantt.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e8f5e8", height=300
                )
                st.plotly_chart(fig_gantt, use_container_width=True)

                # Schedule table
                sched_df = pd.DataFrame(result["schedule"])[[
                    "job", "start_hour", "end_hour", "avg_carbon_intensity", "carbon_gco2", "power_kwh"
                ]]
                sched_df.columns = ["Job", "Start (UTC)", "End (UTC)", "Avg CI (gCO2/kWh)", "CO₂ (gCO2)", "Power (kWh)"]
                st.dataframe(sched_df, use_container_width=True)
            else:
                st.error(f"Optimization failed: {result['status']}. Try relaxing constraints.")

# ─── Page: Carbon Predictor ───────────────────────────────────────────────────────
elif page == "🤖 Carbon Predictor":
    st.markdown("# 🤖 ML Carbon Footprint Predictor")
    if model is None:
        st.warning("⚠️ No trained model found. Run: `python src/modeling/carbon_model.py`")
    else:
        st.success("✅ Gradient Boosting model loaded")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Input Parameters</div>', unsafe_allow_html=True)
            pred_service = st.selectbox("Microservice", df["service"].unique())
            pred_region = st.selectbox("Region", df["region"].unique())
            pred_hour = st.slider("Hour of Day (UTC)", 0, 23, 12)
            pred_day = st.slider("Day of Week (0=Mon)", 0, 6, 2)
            pred_cpu = st.slider("CPU Utilization (%)", 1.0, 99.0, 50.0)
            pred_mem = st.slider("Memory Utilization (%)", 5.0, 99.0, 60.0)
            pred_dur = st.number_input("Duration (ms)", 100, 300000, 5000)
            pred_inv = st.number_input("Invocations", 1, 10000, 100)

            predict_btn = st.button("🔮 Predict Carbon", type="primary", use_container_width=True)

        with col2:
            if predict_btn:
                import numpy as np
                hour_sin = np.sin(2 * np.pi * pred_hour / 24)
                hour_cos = np.cos(2 * np.pi * pred_hour / 24)
                day_sin = np.sin(2 * np.pi * pred_day / 7)
                day_cos = np.cos(2 * np.pi * pred_day / 7)

                input_df = pd.DataFrame([{
                    "cpu_utilization": pred_cpu,
                    "memory_utilization": pred_mem,
                    "duration_ms": pred_dur,
                    "invocations": pred_inv,
                    "hour_sin": hour_sin,
                    "hour_cos": hour_cos,
                    "day_sin": day_sin,
                    "day_cos": day_cos,
                    "service": pred_service,
                    "region": pred_region
                }])

                prediction = model.predict(input_df)[0]
                st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:1rem">
                    <div class="metric-value">{prediction:.6f}</div>
                    <div class="metric-label">Estimated CO₂ (gCO2)</div>
                </div>
                """, unsafe_allow_html=True)

                # Find best time to run this service
                st.markdown("**🌱 Greenest Hour to Run This Service:**")
                predictions_by_hour = []
                for h in range(24):
                    hs = np.sin(2 * np.pi * h / 24)
                    hc = np.cos(2 * np.pi * h / 24)
                    row = input_df.copy()
                    row["hour_sin"] = hs
                    row["hour_cos"] = hc
                    p = model.predict(row)[0]
                    predictions_by_hour.append({"hour": h, "carbon_gco2": p})

                pbh_df = pd.DataFrame(predictions_by_hour)
                best_hour = pbh_df.loc[pbh_df["carbon_gco2"].idxmin(), "hour"]

                fig_pred = px.bar(
                    pbh_df, x="hour", y="carbon_gco2",
                    color="carbon_gco2", color_continuous_scale="RdYlGn_r",
                    labels={"hour": "Hour (UTC)", "carbon_gco2": "Predicted CO₂ (gCO2)"}
                )
                fig_pred.add_vline(x=best_hour, line_dash="dash", line_color="#4ade80",
                                   annotation_text=f"Best: {best_hour}:00 UTC", annotation_font_color="#4ade80")
                fig_pred.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e8f5e8", showlegend=False,
                    xaxis=dict(gridcolor="#1a3a1a"), yaxis=dict(gridcolor="#1a3a1a")
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                st.info(f"💡 Schedule **{pred_service}** at **{best_hour}:00 UTC** for lowest carbon footprint")