import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import io
import time
from datetime import datetime
from matplotlib import colors
from matplotlib.animation import FuncAnimation

st.set_page_config(page_title="Malaria ABM (Professional)", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Utilities & Defaults
# -----------------------
DEFAULT_LOCAL_PATH = "/mnt/data/malaria_climate_synthetic.csv"  # <-- Your uploaded data path (local)
st.markdown("#### Malaria ABM — Full Interactive Dashboard (A+)")
st.write("Use the sidebar to configure models, run/pause animation, and export results. If you are deploying to Streamlit Cloud, upload the CSV in the app UI.")

@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

def safe_load_data(uploader):
    if uploader is not None:
        return pd.read_csv(uploader)
    # Attempt local fallback (useful for local/Colab runs where file exists)
    try:
        df = load_csv_from_path(DEFAULT_LOCAL_PATH)
        st.info(f"Loaded dataset from local path: {DEFAULT_LOCAL_PATH}")
        return df
    except Exception as e:
        st.warning("No file uploaded and local fallback not found. Please upload your dataset.")
        return None

# -----------------------
# Sidebar (global settings)
# -----------------------
with st.sidebar:
    st.header("Model & UI Settings")
    # Model selection
    model_mode = st.selectbox("Mode", ["Interactive Demo (sliders)", "Use Uploaded Dataset / Local CSV"], index=1)
    show_agent_grid = st.checkbox("Show 2D Agent Grid (visualization)", value=True)
    show_SEIR_chart = st.checkbox("Show S/E/I/R time-series", value=True)
    download_option = st.checkbox("Enable Download Results", value=True)
    st.markdown("---")
    st.markdown("**Simulation Defaults**")
    default_population = st.number_input("Default population (agents)", min_value=100, max_value=20000, value=3000, step=100)
    default_days = st.slider("Simulation length (days)", 10, 365, 150)
    st.markdown("---")
    st.markdown("**Animation Settings**")
    animation_speed = st.slider("Animation delay (s)", 0.01, 1.5, 0.05)

# -----------------------
# Data Upload / Load
# -----------------------
st.sidebar.markdown("---")
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (malaria_climate_synthetic.csv)", type=["csv"])
df = safe_load_data(uploaded_file)

if df is not None:
    # Expect columns: date, rainfall_mm, temperature_c, malaria_cases, rain_norm, temp_norm
    df_columns = df.columns.tolist()
    st.sidebar.write(f"Columns: {df_columns}")
    # convert date if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

# -----------------------
# Layout: left controls, center visual, right metrics
# -----------------------
left_col, mid_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.subheader("Model Controls")

    population = st.number_input("Population (N)", min_value=100, max_value=20000, value=int(default_population), step=100)
    days = st.number_input("Days to simulate", min_value=10, max_value=365, value=int(default_days))
    base_prob = st.slider("Base infection probability (base)", 0.0, 0.2, 0.03)
    prog_prob = st.slider("E → I progression prob", 0.0, 0.5, 0.10)
    recovery_prob = st.slider("Recovery prob (I → R)", 0.0, 0.5, 0.05)
    immunity_loss_prob = st.slider("Immunity waning (R → S)", 0.0, 0.05, 0.001)

    st.markdown("---")
    st.subheader("Climate Influence (weights)")
    rain_weight = st.slider("Rainfall weight", 0.0, 3.0, 1.5)
    temp_weight = st.slider("Temperature weight", 0.0, 2.0, 0.5)

    st.markdown("---")
    st.subheader("Interactive Climate Sliders (for demo mode)")
    demo_rain = st.slider("Rain (normalized)", 0.0, 1.0, 0.5)
    demo_temp = st.slider("Temp (normalized)", 0.0, 1.0, 0.5)

    st.markdown("---")
    st.write("**Animation Controls**")
    if 'running' not in st.session_state:
        st.session_state.running = False
    if st.button("Start Simulation"):
        st.session_state.running = True
    if st.button("Pause Simulation"):
        st.session_state.running = False
    if st.button("Reset Simulation"):
        st.session_state.running = False
        if 'results' in st.session_state:
            del st.session_state['results']
        st.experimental_rerun()

with mid_col:
    st.subheader("Visualization")

    # tabs for different visualizations
    tabs = st.tabs(["Dashboard", "SEIR Details", "Agent Grid", "Climate Overlay"])
    tab_dashboard, tab_seir, tab_grid, tab_climate = tabs

    with tab_dashboard:
        st.write("Main infection curve output. Use controls to run the simulation.")

        placeholder_plot = st.empty()
        placeholder_metrics = st.empty()

    with tab_seir:
        seir_plot_placeholder = st.empty()

    with tab_grid:
        grid_plot_placeholder = st.empty()

    with tab_climate:
        climate_plot_placeholder = st.empty()

with right_col:
    st.subheader("Quick Metrics")
    metrics_area = st.empty()
    st.markdown("**Export**")
    # download button placeholder
    download_placeholder = st.empty()

# -----------------------
# ABM Implementation (optimized for display)
# -----------------------
class Agent:
    def __init__(self, uid, init_state="S"):
        self.uid = uid
        self.state = init_state
        self.days = 0

class SimpleABM:
    def __init__(self, N, base_prob, prog_prob, recovery_prob, immunity_loss_prob):
        self.N = N
        self.base_prob = base_prob
        self.prog_prob = prog_prob
        self.recovery_prob = recovery_prob
        self.immunity_loss_prob = immunity_loss_prob
        self.agents = [Agent(i, init_state=("I" if random.random() < 0.02 else "S")) for i in range(self.N)]
        self.history = {"S": [], "E": [], "I": [], "R": []}

    def step(self, infection_prob):
        # each agent acts
        for a in self.agents:
            if a.state == "S":
                if random.random() < infection_prob:
                    a.state = "E"
                    a.days = 0
            elif a.state == "E":
                a.days += 1
                if random.random() < self.prog_prob:
                    a.state = "I"
                    a.days = 0
            elif a.state == "I":
                a.days += 1
                if random.random() < self.recovery_prob:
                    a.state = "R"
                    a.days = 0
            elif a.state == "R":
                if random.random() < self.immunity_loss_prob:
                    a.state = "S"

        # record counts
        counts = {"S":0, "E":0, "I":0, "R":0}
        for a in self.agents:
            counts[a.state] += 1
        for k in counts:
            self.history[k].append(counts[k])
        return counts

# -----------------------
# Simulation runner (handles dataset vs demo mode)
# -----------------------
def run_simulation_demo(N, days, base_prob, prog_prob, recovery_prob, immunity_loss_prob,
                        rain_weight, temp_weight, demo_rain, demo_temp, show_grid, speed):
    model = SimpleABM(N, base_prob, prog_prob, recovery_prob, immunity_loss_prob)
    for t in range(days):
        # compute climate factor from sliders (demo mode, constant)
        factor = 1 + rain_weight * demo_rain + temp_weight * demo_temp
        infection_prob = base_prob * factor
        model.step(infection_prob)
        # streaming visualization handled outside
        yield t, infection_prob, model.history

def run_simulation_with_data(N, days, base_prob, prog_prob, recovery_prob, immunity_loss_prob,
                             rain_weight, temp_weight, data_df, show_grid, speed):
    model = SimpleABM(N, base_prob, prog_prob, recovery_prob, immunity_loss_prob)
    series_len = len(data_df)
    for t in range(days):
        i = t % series_len
        rain_val = data_df.iloc[i].get("rain_norm", 0.5) if data_df is not None else 0.5
        temp_val = data_df.iloc[i].get("temp_norm", 0.5) if data_df is not None else 0.5
        factor = 1 + rain_weight * rain_val + temp_weight * temp_val
        infection_prob = base_prob * factor
        model.step(infection_prob)
        yield t, infection_prob, model.history, rain_val, temp_val

# -----------------------
# Simulation Execution & Plotting loop
# -----------------------
def plot_seir(history, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))
    else:
        fig = ax.figure
    days = len(history['I'])
    x = np.arange(days)
    ax.plot(x, history['S'], label='Susceptible')
    ax.plot(x, history['E'], label='Exposed')
    ax.plot(x, history['I'], label='Infected')
    ax.plot(x, history['R'], label='Recovered')
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)
    return fig

def plot_infection_curve(history, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))
    else:
        fig = ax.figure
    x = np.arange(len(history['I']))
    ax.plot(x, history['I'], label='Infected', color='red')
    ax.set_xlabel("Day")
    ax.set_ylabel("Infected count")
    ax.legend()
    ax.grid(True)
    return fig

def draw_agent_grid(history, N, ax=None, size=50):
    # create a small grid with agent states (approximate)
    grid_side = int(np.ceil(np.sqrt(N)))
    grid = np.zeros((grid_side, grid_side), dtype=int)
    # map states to int
    state_map = {"S":0, "E":1, "I":2, "R":3}
    # use latest snapshot
    latest = {k: history[k][-1] if len(history[k])>0 else 0 for k in history}
    # fill grid: place counts in order (S, E, I, R)
    counts_list = []
    for k in ["S","E","I","R"]:
        counts_list += [k] * latest[k]
    # flatten and fill
    flat = counts_list + ["S"] * (grid_side*grid_side - len(counts_list))
    arr = np.array([state_map[v] if v in state_map else 0 for v in flat])
    arr = arr.reshape((grid_side, grid_side))
    cmap = colors.ListedColormap(['#7fbf7f','#ffc966','#ff6666','#66b3ff'])  # S, E, I, R colors
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    else:
        fig = ax.figure
    ax.imshow(arr, cmap=cmap, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

# -----------------------
# Core: run animation loop (non-blocking via session_state)
# -----------------------
if 'results' not in st.session_state:
    st.session_state['results'] = None

if st.session_state.running:
    # run continuously until paused or done
    placeholder_plot.empty()
    placeholder_metrics.empty()
    seir_plot_placeholder.empty()
    grid_plot_placeholder.empty()
    climate_plot_placeholder.empty()

    # choose the generator
    if model_mode.startswith("Interactive"):
        sim_gen = run_simulation_demo(population, days, base_prob, prog_prob, recovery_prob, immunity_loss_prob,
                                      rain_weight, temp_weight, demo_rain, demo_temp, show_agent_grid, animation_speed)
        for t, infection_prob, history in sim_gen:
            if not st.session_state.running:
                break
            # plots
            fig = plot_infection_curve(history)
            placeholder_plot.pyplot(fig)
            if show_SEIR_chart:
                fig2 = plot_seir(history)
                seir_plot_placeholder.pyplot(fig2)
            if show_agent_grid:
                fig3 = draw_agent_grid(history, population)
                grid_plot_placeholder.pyplot(fig3)
            # metrics
            latest_I = history['I'][-1] if len(history['I'])>0 else 0
            placeholder_metrics.metric("Day", t)
            placeholder_metrics.metric("Infected", int(latest_I))
            time.sleep(animation_speed)
    else:
        # use uploaded/local data
        if df is None:
            st.warning("Please upload dataset or choose Interactive Demo mode.")
        else:
            sim_gen = run_simulation_with_data(population, days, base_prob, prog_prob, recovery_prob, immunity_loss_prob,
                                              rain_weight, temp_weight, df, show_agent_grid, animation_speed)
            for t, infection_prob, history, rain_val, temp_val in sim_gen:
                if not st.session_state.running:
                    break
                fig = plot_infection_curve(history)
                placeholder_plot.pyplot(fig)
                if show_SEIR_chart:
                    fig2 = plot_seir(history)
                    seir_plot_placeholder.pyplot(fig2)
                if show_agent_grid:
                    fig3 = draw_agent_grid(history, population)
                    grid_plot_placeholder.pyplot(fig3)
                # climate plot
                figc, axc = plt.subplots(figsize=(6,2))
                axc.plot(df['rain_norm'].values[:t+1], label='rain_norm (partial)')
                axc.plot(df['temp_norm'].values[:t+1], label='temp_norm (partial)')
                axc.legend()
                climate_plot_placeholder.pyplot(figc)
                placeholder_metrics.metric("Day", t)
                latest_I = history['I'][-1] if len(history['I'])>0 else 0
                placeholder_metrics.metric("Infected", int(latest_I))
                time.sleep(animation_speed)

else:
    # not running – show initial static previews
    if st.session_state['results'] is None:
        st.info("Press 'Start Simulation' to run live animation or 'Run Simulation' button for single-run (below).")

# -----------------------
# Single-run (batch) button (non-interactive)
# -----------------------
if st.button("Run Simulation (single-run)"):
    st.write("Running single-run simulation (fast, non-animated)...")
    if model_mode.startswith("Interactive"):
        gen = run_simulation_demo(population, days, base_prob, prog_prob, recovery_prob, immunity_loss_prob,
                                 rain_weight, temp_weight, demo_rain, demo_temp, show_agent_grid, animation_speed)
        # collect final history
        last_history = None
        for t, infection_prob, history in gen:
            last_history = history
        fig = plot_infection_curve(last_history)
        placeholder_plot.pyplot(fig)
        if show_SEIR_chart:
            fig2 = plot_seir(last_history)
            seir_plot_placeholder.pyplot(fig2)
        if show_agent_grid:
            fig3 = draw_agent_grid(last_history, population)
            grid_plot_placeholder.pyplot(fig3)
        st.success("Single-run finished.")
        st.session_state['results'] = last_history
    else:
        if df is None:
            st.warning("Please upload dataset first.")
        else:
            gen = run_simulation_with_data(population, days, base_prob, prog_prob, recovery_prob, immunity_loss_prob,
                                           rain_weight, temp_weight, df, show_agent_grid, animation_speed)
            last_hist = None
            for t, infection_prob, history, rain_val, temp_val in gen:
                last_hist = history
            fig = plot_infection_curve(last_hist)
            placeholder_plot.pyplot(fig)
            if show_SEIR_chart:
                fig2 = plot_seir(last_hist)
                seir_plot_placeholder.pyplot(fig2)
            if show_agent_grid:
                fig3 = draw_agent_grid(last_hist, population)
                grid_plot_placeholder.pyplot(fig3)
            st.success("Single-run finished.")
            st.session_state['results'] = last_hist

# -----------------------
# Download results
# -----------------------
if download_option and st.session_state.get('results') is not None:
    hist = st.session_state['results']
    # convert to DataFrame
    df_out = pd.DataFrame({
        'S': hist['S'],
        'E': hist['E'],
        'I': hist['I'],
        'R': hist['R'],
        'day': np.arange(len(hist['I']))
    })
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download results (CSV)", csv, file_name=f"abm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# -----------------------
# Final notes
# -----------------------
st.sidebar.markdown("---")
st.sidebar.caption("App created by Nicole K. Rotich — DSA4900 Project")
st.caption("If you deploy to Streamlit Cloud, add a requirements.txt (streamlit, pandas, matplotlib, numpy).")
