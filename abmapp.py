import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abm_model import MalariaModel     # <-- IMPORTANT: Uses your new model file

# ----------------------------------------
# CUSTOM CSS FOR BEAUTIFUL UI
# ----------------------------------------
custom_css = """
<style>
    .main { background-color: #F7FFF7; }
    
    h1, h2, h3 {
        color: #045A70;
        font-weight: 700;
    }

    .sidebar .sidebar-content {
        background-color: #E6F7FF;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background: white;
        border: 1px solid #cccccc50;
        margin-bottom: 10px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Tabs */
    .stTabs [role="tab"] {
        padding: 10px 20px;
        font-weight: bold;
        border-radius: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------------------
# HEADER
# ----------------------------------------
col1, col2 = st.columns([1,4])
with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Malaria_Vector_Mosquito.jpg/512px-Malaria_Vector_Mosquito.jpg",
        width=110
    )

with col2:
    st.title("🦟 Malaria Outbreak Simulation Dashboard")
    st.write("A climate-linked Agent-Based Model (ABM) for malaria prediction in Kisumu County")

# ----------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------
st.sidebar.header("⚙️ Simulation Controls")

uploaded_file = st.sidebar.file_uploader("Upload Climate Dataset (CSV)", type=["csv"])

population_size = st.sidebar.slider("Population Size (Agents)", 500, 10000, 3000)
simulation_days = st.sidebar.slider("Simulation Days", 30, 365, 150)

run_button = st.sidebar.button("▶ Run Simulation")

# ----------------------------------------
# TABS
# ----------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Detailed Outputs", "🧬 Model Description"])

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ["rainfall_mm", "temperature_c", "rain_norm", "temp_norm"]

    if not all(col in df.columns for col in required_cols):
        st.error("❌ The uploaded file is missing required columns!")
        st.stop()
    else:
        st.success("✅ Dataset loaded successfully!")

        with tab1:
            st.subheader("🌦 Climate Dataset Overview")
            st.dataframe(df.head())

            fig, ax = plt.subplots(figsize=(9,3))
            ax.plot(df["rainfall_mm"], label="Rainfall (mm)")
            ax.plot(df["temperature_c"], label="Temperature (°C)")
            ax.legend()
            st.pyplot(fig)

# ----------------------------------------
# RUN MODEL
# ----------------------------------------
if run_button and uploaded_file:

    df = pd.read_csv(uploaded_file)

    rainfall_series = df["rain_norm"].values
    temp_series = df["temp_norm"].values

    model = MalariaModel(population_size)

    infected_counts = []

    for day in range(simulation_days):
        rain = rainfall_series[day % len(rainfall_series)]
        temp = temp_series[day % len(temp_series)]

        model.step(rain, temp)
        infected_counts.append(model.count_state("I"))

    # -----------------------------
    # TAB 1 — MAIN DASHBOARD
    # -----------------------------
    with tab1:
        st.subheader("📈 Infection Over Time")

        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(infected_counts, color="red", linewidth=2)
        ax.set_title("Daily Infected Population Over Time", fontsize=14)
        st.pyplot(fig)

        st.markdown("### 🔎 Key Metrics")
        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Peak Infection", max(infected_counts))

        with colB:
            st.metric("Day of Peak", int(np.argmax(infected_counts)))

        with colC:
            st.metric("Final Day Infected", infected_counts[-1])

    # -----------------------------
    # TAB 2 — DETAILED OUTPUTS
    # -----------------------------
    with tab2:
        st.subheader("📈 Detailed Simulation Data")

        df_output = pd.DataFrame({
            "Day": list(range(simulation_days)),
            "Infected": infected_counts
        })

        st.dataframe(df_output)

        st.download_button(
            "⬇ Download Output CSV",
            data=df_output.to_csv(index=False),
            file_name="malaria_abm_output.csv",
            mime="text/csv"
        )

    # -----------------------------
    # TAB 3 — MODEL DESCRIPTION
    # -----------------------------
    with tab3:
        st.subheader("🧬 Model Structure & Logic")

        st.markdown("""
        ### **SEIR Agent-Based Model**
        The model simulates malaria infection using four human states:

        - **S = Susceptible**  
        - **E = Exposed**  
        - **I = Infected**  
        - **R = Recovered**  

        ### **Climate-Linked Infection Probability**
        Infection probability increases when rainfall and temperature rise:

        ```
        P = 0.03 × (1 + 1.5 × rain_norm + 0.5 × temp_norm)
        ```

        ### **Why ABM?**
        - Captures individual-level variation  
        - Models nonlinear outbreak dynamics  
        - Accounts for changing climate patterns  
        - Simulates real-world malaria transmission more accurately  
        """)

else:
    with tab1:
        st.info("👈 Upload a dataset and press **Run Simulation** to begin.")
