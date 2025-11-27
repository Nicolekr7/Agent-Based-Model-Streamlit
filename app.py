import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Malaria ABM Outbreak Simulator", layout="wide")
st.title("🌍 Malaria Outbreak Prediction Using Agent-Based Modeling (ABM)")
st.write("This dashboard allows you to simulate malaria outbreaks using a Baseline SEIR-ABM and a Climate-Linked ABM model.")


# =========================
# LOAD DATA SECTION
# =========================
st.header("📂 Upload Dataset")

file = st.file_uploader("Upload your malaria_climate_synthetic.csv", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.success("Dataset uploaded successfully!")
    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # =========================
    # VISUALIZATION SECTION
    # =========================
    st.header("📈 Exploratory Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rainfall (mm) vs Malaria Cases")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["rainfall_mm"], label="Rainfall (mm)")
        ax1.plot(df["malaria_cases"], label="Malaria Cases")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.subheader("Temperature (°C)")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["temperature_c"], color="orange", label="Temperature (°C)")
        ax2.legend()
        st.pyplot(fig2)


    # =========================
    # SIDEBAR PARAMETERS
    # =========================
    st.sidebar.header("⚙️ Simulation Settings")

    N = st.sidebar.slider("Population Size (Agents)", 100, 10000, 3000)
    steps = st.sidebar.slider("Simulation Duration (Days)", 50, 300, 150)

    base_infection_prob = st.sidebar.slider("Base Infection Probability", 0.0, 0.2, 0.03)
    prog_prob = st.sidebar.slider("Exposure → Infection Progression Rate", 0.0, 0.5, 0.10)
    recovery_prob = st.sidebar.slider("Recovery Probability", 0.0, 0.2, 0.05)
    immunity_loss_prob = st.sidebar.slider("Immunity Loss Probability", 0.0, 0.01, 0.001)

    st.sidebar.markdown("---")


    # =========================
    # ABM CLASSES
    # =========================
    class HumanAgent:
        def __init__(self, unique_id):
            self.id = unique_id
            self.state = "S"

        def step(self, infection_prob, prog_prob, recovery_prob, immunity_loss_prob):
            if self.state == "S":
                if random.random() < infection_prob:
                    self.state = "E"

            elif self.state == "E":
                if random.random() < prog_prob:
                    self.state = "I"

            elif self.state == "I":
                if random.random() < recovery_prob:
                    self.state = "R"

            elif self.state == "R":
                if random.random() < immunity_loss_prob:
                    self.state = "S"


    class MalariaModel:
        def __init__(self, N):
            self.agents = [HumanAgent(i) for i in range(N)]

        def step(self, infection_prob, prog_prob, recovery_prob, immunity_loss_prob):
            for agent in self.agents:
                agent.step(infection_prob, prog_prob, recovery_prob, immunity_loss_prob)

        def count_infected(self):
            return sum(agent.state == "I" for agent in self.agents)


    # =========================
    # BUTTON: RUN SIMULATIONS
    # =========================
    if st.button("🚀 Run ABM Simulations"):

        st.header("📊 Simulation Results")

        # ========= BASELINE ABM =========
        baseline_model = MalariaModel(N)
        baseline_infected = []

        for t in range(steps):
            baseline_model.step(base_infection_prob, prog_prob, recovery_prob, immunity_loss_prob)
            baseline_infected.append(baseline_model.count_infected())

        # ========= CLIMATE-LINKED ABM =========
        climate_model = MalariaModel(N)
        climate_infected = []

        rain_series = df["rain_norm"].values
        temp_series = df["temp_norm"].values

        for t in range(steps):
            climate_factor = (
                1 + 1.5 * rain_series[t % len(rain_series)] +
                0.5 * temp_series[t % len(temp_series)]
            )
            climate_infection_prob = base_infection_prob * climate_factor

            climate_model.step(
                climate_infection_prob, prog_prob, recovery_prob, immunity_loss_prob
            )
            climate_infected.append(climate_model.count_infected())


        # =========================
        # DISPLAY RESULTS
        # =========================
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(baseline_infected, label="Baseline ABM")
        ax.plot(climate_infected, label="Climate-Linked ABM")
        ax.set_title("Malaria Outbreak Simulation")
        ax.set_xlabel("Day")
        ax.set_ylabel("Number of Infected Agents")
        ax.legend()

        st.pyplot(fig)

        st.success("Simulation completed successfully!")


else:
    st.info("Please upload your CSV dataset to continue.")
