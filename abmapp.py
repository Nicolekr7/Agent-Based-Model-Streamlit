# ======================================================
#  IMPORTS
# ======================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation

# ======================================================
#  PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Malaria ABM Simulation",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
#  HUMAN AGENT CLASS
# ======================================================
class HumanAgent(Agent):
    def __init__(self, unique_id, model, infection_state=0):
        super().__init__(unique_id, model)
        self.infection_state = infection_state  # 0=S,1=E,2=I,3=R
        self.days_in_state = 0

    def step(self):
        climate_factor = (self.model.current_temp + self.model.current_rain) / 2

        base_infection_probability = 0.03
        infection_probability = base_infection_probability + (0.10 * climate_factor)

        # Exposure
        if self.infection_state == 0 and np.random.random() < infection_probability:
            self.infection_state = 1
            self.days_in_state = 0

        # Progression E → I
        elif self.infection_state == 1 and self.days_in_state >= 7:
            self.infection_state = 2
            self.days_in_state = 0

        # Recovery I → R
        elif self.infection_state == 2 and self.days_in_state >= 14:
            self.infection_state = 3
            self.days_in_state = 0

        # Loss of immunity R → S
        elif self.infection_state == 3 and self.days_in_state >= 30:
            self.infection_state = 0
            self.days_in_state = 0

        self.days_in_state += 1

# ======================================================
#  MALARIA MODEL
# ======================================================
class MalariaModel(Model):
    def __init__(self, N, rainfall_series, temp_series):
        super().__init__()
        self.num_agents = N
        self.schedule = RandomActivation(self)

        self.rainfall_series = rainfall_series
        self.temp_series = temp_series
        self.current_day = 0

        for i in range(self.num_agents):
            a = HumanAgent(i, self)
            self.schedule.add(a)

    def step(self):
        if self.current_day < len(self.rainfall_series):
            self.current_rain = self.rainfall_series[self.current_day]
            self.current_temp = self.temp_series[self.current_day]
        else:
            self.current_rain = 0
            self.current_temp = 0

        self.schedule.step()
        self.current_day += 1

    def count_infected(self):
        return sum([1 for a in self.schedule.agents if a.infection_state == 2])


# ======================================================
#  FUNCTION: RUN SIMULATION
# ======================================================
def run_simulation(N, steps, rain, temp):
    model = MalariaModel(N, rain, temp)
    infected_list = []

    for i in range(steps):
        model.step()
        infected_list.append(model.count_infected())

    return infected_list


# ======================================================
#  APP HEADER
# ======================================================
st.markdown(
    """
    <h1 style='text-align:center; color:#b30000;'>🦟 Malaria Outbreak Simulation Using Agent-Based Modeling</h1>
    <p style='text-align:center;'>Interactive ABM simulation driven by rainfall and temperature data</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ======================================================
#  SIDEBAR
# ======================================================
st.sidebar.title("🔧 Simulation Controls")

uploaded = st.sidebar.file_uploader("Upload Climate Dataset (CSV)", type=["csv"])

population = st.sidebar.slider("Population Size (Agents)", 100, 5000, 1000, step=100)
steps = st.sidebar.slider("Simulation Days", 30, 365, 120, step=10)

run_button = st.sidebar.button("▶ Run Simulation")

# ======================================================
#  MAIN LOGIC
# ======================================================
if uploaded is not None:
    df = pd.read_csv(uploaded)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    if run_button:

        with st.spinner("Running simulation… Please wait ⏳"):
            # Normalized values
            rain = df["rain_norm"].values
            temp = df["temp_norm"].values

            infected_curve = run_simulation(population, steps, rain, temp)

        st.success("Simulation complete!")

        # ======================================================
        #  PLOT RESULTS
        # ======================================================
        st.subheader("📈 Infection Trend Over Time")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(infected_curve, linewidth=2, label="Infected")
        ax.set_xlabel("Day")
        ax.set_ylabel("Number of Infected Individuals")
        ax.set_title("Climate-Driven Malaria Simulation")
        ax.grid(True)

        st.pyplot(fig)

else:
    st.info("Please upload a dataset to begin.")

