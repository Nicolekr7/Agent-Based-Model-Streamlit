**Malaria Outbreak Prediction Using Agent-Based Modeling (ABM)**
An Interactive Climate-Linked Simulation System for Kisumu County, Kenya

This repository contains the full implementation of an Agent-Based Modeling (ABM) system for simulating and predicting malaria outbreaks in Kisumu County, Kenya.
It includes data preprocessing, outbreak simulation, climate integration, visualization, and deployment through a fully interactive Streamlit dashboard.

**Project Overview**

Malaria remains a significant public health challenge in the Lake Victoria region. This project applies a climate-sensitive SEIR-based ABM to simulate malaria transmission dynamics at the micro-level, allowing users to explore how changes in rainfall, temperature, and human dynamics influence outbreak patterns.

The model uses:

SEIR agent transitions (Susceptible → Exposed → Infected → Recovered)

Climate-driven infection probability

Individual-level variation in exposure, recovery, and immunity

Visualization of emergent epidemic peaks

**Features**
**Fully Interactive Streamlit Dashboard**

Upload malaria–climate dataset

Visualize rainfall, temperature, malaria trends

Run baseline ABM or climate-linked ABM

Adjust model parameters in real-time

Watch animated epidemic curves

View 2D agent grid showing S/E/I/R state distribution

Generate and download outbreak simulation results

**Climate-Driven Infection Modeling**

Rainfall and temperature weighted transmission

Seasonal cyclic behavior

Effects of climate anomalies on outbreak timing

**Clean and Modern UI**

Fully tabbed layout

Real-time updates

Streamlit Cloud deployment support

**Dataset**

The model uses a synthetic weekly dataset (2015–2022) approximating:

Rainfall (mm)

Temperature (°C)

Malaria cases

Normalized rainfall & temperature

The dataset can be uploaded via the Streamlit UI or included in the repository.
