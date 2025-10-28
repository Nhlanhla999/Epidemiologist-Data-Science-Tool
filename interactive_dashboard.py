import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import time
from datetime import timedelta, datetime

# --- Page Setup ---
st.set_page_config(page_title="Epidemiologist Data Science Tool", layout="wide")
st.title("🦠 Epidemiologist Data Science Tool")

# --- 1. Simulation Settings ---
st.sidebar.header("Simulation Settings")
num_cases = st.sidebar.slider("Number of simulated cases", 50, 1000, 200)
infection_rate = st.sidebar.slider("Infection rate (%)", 1, 100, 10)

np.random.seed(42)
data = pd.DataFrame({
    "Patient ID": range(1, num_cases+1),
    "Age": np.random.randint(0, 90, size=num_cases),
    "Sex": np.random.choice(["M", "F"], size=num_cases),
    "Latitude": np.random.uniform(-30.0, -29.5, size=num_cases),
    "Longitude": np.random.uniform(25.5, 26.5, size=num_cases),
    "Infected": np.random.choice([0,1], size=num_cases, p=[1-infection_rate/100, infection_rate/100]),
    "Diagnosis Date": pd.to_datetime('2025-10-01') + pd.to_timedelta(np.random.randint(0, 30, size=num_cases), unit='d')
})

st.subheader("Sample Data")
st.dataframe(data.head())

# --- 2. Infections Over Time ---
st.subheader("Infections Over Time")
time_series = data.groupby("Diagnosis Date")["Infected"].sum().reset_index()
line_chart = alt.Chart(time_series).mark_line(point=True).encode(
    x=alt.X('Diagnosis Date:T'),
    y=alt.Y('Infected:Q')
)
st.altair_chart(line_chart, use_container_width=True)

# --- 3. Age & Sex Distribution ---
st.subheader("Age Distribution of Infected Cases")
age_chart = alt.Chart(data[data['Infected']==1]).mark_bar().encode(
    x=alt.X('Age:Q', bin=True),
    y=alt.Y('count():Q'),
    color=alt.Color('Sex:N')
)
st.altair_chart(age_chart, use_container_width=True)

# --- 4. Predictive SIR Model ---
st.subheader("Predictive Outbreak Simulation (SIR Model)")
population = st.slider("Population Size", 1000, 100000, 5000)
beta = st.slider("Infection Rate (β)", 0.1, 1.0, 0.3)
gamma = st.slider("Recovery Rate (γ)", 0.01, 0.5, 0.1)
days = st.slider("Days to Simulate", 10, 100, 30)

S = population - 1
I = 1
R = 0
sir_history = []
for day in range(days):
    new_infected = beta * S * I / population
    new_recovered = gamma * I
    S -= new_infected
    I += new_infected - new_recovered
    R += new_recovered
    sir_history.append({"Day": day, "Susceptible": S, "Infected": I, "Recovered": R})

sir_df = pd.DataFrame(sir_history)
sir_chart = alt.Chart(sir_df).transform_fold(
    ['Susceptible', 'Infected', 'Recovered'],
    as_=['Category', 'Count']
).mark_line().encode(
    x=alt.X('Day:Q'),
    y=alt.Y('Count:Q'),
    color=alt.Color('Category:N')
)
st.altair_chart(sir_chart, use_container_width=True)

# --- 5. Control Measure Simulation ---
st.subheader("Apply Control Measures")
vaccination_rate = st.slider("Vaccinated Population (%)", 0, 100, 20)
social_distancing = st.slider("Reduce Infection Rate (%)", 0, 100, 30)
effective_beta = beta * (1 - social_distancing/100)
initial_infected = I
S_control = population * (1 - vaccination_rate/100)
I_control = initial_infected
R_control = population - S_control - I_control

control_history = []
for day in range(days):
    new_infected = effective_beta * S_control * I_control / population
    new_recovered = gamma * I_control
    S_control -= new_infected
    I_control += new_infected - new_recovered
    R_control += new_recovered
    control_history.append({"Day": day, "Susceptible": S_control, "Infected": I_control, "Recovered": R_control})

control_df = pd.DataFrame(control_history)
control_chart = alt.Chart(control_df).transform_fold(
    ['Susceptible', 'Infected', 'Recovered'],
    as_=['Category', 'Count']
).mark_line().encode(
    x=alt.X('Day:Q'),
    y=alt.Y('Count:Q'),
    color=alt.Color('Category:N')
)
st.altair_chart(control_chart, use_container_width=True)

# --- 6. Real-Time Simulation with Clusters & Heatmap (Data Scientist) ---
st.markdown("### 💻 Real-Time Outbreak Simulation with Clusters & Heatmap (For Data Scientists)")

simulate_days = st.slider("Days to Simulate in Real-Time", 5, 30, 15)
speed = st.slider("Animation Speed (seconds per day)", 0.1, 2.0, 0.5)
heat_radius = st.slider("Heatmap Radius", 100, 2000, 500)
heat_intensity = st.slider("Heatmap Intensity", 0.1, 5.0, 1.0)
cluster_threshold = st.slider("Cluster Size Threshold", 5, 50, 10)

infected_sim = data.copy()
infected_sim['Day'] = (infected_sim['Diagnosis Date'] - infected_sim['Diagnosis Date'].min()).dt.days
infected_sim['Status'] = np.where(np.random.rand(len(infected_sim)) < vaccination_rate/100, "Vaccinated", "Susceptible")
infected_sim.loc[infected_sim['Infected']==1, 'Status'] = "Infected"

placeholder_map = st.empty()
placeholder_chart = st.empty()

for day in range(simulate_days):
    daily_data = infected_sim[infected_sim['Day'] <= day].copy()
    
    # Update infected → recovered
    daily_data.loc[daily_data['Status']=="Infected", 'Status'] = np.where(
        np.random.rand(len(daily_data[daily_data['Status']=="Infected"])) < gamma, "Recovered", "Infected"
    )
    
    # Cluster detection
    daily_data['lat_bin'] = (daily_data['Latitude'] * 100).astype(int)
    daily_data['lon_bin'] = (daily_data['Longitude'] * 100).astype(int)
    clusters = daily_data.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='Count')
    alert_clusters = clusters[clusters['Count'] >= cluster_threshold]

    # PyDeck Layers
    layers = [
        # Infected
        pdk.Layer(
            'ScatterplotLayer',
            data=daily_data[daily_data['Status']=="Infected"],
            get_position='[Longitude, Latitude]',
            get_color='[255, 0, 0, 160]',
            get_radius=500,
            pickable=True,
        ),
        # Recovered
        pdk.Layer(
            'ScatterplotLayer',
            data=daily_data[daily_data['Status']=="Recovered"],
            get_position='[Longitude, Latitude]',
            get_color='[0, 255, 0, 160]',
            get_radius=500,
            pickable=True,
        ),
        # Vaccinated
        pdk.Layer(
            'ScatterplotLayer',
            data=daily_data[daily_data['Status']=="Vaccinated"],
            get_position='[Longitude, Latitude]',
            get_color='[0, 0, 255, 160]',
            get_radius=500,
            pickable=True,
        ),
        # Heatmap Layer
        pdk.Layer(
            'HeatmapLayer',
            data=daily_data[daily_data['Status']=="Infected"],
            get_position='[Longitude, Latitude]',
            aggregation='MEAN',
            radius=heat_radius,
            intensity=heat_intensity,
            threshold=0.05,
        )
    ]

    # Egg-like cluster effect (concentric halos)
    for idx, row in alert_clusters.iterrows():
        center = [row['lon_bin']/100, row['lat_bin']/100]
        halo_colors = [(255,140,0,200), (255,165,0,120), (255,215,0,80)]
        halo_radii = [1000, 1500, 2000]
        for color, radius in zip(halo_colors, halo_radii):
            layers.append(
                pdk.Layer(
                    'ScatterplotLayer',
                    data=pd.DataFrame([{'Longitude': center[0], 'Latitude': center[1]}]),
                    get_position='[Longitude, Latitude]',
                    get_color=f"[{color[0]}, {color[1]}, {color[2]}, {color[3]}]",
                    get_radius=radius,
                    pickable=False
                )
            )

    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=daily_data['Latitude'].mean(),
            longitude=daily_data['Longitude'].mean(),
            zoom=8,
            pitch=0,
        ),
        layers=layers,
        tooltip={"text": "Status: {Status}\nLat: {Latitude}\nLon: {Longitude}"}
    )

    placeholder_map.pydeck_chart(deck)
    
    # Cumulative infections chart
    time_series_sim = daily_data.groupby("Diagnosis Date")["Infected"].sum().cumsum().reset_index()
    line_chart_sim = alt.Chart(time_series_sim).mark_line(point=True).encode(
        x=alt.X('Diagnosis Date:T'),
        y=alt.Y('Infected:Q')
    )
    placeholder_chart.altair_chart(line_chart_sim, use_container_width=True)
    
    time.sleep(speed)
