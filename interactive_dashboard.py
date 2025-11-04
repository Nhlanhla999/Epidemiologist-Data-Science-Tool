import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="Epidemiologist Tool", layout="wide")
st.title("🦠 Epidemiologist Data Science Tool")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["📘 Learning", "🧪 Doing"])

# ====================================================
# 📘 LEARNING SECTION
# ====================================================
if section == "📘 Learning":
    st.header("📘 Learn Epidemiological Concepts")
    st.markdown("""
    Welcome to **Learning Mode**! Explore sample data, run experiments, 
    and see how changing parameters affects infection spread.
    """)

    # --- Experiment / Simulation ---
    st.subheader("🧪 Experiment / Simulation")

    # Parameters
    num_cases = st.slider("Number of simulated cases", 50, 1000, 200)
    infection_rate = st.slider("Infection rate (%)", 1, 100, 10)
    days = st.slider("Days to simulate", 10, 100, 30)

    mobile_clinic_infection_types = [
        "URTI", "Gastroenteritis", "Skin Infection", "UTI", "STI",
        "Eye Infection", "Ear Infection", "Minor Wound", "Parasitic Infection",
        "Influenza-like Illness", "Allergic Reaction", "Dental Infection",
        "Nutritional / Hygiene Condition", "Other"
    ]

    # Generate simulated data
    np.random.seed(42)
    data = pd.DataFrame({
        "Patient ID": range(1, num_cases + 1),
        "Age": np.random.randint(0, 90, size=num_cases),
        "Sex": np.random.choice(["M", "F"], size=num_cases),
        "Latitude": np.random.uniform(-30.0, -29.5, size=num_cases),
        "Longitude": np.random.uniform(25.5, 26.5, size=num_cases),
        "Infected": np.random.choice([0, 1], size=num_cases, p=[1 - infection_rate / 100, infection_rate / 100]),
        "Type of Infection": np.random.choice(mobile_clinic_infection_types, size=num_cases),
        "Diagnosis Date": pd.to_datetime('2025-10-01') + pd.to_timedelta(np.random.randint(0, days, size=num_cases), unit='d')
    })

    st.subheader("Sample Patient Data")
    st.dataframe(data.head())

    # Infections over time
    st.subheader("Infections Over Time")
    time_series = data.groupby("Diagnosis Date")["Infected"].sum().reset_index()
    chart = alt.Chart(time_series).mark_line(point=True).encode(
        x='Diagnosis Date:T', y='Infected:Q'
    )
    st.altair_chart(chart, use_container_width=True)

    # Heatmap for simulated infections
    st.subheader("🌍 Heatmap of Simulated Infections")
    heat_radius = st.slider("Heatmap Radius", 100, 2000, 500, key="sim_radius")
    heat_intensity = st.slider("Heatmap Intensity", 0.1, 5.0, 1.0, key="sim_intensity")

    layer = pdk.Layer(
        'HeatmapLayer',
        data=data[data['Infected'] == 1],
        get_position='[Longitude, Latitude]',
        aggregation='MEAN',
        radius=heat_radius,
        intensity=heat_intensity
    )

    view_state = pdk.ViewState(
        latitude=data["Latitude"].mean(),
        longitude=data["Longitude"].mean(),
        zoom=8
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))


# ====================================================
# 🧪 DOING SECTION
# ====================================================
elif section == "🧪 Doing":
    st.header("🧪 Mobile Clinic Operations")
    st.markdown("Upload CSV data to visualize infection spread and demographics.")

    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        required_cols = {"Latitude", "Longitude", "Diagnosis Date"}
        if not required_cols.issubset(data.columns):
            st.error(f"Missing required columns: {required_cols - set(data.columns)}")
            st.stop()

        data["Diagnosis Date"] = pd.to_datetime(data["Diagnosis Date"], errors="coerce")
        data = data.dropna(subset=["Diagnosis Date"])

        # Handle missing columns gracefully
        if "Type of Infection" not in data.columns:
            st.warning("⚠️ 'Type of Infection' column missing — using 'Unknown'.")
            data["Type of Infection"] = "Unknown"

        if "Age" not in data.columns:
            st.warning("⚠️ 'Age' column missing — generating random ages (0–90).")
            np.random.seed(42)
            data["Age"] = np.random.randint(0, 90, len(data))

        if "Sex" not in data.columns:
            st.warning("⚠️ 'Sex' column missing — assigning random M/F values.")
            data["Sex"] = np.random.choice(["M", "F"], len(data))

        # --- Infection summary ---
        st.subheader("🦠 Infection Summary by Type")
        infection_summary = data.groupby("Type of Infection")["Latitude"].count().reset_index().rename(columns={"Latitude": "Cases"})
        chart = alt.Chart(infection_summary).mark_bar().encode(
            x=alt.X("Type of Infection:N", sort='-y'),
            y="Cases:Q",
            color="Type of Infection:N",
            tooltip=["Type of Infection", "Cases"]
        )
        st.altair_chart(chart, use_container_width=True)

        # --- Filter by infection type ---
        st.subheader("🌡️ Filter by Infection Type")
        infection_types = ["All"] + sorted(data["Type of Infection"].unique())
        selected_type = st.selectbox("Select infection type", infection_types)

        if selected_type != "All":
            filtered_data = data[data["Type of Infection"] == selected_type]
        else:
            filtered_data = data

        st.write(f"Showing **{len(filtered_data)}** cases for **{selected_type}** infection type.")

        # --- Age Distribution ---
        st.subheader("📊 Age & Sex Distribution")
        if len(filtered_data) > 0:
            age_chart = alt.Chart(filtered_data).mark_bar().encode(
                x=alt.X("Age:Q", bin=alt.Bin(maxbins=15), title="Age"),
                y=alt.Y("count():Q", title="Number of Cases"),
                color=alt.Color("Sex:N", legend=alt.Legend(title="Sex")),
                tooltip=["Age", "Sex", "count()"]
            ).properties(title=f"Age and Sex Distribution for {selected_type}")
            st.altair_chart(age_chart, use_container_width=True)
        else:
            st.info("No data available for this infection type.")

        # --- Heatmap ---
        st.subheader("🗺️ Geographic Heatmap")
        heat_radius = st.slider("Heatmap Radius", 100, 2000, 500)
        heat_intensity = st.slider("Heatmap Intensity", 0.1, 5.0, 1.0)

        center_lat = filtered_data["Latitude"].mean()
        center_lon = filtered_data["Longitude"].mean()

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=filtered_data,
            get_position='[Longitude, Latitude]',
            radius=heat_radius,
            intensity=heat_intensity
        )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=8
        )

        st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state))

    else:
        st.info("Please upload a CSV file to begin analysis.")

