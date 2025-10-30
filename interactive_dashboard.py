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

    # Mobile clinic infection types
    mobile_clinic_infection_types = [
     "Waterborne Infections", "Vector-Borne Diseases", "Respiratory Infections", 
    "Gastrointestinal Infections", "Skin Infections", "Trauma/Injuries", 
    "Chronic Conditions", "Nutritional Deficiencies", "Vaccine-Preventable Diseases", 
    "Hygiene and Sanitation-Related Issues", "Other"
    ]

    # Generate simulated mobile clinic data
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
    line_chart = alt.Chart(time_series).mark_line(point=True).encode(
        x=alt.X('Diagnosis Date:T'),
        y=alt.Y('Infected:Q')
    )
    st.altair_chart(line_chart, use_container_width=True)

    # Age & Sex Distribution
    st.subheader("Age Distribution of Infected Cases")
    age_chart = alt.Chart(data[data['Infected'] == 1]).mark_bar().encode(
        x=alt.X('Age:Q', bin=True),
        y=alt.Y('count():Q'),
        color=alt.Color('Sex:N')
    )
    st.altair_chart(age_chart, use_container_width=True)

    # Heatmap for simulated data
    st.subheader("🌍 Heatmap of Simulated Infections")
    heat_radius = st.slider("Heatmap Radius", 100, 2000, 500, key="sim_radius")
    heat_intensity = st.slider("Heatmap Intensity", 0.1, 5.0, 1.0, key="sim_intensity")

    layer = pdk.Layer(
        'HeatmapLayer',
        data=data[data['Infected']==1],
        get_position='[Longitude, Latitude]',
        aggregation='MEAN',
        radius=heat_radius,
        intensity=heat_intensity,
        threshold=0.05
    )

    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=data['Latitude'].mean(),
            longitude=data['Longitude'].mean(),
            zoom=8,
            pitch=0
        ),
        layers=[layer],
        tooltip={"text": "Type: {Type of Infection}\nLat: {Latitude}\nLon: {Longitude}"}
    )

    st.pydeck_chart(deck)
    st.info("💡 Adjust sliders above to simulate different numbers of cases, infection rates, and heatmap settings.")


# ====================================================
# 🧪 DOING SECTION (Daily Infection Type + Heatmap)
# ====================================================
elif section == "🧪 Doing":
    st.header("🧪 Mobile Clinic Operations")
    st.markdown("""
    Upload CSV data to visualize infection spread in the field.
    You can see infections by type over time and on the heatmap.
    """)

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader(
        "Upload CSV from health clinic teams (must include Latitude, Longitude, Diagnosis Date, optional Type of Infection)",
        type=["csv"]
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    if st.session_state.uploaded_file is not None:
        st.session_state.uploaded_file.seek(0)
        try:
            data = pd.read_csv(st.session_state.uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            st.error("❌ Uploaded file is empty or invalid CSV.")
            st.stop()

        required_cols = {"Latitude", "Longitude", "Diagnosis Date"}
        if not required_cols.issubset(data.columns):
            st.error(f"❌ Missing required columns: {required_cols - set(data.columns)}")
            st.stop()

        data["Diagnosis Date"] = pd.to_datetime(data["Diagnosis Date"], errors='coerce')
        data = data.dropna(subset=["Diagnosis Date"])

        if "Type of Infection" not in data.columns:
            st.warning("⚠️ No 'Type of Infection' column found — assigning 'Unknown'.")
            data["Type of Infection"] = "Unknown"

        st.subheader("🦠 Infection Summary by Type")
        infection_summary = data.groupby("Type of Infection")["Latitude"].count().reset_index().rename(columns={"Latitude":"Cases"})
        infection_chart = alt.Chart(infection_summary).mark_bar().encode(
            x=alt.X("Type of Infection:N", sort='-y'),
            y=alt.Y("Cases:Q"),
            color="Type of Infection:N",
            tooltip=["Type of Infection", "Cases"]
        )
        st.altair_chart(infection_chart, use_container_width=True)

        # --- Infection Over Time by Type (Daily) ---
        st.subheader("📅 Infections Over Time by Type")
        daily_type_summary = data.groupby(['Diagnosis Date', 'Type of Infection'])['Latitude'].count().reset_index().rename(columns={'Latitude':'Cases'})
        daily_type_chart = alt.Chart(daily_type_summary).mark_bar().encode(
            x=alt.X('Diagnosis Date:T', title="Date"),
            y=alt.Y('Cases:Q'),
            color=alt.Color('Type of Infection:N', legend=alt.Legend(title="Infection Type")),
            tooltip=['Diagnosis Date', 'Type of Infection', 'Cases']
        )
        st.altair_chart(daily_type_chart, use_container_width=True)

        # --- Heatmap Filter by Infection Type ---
        st.subheader("🌡️ Filter Heatmap by Infection Type")
        infection_types = ["All"] + sorted(data["Type of Infection"].unique())
        selected_type = st.selectbox("Select Infection Type", infection_types)

        if selected_type != "All":
            filtered_data = data[data["Type of Infection"] == selected_type]
        else:
            filtered_data = data

        st.write(f"Showing {len(filtered_data)} cases on heatmap.")

        heat_radius = st.slider("Heatmap Radius", 100, 2000, 500)
        heat_intensity = st.slider("Heatmap Intensity", 0.1, 5.0, 1.0)

        if len(filtered_data) > 0:
            center_lat = filtered_data['Latitude'].mean()
            center_lon = filtered_data['Longitude'].mean()
        else:
            center_lat, center_lon = -29.75, 26.0

        layer = pdk.Layer(
            'HeatmapLayer',
            data=filtered_data,
            get_position='[Longitude, Latitude]',
            aggregation='MEAN',
            radius=heat_radius,
            intensity=heat_intensity,
            threshold=0.05
        )

        deck = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=8,
                pitch=0
            ),
            layers=[layer],
            tooltip={"text": "Type: {Type of Infection}\nLat: {Latitude}\nLon: {Longitude}"}
        )

        st.pydeck_chart(deck)

    else:
        st.info("ℹ️ Upload a CSV to view heatmap and charts.")

