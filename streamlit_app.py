import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import os
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from folium import plugins

# Page configuration
st.set_page_config(page_title="Lisbon Road Accidents", layout="wide")

# Title and description
st.title("Lisbon Road Accidents: Interactive Dashboard")
st.markdown(
    "Explore and analyze road accident data from Lisbon."
)

# Educational use notice
st.markdown(
    """
> ⚠️ **Important Notice**  
> This dataset contains real road accident records from Portugal in 2023, provided by **ANSR (National Road Safety Authority)**.  
> It is intended **exclusively for use in the final project of the course "Tools and Techniques for Geospatial ML"**. 
> Redistribution or use for any other purpose is strictly prohibited.
"""
)

# Load dataset
df = pd.read_csv("data/Road_Accidents_Lisbon.csv")

# Sidebar filter by weekday
st.sidebar.header("Filter Options")
weekday_options = df["weekday"].dropna().unique()
selected_weekdays = st.sidebar.multiselect("Filter by Weekday", weekday_options, default=weekday_options)

# Filter dataset
df_filtered = df[df["weekday"].isin(selected_weekdays)]

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_filtered,
    geometry=[Point(xy) for xy in zip(df_filtered["longitude"], df_filtered["latitude"])],
    crs="EPSG:4326"
)

# Center map on Lisbon
center = [gdf["latitude"].mean(), gdf["longitude"].mean()]
m = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")

# Add accident markers
for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=4,
        color="red",
        fill=True,
        fill_opacity=0.6,
        popup=f"ID: {row['id']}<br>Weekday: {row['weekday']}<br>Fatalities: {row['fatalities_30d']}"
    ).add_to(m)

# Map
st.header("Accident Map")
st_data = st_folium(m, width=800, height=600)


# 1) Spatial Analysis ---------------------------------------------------------------------------------------------------

st.subheader("1) Spatial Analysis")

# 1.1) Densit (Heatmap) ---------------------------------------------------------------------------------------------------

df = gdf.copy()  

st.subheader("1.1) Occurrence density map")

needed = ["latitude", "longitude"]
if not all(c in df.columns for c in needed):
    st.error("Missing columns (lat/log)")
else:
    df = df.dropna(subset=["latitude","longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude","longitude"])

    if df.empty:
        st.info("No valid points to show on the map")
    else:
        center = [float(df["latitude"].mean()), float(df["longitude"].mean())]

        st.sidebar.markdown("### Heatmap weight")
        use_weights = st.sidebar.checkbox("Weights (injuries)", value=False)

        if use_weights:
            for c in ["fatalities_30d","serious_injuries_30d","minor_injuries_30d"]:
                if c not in df.columns:
                    df[c] = 0
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            df["weight"] = (
                df["fatalities_30d"]*3   
                + df["serious_injuries_30d"]*2
                + df["minor_injuries_30d"]*1
            )

            if df["weight"].max() > 0:
                df["weight"] = df["weight"] / df["weight"].max()
            else:
                df["weight"] = 0.0

            heat_data = df[["latitude","longitude","weight"]].values.tolist()
        else:
            heat_data = df[["latitude","longitude"]].values.tolist()

        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")

        # HeatMap 
        plugins.HeatMap(
            heat_data,
            radius=14,       
            blur=18,         
            max_zoom=15,     
            min_opacity=0.1, 
        ).add_to(m)

        st_folium(m, height=600, width=None)

# Explanation
st.markdown(
    """
> The analysis of the accident density map made it possible to identify a relevant spatial pattern, with the orange/yellow area standing out as one of the zones with the highest concentration of occurrences. 
> This result highlights a critical point in terms of road safety, which could benefit from specific prevention and risk mitigation measures.
"""
)

# 1.2) Clusters (DBSCAN) ---------------------------------------------------------------------------------------------------

st.subheader("1.2) Clusters")

gdf_meters = gdf.to_crs(epsg=3763)
coords = np.array(list(zip(gdf_meters.geometry.x, gdf_meters.geometry.y)))

# Run DBSCAN: 200m radius, min 2 points per cluster
db = DBSCAN(eps=200, min_samples=2, metric='euclidean').fit(coords)
gdf["Cluster"] = db.labels_.astype(str)

# Check result
gdf["Cluster"].value_counts()

# Preview
gdf.value_counts()

# Clusters summary
cluster_counts = gdf["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]

# Map
st.subheader("Identification of accident clusters with DBSCAN")

# Explanation
st.markdown(
    """
> The identification of critical areas was carried out using the **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) algorithm, which groups nearby points based on density. The method enables the detection of accident concentrations (hotspots) and distinguishes isolated points as noise, making it suitable for irregular spatial patterns.
> Accident points in Lisbon that are less than **200m** apart were grouped together.
"""
)
st.dataframe(cluster_counts)

lisbon_center = [gdf["latitude"].mean(), gdf["longitude"].mean()]
map = folium.Map(location=lisbon_center, zoom_start=14, tiles="CartoDB Positron")

import matplotlib.cm as cm
import matplotlib.colors as colors
unique_clusters = sorted(gdf["Cluster"].unique())
cmap = cm.get_cmap('Set1', len(unique_clusters))
cluster_colors = {cluster: colors.rgb2hex(cmap(i)) for i, cluster in enumerate(unique_clusters)}

for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=6,
        color=cluster_colors[row["Cluster"]],
        fill=True,
        fill_opacity=0.7,
        popup=f"ID: {row['id']}<br>Cluster: {row['Cluster']}"
    ).add_to(map)


# Show map

# 1.2.1) See top 3 to the clusters ------------------------------------------------------------------------------------------

st.subheader("1.2.1) Top 3 Clusters")

gdf = gdf.copy()
gdf["Cluster"] = pd.to_numeric(gdf["Cluster"], errors="coerce")

cluster_counts_no_noise = (
    gdf.loc[(gdf["Cluster"].notna()) & (gdf["Cluster"] >= 0)]
      .groupby("Cluster")
      .size()
      .sort_values(ascending=False)
      .reset_index(name="Count")
)

top_n = 3
top_clusters = cluster_counts_no_noise.head(top_n)["Cluster"].tolist()

st.subheader("🔝 Top 3 clusters (without noise)")
st.dataframe(cluster_counts_no_noise.head(top_n), use_container_width=True)

if len(top_clusters) == 0:
    st.info("No clusters")
else:
    gdf_top = gdf[gdf["Cluster"].isin(top_clusters)].copy()

    center = [
        float(gdf_top["latitude"].mean()) if not gdf_top.empty else 38.7169,
        float(gdf_top["longitude"].mean()) if not gdf_top.empty else -9.1390,
    ]


    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Positron")

    cmap = cm.get_cmap("Set1", max(len(top_clusters), 1))
    cluster_colors = {cl: colors.rgb2hex(cmap(i)) for i, cl in enumerate(top_clusters)}

    for cl in top_clusters:
        sub = gdf_top[gdf_top["Cluster"] == cl]
        count = len(sub)
        color = cluster_colors[cl]

        fg = folium.FeatureGroup(name=f"Cluster {int(cl)} ({count})", show=True)

        for _, row in sub.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"ID: {row.get('id','?')}<br>Cluster: {int(cl)}",
            ).add_to(fg)

        cent_lat = float(sub["latitude"].mean())
        cent_lon = float(sub["longitude"].mean())
        folium.Marker(
            [cent_lat, cent_lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="font-weight:700; color:{color}; background:white; padding:2px 4px; border:1px solid #ccc; border-radius:4px">
                  C{int(cl)} • {count}
                </div>
                """
            ),
        ).add_to(fg)

        fg.add_to(m)

    # Controls and layers

    folium.LayerControl(collapsed=False).add_to(m)
    legend_html = (
        '<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; '
        'background: white; padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px; font-size: 14px;">'
        "<b>Legend (Top 3, without noise)</b><br>"
        + "<br>".join([f'<span style="color:{cluster_colors[cl]};">●</span> Cluster {int(cl)}' for cl in top_clusters])
        + "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, height=650, width=None)



# 2) Temporal Analysis ---------------------------------------------------------------------------------------------------

st.subheader("2) Temporal Analysis")
df = gdf.copy()  

## MONTH

if "month" not in df.columns:
    st.error("Column 'month' not found")
else:
    order = ["Jan","Feb","Mar","Apr","May","Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    counts = df["month"].value_counts().reindex(order, fill_value=0)

    table = counts.rename("Occurrences").reset_index().rename(columns={"index":"month"})

    st.markdown("### Table: Analysis of total accidents by month")
    st.dataframe(table, use_container_width=True)

    st.markdown("### Bar_chart: Total accidents by month")
    st.bar_chart(table.set_index("month"))

## WEEKDAY

if "weekday" not in df.columns:
    st.error("Column 'weekday' not found")
else:
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    counts = df["weekday"].value_counts().reindex(order, fill_value=0)

    table = counts.rename("Occurrences").reset_index().rename(columns={"index":"weekday"})

    st.markdown("### Table: Analysis of total accidents by weekday")
    st.dataframe(table, use_container_width=True)
    
    st.markdown("### Bar_chart: Total accidents by weekday")
    st.bar_chart(table.set_index("weekday"))

## HOUR

if "hour" not in df.columns:
    st.error("Column 'hour' not found")
else:
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(-1).astype(int)
    df = df[df["hour"].between(0,23)]

    counts = df["hour"].value_counts().reindex(range(24), fill_value=0)


    table = counts.rename("Occurrences").reset_index().rename(columns={"index":"hour"})

    st.markdown("### Table: Analysis of total accidents by hour")
    st.dataframe(table, use_container_width=True)

    st.markdown("### Bar_chart: Total accidents by hour")
    st.bar_chart(table.set_index("hour"))



## INJURIES

cols = ["fatalities_30d","serious_injuries_30d","minor_injuries_30d"]
for c in cols:
    if c not in df.columns:
        st.error(f"Column '{c}' no found")

for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

lesion_counts = {
    "Fatalities": (df["fatalities_30d"] > 0).sum(),
    "Serious": (df["serious_injuries_30d"] > 0).sum(),
    "Minor": (df["minor_injuries_30d"] > 0).sum(),
}

table = pd.DataFrame(list(lesion_counts.items()), columns=["Injury Type","Occurrences"])

st.markdown("### Table: Total Accidents and Severity")
st.dataframe(table, use_container_width=True)

st.markdown("### Bar_chart: Total accidents by type of injury")
st.bar_chart(table.set_index("Injury Type"))


## INJURIES: type of severity by weekday

cols = ["weekday", "fatalities_30d", "serious_injuries_30d", "minor_injuries_30d"]
missing = [c for c in cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
else:
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    pivot_day = (
        df.groupby("weekday")[cols[1:]].sum()
          .reindex(order, fill_value=0)
          .reset_index()
    )

    st.markdown("### Table: Total of injuries by weekday")
    st.dataframe(pivot_day, use_container_width=True)

    st.markdown("### Bar_chart: Type of injury by weekday")
    st.bar_chart(pivot_day.set_index("weekday"))

 
# Conclusion
st.markdown(
    """
> **Conclusion**
> The temporal analysis of road accidents recorded in the Lisbon area in 2023, based on data provided by the ANSR (National Road Safety Authority), revealed consistent and relevant patterns in understanding road traffic accidents.
> A higher concentration of accidents was observed in March, May and October, which may be associated with increased traffic flow during seasonal transition periods, as well as with less stable weather conditions that heighten road risk. In contrast, the months of August, December and February recorded fewer accidents, possibly reflecting reduced traffic during holiday periods or adverse weather conditions that discourage circulation.
> At the weekly level, accidents occurred more frequently on Thursdays and Fridays, coinciding with days of heightened commuter and work-related mobility. Weekends, particularly Saturdays and Sundays, registered fewer accidents, which may be linked to the reduction in weekday-related traffic flows.
> In terms of time of day, there was a marked peak in accidents between 16:00 and 19:00, corresponding to the evening rush hour and high traffic intensity. Conversely, the period between 23:00 and 06:00 saw the lowest number of accidents, reflecting lower levels of night-time circulation.
> With regard to injury severity, fatal accidents were concentrated on Tuesdays and Sundays, with no fatalities recorded on Mondays, Fridays and Saturdays. Serious and minor injuries were more prevalent on weekdays, reflecting the impact of heavier traffic linked to professional activities, while weekends registered comparatively fewer such cases.
> In summary, the results confirm that temporal factors — seasonal, weekly and hourly — have a significant influence on the occurrence and severity of road accidents in the Lisbon area. These patterns provide a robust basis for designing preventive and mitigation strategies tailored to the periods and contexts of greatest risk.
"""
)


