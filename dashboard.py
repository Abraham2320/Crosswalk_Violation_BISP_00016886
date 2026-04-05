"""
Crosswalk Violation Analytics Dashboard
Run with: streamlit run dashboard.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent / "crosswalk_violations.db"

st.set_page_config(
    page_title="Crosswalk Violation Analytics Dashboard",
    page_icon="🚦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_violations() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=[
            "id", "vehicle_id", "plate_number", "timestamp",
            "confidence", "status", "location", "violation_type",
            "pedestrian_direction", "frame_image_path",
        ])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM violations ORDER BY timestamp DESC", conn)
    conn.close()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")

raw = load_violations()

if not raw.empty and "timestamp" in raw.columns:
    min_date = raw["timestamp"].min().date()
    max_date = raw["timestamp"].max().date()
else:
    from datetime import date
    min_date = max_date = date.today()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.3, 0.05)

plate_filter = st.sidebar.radio(
    "Plate captured",
    ["All", "Plate captured", "No plate"],
)

# Apply filters
df = raw.copy()
if not df.empty and "timestamp" in df.columns:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        df = df[
            (df["timestamp"].dt.date >= start) &
            (df["timestamp"].dt.date <= end)
        ]
    df = df[df["confidence"] >= min_conf]
    if plate_filter == "Plate captured":
        df = df[df["plate_number"].notna() & (df["plate_number"] != "")]
    elif plate_filter == "No plate":
        df = df[df["plate_number"].isna() | (df["plate_number"] == "")]

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------

st.title("Crosswalk Violation Analytics Dashboard")

# ---------------------------------------------------------------------------
# A) KPI cards
# ---------------------------------------------------------------------------

total = len(df)
unique_vehicles = df["vehicle_id"].nunique() if total else 0

if total and "timestamp" in df.columns and df["timestamp"].notna().any():
    hour_counts = df["timestamp"].dt.hour.value_counts()
    peak_hour = int(hour_counts.idxmax()) if not hour_counts.empty else 0
    peak_hour_label = f"{peak_hour:02d}:00"
else:
    peak_hour_label = "N/A"

if total:
    plates_detected = df["plate_number"].notna() & (df["plate_number"] != "")
    plate_rate = f"{plates_detected.mean() * 100:.1f}%"
else:
    plate_rate = "N/A"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Violations", total)
col2.metric("Unique Vehicles", unique_vehicles)
col3.metric("Peak Violation Hour", peak_hour_label)
col4.metric("Plate Recognition Rate", plate_rate)

st.divider()

# ---------------------------------------------------------------------------
# B) Violations over time — line chart
# ---------------------------------------------------------------------------

st.subheader("Violations Over Time (by Hour of Day)")

if total and "timestamp" in df.columns and df["timestamp"].notna().any():
    hourly = (
        df.groupby(df["timestamp"].dt.hour)
        .size()
        .reindex(range(24), fill_value=0)
        .rename_axis("Hour")
        .reset_index(name="Violations")
    )
    st.line_chart(hourly.set_index("Hour"))
else:
    st.info("No data available for the selected filters.")

# ---------------------------------------------------------------------------
# C) Top offending vehicles — bar chart
# ---------------------------------------------------------------------------

st.subheader("Top Offending Vehicles")

if total:
    if plates_detected.any():
        label_col = df["plate_number"].where(
            df["plate_number"].notna() & (df["plate_number"] != ""),
            other="Vehicle #" + df["vehicle_id"].astype(str),
        )
    else:
        label_col = "Vehicle #" + df["vehicle_id"].astype(str)

    top_offenders = (
        label_col.value_counts()
        .head(10)
        .rename_axis("Vehicle")
        .reset_index(name="Violations")
    )
    st.bar_chart(top_offenders.set_index("Vehicle"))
else:
    st.info("No data available.")

# ---------------------------------------------------------------------------
# D) Heatmap — day of week × hour
# ---------------------------------------------------------------------------

st.subheader("Violations by Day & Hour (Heatmap)")

if total and "timestamp" in df.columns and df["timestamp"].notna().any():
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heat = (
        df.assign(
            day=df["timestamp"].dt.dayofweek,
            hour=df["timestamp"].dt.hour,
        )
        .groupby(["day", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    heat.index = [day_names[i] for i in heat.index]
    heat.columns = [f"{h:02d}h" for h in heat.columns]

    st.dataframe(
        heat.style.background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True,
    )
else:
    st.info("No data available.")

st.divider()

# ---------------------------------------------------------------------------
# E) Violation log table — paginated
# ---------------------------------------------------------------------------

st.subheader("Violation Log")

ROWS_PER_PAGE = 20

if total:
    display_cols = [
        c for c in ["id", "vehicle_id", "plate_number", "timestamp",
                     "confidence", "status", "location"]
        if c in df.columns
    ]
    display_df = df[display_cols].copy()
    if "timestamp" in display_df.columns:
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    total_pages = max(1, (total + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * ROWS_PER_PAGE
    end = start + ROWS_PER_PAGE

    st.dataframe(display_df.iloc[start:end], use_container_width=True)
    st.caption(f"Showing rows {start + 1}–{min(end, total)} of {total}")
else:
    st.info("No violations recorded yet.")

st.divider()

# ---------------------------------------------------------------------------
# G) AI Summary
# ---------------------------------------------------------------------------

st.subheader("AI Traffic Safety Summary")

if st.button("Generate AI Summary", type="primary"):
    if total == 0:
        st.warning("No data to summarise — run the detection system first.")
    else:
        with st.spinner("Generating summary with Claude..."):
            hour_dist = ""
            if "timestamp" in df.columns and df["timestamp"].notna().any():
                top_hours = (
                    df["timestamp"].dt.hour.value_counts()
                    .head(5)
                    .to_dict()
                )
                hour_dist = ", ".join(f"{h:02d}:00 ({c} violations)"
                                      for h, c in sorted(top_hours.items()))

            top5_vehicles = (
                df["vehicle_id"].value_counts()
                .head(5)
                .to_dict()
            )
            top5_str = ", ".join(f"ID {v} ({c}x)" for v, c in top5_vehicles.items())

            prompt = f"""You are a traffic safety analyst for a crosswalk violation detection system in Tashkent, Uzbekistan.

Violation dataset summary (after applied filters):
- Total violations: {total}
- Unique vehicles involved: {unique_vehicles}
- Plate recognition rate: {plate_rate}
- Peak hours: {hour_dist or 'insufficient timestamp data'}
- Top offending vehicles: {top5_str or 'N/A'}
- Date range: {min_date} to {max_date}

Based on this data:
1. Summarise the key violation patterns observed.
2. Identify the highest-risk time windows.
3. Suggest 3 specific, actionable traffic management recommendations for the crosswalk authority.

Be concise, factual, and actionable. Write in a formal but readable tone."""

            try:
                client = anthropic.Anthropic()
                message = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = message.content[0].text
                st.info(response_text)
            except anthropic.AuthenticationError:
                st.error(
                    "ANTHROPIC_API_KEY is missing or invalid. "
                    "Set it with: `set ANTHROPIC_API_KEY=sk-ant-...`"
                )
            except Exception as exc:
                st.error(f"API call failed: {exc}")
