import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import re
import json

# Streamlit app configuration
st.set_page_config(page_title="User Activity Visualization", layout="wide")
st.title("User Activity Visualization")

# Input for date_suffix
date_suffix = st.text_input("Enter date (DD-MM-YYYY)", value="11-06-2025")

# Validate date format
try:
    datetime.strptime(date_suffix, "%d-%m-%Y")
except ValueError:
    st.error("Invalid date format. Please use DD-MM-YYYY (e.g., 11-06-2025)")
    st.stop()

# --- Load Data ---
input_dir = os.path.join("logs", date_suffix)
linked_dir = os.path.join("linked_logs", date_suffix)
grouped_file = os.path.join(linked_dir, f"grouped_records_by_number_{date_suffix}.txt")

if not os.path.exists(grouped_file):
    st.warning(f"Grouped records file not found at {grouped_file}")
    st.stop()

# Load CSVs for other visualizations
df_transactions = pd.read_csv(os.path.join(input_dir, f"transactions_{date_suffix}.csv")) if os.path.exists(os.path.join(input_dir, f"transactions_{date_suffix}.csv")) else pd.DataFrame()
df_ai_logs = pd.read_csv(os.path.join(input_dir, f"stoxbot_{date_suffix}.csv")) if os.path.exists(os.path.join(input_dir, f"stoxbot_{date_suffix}.csv")) else pd.DataFrame()
df_comm_logs = pd.read_csv(os.path.join(input_dir, f"stoxbotcomm_{date_suffix}.csv")) if os.path.exists(os.path.join(input_dir, f"stoxbotcomm_{date_suffix}.csv")) else pd.DataFrame()

# --- Helper Function for Parsing Grouped File ---
def parse_grouped_file(grouped_file):
    activities = []
    current_number = None
    activity_pattern = r'\[(TRANSACTION|STOXBOT|STOXBOTCOMM)\]\s+(.+)$'
    number_pattern = r'=== Records for Number: (\d+) ==='
    time_pattern = r'"time":\s*"(\d{4}-\d{2}-\d{2} T \d{2}:\d{2}:\d{2}\.\d{3} Z)"'

    with open(grouped_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            number_match = re.match(number_pattern, line)
            if number_match:
                current_number = number_match.group(1)
                continue
            activity_match = re.match(activity_pattern, line)
            if activity_match and current_number:
                source = activity_match.group(1)
                json_str = activity_match.group(2)
                time_match = re.search(time_pattern, json_str)
                if time_match:
                    raw_time = time_match.group(1)
                    try:
                        parsed_time = pd.to_datetime(raw_time, format="%Y-%m-%d T %H:%M:%S.%f Z", errors='coerce')
                        if pd.notna(parsed_time):
                            activities.append({
                                'mobile': current_number,
                                'time': parsed_time,
                                'source': source,
                                'raw_time': raw_time
                            })
                    except:
                        pass
    return pd.DataFrame(activities)

# Parse grouped file for all activities
df_activities = parse_grouped_file(grouped_file)

if df_activities.empty:
    st.warning("No valid activities found in the grouped records file.")
    st.stop()

# --- Visualization 1: Number of Users vs. Timing Hours ---
def prepare_hourly_user_data(df_transactions, df_ai_logs, df_comm_logs):
    activities = []
    for _, row in df_transactions.iterrows():
        if pd.notna(row['time']):
            mobile = row['second_auth'] if pd.notna(row['second_auth']) and row['second_auth'] != '' else row['mobile']
            if pd.notna(mobile) and mobile != '':
                activities.append({'time': row['time'], 'mobile': str(mobile).strip()})
    for _, row in df_ai_logs.iterrows():
        if pd.notna(row['time']) and pd.notna(row['mobile']) and row['mobile'] != '':
            activities.append({'time': row['time'], 'mobile': str(row['mobile']).strip()})
    for _, row in df_comm_logs.iterrows():
        if pd.notna(row['time']) and pd.notna(row['mo']) and row['mo'] != '':
            activities.append({'time': row['time'], 'mobile': str(row['mo']).strip()})
    df_activities_csv = pd.DataFrame(activities)
    df_activities_csv['time'] = pd.to_datetime(df_activities_csv['time'], format="%Y-%m-%d T %H:%M:%S.%f Z", errors='coerce')
    df_activities_csv['hour'] = df_activities_csv['time'].dt.hour
    hourly_users = df_activities_csv.groupby('hour')['mobile'].nunique().reset_index()
    hourly_users.columns = ['Hour', 'Unique Users']
    return hourly_users

st.subheader("Number of Unique Users Active per Hour")
if not (df_transactions.empty and df_ai_logs.empty and df_comm_logs.empty):
    df_hourly_users = prepare_hourly_user_data(df_transactions, df_ai_logs, df_comm_logs)
    if not df_hourly_users.empty:
        fig1 = px.histogram(
            df_hourly_users,
            x='Hour',
            y='Unique Users',
            title=f"Number of Unique Users Active per Hour on {date_suffix}",
            labels={'Hour': 'Hour of Day (24-hour)', 'Unique Users': 'Number of Unique Users'},
            nbins=24
        )
        fig1.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1), bargap=0.1)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No hourly user data available for visualization.")
else:
    st.warning("No CSV data available for hourly visualization.")

# --- Visualization 2: Last Activity Source Distribution ---
def prepare_last_activity_data(df_activities):
    last_activities = []
    for number in df_activities['mobile'].unique():
        number_activities = df_activities[df_activities['mobile'] == number]
        valid_activities = number_activities[pd.notna(number_activities['time'])]
        if not valid_activities.empty:
            latest_activity = valid_activities.loc[valid_activities['time'].idxmax()]
            last_activities.append({
                'mobile': number,
                'source': latest_activity['source'],
                'last_time': latest_activity['raw_time']
            })
    st.subheader("Debug: Last Activity per Mobile Number")
    df_debug = pd.DataFrame(last_activities)[['mobile', 'source', 'last_time']]
    st.write(df_debug)
    df_last_activities = pd.DataFrame(last_activities)
    source_counts = df_last_activities['source'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Count']
    return source_counts

st.subheader("Distribution of Last Activity Source per User")
df_last_activity = prepare_last_activity_data(df_activities)
if not df_last_activity.empty:
    fig2 = px.pie(
        df_last_activity,
        names='Source',
        values='Count',
        title=f"Distribution of Last Activity Source per User on {date_suffix}"
    )
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No last activity data available.")

# --- Visualization 3: Activity Count per Source by Hour ---
st.subheader("Activity Count per Source by Hour")
df_activities['hour'] = df_activities['time'].dt.hour
activity_counts = df_activities.groupby(['hour', 'source']).size().reset_index(name='Count')
if not activity_counts.empty:
    fig3 = px.line(
        activity_counts,
        x='hour',
        y='Count',
        color='source',
        title=f"Activity Count per Source over Time on {date_suffix}",
        labels={'hour': 'Hour of Day', 'Count': 'Number of Activities', 'source': 'Source'}
    )
    fig3.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No activity count data available for visualization.")

# --- Visualization 4: Top 5 Active Users ---
st.subheader("Top 5 Active Users")
user_counts = df_activities['mobile'].value_counts().head(5).reset_index()
user_counts.columns = ['Number', 'Activity Count']
if not user_counts.empty:
    fig4 = px.bar(
        user_counts,
        x='Number',
        y='Activity Count',
        title=f"Top 5 Active Users on {date_suffix}",
        labels={'Number': 'Mobile Number', 'Activity Count': 'Total Activities'},
        text_auto=True
    )
    fig4.update_traces(textposition='auto')
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("No user activity data available for visualization.")

# --- Visualization 5: Activity Heatmap by Hour and Source ---
st.subheader("Activity Heatmap by Hour and Source")
heatmap_data = df_activities.pivot_table(
    index='source',
    columns='hour',
    values='mobile',
    aggfunc='count',
    fill_value=0
).reset_index()
if not heatmap_data.empty:
    fig5 = go.Figure(data=go.Heatmap(
        z=heatmap_data.drop(columns='source').values,
        x=heatmap_data.columns[1:],
        y=heatmap_data['source'],
        colorscale='Viridis',
        text=heatmap_data.drop(columns='source').values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig5.update_layout(
        title=f"Activity Heatmap by Hour and Source on {date_suffix}",
        xaxis_title="Hour of Day",
        yaxis_title="Source"
    )
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.warning("No heatmap data available for visualization.")

# --- Visualization 6: User Activity Timeline ---
st.subheader("User Activity Timeline")
mobile_numbers = df_activities['mobile'].unique()
selected_number = st.selectbox("Select a Mobile Number", options=mobile_numbers)
if selected_number:
    user_activities = df_activities[df_activities['mobile'] == selected_number]
    if not user_activities.empty:
        fig6 = px.scatter(
            user_activities,
            x='time',
            y='source',
            title=f"Activity Timeline for {selected_number} on {date_suffix}",
            labels={'time': 'Time', 'source': 'Activity Source'},
            color='source',
            size_max=10
        )
        fig6.update_traces(marker=dict(size=10))
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.warning(f"No activities found for {selected_number}.")