import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import json
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
from collections import defaultdict, Counter
from sklearn.metrics import classification_report
# Streamlit app configuration
st.set_page_config(page_title="User Activity ML Analysis", layout="wide")
st.title("User Activity ML Analysis")

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

# Load CSVs
df_transactions = pd.read_csv(os.path.join(input_dir, f"transactions_{date_suffix}.csv")) if os.path.exists(os.path.join(input_dir, f"transactions_{date_suffix}.csv")) else pd.DataFrame()
df_ai_logs = pd.read_csv(os.path.join(input_dir, f"stoxbot_{date_suffix}.csv")) if os.path.exists(os.path.join(input_dir, f"stoxbot_{date_suffix}.csv")) else pd.DataFrame()
df_comm_logs = pd.read_csv(os.path.join(input_dir, f"stoxbotcomm_{date_suffix}.csv")) if os.path.exists(os.path.join(input_dir, f"stoxbotcomm_{date_suffix}.csv")) else pd.DataFrame()

# --- Parse Grouped File ---
def parse_grouped_file(grouped_file):
    activities = []
    activity_pattern = r'\[(TRANSACTION|STOXBOT|STOXBOTCOMM)\]\s+(.+)$'
    number_pattern = r'=== Records for Number: (\d+) ==='
    time_pattern = r'"time":\s*"(\d{4}-\d{2}-\d{2} T \d{2}:\d{2}:\d{2}\.\d{3} Z)"'
    current_number = None

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

df_activities = parse_grouped_file(grouped_file)
if df_activities.empty:
    st.warning("No valid activities found in the grouped records file.")
    st.stop()

# --- Feature Engineering ---
def prepare_features(df_activities):
    features = []
    current_time = datetime.strptime(f"{date_suffix} 23:59:59", "%d-%m-%Y %H:%M:%S")

    for number in df_activities['mobile'].unique():
        number_activities = df_activities[df_activities['mobile'] == number].sort_values('time')
        feature_dict = {'mobile': number}

        # Basic counts
        feature_dict['total_activities'] = len(number_activities)
        feature_dict['txn_count'] = len(number_activities[number_activities['source'] == 'TRANSACTION'])
        feature_dict['stoxbot_count'] = len(number_activities[number_activities['source'] == 'STOXBOT'])
        feature_dict['comm_count'] = len(number_activities[number_activities['source'] == 'STOXBOTCOMM'])

        # Proportions
        feature_dict['txn_proportion'] = feature_dict['txn_count'] / feature_dict['total_activities'] if feature_dict['total_activities'] > 0 else 0
        feature_dict['stoxbot_proportion'] = feature_dict['stoxbot_count'] / feature_dict['total_activities'] if feature_dict['total_activities'] > 0 else 0

        # Time-based features
        if not number_activities.empty:
            last_time = number_activities['time'].iloc[-1]
            feature_dict['hours_since_last_activity'] = (current_time - last_time).total_seconds() / 3600
            time_diffs = number_activities['time'].diff().dt.total_seconds() / 3600
            feature_dict['avg_time_between_activities'] = time_diffs.mean() if len(time_diffs) > 1 else 0
        else:
            feature_dict['hours_since_last_activity'] = np.nan
            feature_dict['avg_time_between_activities'] = np.nan

        # Sequence features
        sequence = number_activities['source'].tolist()
        transitions = [(sequence[i], sequence[i+1]) for i in range(len(sequence)-1)]
        feature_dict['stoxbot_to_txn'] = sum(1 for t in transitions if t == ('STOXBOT', 'TRANSACTION'))
        feature_dict['comm_to_txn'] = sum(1 for t in transitions if t == ('STOXBOTCOMM', 'TRANSACTION'))
        feature_dict['has_txn'] = int(feature_dict['txn_count'] > 0)

        # Churn target
        feature_dict['churn'] = 1 if feature_dict['hours_since_last_activity'] > 6 else 0

        features.append(feature_dict)

    return pd.DataFrame(features)

df_features = prepare_features(df_activities)

# --- ML Task 1: Unsupervised Clustering for Drop-off Patterns ---
st.subheader("Unsupervised Clustering for Drop-off Patterns")
def cluster_drop_off_patterns(df_features):
    feature_cols = [
        'total_activities', 'txn_proportion', 'stoxbot_proportion',
        'hours_since_last_activity', 'avg_time_between_activities',
        'stoxbot_to_txn', 'comm_to_txn', 'has_txn'
    ]
    X = df_features[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_features['cluster'] = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, df_features['cluster'])
    st.write(f"Silhouette Score: {silhouette:.3f}")

    # Visualize clusters
    fig = px.scatter(
        df_features,
        x='total_activities',
        y='txn_proportion',
        color='cluster',
        hover_data=['mobile', 'hours_since_last_activity'],
        title=f"User Clusters Based on Activity Patterns ({date_suffix})",
        labels={'total_activities': 'Total Activities', 'txn_proportion': 'Transaction Proportion'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display cluster summary
    cluster_summary = df_features.groupby('cluster').mean(numeric_only=True)[feature_cols].round(2)
    st.write("Cluster Summary (Mean Values):", cluster_summary)

    # Save results
    os.makedirs(linked_dir, exist_ok=True)
    df_features[['mobile', 'cluster']].to_csv(os.path.join(linked_dir, f"clusters_{date_suffix}.csv"), index=False)
    st.write(f"Cluster assignments saved to: {os.path.join(linked_dir, f'clusters_{date_suffix}.csv')}")

    return df_features

if st.button("Run Clustering"):
    df_features = cluster_drop_off_patterns(df_features)

# --- ML Task 2: Churn Prediction ---
st.subheader("Churn Prediction")
def predict_churn(df_features):
    feature_cols = [
        'total_activities', 'txn_count', 'stoxbot_count', 'comm_count',
        'hours_since_last_activity', 'avg_time_between_activities',
        'stoxbot_to_txn', 'comm_to_txn'
    ]
    X = df_features[feature_cols].fillna(0)
    y = df_features['churn']

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Classification Report:")
    st.write(pd.DataFrame(report).transpose().round(2))

    # Predict for all
    df_features['churn_probability'] = model.predict_proba(X)[:, 1]

    # Visualize churn probabilities
    fig = px.histogram(
        df_features,
        x='churn_probability',
        title=f"Distribution of Churn Probabilities ({date_suffix})",
        labels={'churn_probability': 'Churn Probability'},
        nbins=20
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display high-risk users
    high_risk = df_features[df_features['churn_probability'] > 0.7][['mobile', 'churn_probability']].head(10)
    st.write("Top 10 High-Risk Users for Churn:", high_risk)

    # Save results
    df_features[['mobile', 'churn_probability']].to_csv(
        os.path.join(linked_dir, f"churn_predictions_{date_suffix}.csv"), index=False
    )
    st.write(f"Churn predictions saved to: {os.path.join(linked_dir, f'churn_predictions_{date_suffix}.csv')}")

if st.button("Run Churn Prediction"):
    predict_churn(df_features)

# --- ML Task 3: Anomaly Detection ---
st.subheader("Anomaly Detection")
def detect_anomalies(df_features):
    feature_cols = [
        'total_activities', 'txn_count', 'stoxbot_count', 'comm_count',
        'avg_time_between_activities'
    ]
    X = df_features[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42)
    df_features['is_anomaly'] = model.fit_predict(X_scaled)
    df_features['anomaly_score'] = model.decision_function(X_scaled)

    anomalies = df_features[df_features['is_anomaly'] == -1][['mobile', 'total_activities', 'txn_count', 'stoxbot_count', 'comm_count', 'anomaly_score']]
    st.write(f"Detected {len(anomalies)} anomalies.")
    st.write("Anomalous Users:", anomalies.head(10))

    # Visualize anomalies
    fig = px.scatter(
        df_features,
        x='total_activities',
        y='stoxbot_count',
        color='is_anomaly',
        hover_data=['mobile', 'anomaly_score'],
        title=f"Anomaly Detection Results ({date_suffix})",
        labels={'total_activities': 'Total Activities', 'stoxbot_count': 'StoxBot Activities', 'is_anomaly': 'Anomaly (1=Normal, -1=Anomaly)'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Save results
    anomalies.to_csv(os.path.join(linked_dir, f"anomalies_{date_suffix}.csv"), index=False)
    st.write(f"Anomalies saved to: {os.path.join(linked_dir, f'anomalies_{date_suffix}.csv')}")

if st.button("Run Anomaly Detection"):
    detect_anomalies(df_features)

# --- ML Task 4: Activity Sequence Analysis (Markov Chains) ---
st.subheader("Activity Sequence Analysis (Markov Chains)")
def analyze_sequences(df_activities):
    transitions = defaultdict(Counter)
    for number in df_activities['mobile'].unique():
        sequence = df_activities[df_activities['mobile'] == number]['source'].tolist()
        for i in range(len(sequence)-1):
            transitions[sequence[i]][sequence[i+1]] += 1

    # Normalize to probabilities
    transition_matrix = {}
    for state in transitions:
        total = sum(transitions[state].values())
        transition_matrix[state] = {next_state: count/total for next_state, count in transitions[state].items()}

    # Identify drop-off points
    drop_offs = []
    for state in transition_matrix:
        if 'TRANSACTION' not in transition_matrix[state]:
            drop_offs.append(f"From {state}: No transition to TRANSACTION (Probabilities: {transition_matrix[state]})")

    # Display transition matrix
    st.write("Transition Matrix (Probabilities):")
    st.json(transition_matrix)

    # Display drop-offs
    st.write("Drop-off Points (No Transition to TRANSACTION):")
    st.write(drop_offs)

    # Save results
    with open(os.path.join(linked_dir, f"transition_matrix_{date_suffix}.txt"), 'w') as f:
        f.write(json.dumps(transition_matrix, indent=2))
    with open(os.path.join(linked_dir, f"drop_offs_{date_suffix}.txt"), 'w') as f:
        f.write("\n".join(drop_offs))
    st.write(f"Transition matrix saved to: {os.path.join(linked_dir, f'transition_matrix_{date_suffix}.txt')}")
    st.write(f"Drop-off points saved to: {os.path.join(linked_dir, f'drop_offs_{date_suffix}.txt')}")

if st.button("Run Sequence Analysis"):
    analyze_sequences(df_activities)