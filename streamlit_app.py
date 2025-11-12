###############################
# Loan Repayment Clustering App
# Beautiful Modern Streamlit UI
###############################

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

# Set visual theme
sns.set_style("whitegrid")

# ======================================
# CUSTOM MANUAL F1 SCORE
# ======================================
def manual_f1_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# ======================================
# STREAMLIT APP CONFIG
# ======================================
st.set_page_config(
    page_title="Loan Repayment Clustering (KMeans)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------
# HEADER
# --------------------------------------
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>📊 Loan Repayment Clustering App</h1>
<p style='text-align:center; font-size:18px; color:gray;'>Explore loan repayment patterns using KMeans clustering with interactive controls.</p>
""", unsafe_allow_html=True)

st.write("---")

# ======================================
# SIDEBAR
# ======================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])

# ======================================
# MAIN APP LOGIC
# ======================================
if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("---")

    # ======================================
    # EDA SECTION
    # ======================================
    st.header("🔍 Exploratory Data Analysis")

    # Plot 1: Repayment Status Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Repayment Status")
        fig, ax = plt.subplots()
        sns.countplot(x='not.fully.paid', data=df, palette="Set2")
        st.pyplot(fig)

    with col2:
        st.subheader("2️⃣ Loan Purpose Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='purpose', data=df, palette="tab10")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # FICO Distribution
    st.subheader("3️⃣ FICO Distribution by Repayment Status")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='fico', hue='not.fully.paid', palette="Set2", kde=False)
    st.pyplot(fig)

    # ======================================
    # PREPROCESSING
    # ======================================
    st.header("⚙️ Preprocessing")

    df_encoded = pd.get_dummies(df, columns=['purpose'], drop_first=True)

    X = df_encoded.drop('not.fully.paid', axis=1)
    y = df_encoded['not.fully.paid']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================================
    # KMEANS + SLIDER
    # ======================================
    st.header("🤖 KMeans Clustering (with Slider)")

    k_value = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=2)

    model = KMeans(n_clusters=k_value, random_state=42, n_init=20)
    model.fit(X_scaled)

    df_encoded['cluster'] = model.labels_

    custom_f1 = manual_f1_score(y, df_encoded['cluster'])

    st.success(f"📌 F1 Score for k = {k_value}: **{custom_f1:.4f}**")

    # ======================================
    # SCATTERPLOT OF CLUSTERS
    # ======================================
    st.subheader("🎨 Color-Coded Scatterplot (FICO vs Interest Rate)")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(
        data=df_encoded,
        x="fico",
        y="int.rate",
        hue="cluster",
        palette="viridis",
        s=60,
        alpha=0.8
    )
    plt.title("Cluster Visualization (FICO vs Interest Rate)")
    st.pyplot(fig)

    # ======================================
    # F1 vs K PLOT
    # ======================================
    st.header("📈 F1 Score vs Number of Clusters (k)")

    f1_scores = []
    cluster_range = range(2, 11)

    for k in cluster_range:
        model_k = KMeans(n_clusters=k, random_state=42, n_init=20)
        model_k.fit(X_scaled)
        preds = model_k.labels_
        score = manual_f1_score(y, preds)
        f1_scores.append(score)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(cluster_range, f1_scores, marker="o", color="#FF5722")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("F1 Score")
    ax.set_title("How Cluster Count Affects F1 Score")
    st.pyplot(fig)

    # ======================================
    # CONFUSION MATRIX + REPORT
    # ======================================
    st.header("📊 Model Evaluation")
    cm = confusion_matrix(y, df_encoded['cluster'])

    st.subheader("Confusion Matrix")
    st.write(cm)

    st.subheader("Classification Report")
    st.text(classification_report(y, df_encoded['cluster']))

else:
    st.warning("⬅️ Upload a CSV file from the sidebar to begin.")

