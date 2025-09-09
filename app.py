import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import io

# --- Custom CSS for stylish download buttons ---
st.markdown("""
<style>
.download-container {
    background: #f9f9f9;
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    transition: box-shadow 0.3s ease;
}
.download-container:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
}
.download-label {
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 8px;
    color: #222222;
}
div.stDownloadButton > button {
    background-color: #0072C6 !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    box-shadow: none !important;
    transition: background-color 0.3s ease !important;
    width: 100% !important;
}
div.stDownloadButton > button:hover {
    background-color: #005a9e !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Advanced Cancer Data Analysis Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filter Data")
max_survival = st.sidebar.slider("Max Survival Time (days)", 0, 2000, 1000)
selected_diagnosis = st.sidebar.multiselect("Diagnosis", ["M", "B"], default=["M", "B"])

@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    np.random.seed(42)
    data["survival_time"] = np.random.exponential(scale=365 * 2, size=len(data))
    data["event"] = np.random.binomial(1, 0.3, size=len(data))
    return data

data = load_data()

# Filter data based on sidebar selections
filtered_data = data[
    (data["survival_time"] <= max_survival) & (data["diagnosis"].isin(selected_diagnosis))
]

st.write(f"Dataset contains {filtered_data.shape[0]} records after filtering.")

# Feature selection multiselect for modeling
all_features = list(filtered_data.columns.difference(["diagnosis", "survival_time", "event"]))
selected_features = st.sidebar.multiselect("Select Features for Modeling", all_features, default=all_features)

# Kaplan-Meier survival curves
st.header("Kaplan-Meier Survival Curves by Diagnosis")
with st.expander("What is a Kaplan-Meier curve?"):
    st.markdown("""
    The Kaplan-Meier curve shows the probability of survival over time for different groups.
    Here, survival probabilities are plotted for each diagnosis group.
    """)

kmf = KaplanMeierFitter()
fig_km, ax_km = plt.subplots(figsize=(10, 6))
for diagnosis in filtered_data["diagnosis"].unique():
    kmf.fit(filtered_data.loc[filtered_data["diagnosis"] == diagnosis, "survival_time"],
            event_observed=filtered_data.loc[filtered_data["diagnosis"] == diagnosis, "event"],
            label=diagnosis)
    kmf.plot_survival_function(ax=ax_km)
ax_km.set_title("Kaplan-Meier Survival Curves")
ax_km.set_xlabel("Survival Time (days)")
ax_km.set_ylabel("Survival Probability")
st.pyplot(fig_km)

# Predictive Modeling & SHAP explanations
st.header("Predictive Modeling & Explainability with SHAP")

if not selected_features:
    st.warning("Please select at least one feature for modeling.")
else:
    X = filtered_data[selected_features].copy()

    # Encode categorical columns as numeric
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.factorize(X[col])[0]

    y = filtered_data["diagnosis"].map({"M": 1, "B": 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = X_test.reset_index(drop=True)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Show metrics in columns
    st.subheader("Model Performance Metrics")
    precision, recall, f1, support = classification_report(y_test, y_pred, output_dict=True)["1"].values()

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision (Class M)", f"{precision:.2f}")
    col2.metric("Recall (Class M)", f"{recall:.2f}")
    col3.metric("F1 Score (Class M)", f"{f1:.2f}")

    with st.expander("Classification Report Details"):
        st.text(classification_report(y_test, y_pred))

    # SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap_values_class1 = shap_values.values[:, :, 1]

    # SHAP plots side by side
    st.subheader("SHAP Feature Importance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bar Plot: Average impact of features on model output**")
        fig_shap_bar, ax_bar = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values_class1, X_test, plot_type="bar", show=False)
        st.pyplot(fig_shap_bar)

    with col2:
        st.markdown("**Scatter Plot: Impact of features across samples**")
        fig_shap_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values_class1, X_test, show=False)
        st.pyplot(fig_shap_scatter)

    with st.expander("How to interpret SHAP plots"):
        st.markdown("""
        - **Bar plot** shows the mean absolute SHAP value for each feature, summarizing overall importance.
        - **Scatter plot** shows SHAP values for each sample and feature. Color usually represents the feature value (red high, blue low).
        - Features with wider spread and stronger impact are more important.
        """)

    # Export options
    st.subheader("Export Data & Plots")

    # Export filtered data CSV
    csv_data = filtered_data.to_csv(index=False).encode('utf-8')
    # Export model feature importance as CSV (mean absolute SHAP values)
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)
    shap_csv = shap_df.to_csv(index=False).encode('utf-8')

    # Export SHAP plots as PNG
    def fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf

    bar_png = fig_to_bytes(fig_shap_bar)
    scatter_png = fig_to_bytes(fig_shap_scatter)
    km_png = fig_to_bytes(fig_km)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        st.markdown('<div class="download-label">Filtered Data CSV</div>', unsafe_allow_html=True)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="filtered_cancer_data.csv",
            mime="text/csv",
            key="download_csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        st.markdown('<div class="download-label">SHAP Feature Importance CSV</div>', unsafe_allow_html=True)
        st.download_button(
            label="Download CSV",
            data=shap_csv,
            file_name="shap_feature_importance.csv",
            mime="text/csv",
            key="download_shap_csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        st.markdown('<div class="download-label">Kaplan-Meier Plot PNG</div>', unsafe_allow_html=True)
        st.download_button(
            label="Download PNG",
            data=km_png,
            file_name="kaplan_meier_plot.png",
            mime="image/png",
            key="download_km"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        st.markdown('<div class="download-label">SHAP Bar Plot PNG</div>', unsafe_allow_html=True)
        st.download_button(
            label="Download PNG",
            data=bar_png,
            file_name="shap_bar_plot.png",
            mime="image/png",
            key="download_bar"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        st.markdown('<div class="download-label">SHAP Scatter Plot PNG</div>', unsafe_allow_html=True)
        st.download_button(
            label="Download PNG",
            data=scatter_png,
            file_name="shap_scatter_plot.png",
            mime="image/png",
            key="download_scatter"
        )
        st.markdown('</div>', unsafe_allow_html=True)
