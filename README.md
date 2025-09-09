# 🧬 Breast Cancer Data Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red)

An interactive Streamlit web application and exploratory notebook for analyzing breast cancer data using survival analysis, machine learning, and SHAP-based explainability.

---

## 📊 Features

### ✅ Streamlit App (`app.py`)
- 📈 **Kaplan-Meier survival curves** stratified by diagnosis
- 🌲 **Random Forest classifier** to predict malignant vs benign tumors
- 🧠 **SHAP plots** for feature importance and model transparency
- 💾 **Downloadable outputs**:
  - Filtered dataset CSV
  - SHAP values CSV
  - SHAP bar & scatter plots
  - Kaplan-Meier plots

### 📓 Colab Notebook (`notebooks/exploration.ipynb`)
- 🔍 Data cleaning & preprocessing
- 📊 Exploratory Data Analysis (EDA)
- ⚰️ Kaplan-Meier survival analysis
- 📉 Cox Proportional Hazards model
- 🤖 Machine learning model with SHAP explanations

---

## 🗂️ Project Structure
Breast-Cancer-Data-Analysis/
│
├── app.py # Streamlit web app
├── data.csv # Breast cancer dataset
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── notebooks/
└── exploration.ipynb # Full notebook analysis

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- pip installed

### 📦 Installation

1. Clone the repository:
   git clone https://github.com/Monika-Bhardwaj/Breast-Cancer-Data-Analysis.git
   cd Breast-Cancer-Data-Analysis

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run app.py

---

🧪 Dataset

Based on the Wisconsin Breast Cancer Dataset, with synthetic survival data added:

diagnosis: Target variable (M = Malignant, B = Benign)

radius_mean, texture_mean, etc.: Tumor features

survival_time: Simulated time in days

event: 1 = death, 0 = censored (simulated)

📈 Visualizations

Survival Curves: Kaplan-Meier plots by diagnosis

Feature Importance: SHAP bar and scatter plots

Classification Metrics: Precision, recall, F1 score

📋 Requirements

Key Python libraries used:

streamlit
pandas
numpy
matplotlib
lifelines
scikit-learn
shap
seaborn


Install via:

pip install -r requirements.txt

📤 Exports

From the Streamlit app, users can download:

Filtered dataset (.csv)

SHAP feature importance (.csv)

Visualizations (.png)

💡 Future Enhancements

Integrate real clinical survival data

Deploy on Streamlit Cloud
 or Hugging Face Spaces

Enable CSV uploads for custom analysis

Add XGBoost or Logistic Regression options

📜 License

This project is licensed under the [MIT License](LICENSE).


👤 Author

Monika Bhardwaj


GitHub: [Monika-Bhardwaj](https://github.com/Monika-Bhardwaj)


🌐 Live Demo 

👉 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://breast-cancer-data-analysis-8expporliqyetd6xpotemo.streamlit.app/)


---

