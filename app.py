import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_detector_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🛡️ Детекция мошенничества по кредитной карте")
st.markdown("Введите параметры транзакции (30 признаков) и нажмите **Проверить**.")

st.sidebar.header("📊 Введите признаки")

input_values = []
for i in range(1, 29):
    val = st.sidebar.number_input(f"V{i}", value=0.0, step=0.01, format="%.4f", key=f"v_{i}")
    input_values.append(val)

time = st.sidebar.number_input("Time (секунды)", value=0.0, step=1.0, format="%.0f", key="time")
amount = st.sidebar.number_input("Amount (сумма)", value=0.0, step=0.01, format="%.2f", key="amount")
input_values.extend([time, amount])

if st.sidebar.button("🚀 Проверить транзакцию", type="primary"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    
    st.subheader("📋 Результат проверки")
    col1, col2 = st.columns(2)
    if prediction == 1:
        col1.error("❌ Транзакция **ПОДОЗРИТЕЛЬНАЯ** (мошенничество)")
    else:
        col1.success("✅ Транзакция **НОРМАЛЬНАЯ** (безопасная)")
    col2.metric("Вероятность мошенничества", f"{proba * 100:.2f}%")
    
    with st.expander("🔍 Посмотреть введённые признаки"):
        st.write(pd.DataFrame([input_values], columns=[f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]))
else:
    st.info("ℹ️ Заполните параметры транзакции в левой панели и нажмите кнопку.")

st.caption("Проект по машинному обучению — Детекция мошенничества (F1-Score: 0.85)")
