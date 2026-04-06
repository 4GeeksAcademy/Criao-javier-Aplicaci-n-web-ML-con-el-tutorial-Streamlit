# from utils import db_connect
# engine = db_connect()

# your code here
import streamlit as st
import pickle
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predictor de Vinos", page_icon="🍷", layout="wide")

st.markdown("""
<style>
    .titulo { color: #722F37; text-align: center; font-family: 'Arial'; }
    .prediccion { font-size: 24px; color: #27AE60; font-weight: bold; text-align: center; padding: 20px; border-radius: 10px; background-color: #EAFAF1; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    with open("models/wine_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# --- INTERFAZ DE USUARIO ---
st.markdown("<h1 class='titulo'>🍷 Predictor de Calidad de Vino</h1>", unsafe_allow_html=True)
st.write("Ajusta las características químicas del vino para predecir su calidad (escala del 0 al 10).")
st.divider()

# Layout en 3 columnas para organizar los 11 sliders
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", min_value=4.0, max_value=16.0, value=8.0, step=0.1)
    residual_sugar = st.slider("Residual Sugar", min_value=0.0, max_value=16.0, value=2.5, step=0.1)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", min_value=6.0, max_value=290.0, value=46.0, step=1.0)
    sulphates = st.slider("Sulphates", min_value=0.3, max_value=2.0, value=0.6, step=0.01)

with col2:
    volatile_acidity = st.slider("Volatile Acidity", min_value=0.1, max_value=2.0, value=0.5, step=0.01)
    chlorides = st.slider("Chlorides", min_value=0.01, max_value=0.65, value=0.08, step=0.001)
    density = st.slider("Density", min_value=0.990, max_value=1.004, value=0.996, step=0.001)
    alcohol = st.slider("Alcohol", min_value=8.0, max_value=15.0, value=10.4, step=0.1)

with col3:
    citric_acid = st.slider("Citric Acid", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", min_value=1.0, max_value=75.0, value=14.0, step=1.0)
    pH = st.slider("pH", min_value=2.7, max_value=4.0, value=3.3, step=0.01)

st.divider()

# --- PREDICCIÓN ---
_, col_btn, _ = st.columns([1, 1, 1])

with col_btn:
    predict_btn = st.button("Hacer Predicción", use_container_width=True)

if predict_btn:
    features = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                 chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                 pH, sulphates, alcohol]]
    
    column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                    'pH', 'sulphates', 'alcohol']
    
    df_features = pd.DataFrame(features, columns=column_names)
    prediction = model.predict(df_features)[0]
    
    st.markdown(f"<div class='prediccion'>La calidad predicha de este vino es: {prediction}</div>", unsafe_allow_html=True)
    st.balloons()