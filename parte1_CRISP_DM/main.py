import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Proyecto Integrador: Análisis de datos de vinos")

st.caption(
    'El presente tablero forma parte del proyecto integrador de la asignatura "Introducción a la Ciencia de Datos y sus Metodologías" del programa de la Maestría en Ciencia de Datos de la Universidad de Sonora. La realización de este trabajo fue a cargo de Santiago Robles, Yanet Hernández, María Elena Manzanares y Martín Vega.  El modelo de árbol de decisión aquí presentado fue entrenado a partir de los datos de calidad de vino rojo de la UCI.'
)

col1, col2, col3 = st.columns(3)
with col1:
    fixed_acidity = st.number_input("Insert fixed acidity", min_value=0.0)
with col2:
    volatile_acidity = st.number_input("Insert volatile acidity", min_value=0.0)
with col3:
    citric_acid = st.number_input("Insert citric acid", min_value=0.0)

col1, col2, col3 = st.columns(3)
with col1:
    residual_sugar = st.number_input("Insert residual sugar", min_value=0.0)
with col2:
    chlorides = st.number_input("Insert chlorides", min_value=0.0)
with col3:
    free_sulfur_dioxide = st.number_input("Insert free sulfur dioxide", min_value=0.0)

col1, col2, col3 = st.columns(3)
with col1:
    total_sulfur_dioxide = st.number_input("Insert total sulfur dioxide", min_value=0.0)
with col2:
    density = st.number_input("Insert density", min_value=0.0)
with col3:
    pH = st.number_input("Insert pH", min_value=0.0)

col1, col2 = st.columns(2)
with col1:
    sulphates = st.number_input("Insert sulphates", min_value=0.0)
with col2:
    alcohol = st.number_input("Insert alcohol", min_value=0.0)

data_dict = {
    "fixed acidity": [fixed_acidity],
    "volatile acidity	": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol],
}

df = pd.DataFrame.from_dict(data_dict)
st.dataframe(df)
model_name = "model.pkl"
clf = pickle.load(open(model_name, "rb"))

col1, col2, col3 = st.columns(3)
with col1:
    st.text(" ")
with col2:
    if st.button("Predict"):
        prediction = clf.predict(df)
        st.metric("Clasification", prediction)

with col3:
    st.text(" ")
