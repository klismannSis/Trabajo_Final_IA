
import streamlit as st
import pandas as pd
import joblib

st.title("🔍 Predicción de Satisfacción del Cliente")

# Cargar modelo
model = joblib.load("app/model.pkl")
label_encoders = joblib.load("app/label_encoders.pkl")

# Crear formulario
def user_input():
    genero = st.selectbox("Género", ["Male", "Female"])
    edad = st.slider("Edad", 10, 90, 35)
    tipo_viaje = st.selectbox("Tipo de viaje", ["Business travel", "Personal travel"])
    clase = st.selectbox("Clase", ["Eco", "Business", "Eco Plus"])
    puntualidad = st.slider("Llegó a tiempo", 0, 5, 3)
    entretenimiento = st.slider("Entretenimiento a bordo", 0, 5, 3)
    limpieza = st.slider("Limpieza", 0, 5, 3)

    data = {
        "Gender": genero,
        "Age": edad,
        "Type of Travel": tipo_viaje,
        "Class": clase,
        "Flight Distance": 1000,
        "Inflight entertainment": entretenimiento,
        "On-board service": 3,
        "Cleanliness": limpieza,
        "Arrival Delay in Minutes": 15,
        "Departure Delay in Minutes": 10
    }
    return pd.DataFrame([data])

input_df = user_input()

# Codificar datos
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
# Ajustar columnas faltantes
missing_cols = set(model.feature_names_in_) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

input_df = input_df[model.feature_names_in_]

# Predicción
if st.button("🔮 Predecir satisfacción"):
    pred = model.predict(input_df)[0]
    resultado = "Satisfecho" if pred == 1 else "Insatisfecho"
    st.success(f"✅ Resultado: El cliente probablemente estará **{resultado}**")
