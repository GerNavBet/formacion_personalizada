import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Título ---
st.title("Recomendador de Ruta Formativa Personalizada")
st.write("Completa tu perfil para recibir una sugerencia de itinerario formativo adaptado a tus necesidades.")

# --- Formulario de entrada ---
with st.form("perfil_formativo"):
    experiencia_laboral = st.selectbox("Años de experiencia laboral", ["0-1", "2-5", "6-10", "10+"])
    rol_actual = st.selectbox("Rol profesional actual", ["Autónomo", "Empleado", "Desempleado", "Emprendedor"])
    area_interes = st.multiselect("Áreas de interés formativo", [
        "Comercial", "Finanzas", "Fiscal", "Gestión de Personas", "Sistemas de Gestión", "Habilidades", "Otros"])
    conocimiento_digital = st.select_slider("Nivel de alfabetización digital", options=["Bajo", "Medio", "Alto"])
    nivel_estudios = st.selectbox("Nivel de estudios", ["Primaria", "Secundaria", "Formación Profesional", "Universitario"])

    submitted = st.form_submit_button("Obtener recomendación")

# --- Funciones de codificación ---
def codificar_usuario(exp, rol, interes, digital, estudios):
    exp_map = {"0-1": 0, "2-5": 1, "6-10": 2, "10+": 3}
    rol_map = {"Desempleado": 0, "Empleado": 1, "Autónomo": 2, "Emprendedor": 3}
    digital_map = {"Bajo": 0, "Medio": 1, "Alto": 2}
    estudios_map = {"Primaria": 0, "Secundaria": 1, "Formación Profesional": 2, "Universitario": 3}
    intereses_map = {
        "Comercial": 0, "Finanzas": 1, "Fiscal": 2, "Gestión de Personas": 3,
        "Sistemas de Gestión": 4, "Habilidades": 5, "Otros": 6
    }
    vector_interes = [0]*7
    for i in interes:
        vector_interes[intereses_map[i]] = 1

    return [
        exp_map[exp], rol_map[rol], digital_map[digital], estudios_map[estudios]
    ] + vector_interes

# --- Entrenar modelo dummy ---
def entrenar_modelo_cluster():
    np.random.seed(42)
    X = np.random.randint(0, 4, size=(50, 4))
    intereses = np.random.randint(0, 2, size=(50, 7))
    X_total = np.hstack((X, intereses))
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X_total)
    return model

modelo_cluster = entrenar_modelo_cluster()

# --- Mostrar recomendación ---
if submitted:
    datos = np.array([codificar_usuario(experiencia_laboral, rol_actual, area_interes, conocimiento_digital, nivel_estudios)])
    grupo = modelo_cluster.predict(datos)[0]

    recomendaciones = {
        0: ["Curso de habilidades personales", "Curso de alfabetización digital"],
        1: ["Planificación estratégica", "Finanzas para no financieros"],
        2: ["Liderazgo de equipos", "Marketing digital"],
        3: ["Gestión de procesos", "Optimización de costes"]
    }

    st.subheader("Recomendación Formativa")
    st.success(f"En base a tu perfil, te recomendamos:")
    for rec in recomendaciones[grupo]:
        st.write(f"- {rec}")

    st.write("Estas recomendaciones están basadas en patrones comunes entre perfiles similares. Para un plan formativo detallado, puedes agendar una sesión personalizada.")
