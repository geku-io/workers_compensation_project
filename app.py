import streamlit as st
import analysis_and_model
import presentation

st.set_page_config(page_title="Workers Compensation", layout="wide")

st.sidebar.title("Навигация")

page = st.sidebar.radio(
    "Выберите страницу:",
    ["Анализ и модель", "Презентация"]
)

if page == "Анализ и модель":
    analysis_and_model.analysis_and_model_page()
else:
    presentation.presentation_page()