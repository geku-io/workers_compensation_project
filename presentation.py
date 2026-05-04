import streamlit as st


def presentation_page():
    st.title("Презентация проекта")

    st.header("Цель")
    st.write("Предсказать стоимость страхового возмещения")

    st.header("Данные")
    st.write("100 000 записей, 14 признаков")

    st.header("Этапы работы")
    st.write("""
    - Загрузка данных  
    - Предобработка  
    - Обучение модели  
    - Оценка качества  
    """)

    st.header("Модель")
    st.write("Random Forest")

    st.header("Метрики")
    st.write("MAE, RMSE, R²")

    st.header("Итог")
    st.success("Модель успешно предсказывает стоимость страховых выплат")