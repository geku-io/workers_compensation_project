import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def load_data():
   data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
   return data.frame


def preprocess_data(df):
   data = df.copy()

   data['DateTimeOfAccident'] = pd.to_datetime(data['DateTimeOfAccident'])
   data['DateReported'] = pd.to_datetime(data['DateReported'])

   data['AccidentMonth'] = data['DateTimeOfAccident'].dt.month
   data['AccidentDayOfWeek'] = data['DateTimeOfAccident'].dt.dayofweek
   data['ReportingDelay'] = (
      data['DateReported'] - data['DateTimeOfAccident']
   ).dt.days

   data = data.drop(columns=['DateTimeOfAccident', 'DateReported'])

   categorical_columns = [
      'Gender',
      'MaritalStatus',
      'PartTimeFullTime',
      'ClaimDescription'
   ]

   label_encoders = {}
   for col in categorical_columns:
      le = LabelEncoder()
      data[col] = le.fit_transform(data[col].astype(str))
      label_encoders[col] = le

   data = data.fillna(0)

   data = data[data['UltimateIncurredClaimCost'] >= 0]

   numerical_features = [
      'Age', 'DependentChildren', 'DependentsOther',
      'WeeklyPay', 'HoursWorkedPerWeek', 'DaysWorkedPerWeek',
      'InitialCaseEstimate', 'AccidentMonth',
      'AccidentDayOfWeek', 'ReportingDelay'
   ]

   scaler = StandardScaler()
   data[numerical_features] = scaler.fit_transform(data[numerical_features])

   return data, scaler, label_encoders


def train_model(data):
   X = data.drop(columns=['UltimateIncurredClaimCost'])
   y = data['UltimateIncurredClaimCost']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
   y_pred = model.predict(X_test)

   mae = mean_absolute_error(y_test, y_pred)
   mse = mean_squared_error(y_test, y_pred)
   rmse = np.sqrt(mse)
   r2 = r2_score(y_test, y_pred)

   return mae, rmse, r2, y_pred


def analysis_and_model_page():
   st.title("Прогнозирование стоимости страховых выплат")

   if st.button("Загрузить данные"):
      df = load_data()
      st.session_state['df'] = df

   if 'df' in st.session_state:
      df = st.session_state['df']

      st.subheader("Сырые данные")
      st.write(df.head())

      st.subheader("Статистика")
      st.write(df.describe())

      data, scaler, encoders = preprocess_data(df)

      st.subheader("После предобработки")
      st.write(data.head())

      model, X_train, X_test, y_train, y_test = train_model(data)

      mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

      st.subheader("Метрики")
      st.write(f"MAE: {mae:.2f}")
      st.write(f"RMSE: {rmse:.2f}")
      st.write(f"R²: {r2:.4f}")

      st.subheader("Предсказания vs Реальные значения")
      fig, ax = plt.subplots()
      ax.scatter(y_test, y_pred, alpha=0.3)
      ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
      st.pyplot(fig)

      st.subheader("Важность признаков")
      importance = pd.DataFrame({
         'feature': X_train.columns,
         'importance': model.feature_importances_
      }).sort_values('importance', ascending=False)

      st.write(importance.head(10))

      st.header("Предсказание")

      with st.form("form"):
         age = st.number_input("Возраст", 13, 76, 35)
         weekly_pay = st.number_input("Зарплата", 0, 5000, 500)
         initial_estimate = st.number_input("Начальная оценка", 0, 50000, 5000)

         submit = st.form_submit_button("Предсказать")

         if submit:
            sample = X_train.iloc[0].copy()

            sample['Age'] = age
            sample['WeeklyPay'] = weekly_pay
            sample['InitialCaseEstimate'] = initial_estimate

            prediction = model.predict([sample])[0]

            st.success(f"Прогноз: ${prediction:.2f}")