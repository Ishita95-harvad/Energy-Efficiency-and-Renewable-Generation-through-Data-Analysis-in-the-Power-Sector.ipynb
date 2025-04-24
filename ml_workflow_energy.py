
# 📦 Google Colab: Energy Forecasting ML Workflow

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load Data
df = pd.read_csv("https://raw.githubusercontent.com/datablist/sample-csv-files/main/files/people/people-100.csv")  # Replace with your dataset

# Simulated columns for energy project example
df['Date'] = pd.date_range(start="2023-01-01", periods=len(df), freq='D')
df['Solar'] = np.random.uniform(100, 300, size=len(df))
df['Wind'] = np.random.uniform(50, 200, size=len(df))
df['Thermal'] = np.random.uniform(300, 700, size=len(df))
df['Total_Generation'] = df['Solar'] + df['Wind'] + df['Thermal']

# Step 3: Feature Engineering
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Step 4: Train-Test Split
features = ['Solar', 'Wind', 'Thermal', 'Month', 'DayOfWeek']
target = 'Total_Generation'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 6: Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

# Step 7: Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:50], label='Actual')
plt.plot(y_pred[:50], label='Predicted')
plt.title('Forecast vs Actual')
plt.legend()
plt.show()
