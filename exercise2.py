import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("weight-height(1).csv")


X = df[['Height']]
y = df['Weight']


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label="data")
plt.plot(X, y_pred, color='red', label="regression line")
plt.xlabel("height")
plt.ylabel("weight")
plt.title("linear regression: height vs weight")
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R square Score: {r2:.4f}")