import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)  # Load diabetes dataset
df = data['frame']  # Get DataFrame from dataset
print(data.DESCR)  # Print dataset description
print(df.head())  # Show first few rows

plt.hist(df["target"], 25)  # Plot histogram of target
plt.xlabel("target")
plt.title("Target Distribution")
plt.show()

sns.heatmap(df.corr().round(2), annot=True)  # Correlation heatmap
plt.title("Correlation Matrix")
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'])  # Scatter plot: bmi vs target
plt.xlabel('bmi')
plt.ylabel('target')

plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'])  # Scatter plot: s5 vs target
plt.xlabel('s5')
plt.ylabel('target')
plt.tight_layout()
plt.show()

X_base = df[['bmi', 's5']]  # Base features
y = df['target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=5)  # Train/test split

model = LinearRegression()  # Initialize linear regression
model.fit(X_train, y_train)  # Train model

y_train_pred = model.predict(X_train)  # Predict train
y_test_pred = model.predict(X_test)  # Predict test

rmse_base = np.sqrt(mean_squared_error(y_train, y_train_pred))  # Train RMSE base
r2_base = r2_score(y_train, y_train_pred)  # Train R2 base
rmse_base_test = np.sqrt(mean_squared_error(y_test, y_test_pred))  # Test RMSE base
r2_base_test = r2_score(y_test, y_test_pred)  # Test R2 base

print("Base RMSE Train:", rmse_base, "R2:", r2_base)
print("Base RMSE Test:", rmse_base_test, "R2:", r2_base_test)

"""
a) We add 'bp' because blood pressure is medically relevant and moderately correlated with target.
"""

X_bp = df[['bmi', 's5', 'bp']]  # Add bp to features

X_train, X_test, y_train, y_test = train_test_split(X_bp, y, test_size=0.2, random_state=5)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

rmse_bp = np.sqrt(mean_squared_error(y_test, y_test_pred))  # RMSE after adding bp
r2_bp = r2_score(y_test, y_test_pred)  # R2 after adding bp

print("With bp RMSE Test:", rmse_bp, "R2:", r2_bp)

"""
b) Adding 'bp' slightly improves model performance with lower RMSE and higher R2.
"""

X_all = df.drop(columns=["target"])  # Use all features
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=5)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

rmse_all = np.sqrt(mean_squared_error(y_test, y_test_pred))  # RMSE with all features
r2_all = r2_score(y_test, y_test_pred)  # R2 with all features

print("All Features RMSE Test:", rmse_all, "R2:", r2_all)

"""
d) Using all variables improves prediction the most with highest R2 and lowest RMSE.
"""


