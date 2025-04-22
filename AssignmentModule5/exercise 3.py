import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


auto = pd.read_csv("Auto.csv", na_values="?").dropna()  # Load Auto.csv and drop missing
print(auto.head())  # Show data

X = auto.drop(columns=["mpg", "name", "origin"])  # Exclude non-numeric and target
y = auto["mpg"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)  # Train/test split

alphas = np.logspace(-3, 2, 50)  # Alpha range
r2_ridge = []  # Ridge R2 scores
r2_lasso = []  # Lasso R2 scores

for alpha in alphas:  # Loop over alpha values
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    r2_ridge.append(ridge.score(X_test, y_test))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    r2_lasso.append(lasso.score(X_test, y_test))

plt.plot(alphas, r2_ridge, label='Ridge')  # Plot Ridge scores
plt.plot(alphas, r2_lasso, label='Lasso')  # Plot Lasso scores
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('R2 Score vs Alpha')
plt.legend()
plt.show()

best_alpha_ridge = alphas[np.argmax(r2_ridge)]  # Best alpha for Ridge
best_alpha_lasso = alphas[np.argmax(r2_lasso)]  # Best alpha for Lasso

print("Best Ridge alpha:", best_alpha_ridge, "R2:", max(r2_ridge))
print("Best Lasso alpha:", best_alpha_lasso, "R2:", max(r2_lasso))

"""
Best alpha for Ridge and Lasso found by plotting and selecting value with highest R2 score.
"""
