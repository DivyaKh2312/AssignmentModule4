import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



x_value = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in x_value:
    d1 = np.random.randint(1, 7, size=n)
    d2 = np.random.randint(1, 7, size=n)
    s = d1 + d2
    h, h2 = np.histogram(s, bins=range(2, 14))

    plt.bar(h2[:-1], h / n)
    plt.title(f"sum of thrown dice : {n}")
    plt.xlabel("sum")
    plt.ylabel("relative frequency")
    plt.xticks(range(2, 13))
    plt.show()

