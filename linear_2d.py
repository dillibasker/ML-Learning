import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("train_2d.csv")
x1 = df["X1"].values
x2 = df["X2"].values
y = df["y"].values

# DESIGN MATRIX (2D)
X = np.column_stack((np.ones(len(x1)), x1, x2))

# TRAINING (Normal Equation)
w = np.linalg.pinv(X) @ y

# PREDICTION
y_pred = X @ w

# ERROR
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)

print("Weights:", w)
print("MSE:", mse)
print("RMSE:", rmse)
