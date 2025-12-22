import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# LOAD DATA
train_df = pd.read_csv("train_2d.csv")
test_df = pd.read_csv("test_2d.csv")

X_train_raw = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test_raw = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

degree = 3  # change manually

n_features = X_train_raw.shape[1]

# DESIGN MATRIX (GENERAL)
def design_matrix_nd(X, degree):
    n_samples, n_features = X.shape
    Phi = np.ones((n_samples, 1))

    for d in range(1, degree + 1):
        for f in range(n_features):
            Phi = np.column_stack((Phi, X[:, f] ** d))

    return Phi

# TRAIN
X_train = design_matrix_nd(X_train_raw, degree)
w = np.linalg.pinv(X_train) @ y_train

# PREDICT
y_train_pred = X_train @ w
X_test = design_matrix_nd(X_test_raw, degree)
y_test_pred = X_test @ w

# METRICS
train_mse = np.mean((y_train - y_train_pred) ** 2)
train_rmse = np.sqrt(train_mse)

test_mse = np.mean((y_test - y_test_pred) ** 2)
test_rmse = np.sqrt(test_mse)

# PRINT RESULTS
print("Degree:", degree)
print("Number of Features:", n_features)
print("Weights:", w)
print("Train MSE:", train_mse)
print("Train RMSE:", train_rmse)
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)

# ================= VISUALIZATION =================

# 1D → 2D PLOT
if n_features == 1:
    x_train = X_train_raw[:, 0]
    x_test = X_test_raw[:, 0]

    plt.scatter(x_train, y_train, color='blue', label='Train')
    plt.scatter(x_test, y_test, color='red', label='Test')

    x_line = np.linspace(min(x_train), max(x_train), 300).reshape(-1, 1)
    y_line = design_matrix_nd(x_line, degree) @ w

    plt.plot(x_line, y_line, color='green', label='Regression Line')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(f"1D Polynomial Regression (Degree {degree})")
    plt.legend()
    plt.show()

# 2D → 3D PLOT
elif n_features == 2:
    x1 = X_train_raw[:, 0]
    x2 = X_train_raw[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, x2, y_train, color='blue', label='Train')
    ax.scatter(X_test_raw[:,0], X_test_raw[:,1], y_test, color='red', label='Test')

    x1_line, x2_line = np.meshgrid(
        np.linspace(min(x1), max(x1), 30),
        np.linspace(min(x2), max(x2), 30)
    )

    grid = np.column_stack((x1_line.ravel(), x2_line.ravel()))
    y_surface = design_matrix_nd(grid, degree) @ w

    ax.plot_surface(x1_line, x2_line,
                    y_surface.reshape(x1_line.shape),
                    alpha=0.5)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    ax.set_title(f"2D Polynomial Regression (Degree {degree})")
    plt.show()

# ≥3D → NO PLOT
else:
    print("Visualization skipped (dimension > 2)")
