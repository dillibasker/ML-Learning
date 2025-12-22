import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n_samples = 80          # number of data points
n_features = 2    # CHANGE THIS: 1, 2, 3, 4...
degree = 2              # 1 = linear, >1 = polynomial
noise_std = 0.5         # noise level
random_seed = 42

np.random.seed(random_seed)

# Generate input features
X = np.random.uniform(-1, 1, size=(n_samples, n_features))

# True weights (unknown in real world)
true_w = np.random.randn(1 + n_features * degree)

# Design matrix generator
def design_matrix_nd(X, degree):
    n_samples, n_features = X.shape
    Phi = np.ones((n_samples, 1))  # bias

    for d in range(1, degree + 1):
        for f in range(n_features):
            Phi = np.column_stack((Phi, X[:, f] ** d))

    return Phi

# Generate output with noise
Phi_true = design_matrix_nd(X, degree)
y = Phi_true @ true_w + np.random.normal(0, noise_std, size=n_samples)

# PANDAS DATAFRAME 
columns = [f"X{i+1}" for i in range(n_features)]
df = pd.DataFrame(X, columns=columns)
df["y"] = y

# Train-Test Split
train_df = df.sample(frac=0.75, random_state=1)
test_df = df.drop(train_df.index)

X_train = train_df.iloc[:, :-1].values
y_train = train_df["y"].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df["y"].values

#TRAINING 
Phi_train = design_matrix_nd(X_train, degree)
w = np.linalg.pinv(Phi_train) @ y_train

#PREDICTION 
y_train_pred = Phi_train @ w
Phi_test = design_matrix_nd(X_test, degree)
y_test_pred = Phi_test @ w

#ERROR METRICS
train_mse = np.mean((y_train - y_train_pred) ** 2)
train_rmse = np.sqrt(train_mse)

test_mse = np.mean((y_test - y_test_pred) ** 2)
test_rmse = np.sqrt(test_mse)

#  RESULTS 
print("\n===== MODEL SUMMARY =====")
print("Number of samples:", n_samples)
print("Number of features:", n_features)
print("Polynomial degree:", degree)
print("Weights:", w)
print("Train MSE:", train_mse)
print("Train RMSE:", train_rmse)
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)

# VISUALIZATION 

#  1D INPUT → 2D PLOT
if n_features == 1:
    x_train = X_train[:, 0]
    x_test = X_test[:, 0]

    plt.scatter(x_train, y_train, color='blue', label='Train')
    plt.scatter(x_test, y_test, color='red', label='Test')

    x_line = np.linspace(min(x_train), max(x_train), 300).reshape(-1, 1)
    y_line = design_matrix_nd(x_line, degree) @ w

    plt.plot(x_line, y_line, color='green', label='Regression Curve')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("1D Polynomial Regression")
    plt.legend()
    plt.show()

#  2D INPUT → 3D PLOT
elif n_features == 2:
    x1 = X_train[:, 0]
    x2 = X_train[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, x2, y_train, color='blue', label='Train')
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='red', label='Test')

    x1_grid, x2_grid = np.meshgrid(
        np.linspace(x1.min(), x1.max(), 30),
        np.linspace(x2.min(), x2.max(), 30)
    )

    grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
    y_surface = design_matrix_nd(grid, degree) @ w

    ax.plot_surface(
        x1_grid, x2_grid,
        y_surface.reshape(x1_grid.shape),
        alpha=0.5
    )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    ax.set_title("2D Polynomial Regression")
    plt.show()

#  ≥3D → NO VISUALIZATION
else:
    print("\nVisualization skipped (dimension > 2)")
