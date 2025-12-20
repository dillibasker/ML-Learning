import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#extraxt data
train_df = pd.read_csv("train_1b.csv")
x_train = train_df["X"].values
y_train = train_df["y"].values

test_df = pd.read_csv("test.csv")
x_test = test_df["X"].values
y_test = test_df["y"].values

degree = 3   

# DESIGN MATRIX
def design_matrix(x, degree):
    X = np.ones((len(x), 1))  # bias term
    for d in range(1, degree + 1):
        X = np.column_stack((X, x ** d))
    return X

# TRAINING
X_train = design_matrix(x_train, degree)
w = np.linalg.pinv(X_train) @ y_train

# PREDICTION
y_train_pred = X_train @ w
X_test = design_matrix(x_test, degree)
y_test_pred = X_test @ w

# ERROR METRICS
train_mse = np.mean((y_train - y_train_pred) ** 2)
train_rmse = np.sqrt(train_mse)

test_mse = np.mean((y_test - y_test_pred) ** 2)
test_rmse = np.sqrt(test_mse)

print("Degree:", degree)
print("Weights:", w)
print("Train MSE:", train_mse)
print("Train RMSE:", train_rmse)
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)

plt.scatter(x_train, y_train, color='blue', label='Train Data')
plt.scatter(x_test, y_test, color='red', label='Test Data')

x_line = np.linspace(min(np.concatenate([x_train,x_test])),
                     max(np.concatenate([x_train,x_test])), 300)
y_line = design_matrix(x_line, degree) @ w
plt.plot(x_line, y_line, color='green', label='Regression Line')

plt.xlabel("X")
plt.ylabel("y")
plt.title(f"Polynomial Regression (Degree {degree})")
plt.legend()
plt.show()