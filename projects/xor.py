import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD XOR DATA
data = pd.read_csv("xor_data.csv")

X = data[['x1', 'x2']].values
y = data['label'].values.reshape(-1, 1)

# 2. ACTIVATION FUNCTIONS
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# 3. INITIALIZATION
np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

learning_rate = 0.1   # FIXED
epochs = 10000        # MORE TRAINING

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# 4. TRAINING
for epoch in range(epochs):
    # Forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)

    # Backprop
    dz2 = y_pred - y
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = dz2 @ W2.T * sigmoid_derivative(a1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# 5. FINAL ACCURACY
y_final = (y_pred >= 0.5).astype(int)
accuracy = np.mean(y_final == y) * 100

print("Final Predictions:")
for i in range(len(X)):
    print(X[i], "Pred:", y_final[i][0], "Actual:", y[i][0])

print(f"\nOverall Accuracy: {accuracy:.2f}%")

# 6. FULL DECISION REGION PLOT

# Create grid
xx, yy = np.meshgrid(
    np.linspace(-0.2, 1.2, 300),
    np.linspace(-0.2, 1.2, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Predict on grid
z1_grid = sigmoid(grid @ W1 + b1)
z2_grid = sigmoid(z1_grid @ W2 + b2)
grid_pred = (z2_grid >= 0.5).astype(int)
grid_pred = grid_pred.reshape(xx.shape)

# Plot decision regions
plt.contourf(xx, yy, grid_pred, alpha=0.3, cmap="coolwarm")

# Plot actual XOR points
plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap="coolwarm", edgecolors="k", s=100)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title(f"XOR Decision Boundary (Accuracy = {accuracy:.2f}%)")
plt.show()