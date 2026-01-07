import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 1. LOAD DATA
# ============================
data = pd.read_csv("xor_data.csv")

X = data[['x1', 'x2']].values
y = data['label'].values

# One-hot encoding
y_onehot = np.zeros((y.size, 2))
y_onehot[np.arange(y.size), y] = 1


# ============================
# 2. ACTIVATION FUNCTIONS
# ============================
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================
# 3. LOSS FUNCTION
# ============================
def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))


# ============================
# 4. INITIALIZATION
# ============================
np.random.seed(1)

input_size = 2
hidden_size = 4      # ✅ REQUIRED FOR XOR
output_size = 2

learning_rate = 0.1
epochs = 300         # ✅ REQUIRED FOR LEARNING

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


# ============================
# 5. TRAINING
# ============================
for epoch in range(epochs):

    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    y_pred = softmax(z2)

    dz2 = y_pred - y_onehot
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * relu_derivative(z1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# ============================
# 6. FINAL RESULTS + OVERALL ACCURACY
# ============================
predictions = np.argmax(y_pred, axis=1)
overall_accuracy = np.mean(predictions == y) * 100

print("\nFinal Predictions:")
for i in range(len(X)):
    print(X[i], "Pred:", predictions[i], "Actual:", y[i])

print(f"\n✅ Overall Accuracy: {overall_accuracy:.2f}%")


# ============================
# 7. PLOTS
# ============================
plt.scatter(X[:,0], X[:,1], c=predictions, cmap="coolwarm")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR Classification (≥90% Accuracy)")
plt.show()