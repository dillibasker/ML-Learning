import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Step 1: Training Data (2D)
# -----------------------------------
X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [6, 5],
    [7, 6],
    [8, 5]
])

y = np.array([0, 0, 0, 1, 1, 1])

# -----------------------------------
# Step 2: Separate classes
# -----------------------------------
X0 = X[y == 0]
X1 = X[y == 1]

# -----------------------------------
# Step 3: Mean and variance per feature
# -----------------------------------
mean0 = X0.mean(axis=0)
var0  = X0.var(axis=0)

mean1 = X1.mean(axis=0)
var1  = X1.var(axis=0)

# -----------------------------------
# Step 4: Prior probabilities
# -----------------------------------
prior0 = len(X0) / len(X)
prior1 = len(X1) / len(X)

# -----------------------------------
# Step 5: 1D Gaussian PDF
# -----------------------------------
def gaussian_pdf(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))

# -----------------------------------
# Step 6: Posterior computation
# -----------------------------------
def posterior(x, mean, var, prior):
    return gaussian_pdf(x[0], mean[0], var[0]) * \
           gaussian_pdf(x[1], mean[1], var[1]) * prior

# -----------------------------------
# Step 7: Decision boundary grid
# -----------------------------------
x1_range = np.linspace(0, 9, 200)
x2_range = np.linspace(0, 9, 200)

X1g, X2g = np.meshgrid(x1_range, x2_range)

Z = np.zeros_like(X1g)

for i in range(X1g.shape[0]):
    for j in range(X1g.shape[1]):
        point = np.array([X1g[i, j], X2g[i, j]])
        p0 = posterior(point, mean0, var0, prior0)
        p1 = posterior(point, mean1, var1, prior1)
        Z[i, j] = p0 - p1

# -----------------------------------
# Step 8: TEST DATA
# -----------------------------------
test_point = np.array([4, 4])

p0_test = posterior(test_point, mean0, var0, prior0)
p1_test = posterior(test_point, mean1, var1, prior1)

predicted_class = 0 if p0_test > p1_test else 1

# -----------------------------------
# Step 9: Plot
# -----------------------------------
plt.contour(X1g, X2g, Z, levels=[0], colors='black')

plt.scatter(X0[:,0], X0[:,1], color='blue', label='Class 0 (Train)')
plt.scatter(X1[:,0], X1[:,1], color='red', label='Class 1 (Train)')

plt.scatter(test_point[0], test_point[1], color='green', s=100, label='Test Point')

plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.title("2D Naive Bayes Decision Boundary (Manual Gaussian)")
plt.legend()
plt.grid()
plt.show()

# -----------------------------------
# Step 10: Output
# -----------------------------------
print("Test Point:", test_point)
print("P(Class 0 | x) =", round(p0_test, 6))
print("P(Class 1 | x) =", round(p1_test, 6))
print("Predicted Class =", predicted_class)
