import numpy as np
import matplotlib.pyplot as plt

X = np.array([1.0, 1.5, 2.0, 5.0, 5.5, 6.0])

means = np.array([1.5, 5.5])
variances = np.array([0.5, 0.5])
weights = np.array([0.4, 0.6])

def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))

print("\n--- GMM PROBABILITY CALCULATION ---\n")

gmm_probs = []

for x in X:
    print(f"\nData Point: {x}")

    component_probs = []
    for i in range(2):
        p = gaussian(x, means[i], variances[i])
        weighted_p = weights[i] * p

        component_probs.append(weighted_p)

        print(f"  Gaussian {i+1}:")
        print(f"     Mean = {means[i]}, Variance = {variances[i]}")
        print(f"     P(x) = {p:.6f}")
        print(f"     Weight = {weights[i]}")
        print(f"     Weighted P = {weighted_p:.6f}")

    total_prob = sum(component_probs)
    gmm_probs.append(total_prob)

    print(f"  --> Final GMM Probability = {total_prob:.6f}")

plt.plot(X, gmm_probs, marker='o')
plt.xlabel("Data")
plt.ylabel("GMM Probability")
plt.title("Gaussian Mixture Model Probability Curve")
plt.show()