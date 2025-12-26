import numpy as np
import matplotlib.pyplot as plt

# ---------------- DATA ----------------
X = np.array([
    [1, 2], [2, 3], [3, 4],
    [6, 7], [7, 8], [8, 9],
    [9,10] , [10,11] , [11,12]
])

# Number of clusters
K = 4

# Randomly initialize centroids
np.random.seed(42)
centroids = X[np.random.choice(len(X), K, replace=False)]

# ---------------- K-MEANS ----------------
def kmeans(X, centroids, max_iter=10):
    for _ in range(max_iter):
        clusters = {i: [] for i in range(K)}

        # Assign points to nearest centroid
        for point in X:
            distances = [np.linalg.norm(point - c) for c in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        # Update centroids
        new_centroids = []
        for i in range(K):
            new_centroids.append(np.mean(clusters[i], axis=0))
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


clusters, centroids = kmeans(X, centroids)

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(7, 7))
colors = ['blue', 'green', 'orange', 'purple']

for idx, points in clusters.items():
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], color=colors[idx], label=f'Cluster {idx+1}')

    # Draw cluster boundary
    center = centroids[idx]
    radius = max(np.linalg.norm(points - center, axis=1)) + 0.5
    circle = plt.Circle(center, radius, fill=False, linestyle='--', color=colors[idx])
    plt.gca().add_patch(circle)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1],
            color='red', marker='X', s=200, label='Centroids')

plt.title(f"K-Means Clustering (K = {K})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
