import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))

    distances.sort(key=lambda x: x[0])
    labels = [label for _, label in distances[:k]]
    return Counter(labels).most_common(1)[0][0]


# ---------------- DATA ----------------
training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = np.array([4, 5])
k = 3

prediction = knn_predict(training_data, training_labels, test_point, k)
print("New comer joins group:", prediction)

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(7, 7))

group_A = np.array([p for p, l in zip(training_data, training_labels) if l == 'A'])
group_B = np.array([p for p, l in zip(training_data, training_labels) if l == 'B'])

plt.scatter(group_A[:, 0], group_A[:, 1], color='blue', label='Class A')
plt.scatter(group_B[:, 0], group_B[:, 1], color='green', label='Class B')

new_color = 'blue' if prediction == 'A' else 'green'
plt.scatter(test_point[0], test_point[1],
            color=new_color, marker='X', s=150, label='New Comer')

# -------- Joined group --------
joined_group = group_A if prediction == 'A' else group_B
center = joined_group.mean(axis=0)

# Radius of group circle
radius = max(np.linalg.norm(joined_group - center, axis=1)) + 0.5

# -------- Compute edge joining point --------
direction = center - test_point
direction_unit = direction / np.linalg.norm(direction)

edge_point = center - direction_unit * radius

# Dotted line: new comer → circle edge
plt.plot(
    [test_point[0], edge_point[0]],
    [test_point[1], edge_point[1]],
    linestyle=':',
    linewidth=2,
    color=new_color,
    label='Join Path'
)

# Draw dotted group boundary
circle = plt.Circle(
    center,
    radius,
    fill=False,
    linestyle=':',
    linewidth=2,
    color=new_color,
    label=f'Group {prediction} Boundary'
)
plt.gca().add_patch(circle)

plt.title(f"New Comer Joining Group {prediction}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.legend()
plt.show()
