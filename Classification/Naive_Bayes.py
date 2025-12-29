import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Dataset
data = {
    'Email_ID': [1, 2, 3, 4],
    'Words': ['win money free', 'win lottery', 'hello friend', 'meeting schedule'],
    'Class': ['Spam', 'Spam', 'Not Spam', 'Not Spam']
}

df = pd.DataFrame(data)

# Step 2: Vocabulary
vocab = list(set(" ".join(df['Words']).split()))
V = len(vocab)

# Step 3: Prior Probabilities
classes = df['Class'].unique()
priors = {c: len(df[df['Class'] == c]) / len(df) for c in classes}

# Step 4: Likelihood with Laplace Smoothing
likelihoods = {}

for c in classes:
    words = " ".join(df[df['Class'] == c]['Words']).split()
    total_words = len(words)
    likelihoods[c] = {}

    for word in vocab:
        count = words.count(word)
        likelihoods[c][word] = (count + 1) / (total_words + V)

# Step 5: New Email Prediction
new_email = "hello welcome"
new_words = new_email.split()

posterior = {}
contributions = {}

for c in classes:
    posterior[c] = priors[c]
    contributions[c] = [priors[c]]

    for word in new_words:
        prob = likelihoods[c].get(word, 1 / V)
        posterior[c] *= prob
        contributions[c].append(prob)

# Step 6: Prediction Result
predicted_class = max(posterior, key=posterior.get)

print("New Email:", new_email)
print("\nPosterior Probabilities:")
for c in posterior:
    print(f"P({c} | Email) = {posterior[c]:.6f}")

print("\nPredicted Class:", predicted_class)

# Step 7: Visualization
plt.figure(figsize=(8,5))
for c in classes:
    log_probs = np.log(contributions[c])
    plt.plot(log_probs.cumsum(), marker='o', label=c)

plt.xticks(range(len(new_words) + 1), ['Prior'] + new_words)
plt.xlabel("Words")
plt.ylabel("Log Cumulative Probability")
plt.title("Naive Bayes Email Classification")
plt.legend()
plt.grid()
plt.show()
