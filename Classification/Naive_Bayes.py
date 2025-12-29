import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'Email_ID': [1, 2, 3, 4],
    'Words': ['win money free', 'win lottery', 'hello friend', 'meeting schedule'],
    'Class': ['Spam', 'Spam', 'Not Spam', 'Not Spam']
}

df = pd.DataFrame(data)
vocab = list(set(" ".join(df['Words']).split()))

classes = df['Class'].unique()
priors = {c: len(df[df['Class']==c])/len(df) for c in classes}

likelihoods = {}
V = len(vocab)

for c in classes:
    words_in_class = " ".join(df[df['Class']==c]['Words']).split()
    total_words = len(words_in_class)
    likelihoods[c] = {}
    for word in vocab:
        count = words_in_class.count(word)
        likelihoods[c][word] = (count + 1) / (total_words + V)  # Laplace smoothing

new_email = "win free lottery"
new_words = new_email.split()

posterior = {}
contributions = {}  # For graph
for c in classes:
    posterior[c] = priors[c]
    contributions[c] = [priors[c]]  # Start with prior
    for word in new_words:
        word_prob = likelihoods[c].get(word, 1/V)
        posterior[c] *= word_prob
        contributions[c].append(word_prob)

predicted_class = max(posterior, key=posterior.get)

print(f"New Email: '{new_email}'\n")
print("Posterior Probabilities:")
for c in posterior:
    print(f"P({c}|Email) = {posterior[c]:.6f}")
print(f"\nPredicted Class: {predicted_class}")

fig, ax = plt.subplots(figsize=(8,5))

for c in classes:
    
    log_contrib = np.log(contributions[c])
    ax.plot(range(len(log_contrib)), log_contrib.cumsum(), marker='o', label=c)

ax.set_xticks(range(len(contributions[classes[0]])))
ax.set_xticklabels(['Prior'] + new_words)
ax.set_ylabel("Log Cumulative Probability")
ax.set_xlabel("Factors (Words)")
ax.set_title("Naive Bayes Posterior Probability Contributions")
ax.legend()
ax.grid(True)
plt.show()