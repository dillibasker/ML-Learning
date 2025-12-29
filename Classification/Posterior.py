import numpy as np
import matplotlib.pyplot as plt

P_D = 0.01          
P_H = 1 - P_D       
P_Pos_given_D = 0.99
P_Neg_given_H = 0.95
P_Pos_given_H = 1 - P_Neg_given_H

P_Pos = (P_Pos_given_D * P_D) + (P_Pos_given_H * P_H)

P_D_given_Pos = (P_Pos_given_D * P_D) / P_Pos

print("\n--- POSTERIOR PROBABILITY CALCULATION ---\n")

print(f"P(Disease) = {P_D}")
print(f"P(Healthy) = {P_H}")
print(f"P(Positive | Disease) = {P_Pos_given_D}")
print(f"P(Positive | Healthy) = {P_Pos_given_H}")

print("\nStep 1: Total Probability of Positive")
print(f"P(Positive) = (0.99 × 0.01) + (0.05 × 0.99)")
print(f"P(Positive) = {P_Pos:.5f}")

print("\nStep 2: Posterior Probability")
print(f"P(Disease | Positive) = (0.99 × 0.01) / {P_Pos:.5f}")
print(f"P(Disease | Positive) = {P_D_given_Pos:.5f}")

print("\nFinal Answer:")
print(f"Posterior Probability = {P_D_given_Pos*100:.2f}%")

labels = ['Disease', 'Healthy']
prior_probs = [P_D, P_H]
posterior_probs = [P_D_given_Pos, 1 - P_D_given_Pos]

x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, prior_probs, width, label='Prior Probability')
plt.bar(x + width/2, posterior_probs, width, label='Posterior Probability')

plt.xlabel('Condition')
plt.ylabel('Probability')
plt.title('Prior vs Posterior Probability')
plt.xticks(x, labels)
plt.legend()
plt.show()