import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)

sunny_humidity = np.random.normal(40, 5, 30)
rainy_humidity = np.random.normal(80, 5, 30)

data = pd.DataFrame({
    'Weather': ['Sunny']*30 + ['Rainy']*30,
    'Humidity': np.concatenate([sunny_humidity, rainy_humidity])
})

weathers = data['Weather'].unique()
priors = {w: len(data[data['Weather']==w])/len(data) for w in weathers}

mean_std = {w: (data[data['Weather']==w]['Humidity'].mean(),
                data[data['Weather']==w]['Humidity'].std()) for w in weathers}

def likelihood(x, mean, std):
    return (1/(np.sqrt(2*np.pi)*std)) * np.exp(-((x-mean)**2)/(2*std**2))

new_humidity = float(input("Enter New Day Humidity (%): "))

posterior = {}
for w in weathers:
    mean, std = mean_std[w]
    posterior[w] = likelihood(new_humidity, mean, std) * priors[w]

# Normalize Posterior
total = sum(posterior.values())
for w in posterior:
    posterior[w] /= total

predicted = max(posterior, key=posterior.get)

print(f"\nNew Day Humidity: {new_humidity}%")
for w in posterior:
    print(f"{w}: Posterior = {posterior[w]:.4f}")
print(f"Predicted Weather: {predicted}")

humidity_range = np.linspace(20, 100, 300)
plt.figure(figsize=(9,6))
for w in weathers:
    mean, std = mean_std[w]
    plt.plot(humidity_range, likelihood(humidity_range, mean, std) * priors[w], label=w)

for w in weathers:
    mean, std = mean_std[w]
    y = likelihood(new_humidity, mean, std) * priors[w]
    plt.scatter(new_humidity, y, s=80, zorder=5)

plt.axvline(new_humidity, linestyle='--', label='New Day Humidity')
plt.title('Bayes Classification')
plt.xlabel('Humidity (%)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
