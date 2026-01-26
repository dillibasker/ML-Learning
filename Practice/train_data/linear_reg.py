import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data.csv")
df.info()
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.nunique())
print(df.describe())
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,8))
plt.title("Feature Correlation Heatmap")
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()