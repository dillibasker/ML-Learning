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