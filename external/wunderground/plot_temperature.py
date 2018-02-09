import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

file = '2018-01-31-2016-09-19.csv'

df = pd.read_csv(file, header=0, encoding='utf-8')

df['tempi'] = pd.to_numeric(df['tempi'], errors='coerce')

df.dropna(axis=0, inplace=True)

print(df.info())
print(df.head())

temp_mean = df.tempm

plt.plot(temp_mean)
plt.show()