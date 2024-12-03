import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("co2_mm_mlo.csv", skiprows=52, usecols=[0, 3], delimiter=",")
df.columns = ['year', 'average']  

filtered_data = df[(df['year'] >= 1981) & (df['year'] <= 1990)]
 

plt.figure(figsize=(10, 6))
plt.scatter(filtered_data.index, filtered_data['average'], label='CO2 Levels (1981-1990)', marker='+')
plt.xlabel('Index')
plt.ylabel('CO2 Levels (ppm)')
plt.title('CO2 Measurements from 1981 to 1990')
plt.legend()
plt.grid()
plt.show()

