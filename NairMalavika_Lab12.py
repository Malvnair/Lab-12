import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("co2_mm_mlo.csv", skiprows=51, delimiter= ",")
filtered_data = df[(df.index >= 1981) & (df.index <= 1990)].reset_index()

# Plot using 'year' and 'average'
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(filtered_data['year'], filtered_data['average'], label='CO2 Levels (1981-1990)')
plt.xlabel('Year')
plt.ylabel('CO2 Levels (ppm)')
plt.title('CO2 Measurements from 1981 to 1990')
plt.legend()
plt.grid()
plt.show()
