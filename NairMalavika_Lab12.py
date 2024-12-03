import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("co2_mm_mlo.csv", skiprows=52, usecols=[0, 3], delimiter=",")
df.columns = ['year', 'average']  

filtered_data = df[(df['year'] >= 1981) & (df['year'] <= 1990)]
 

plt.figure(figsize=(10, 6))
plt.scatter(filtered_data.index, filtered_data['average'], marker='+')
plt.xlabel('Index')
plt.ylabel('CO2 Levels (ppm)')
plt.title('CO2 Measurements from 1981 to 1990')
plt.grid()
plt.show()


x = filtered_data['year'].to_numpy()
y = filtered_data['average'].to_numpy()

# Fit a low-order polynomial
degree = 2
coefficients = np.polynomial.polynomial.polyfit(x, y, deg=degree)
trend = np.polynomial.polynomial.polyval(x, coefficients)
residuals = y - trend

# Plot 
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top panel: Original data and fitted trend
axs[0].scatter(x, y, marker='+', color='blue')
axs[0].plot(x, trend, label=f'Fitted Polynomial Trend (degree {degree})', color='red', linewidth=2)
axs[0].set_ylabel('CO2 Levels (ppm)')
axs[0].set_title('Analysis of CO2 Measurements: Long-Term Trend (1981–1990)')
axs[0].legend()
axs[0].grid()

# Bottom panel: Residuals
axs[1].scatter(x, residuals, label='Residuals', marker='o', color='green')
axs[1].hlines(0, xmin=x.min(), xmax=x.max(), color='black', linestyles='dashed', linewidth=1)
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Residuals (ppm)')
axs[1].set_title('Analysis of CO2 Measurements: Residuals (1981–1990)')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()