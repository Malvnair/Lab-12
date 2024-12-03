import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq



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








years = filtered_data.index
co2_levels = filtered_data['average']
poly_order = 2  

poly_coeffs = np.polynomial.polynomial.polyfit(years, co2_levels, deg=poly_order)

trend = np.polynomial.polynomial.polyval(years, poly_coeffs)

residuals = co2_levels - trend

# Plot the results
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top panel: Original data with trend
axs[0].scatter(years, co2_levels, marker='+', label='Original Data')
axs[0].plot(years, trend, color='red', label=f'Order {poly_order} Polynomial Fit')
axs[0].set_ylabel('CO2 Levels (ppm)')
axs[0].set_title('Analysis of CO2 Measurements: General Trend')
axs[0].legend()
axs[0].grid()

# Bottom panel: Residuals
axs[1].scatter(years, residuals, marker='+', color='blue')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Residuals (ppm)')
axs[1].set_title('Analysis of CO2 Measurements: Residuals')
axs[1].grid()

plt.tight_layout()
plt.savefig("NairMalavika_Lab12_Fig1.png")
plt.show()