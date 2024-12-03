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






def sinusoidal_function(t, A, T, phi):
    
    return A * np.sin(2 * np.pi * t / T + phi)

# parameters 
A_trial = 3.5
T_trial = 12.05  
phi_trial = 8 


sinusoidal_fit = sinusoidal_function(years, A_trial, T_trial, phi_trial)

# Plot 
plt.figure(figsize=(12, 6))
plt.scatter(years, residuals, marker='+', color='blue', label='Residuals')
plt.plot(years, sinusoidal_fit, color='red', linestyle='--', label='Sinusoidal Fit')
plt.xlabel('Index')
plt.ylabel('Residuals (ppm)')
plt.title('Residuals with Sinusoidal Fit')
plt.legend()
plt.grid()
plt.show()


print("The attempt to fit a sinusoidal function to the residuals suggests that the polynomial fit may not fully capture the periodic variations in the data. It might be worthwhile to revisit the polynomial fit or explore a combined approach that accounts for both long-term trends and periodic oscillations.")




residuals_array = residuals.to_numpy()  
N = len(residuals_array)  
d = 1  #

# Compute the FFT
fft_values = fft(residuals_array)  
frequencies = fftfreq(N, d=d) 
# Only  positive 
positive_freqs = frequencies[:N // 2]  
fft_magnitudes = np.abs(fft_values[:N // 2])  

power_spectrum = fft_magnitudes**2

dominant_freq_index = np.argmax(power_spectrum[1:]) + 1  # Skip zero frequency
dominant_frequency = positive_freqs[dominant_freq_index]
dominant_period = 1 / dominant_frequency  

# Print 
print(f"Dominant Frequency: {dominant_frequency:.2f} (1/Index)")

# Plot 
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, power_spectrum, label="Power Spectrum")
plt.axvline(x=dominant_frequency, color='red', linestyle='--', label=f"Dominant Frequency = {dominant_frequency:.4f}")
plt.xlabel("Frequency (1/Index)")
plt.ylabel("Power")
plt.title("FFT Power Spectrum of Residuals")
plt.legend()
plt.grid()
plt.show()

print(f"Trial-and-Error Period: {T_trial:.2f}, FFT-Derived Period: {dominant_period:.2f}. "
      f"\nTherefore the period I obtained does agree my trial and error with a difference of {abs(T_trial - dominant_period):.2f}. ")
