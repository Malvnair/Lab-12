import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq





df = pd.read_csv("co2_mm_mlo.csv", skiprows=52, usecols=[0, 1, 3], names=['year', 'month', 'average'], delimiter=",")
filtered_data = df[(df['year'] >= 1981) & (df['year'] < 1990)]

filtered_data['time'] = (filtered_data['year'] - 1981) * 12 + filtered_data['month']

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['time'], filtered_data['average'], marker='+')
plt.xlabel('Year)')
plt.ylabel('CO2 (ppm)')
plt.title('CCO2 Measurements from 1981 to 1990')
plt.xticks(
    ticks=range(0, (1989 - 1981 + 1) * 12, 12), 
    labels=range(1981, 1990),  
)
plt.show()









# years = filtered_data.index
# co2_levels = filtered_data['average']
# poly_order = 2  

# poly_coeffs = np.polynomial.polynomial.polyfit(years, co2_levels, deg=poly_order)

# trend = np.polynomial.polynomial.polyval(years, poly_coeffs)

# residuals = co2_levels - trend

# # Plot the results
# fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# # Top panel: Original data with trend
# axs[0].scatter(years, co2_levels, marker='+', label='Original Data')
# axs[0].plot(years, trend, color='red', label=f'Order {poly_order} Polynomial Fit')
# axs[0].set_ylabel('CO2 Levels (ppm)')
# axs[0].set_title('Analysis of CO2 Measurements: General Trend')
# axs[0].legend()
# axs[0].grid()

# # Bottom panel: Residuals
# axs[1].scatter(years, residuals, marker='+', color='blue')
# axs[1].set_xlabel('Year')
# axs[1].set_ylabel('Residuals (ppm)')
# axs[1].set_title('Analysis of CO2 Measurements: Residuals')
# axs[1].grid()

# plt.tight_layout()
# plt.savefig("NairMalavika_Lab12_Fig1.png")
# plt.show()






# def sinusoidal_function(t, A, T, phi):
    
#     return A * np.sin(2 * np.pi * t / T + phi)

# # parameters 
# A_trial = 3.5
# T_trial = 12.05  
# phi_trial = 8 


# sinusoidal_fit = sinusoidal_function(years, A_trial, T_trial, phi_trial)

# # Plot 
# plt.figure(figsize=(12, 6))
# plt.scatter(years, residuals, marker='+', color='blue', label='Residuals')
# plt.plot(years, sinusoidal_fit, color='red', linestyle='--', label='Sinusoidal Fit')
# plt.xlabel('Index')
# plt.ylabel('Residuals (ppm)')
# plt.title('Residuals with Sinusoidal Fit')
# plt.legend()
# plt.grid()
# plt.show()


# print("The attempt to fit a sinusoidal function to the residuals suggests that the polynomial fit may not fully capture the periodic variations in the data. It might be worthwhile to revisit the polynomial fit or explore a combined approach that accounts for both long-term trends and periodic oscillations.")




# residuals_array = residuals.to_numpy()  
# N = len(residuals_array)  
# d = 1  #

# # Compute the FFT
# fft_values = fft(residuals_array)  
# frequencies = fftfreq(N, d=d) 
# # Only  positive 
# positive_freqs = frequencies[:N // 2]  
# fft_magnitudes = np.abs(fft_values[:N // 2])  

# power_spectrum = fft_magnitudes**2

# dominant_freq_index = np.argmax(power_spectrum[1:]) + 1  # Skip zero frequency
# dominant_frequency = positive_freqs[dominant_freq_index]
# dominant_period = 1 / dominant_frequency  

# # Plot 
# plt.figure(figsize=(10, 6))
# plt.plot(positive_freqs, power_spectrum, label="Power Spectrum")
# plt.axvline(x=dominant_frequency, color='red', linestyle='--', label=f"Dominant Frequency = {dominant_frequency:.4f}")
# plt.xlabel("Frequency")
# plt.ylabel("Power")
# plt.title("FFT Power Spectrum of Residuals")
# plt.legend()
# plt.grid()
# plt.show()

# print(f"Trial-and-Error Period: {T_trial:.2f}, FFT-Derived Period: {dominant_period:.2f}. "
#       f"\nTherefore the period I obtained does agree my trial and error with a difference of {abs(T_trial - dominant_period):.2f}. ")




# def combined_model(t, A, T, phi, poly_coeffs):
#     poly_part = np.polynomial.polynomial.polyval(years, poly_coeffs)
#     sin_part = A * np.sin(2 * np.pi * t / T + phi)
#     return poly_part + sin_part

# # Use polynomial coefficients and sinusoidal parameters
# A_combined = A_trial
# T_combined = T_trial
# phi_combined = phi_trial

# # Fit combined model to 1981-1990 data
# years_extended = df['year']
# model_fit = combined_model(filtered_data.index, A_combined, T_combined, phi_combined, poly_coeffs)

# # Plot the model and the original data
# plt.figure(figsize=(12, 6))
# plt.scatter(filtered_data.index, filtered_data['average'], marker='+', color='blue', label='Original Data (1981-1990)')
# plt.plot(filtered_data.index, model_fit, color='red', linestyle='--', label='Combined Model')
# plt.xlabel('Year')
# plt.ylabel('CO2 Levels (ppm)')
# plt.title('Combined Polynomial + Sinusoidal Model for CO2 Concentration')
# plt.legend()
# plt.grid()
# plt.show()

# # Evaluate the model outside the 1981–1990 range
# outside_data = df[(df['year'] < 1981) | (df['year'] > 1990)]
# outside_years = outside_data.index
# outside_model_fit = combined_model(outside_years, A_combined, T_combined, phi_combined, poly_coeffs)

# # Predict when CO2 first reached 400 ppm
# from scipy.optimize import fsolve

# def co2_400_eq(year):
#     return combined_model(year, A_combined, T_combined, phi_combined, poly_coeffs) - 400

# # Find the year
# year_400 = fsolve(co2_400_eq, x0=2014)  # Start the search around 2010
# print(f"The model predicts CO2 first reached 400 ppm in the year: {year_400[0]:.2f}")

# # Compare with actual date 
# actual_year_400 = 2014
# print(f"Actual year CO2 reached 400 ppm: {actual_year_400}")
# print(f"Difference between predicted and actual year: {year_400[0] - actual_year_400:.2f} years")

# # Plot model vs. data including prediction of 400 ppm
# plt.figure(figsize=(12, 6))
# plt.scatter(years_extended, df['average'], marker='+', color='blue', label='Original Data')
# plt.plot(years_extended, combined_model(years_extended, A_combined, T_combined, phi_combined, poly_coeffs), color='red', label='Combined Model')
# plt.axhline(400, color='green', linestyle='--', label='400 ppm Threshold')
# plt.axvline(year_400[0], color='purple', linestyle='--', label=f'Predicted Year: {year_400[0]:.2f}')
# plt.xlabel('Year')
# plt.ylabel('CO2 Levels (ppm)')
# plt.title('CO2 Concentration Prediction with Combined Model')
# plt.legend()
# plt.grid()
# plt.show()
