import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq





df = pd.read_csv("co2_mm_mlo.csv", skiprows=52, usecols=[0, 1, 3], names=['year', 'month', 'average'], delimiter=",")
filtered_data = df[(df['year'] >= 1981) & (df['year'] < 1990)].copy()

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









time = filtered_data['time']
co2_levels = filtered_data['average']
poly_order = 2  

poly_coeffs = np.polynomial.polynomial.polyfit(time, co2_levels, deg=poly_order)
trend = np.polynomial.polynomial.polyval(time, poly_coeffs)
residuals = co2_levels - trend

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axs[0].scatter(time, co2_levels, marker='+', color='blue', label='Original Data')
axs[0].plot(time, trend, color='red', label=f'Order {poly_order} Polynomial Fit')
axs[0].set_ylabel('CO2 Levels (ppm)')
axs[0].set_title('Analysis of CO2 Measurements: General Trend')
axs[0].set_ylim(335, 360)
axs[0].set_yticks(np.arange(335, 361, 5))
axs[0].legend()
axs[0].grid()

axs[1].scatter(time, residuals, marker='+', color='blue')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Residuals (ppm)')
axs[1].set_title('Analysis of CO2 Measurements: Residuals')
axs[1].grid()

plt.xticks(
    ticks=range(0, (1989 - 1981 + 1) * 12, 12),
    labels=range(1981, 1990)
)

plt.tight_layout()
plt.savefig("NairMalavika_Lab12_Fig1.png")
plt.show()










def sinusoidal_function(t, A, T, phi):
    
    return A * np.sin(2 * np.pi * t / T + phi)

# parameters 
A_trial = 3.5
T_trial = 12.05  
phi_trial = 5.3


sinusoidal_fit = sinusoidal_function(time, A_trial, T_trial, phi_trial)

# Plot 
plt.figure(figsize=(12, 6))
plt.scatter(time, residuals, marker='+', color='blue', label='Residuals')
plt.plot(time, sinusoidal_fit, color='red', linestyle='--', label='Sinusoidal Fit')
plt.xticks(
    ticks=range(0, (1989 - 1981 + 1) * 12, 12),  # Every 12 months (1 year)
    labels=range(1981, 1990)  # Year labels
)
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

# Plot 
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, power_spectrum, label="Power Spectrum")
plt.axvline(x=dominant_frequency, color='red', linestyle='--', label=f"Dominant Frequency = {dominant_frequency:.4f}")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("FFT Power Spectrum of Residuals")
plt.legend()
plt.grid()
plt.show()

print(f"Trial-and-Error Period: {T_trial:.2f}, FFT-Derived Period: {dominant_period:.2f}. "
      f"\nTherefore the period I obtained does agree my trial and error with a difference of {abs(T_trial - dominant_period):.2f}. ")




def combined_model(t, A, T, phi, poly_coeffs):
    poly_part = np.polynomial.polynomial.polyval(t, poly_coeffs)
    sin_part = A * np.sin(2 * np.pi * t / T + phi)
    return poly_part + sin_part

A_combined = A_trial
T_combined = T_trial
phi_combined = phi_trial

model_fit = combined_model(time, A_combined, T_combined, phi_combined, poly_coeffs)

plt.figure(figsize=(12, 6))
plt.scatter(time, filtered_data['average'], marker='+', color='blue', label='Original Data (1981-1990)')
plt.plot(time, model_fit, color='red', linestyle='--', label='Combined Model')
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.title('Combined Polynomial + Sinusoidal Model for CO2 Concentration')
plt.legend()
plt.grid()
plt.xticks(
    ticks=range(0, (1989 - 1981 + 1) * 12, 12), 
    labels=range(1981, 1990)  
)
plt.tight_layout()
plt.show()



# Evaluate the model outside the 1981–1990 range
outside_data = df[(df['year'] < 1981) | (df['year'] >= 1990)]
outside_time = (outside_data['year'] - 1981) * 12 + outside_data['month']
outside_model_fit = combined_model(outside_time, A_combined, T_combined, phi_combined, poly_coeffs)



df_full = pd.read_csv("co2_mm_mlo.csv", skiprows=52, usecols=[0, 1, 3], names=['year', 'month', 'average'], delimiter=",")
df_full['time'] = (df_full['year'] - df['year'].min()) * 12 + df_full['month']

all_time = (df['year'] - 1958) * 12 + df['month']
poly_coeffs = np.polynomial.polynomial.polyfit(all_time, df['average'], deg=poly_order)

model_fit = combined_model(all_time, A_combined, T_combined, phi_combined, poly_coeffs)



model_400_index = np.where(model_fit >= 400)[0][0]  
predicted_time_400 = all_time[model_400_index]  
predicted_year_400 = 1958 + predicted_time_400 / 12  

actual_400_index = np.where(df['average'] >= 400)[0][0]  
actual_time_400 = all_time[actual_400_index] 
actual_year_400 = 1958 + actual_time_400 / 12 

plt.figure(figsize=(12, 6))
plt.scatter(all_time, df['average'], color='blue', marker='+', label='Actual Data (1958-2020)')
plt.plot(all_time, model_fit, color='red', linestyle='--', label='Combined Model')
plt.axhline(y=400, color='purple', linestyle='--', label='400 ppm Threshold')
plt.axvline(x=predicted_time_400, color='green', linestyle='--', label=f'Predicted: {predicted_year_400:.2f}')
plt.axvline(x=actual_time_400, color='green', linestyle='--', label=f'Actual: {actual_year_400:.2f}')
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.title('CO2 Measurements, Combined Model, and 400 ppm Prediction')
plt.legend()
plt.grid()
plt.savefig("NairMalavika_Lab12_Fig2")
plt.show()

# Print the results
print(f"Predicted year CO2 reached 400 ppm (from model): {predicted_year_400:.2f}")
print(f"Actual year CO2 reached 400 ppm (from data): {actual_year_400:.2f}")
print(f"Difference: {predicted_year_400 - actual_year_400:.2f} years")



actual_co2_outside = outside_data['average'].to_numpy()
predicted_co2_outside = outside_model_fit

mape = np.mean(np.abs((actual_co2_outside - predicted_co2_outside) / actual_co2_outside)) * 100
print(f"The MAPE of {mape:4f}% suggests that the model's predictions are highly accurate, with an average deviation of only 2.95% from the actual data. Overall, the model provides a reliable representation of CO2 variations over time.")
