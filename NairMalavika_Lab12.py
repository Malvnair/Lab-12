import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq

################
#####PART1######
################

# Load and filter the dataset
df = pd.read_csv("co2_mm_mlo.csv", skiprows=52, usecols=[0, 1, 3], names=['year', 'month', 'average'], delimiter=",")
filtered_data = df[(df['year'] >= 1981) & (df['year'] < 1990)].copy()

# Calculate time index relative to 1981
filtered_data['time'] = (filtered_data['year'] - 1981) * 12 + filtered_data['month']

# Plot the CO2 data
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['time'], filtered_data['average'], marker='+')
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.title('CO2 Measurements from 1981 to 1990')
plt.xticks(
    ticks=range(0, (1989 - 1981 + 1) * 12, 12), 
    labels=range(1981, 1990),  
)
plt.show()

# Extract time and CO2 levels for polynomial fitting
time = filtered_data['time']
co2_levels = filtered_data['average']
# Order of the polynomial fit
poly_order = 2  

# Fit polynomial
poly_coeffs = np.polynomial.polynomial.polyfit(time, co2_levels, deg=poly_order)
trend = np.polynomial.polynomial.polyval(time, poly_coeffs)
# Calculate residuals
residuals = co2_levels - trend

# Plot
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# Plot data and trend
axs[0].scatter(time, co2_levels, marker='+', color='blue', label='Original Data')
axs[0].plot(time, trend, color='red', label=f'Order {poly_order} Polynomial Fit')
axs[0].set_ylabel('CO2 (ppm)')
axs[0].set_title('Analysis of CO2 Measurements: General Trend')
axs[0].set_ylim(335, 360)
axs[0].set_yticks(np.arange(335, 361, 5))
axs[0].legend()
axs[0].grid()
# Plot residuals
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




################
#####PART2######
################




# Define a sinusoidal function
def sinusoidal_function(t, A, T, phi):
    """Sinusoidal model function.

    Args:
        t: Time variable.
        A: Amplitude of the sinusoidal wave.
        T: Period of the sinusoidal wave.
        phi: Phase shift of the sinusoidal wave.

    Returns:
        The sinusoidal wave evaluated at t.
    """
    return A * np.sin(2 * np.pi * t / T + phi)

# Trial-and-Error parameters for sinusoidal fit
A_trial = 3.5
T_trial = 12.05  # multplied by 12 because of month calculations
phi_trial = 5.3

# Call functio to create sinusoidal fit using parameters
sinusoidal_fit = sinusoidal_function(time, A_trial, T_trial, phi_trial)

# Plot residuals and sinusoidal fit
plt.figure(figsize=(12, 6))
plt.scatter(time, residuals, marker='+', color='blue', label='Residuals')
plt.plot(time, sinusoidal_fit, color='red', linestyle='--', label='Sinusoidal Fit')
plt.xticks(
    ticks=range(0, (1989 - 1981 + 1) * 12, 12),  
    labels=range(1981, 1990)  
)
plt.xlabel('Year')
plt.ylabel('Residuals (ppm)')
plt.title('Residuals with Sinusoidal Fit')
plt.legend()
plt.grid()
plt.show()

print("TThe sinusoidal fit to the residuals indicates that the polynomial trend doesn't fully capture the periodic patterns in the data. It seems a combined model accounting for both long-term trends and periodic oscillations would be more effective.")

# Convert residuals to numpy array for FFT analysis
residuals_array = residuals.to_numpy()  
N = len(residuals_array)  
d = 0.1  
print(N)

# Compute the FFT
fft_values = fft(residuals_array)  
frequencies = fftfreq(N, d=d) 

# Take only positive frequencies
positive_freqs = frequencies[:N // 2]   
fft_magnitudes = np.abs(fft_values[:N // 2])  
power_spectrum = fft_magnitudes

# Identify the dominant frequency
dominant_freq_index = np.argmax(power_spectrum[1:]) + 1  
dominant_frequency = positive_freqs[dominant_freq_index]
dominant_period = 1 / dominant_frequency  

# Plot power spectrum
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, power_spectrum, label="Power Spectrum")
plt.xlabel("Frequency (1/Year)")
plt.ylabel("Power")
plt.title("FFT Power Spectrum of Residuals")
plt.legend()
plt.grid()
plt.show()

# Print statement to compare periods fromt trial and error with FFT
print(f"\n\nTrial-and-Error Period: {T_trial/12:.2f}, FFT-Derived Period: {dominant_period:.2f}. " #divide by 12 to put in terms of years
      f"\nTherefore the period I obtained does agree with my trial and error with a difference of {abs(T_trial/12 - dominant_period):.2f}. ")



################
#####PART3######
################






# Define a combined polynomial + sinusoidal model
def combined_model(t, A, T, phi, poly_coeffs):
    """Combined model with polynomial and sinusoidal components.

    Args:
        t: Time variable.
        A: Amplitude of the sinusoidal wave.
        T: Period of the sinusoidal wave.
        phi: Phase shift of the sinusoidal wave.
        poly_coeffs: Polynomial coefficients.

    Returns:
        Combined model evaluated at t.
    """
    poly_part = np.polynomial.polynomial.polyval(t, poly_coeffs)
    sin_part = A * np.sin(2 * np.pi * t / T + phi)
    return poly_part + sin_part

# Use trial parameters for combined model
A_combined = A_trial
T_combined = T_trial
phi_combined = phi_trial

# Call function to fit combined model to data
model_fit = combined_model(time, A_combined, T_combined, phi_combined, poly_coeffs)

# Plot original data and combined model
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



# Refit the model for the full dataset
df_full = pd.read_csv("co2_mm_mlo.csv", skiprows=52, usecols=[0, 1, 3], names=['year', 'month', 'average'], delimiter=",")
df_full['time'] = (df_full['year'] - df['year'].min()) * 12 + df_full['month']

# Calculate time index
all_time = (df['year'] - 1958) * 12 + df['month']
poly_coeffs = np.polynomial.polynomial.polyfit(all_time, df['average'], deg=poly_order)

# Call function to get combined model fit for all data
model_fit = combined_model(all_time, A_combined, T_combined, phi_combined, poly_coeffs)

# Adjust the all_time variable to years
years = 1958 + all_time / 12

# Predict the year CO2 reached 400 ppm using the model
model_400_index = np.where(model_fit >= 400)[0][0]
predicted_time_400 = years[model_400_index]
predicted_year_400 = predicted_time_400

# Find the actual year CO2 reached 400 ppm from the data
actual_400_index = np.where(df['average'] >= 400)[0][0]
actual_time_400 = years[actual_400_index]
actual_year_400 = actual_time_400

# Plot the full dataset, model, and 400 ppm threshold with years on x-axis
plt.figure(figsize=(12, 6))
plt.scatter(years, df['average'], color='blue', marker='+', label='Actual Data (1958-2020)')
plt.plot(years, model_fit, color='red', linestyle='--', label='Combined Model')
plt.axhline(y=400, color='purple', linestyle='--', label='400 ppm Threshold')
plt.axvline(x=predicted_year_400, color='green', linestyle='--', label=f'Predicted: {predicted_year_400:.2f}')
plt.axvline(x=actual_year_400, color='green', linestyle='--', label=f'Actual: {actual_year_400:.2f}')
plt.xlabel('Year')
plt.ylabel('CO2 (ppm)')
plt.title('CO2 Measurements, Combined Model, and 400 ppm Prediction')
plt.legend()
plt.grid()
plt.savefig("NairMalavika_Lab12_Fig2.png")
plt.show()

# Print the results
print(f"\n\nAccording to the model, CO2 levels reached 400 ppm in {predicted_year_400:.2f}. The data shows this actually happened in {actual_year_400:.2f}, with a difference of {predicted_year_400 - actual_year_400:.2f} years.")




# Look outside the 1981–1990 range
outside_data = df[(df['year'] < 1981) | (df['year'] >= 1990)]
outside_time = (outside_data['year'] - 1981) * 12 + outside_data['month']
outside_model_fit = combined_model(outside_time, A_combined, T_combined, phi_combined, poly_coeffs)

# Calculate and print the MAPE for predictions outside the fitted range
actual_co2_outside = outside_data['average'].to_numpy()
predicted_co2_outside = outside_model_fit
mape = np.mean(np.abs((actual_co2_outside - predicted_co2_outside) / actual_co2_outside)) * 100

# Print the results
print(f"\nThe model predicts CO2 variations with a MAPE of {mape:.2f}%, showing a high level of accuracy and only minor deviations from the actual data.")

