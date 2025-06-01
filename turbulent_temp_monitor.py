
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("sample_temperature_data.xlsx")
df['Temperature'] = df['Temperature'].fillna(0)
df['Time'] = pd.to_datetime(df['Time'])
temp = df['Temperature']
time = df['Time']

# Rolling stats
window = 12  # 1 hour window (12 samples of 5-min)
rolling_mean = temp.rolling(window=window).mean()
rolling_max = temp.rolling(window=window).max()
rolling_min = temp.rolling(window=window).min()

# Deviations
delta_upper = rolling_max - rolling_mean
delta_lower = rolling_mean - rolling_min

# Adaptive thresholds
adaptive_upper_thresh = delta_upper.rolling(window=6).mean() + 2 * delta_upper.rolling(window=6).std()
adaptive_lower_thresh = delta_lower.rolling(window=6).mean() + 2 * delta_lower.rolling(window=6).std()

# Derivatives
delta_T = temp.diff()
delta2_T = temp.diff().diff()

# System state classification
system_state = []
for i in range(len(temp)):
    t = temp[i]
    dT = delta_T[i]
    d2T = delta2_T[i]
    du = delta_upper[i]
    dl = delta_lower[i]
    au_thresh = adaptive_upper_thresh[i] if not np.isnan(adaptive_upper_thresh[i]) else 3
    al_thresh = adaptive_lower_thresh[i] if not np.isnan(adaptive_lower_thresh[i]) else 3

    if t == 0:
        state = "Invalid"
    elif abs(d2T) > 0.75 and abs(dT) > 0.5:
        state = "Spike"
    elif abs(dT) > 0.5 and du > au_thresh:
        state = "Critical"
    elif (du >= 1.5 and du <= 3) or (dl >= 1.5 and dl <= 3):
        state = "Warning"
    else:
        state = "Normal"
    system_state.append(state)

df['State'] = system_state

# Plot 1: Temp vs Time with rolling stats
plt.figure(figsize=(10, 5))
plt.plot(time, temp, label='Temperature', marker='o')
plt.plot(time, rolling_mean, '--', label='Rolling Mean')
plt.plot(time, rolling_max, '--', label='Rolling Max')
plt.plot(time, rolling_min, '--', label='Rolling Min')
plt.title("Temperature vs Time with Rolling Stats")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("plot1_temp_vs_time.png")
plt.close()

# Plot 2: First Derivative
plt.figure(figsize=(10, 4))
plt.plot(time, delta_T, label="ΔT (1st Derivative)", color='orange')
plt.axhline(0, linestyle='--', color='gray')
plt.title("1st Derivative of Temperature")
plt.xlabel("Time")
plt.ylabel("ΔT")
plt.tight_layout()
plt.savefig("plot2_first_derivative.png")
plt.close()

# Plot 3: Second Derivative
plt.figure(figsize=(10, 4))
plt.plot(time, delta2_T, label="Δ²T (2nd Derivative)", color='goldenrod')
plt.axhline(0, linestyle='--', color='gray')
plt.title("2nd Derivative of Temperature")
plt.xlabel("Time")
plt.ylabel("Δ²T")
plt.tight_layout()
plt.savefig("plot3_second_derivative.png")
plt.close()

# Plot 4: ΔUpper and ΔLower deviation
plt.figure(figsize=(10, 4))
plt.plot(time, delta_upper, label='ΔUpper')
plt.plot(time, delta_lower, label='ΔLower')
plt.plot(time, adaptive_upper_thresh, '--', label='Adaptive Upper Thresh')
plt.plot(time, adaptive_lower_thresh, '--', label='Adaptive Lower Thresh')
plt.title("Deviation from Rolling Mean (ΔUpper, ΔLower)")
plt.xlabel("Time")
plt.ylabel("Deviation")
plt.legend()
plt.tight_layout()
plt.savefig("plot4_deviation.png")
plt.close()

# Plot 5: System Health Timeline
state_colors = {'Normal': 'green', 'Warning': 'orange', 'Critical': 'red', 'Spike': 'purple', 'Invalid': 'gray'}
color_map = df['State'].map(state_colors)

plt.figure(figsize=(10, 2))
plt.scatter(time, [1]*len(df), c=color_map, label=None)
plt.yticks([])
plt.title("System Health Timeline")
plt.xlabel("Time")
plt.tight_layout()
plt.savefig("plot5_system_health_timeline.png")
plt.close()
