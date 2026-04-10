import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter

data = []
with open("datasets/MERGED_TP_J1_OSTM_OST_GMSL_ASCII_V52/GMSL_TPJAOS_5.2.txt") as f:
    for line in f:
        if line.startswith("HDR"):
            continue
        parts = line.strip().split()
        if len(parts) < 12:
            continue
        data.append(parts)

df = pd.DataFrame(data).astype(float)

time = df[2]  # fractional year
gmsl = df[11] # global mean sea level (smoothed col)

# compute linear trend
coeffs = np.polyfit(time, gmsl, 1)
trend = coeffs[0]
print('sea level rise rate (mm/year):', trend)

# compute acceleration
coeffs_quad = np.polyfit(time, gmsl, 2)
acceleration = 2 * coeffs_quad[0]
print('acceleration (mm/year^2):', acceleration)

# calculate derivative with smoothed data
smoothed = savgol_filter(gmsl, 51, 3)
rate = np.gradient(smoothed, time)
# ommit first point
time_rate = time[1:]
rate = rate[1:]

# create plot
fig, ax1 = plt.subplots()
# sea level on left
ax1.plot(time, gmsl, label="Sea Level", color="blue")
ax1.set_xlabel("Year")
ax1.set_ylabel("Sea Level (mm)", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')

# acceleration on right
ax2 = ax1.twinx()
ax2.plot(time_rate, rate, label="Rate of Rise", color="red")
ax2.set_ylabel("Rate (mm/year)", color="red")
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Global Mean Sea Level and Rate of Rise")
plt.show()

