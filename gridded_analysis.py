import xarray as xr
import pandas as pd
import glob
import re
import numpy as np

import matplotlib.pyplot as plt

# load dataset from all files
files = sorted(glob.glob("datasets/NASA_SSH_REF_SIMPLE_GRID_V1/NASA-SSH_alt_ref_simple_grid_v1_*.nc"))
datasets = []
for f in files:
    # extract date from filename
    date_str = re.search(r'_(\d{8})\.nc$', f).group(1)
    time = pd.to_datetime(date_str, format="%Y%m%d")

    ds = xr.open_dataset(f)

    ds = ds.expand_dims(time=[time])
    datasets.append(ds)

ds = xr.concat(datasets, dim="time").chunk({"time": 10})
# print info
# print(ds)

# convert longitude from 0, 360 to -180, 180
ds = ds.rename({"latitude": "lat", "longitude": "lon"})
ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
ds = ds.sortby("lon")

# handle missing values
sla = ds.ssha.where(ds.ssha < 1e10)

# compute global mean sea level
weights = np.cos(np.deg2rad(ds.lat))
weights.name = "weights"
gmsl = sla.weighted(weights).mean(dim=["lat", "lon"])
gmsl = gmsl * 1000 # convert to mm
gmsl = gmsl.compute()

# plot global time series
plt.figure()
plt.plot(ds.time, gmsl)
plt.xlabel("Year")
plt.ylabel("Sea Level (mm)")
plt.title("Global Mean Sea Level")
plt.show()

# calcluate trend and acceleration
t = np.arange(len(ds.time))

coeffs_linear = np.polyfit(t, gmsl, 1)
coeffs_quad = np.polyfit(t, gmsl, 2)

trend = coeffs_linear[0]
acceleration = 2 * coeffs_quad[0]

print("Trend (mm/timestep):", trend)
print("Acceleration:", acceleration)

# regional analysis
regions = { # define regions to analyze
    "Equatorial Pacific": dict(lat=slice(-10, 10), lon=slice(150, -90)),
    "North Atlantic": dict(lat=slice(0, 60), lon=slice(-80, 0)),
    "Indian Ocean": dict(lat=slice(-30, 30), lon=slice(40, 120)),
}

regional_means = {}
for name, bounds in regions.items():
    subset = sla.sel(**bounds)
    regional_means[name] = subset.weighted(weights).mean(dim=["lat", "lon"]) * 1000

# compare regional and global means
plt.figure()
plt.plot(ds.time, gmsl, label="Global", linewidth=3)

for name, data in regional_means.items():
    plt.plot(ds.time, data, label=name)

plt.legend()
plt.title("Regional vs Global Sea level")
plt.ylabel("Sea Level (mm)")
plt.show()

