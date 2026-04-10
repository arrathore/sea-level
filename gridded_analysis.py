import xarray as xr
import pandas as pd
import glob
import re
import numpy as np

import matplotlib.pyplot as plt

# load dataset from all files
print("loading data...")
files = sorted(glob.glob("datasets/NASA_SSH_REF_SIMPLE_GRID_V1/NASA-SSH_alt_ref_simple_grid_v1_*.nc"))
datasets = []
for f in files:
    # extract date from filename
    date_str = re.search(r'_(\d{8})\.nc$', f).group(1)
    time = pd.to_datetime(date_str, format="%Y%m%d")

    ds = xr.open_dataset(f)

    ds = ds.expand_dims(time=[time])
    datasets.append(ds)

print("processing...")    
ds = xr.concat(datasets, dim="time").chunk({"time": 10})
# print info
# print(ds)

# convert longitude from 0, 360 to -180, 180
ds = ds.rename({"latitude": "lat", "longitude": "lon"})
ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
ds = ds.sortby("lon")

# handle missing values
sla = ds.ssha.where(ds.ssha < 1e10)

# remove seasonal cycle from xarray DataArray
def deseason(da, freq="dayofyear", smooth=90):
    valid = {"month", "dayofyear", "season"}
    if freq not in valid:
        raise ValueError(f"freq must be one of {valid}, got '{freq}'")
    if smooth is not None and (not isinstance(smooth, int) or smooth < 2):
        raise ValueError("smooth must be an integer >= 2 or None")
    
    grouped = da.groupby(f"time.{freq}")
    anomaly = grouped - grouped.mean("time")

    # apply smoothing
    if smooth is not None:
        anomaly = anomaly.rolling(time=smooth, center=True, min_periods=smooth // 2).mean()
    
    return anomaly

# compute global mean sea level
weights = np.cos(np.deg2rad(ds.lat))
weights.name = "weights"
gmsl = sla.weighted(weights).mean(dim=["lat", "lon"])
gmsl = gmsl * 1000 # convert to mm
gmsl = gmsl.compute()

# deseason
gmsl_da = xr.DataArray(gmsl, coords = [ds.time], dims=["time"])
gmsl_deseasoned = deseason(gmsl_da)

# plot global time series
plt.figure()
plt.plot(ds.time, gmsl)
plt.xlabel("Year")
plt.ylabel("Sea Level Anomaly (mm)")
plt.title("Global Mean Sea Level")
plt.show()

# plot deseasoned
plt.figure()
plt.plot(ds.time, gmsl_deseasoned)
plt.xlabel("Year")
plt.ylabel("Sea Level Anomaly (mm)")
plt.title("Global Mean Sea Level (Deseasoned)")
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

# calculate means
regional_means = {}
for name, bounds in regions.items():
    subset = sla.sel(**bounds)
    regional_means[name] = subset.weighted(weights).mean(dim=["lat", "lon"]) * 1000

# deseason
regional_deseasoned = {
    name: deseason(xr.DataArray(data.values, coords=[ds.time], dims=["time"]))
    for name, data in regional_means.items()
}

# compare regional and global means
plt.figure()
plt.plot(ds.time, gmsl, label="Global", linewidth=3)

for name, data in regional_means.items():
    plt.plot(ds.time, data, label=name)

plt.legend()
plt.title("Regional vs Global Sea level")
plt.ylabel("Sea Level Anomaly (mm)")
plt.show()

# plot deseasoned
plt.figure()
plt.plot(ds.time, gmsl_deseasoned, label="Global", linewidth=3)

for name, data in regional_deseasoned.items():
    plt.plot(ds.time, data, label=name)

plt.legend()
plt.title("Regional vs Global Sea level (Deseasoned)")
plt.ylabel("Sea Level Anomaly (mm)")
plt.show()

