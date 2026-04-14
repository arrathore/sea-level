import xarray as xr
import pandas as pd
import glob
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pyproj
pyproj.datadir.set_data_dir('/opt/local/lib/proj9/share/proj/')
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

# map total change to show differences in regions

# calcluate per-pixel sea level change using mean of first and last n_years
def compute_endpoint_trend(sla, n_years=5, use_means=True):
    if use_means:
        timesteps_per_year = len(sla.time) / (
            (sla.time[-1] - sla.time[0]).values / np.timedelta64(1, 'D') / 365.25
        )
        n = int(n_years * timesteps_per_year)

        start = sla.isel(time=slice(0, n)).mean("time")
        end = sla.isel(time=slice(-n, None)).mean("time")
    else:
        start = sla.isel(time=0)
        end = sla.isel(time=-1)

    return (end - start) * 100 # convert to cm

# plot the map
def plot_sea_level_change_map(sla, n_years=5, vmax=None, use_means=True):
    change = compute_endpoint_trend(sla, n_years=n_years).compute()

    # calculate scale
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(change.values), 95))

    fig, ax = plt.subplots(
        figsize=(16, 8),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    img = ax.pcolormesh(
        sla.lon, sla.lat, change,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        shading="auto",
    )

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    start_year = int(sla.time[0].dt.year)
    end_year = int(sla.time[-1].dt.year)

    plt.colorbar(img, ax=ax, orientation="horizontal", pad=0.05,
                 label="Sea Level Change (mm)", shrink=0.6)
    ax.set_title(
        f"Sea Level Change {end_year-n_years}-{end_year}\n"
        f"(mean of first vs. last {n_years} years)",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

plot_sea_level_change_map(sla, n_years=10, use_means=True)

