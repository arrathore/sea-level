import xarray as xr
import pandas as pd
import glob
import re
import numpy as np

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

# print info
ds = xr.concat(datasets, dim="time").chunk({"time": 10})
print(ds)

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
print('gmsl:', gmsl[:10])


