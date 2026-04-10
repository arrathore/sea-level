import xarray as xr
import pandas as pd
import glob
import re

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
ds = xr.concat(datasets, dim="time")
print(ds)



