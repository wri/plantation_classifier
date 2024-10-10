from pyproj import Proj, transform
import pandas as pd
import rasterio as rs
import numpy as np
pd.options.mode.copy_on_write = False 

local_folder = ("/Volumes/John/tof-output-2022/")
train_df = "/Volumes/John/data/NE_forest_reserves.csv" # plot level output from CEO
deploy_df = "/Volumes/John/dbs/america-africa-europe.csv"  # TTC tile grid 
output_df = f"{train_df[:-4]}_ard.csv"

def image_latlon_pxpy(local_folder, X, Y, latitude, longitude):
    X = str(X)
    Y = str(Y)
    fname = f"{local_folder}/{X}/{Y}/{X}X{Y}Y_FINAL.tif"
    dataset = rs.open(fname)
    px = longitude
    py = latitude
    px_pc = (px - dataset.bounds.left) / (dataset.bounds.right - dataset.bounds.left)
    py_pc = (dataset.bounds.top - py) / (dataset.bounds.top - dataset.bounds.bottom)
    return (np.floor(px_pc*dataset.width), np.floor(py_pc*dataset.height)), dataset.height, dataset.width


train_df = pd.read_csv(train_df)
deploy_df = pd.read_csv(deploy_df)

train_df['X_tile'] = 0
train_df['Y_tile'] = 0
train_df['X_px'] = 0
train_df['Y_px'] = 0
train_df['X'] = 0.
train_df['Y'] = 0.


deploy_lons = deploy_df.X
deploy_lats = deploy_df.Y

for i, val in train_df.iterrows():
    try:
        lon = val.lon
        lat = val.lat
        x_tile = np.argmin(abs(lon - deploy_lons))
        lon_tile = deploy_df.X[x_tile]
        x_tile = deploy_df.X_tile[x_tile]
        
        y_tile = np.argmin(abs(lat - deploy_lats))
        lat_tile = deploy_df.Y[y_tile]
        y_tile = deploy_df.Y_tile[y_tile]
        l, w, h = image_latlon_pxpy(local_folder, x_tile, y_tile, lat, lon)
        if l[0] > 16 and l[1] > 16:
            if l[0] < (w - 16) and l[1] < (h - 16): 
                train_df.iloc[i, train_df.columns.get_loc('X_tile')] = x_tile
                train_df.iloc[i, train_df.columns.get_loc('Y_tile')] = y_tile

                train_df.iloc[i, train_df.columns.get_loc('X')] = lon_tile
                train_df.iloc[i, train_df.columns.get_loc('Y')] = lat_tile

                train_df.iloc[i, train_df.columns.get_loc('X_px')] = l[0]
                train_df.iloc[i, train_df.columns.get_loc('Y_px')] = l[1]
    except:
        continue
train_df.to_csv(output_df, index = False)