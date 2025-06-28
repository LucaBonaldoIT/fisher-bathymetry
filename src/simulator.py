import xarray as xr
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, nc_path: str, spacing: float = 1.0):
        self.lake_ds = xr.open_dataset(nc_path)
        self.spacing = spacing
        self.geod = pyproj.Geod(ellps='WGS84')

        lons = self.lake_ds['lon'].values
        lats = self.lake_ds['lat'].values
        
        self.grid_points = np.vstack([lons.ravel(), lats.ravel()]).T
        
        self.kdtree = cKDTree(self.grid_points)

    def simulate(self, waypoints: list[Point]) -> xr.Dataset:
        if len(waypoints) < 2:
            return pd.DataFrame()

        all_path_points = []
        for i in range(len(waypoints) - 1):
            start_wp, end_wp = waypoints[i], waypoints[i+1]
            fwd_azimuth, back_azimuth, distance = self.geod.inv(start_wp.x, start_wp.y, end_wp.x, end_wp.y)
            if distance < self.spacing:
                if i == 0: all_path_points.append((start_wp.x, start_wp.y))
                all_path_points.append((end_wp.x, end_wp.y))
                continue
            num_intermediate_points = int(distance / self.spacing)
            segment_points = self.geod.npts(start_wp.x, start_wp.y, end_wp.x, end_wp.y, npts=num_intermediate_points)
            if i == 0: all_path_points.append((start_wp.x, start_wp.y))
            all_path_points.extend(segment_points)
            all_path_points.append((end_wp.x, end_wp.y))
        
        unique_points = list(dict.fromkeys(all_path_points))
        lons_to_sample, lats_to_sample = zip(*unique_points)

        query_points = np.vstack([lons_to_sample, lats_to_sample]).T
        distances, flat_indices = self.kdtree.query(query_points)

        y_indices, x_indices = np.unravel_index(flat_indices, self.lake_ds['lon'].shape)

        y_indexer = xr.DataArray(y_indices, dims="points")
        x_indexer = xr.DataArray(x_indices, dims="points")
        
        sampled_data = self.lake_ds.isel(y=y_indexer, x=x_indexer)
        
        return sampled_data




    def score(self, waypoints: list[Point]): ...