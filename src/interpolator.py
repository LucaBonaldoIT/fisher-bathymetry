import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from shapely.geometry import Point, MultiPoint, Polygon

class Interpolator:
    def __init__(
        self,
        nc_path: str,
        grid_resolution: tuple = (200, 200),
        method: str = 'linear'
    ):
        """
        Initialize with a NetCDF bathymetry file to define the lake polygon.

        Parameters:
        - nc_path: path to NetCDF file containing 'depth', 'lon', 'lat'
        - grid_resolution: (nx, ny) tuple for output grid
        - method: interpolation method for griddata
        """
        # Load true bathymetry
        ds = xr.open_dataset(nc_path)
        depth = ds['depth'].values
        lon = ds['lon'].values
        lat = ds['lat'].values

        # Build lake polygon from valid depth points
        pts = np.column_stack((lon[~np.isnan(depth)], lat[~np.isnan(depth)]))
        self.lake_poly = MultiPoint(pts).convex_hull

        # Grid setup
        self.nx, self.ny = grid_resolution
        self.method = method
        minx, miny, maxx, maxy = self.lake_poly.bounds
        self.grid_x = np.linspace(minx, maxx, self.nx)
        self.grid_y = np.linspace(miny, maxy, self.ny)
        self.grid_lon, self.grid_lat = np.meshgrid(self.grid_x, self.grid_y)

    def interpolate(self, rec_dataset):
        """
        Interpolate reconstructed depths onto the lake grid.

        Parameters:
        - rec_dataset: xarray Dataset with 'lon','lat','depth' from sim.simulate()

        Returns:
        - grid_lon, grid_lat: meshgrid arrays
        - grid_z: interpolated depth array, NaN outside lake
        """
        # Extract simulated points
        lon_r = rec_dataset['lon'].values
        lat_r = rec_dataset['lat'].values
        depth_r = rec_dataset['depth'].values
        pts_r = np.column_stack((lon_r, lat_r))

        # Interpolate
        grid_z = griddata(
            pts_r,
            depth_r,
            (self.grid_lon, self.grid_lat),
            method=self.method
        )

        # Mask outside lake
        flat_xy = np.column_stack((self.grid_lon.ravel(), self.grid_lat.ravel()))
        mask = np.array([self.lake_poly.contains(Point(x, y)) for x, y in flat_xy])
        mask = mask.reshape(self.grid_lon.shape)
        grid_z[~mask] = np.nan

        return self.grid_lon, self.grid_lat, grid_z