from planners.zigzag_planner import ZigZagPlanner
from planners.triangle_planner import TrianglePlanner
from simulator import Simulator

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyproj
from shapely.geometry import Point, MultiPoint
from scipy.interpolate import griddata

from shapely.geometry import Polygon, LineString, box
from shapely.affinity import rotate, translate
import numpy as np

from shapely.geometry import Polygon, LineString, MultiPoint
from shapely.affinity import rotate
from shapely.ops import transform
import matplotlib.pyplot as plt

import numpy as np
import pyproj
from shapely.geometry import Polygon, LineString, MultiPoint, Point
from shapely.affinity import rotate
from shapely.ops import transform

nc_path = 'lake_depth.nc'       # input NetCDF of true bathymetry
ds = xr.open_dataset(nc_path)
depth_true = ds['depth'].values
lon = ds['lon'].values
lat = ds['lat'].values

pts = np.column_stack((lon[~np.isnan(depth_true)], lat[~np.isnan(depth_true)]))
lake_poly_ll = MultiPoint(pts).convex_hull

planner = ZigZagPlanner(
    lake_poly_ll, spacing_m=50.0, heading_deg=45
)

waypoints = planner.plan()

sim = Simulator(nc_path)

rec = sim.simulate(waypoints)
print(rec)

rec_lon = rec['lon'].values
rec_lat = rec['lat'].values
rec_depth = rec['depth'].values

pts_rec = np.column_stack((rec_lon, rec_lat))
vals_rec = rec_depth

# --- Create regular grid over bounding box ---
minx, miny, maxx, maxy = lake_poly_ll.bounds
nx, ny = 200, 200
grid_x = np.linspace(minx, maxx, nx)
grid_y = np.linspace(miny, maxy, ny)
grid_lon, grid_lat = np.meshgrid(grid_x, grid_y)

# --- Interpolate onto grid ---
grid_z = griddata(pts_rec, vals_rec, (grid_lon, grid_lat), method='linear')

# --- Mask outside lake ---n
flat_xy = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))
mask_flat = np.array([lake_poly_ll.contains(Point(x, y)) for x, y in flat_xy])
mask = mask_flat.reshape(grid_lon.shape)
grid_z[~mask] = np.nan

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.pcolormesh(grid_lon, grid_lat, grid_z, shading='auto')
plt.plot(pts_rec[:, 0], pts_rec[:, 1], 'k.', markersize=2, alpha=0.5)
# outline lake polygon
x_poly, y_poly = lake_poly_ll.exterior.xy
plt.plot(x_poly, y_poly, 'r-', linewidth=1)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Linearly Interpolated Reconstructed Depth Inside Lake')
plt.colorbar(label='Depth (m)')
plt.show()


# df = rec.to_dataframe().reset_index()

# plt.figure(figsize=(10, 6))
# sc = plt.scatter(df['lon'], df['lat'], c=df['depth'], cmap='viridis', s=5)
# plt.colorbar(sc, label='Depth')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Depth map')
# plt.show()


# ds_survey = sim.simulate(waypoints)
# print(ds_survey)

# # Calculate the score
# score = sim.score(waypoints)
# print(f"Score: {score}")

# 4) (Optional) Quick plot in lon/lat to see the zig-zag path
# xs_poly, ys_poly = lake_poly_ll.exterior.xy

# plt.figure(figsize=(7, 6))
# plt.plot(xs_poly, ys_poly, color="navy", linewidth=1, label="Lake Boundary")

# # Extract arrays of lon/lat for the waypoint sequence
# xs_way = [p.x for p in waypoints]
# ys_way = [p.y for p in waypoints]
# plt.plot(xs_way, ys_way, linestyle="-", color="firebrick", linewidth=1, label="Survey Path")

# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Bathymetric Survey Waypoints (zig-zag)")
# plt.legend(fontsize="small")
# plt.axis("equal")
# plt.show()
