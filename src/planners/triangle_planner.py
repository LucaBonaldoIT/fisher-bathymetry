import numpy as np
import pyproj
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate
from shapely.ops import transform

class TrianglePlanner:
    """
    Given a Polygon in WGS84 (lon/lat), this planner will:
      1. Project the polygon into a local UTM zone (meters).
      2. Generate a triangular grid of waypoints (triangle corners) spaced by `spacing_m`.
      3. Reproject those waypoints back to WGS84.

    Only the triangle corner points that fall inside the polygon are returned,
    in no particular speed-optimal traversal order (just the grid ordering).
    """

    def __init__(self, polygon_ll: Polygon, spacing_m: float, heading_deg: float = 0.0):
        """
        :param polygon_ll: Shapely Polygon in WGS84 (lon/lat).
        :param spacing_m: desired spacing between triangle corners (meters).
        :param heading_deg: rotation of the triangular grid (degrees clockwise from East).
        """
        if not polygon_ll.is_valid or polygon_ll.is_empty:
            raise ValueError("Provided polygon must be a valid, non-empty Shapely Polygon.")

        self.polygon_ll = polygon_ll
        self.spacing = float(spacing_m)
        self.heading = float(heading_deg) % 360.0

        # Determine UTM zone from centroid longitude
        lon_c, lat_c = polygon_ll.centroid.x, polygon_ll.centroid.y
        zone = int(np.floor((lon_c + 180) / 6) % 60) + 1
        is_north = (lat_c >= 0)
        crs_dict = {'proj': 'utm', 'zone': zone, 'datum': 'WGS84'}
        crs_dict['north' if is_north else 'south'] = True
        self.crs_utm = pyproj.CRS.from_dict(crs_dict)

        # Define transformers
        self._to_utm = pyproj.Transformer.from_crs(
            'EPSG:4326', self.crs_utm, always_xy=True
        ).transform
        self._to_wgs = pyproj.Transformer.from_crs(
            self.crs_utm, 'EPSG:4326', always_xy=True
        ).transform

        self._waypoints_ll = []

    def plan(self):
        # (1) Project polygon to UTM
        poly_utm = transform(self._to_utm, self.polygon_ll)
        # (2) Rotate polygon so grid aligns with axes
        poly_rot = rotate(poly_utm, -self.heading, origin='centroid', use_radians=False)
        minx, miny, maxx, maxy = poly_rot.bounds

        # (3) Triangular lattice parameters
        dx = self.spacing
        dy = self.spacing * np.sqrt(3) / 2

        # (4) Generate grid points
        points_utm = []
        row = 0
        y = miny
        epsilon = 1e-6
        while y <= maxy + epsilon:
            x_start = minx + (row % 2) * (dx / 2)
            xs = np.arange(x_start, maxx + dx, dx)
            for x in xs:
                pt = Point(x, y)
                if poly_rot.contains(pt):
                    points_utm.append(pt)
            y += dy
            row += 1

        # (5) Rotate points back and reproject to WGS84
        waypoints = []
        centroid = poly_utm.centroid
        for pt in points_utm:
            # undo rotation
            pt_unrot = rotate(pt, self.heading, origin=centroid, use_radians=False)
            lon, lat = transform(self._to_wgs, pt_unrot).coords[0]
            waypoints.append(Point(lon, lat))

        self._waypoints_ll = waypoints
        return waypoints

    def get_waypoints(self):
        """: List of Shapely Point(lon, lat) in the grid, after plan() has been called. """
        return self._waypoints_ll
