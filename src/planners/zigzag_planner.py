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

class ZigZagPlanner:
    """
    Given a Polygon in WGS84 (lon/lat), this planner will:
      1. Project the polygon into a local UTM zone (meters).
      2. Generate parallel transects (spaced in meters) at a given heading.
      3. Reproject those transect LineStrings back to WGS84.
      4. Produce a single ordered list of Points (lon, lat) that zig-zag along each transect.
    """

    def __init__(self, polygon_ll: Polygon, spacing_m: float, heading_deg: float = 0.0):
        """
        :param polygon_ll: Shapely Polygon in WGS84 (lon/lat).
        :param spacing_m: desired spacing between lines (in meters).
        :param heading_deg: orientation of transects (degrees clockwise from East).
        """
        if not polygon_ll.is_valid or polygon_ll.is_empty:
            raise ValueError("Provided polygon must be a valid, non-empty Shapely Polygon.")

        self.polygon_ll = polygon_ll
        self.spacing = float(spacing_m)
        self.heading = float(heading_deg) % 360.0

        # Determine UTM zone from centroid
        lon_center, lat_center = polygon_ll.centroid.x, polygon_ll.centroid.y
        utm_zone = int(np.floor((lon_center + 180) / 6) % 60) + 1
        is_northern = (lat_center >= 0)

        # Build a PROJ CRS for that UTM zone
        self.crs_utm = pyproj.CRS.from_dict({
            'proj': 'utm',
            'zone': utm_zone,
            'datum': 'WGS84',
            'south' if not is_northern else 'north': True
        })

        # Create transformers: WGS84→UTM and UTM→WGS84
        self.project_to_utm = pyproj.Transformer.from_crs(
            "EPSG:4326", self.crs_utm, always_xy=True
        ).transform
        self.project_to_wgs = pyproj.Transformer.from_crs(
            self.crs_utm, "EPSG:4326", always_xy=True
        ).transform

        # Will hold the final list of Shapely Point(lon, lat) in proper order:
        self._waypoints_ll = []

    def plan(self):
        """
        1. Project the polygon into UTM (meters).
        2. Rotate it so that transects align with the X-axis in UTM.
        3. Walk horizontal lines (spaced by self.spacing m) through the rotated polygon, intersect.
        4. Rotate those intersections back, reproject each to WGS84.
        5. Build a zig-zag ordering of endpoints (lon, lat) as Shapely Points.
        """
        # (1) Project polygon to UTM
        poly_utm = transform(self.project_to_utm, self.polygon_ll)

        # (2) Rotate polygon by –heading so that “new” transects will be horizontal lines
        poly_utm_rot = rotate(poly_utm, -self.heading, origin='centroid', use_radians=False)

        # (3) Compute bounds in rotated UTM space
        minx, miny, maxx, maxy = poly_utm_rot.bounds
        # For debugging, you might uncomment:
        # print("Rotated UTM bounds:", minx, miny, maxx, maxy)
        # print(f"Spacing = {self.spacing} m ; Heading = {self.heading}°")

        # (4) Generate all horizontal lines (in rotated UTM) spaced by self.spacing
        lines_utm_rot = []
        y = miny
        epsilon = self.spacing * 1e-6  # tiny fudge so last row isn’t skipped
        while y <= maxy + epsilon:
            scan_line = LineString([ (minx - 10.0, y), (maxx + 10.0, y) ])
            intersection = poly_utm_rot.intersection(scan_line)
            if not intersection.is_empty:
                # Could be a single LineString or a MultiLineString
                if intersection.geom_type == "LineString":
                    lines_utm_rot.append(intersection)
                else:
                    for geom in getattr(intersection, "geoms", []):
                        if geom.geom_type == "LineString":
                            lines_utm_rot.append(geom)
            y += self.spacing

        # (5) Rotate each UTM-space transect back by +heading (so they lie at the correct angle)
        #     around the same centroid used earlier. This yields a list of LineStrings in UTM coords.
        transects_utm = [
            rotate(line, self.heading, origin=poly_utm.centroid, use_radians=False)
            for line in lines_utm_rot
        ]

        # (6) Reproject each UTM-space LineString back to WGS84
        transects_ll = [ transform(self.project_to_wgs, tr) for tr in transects_utm ]

        # (7) Build a single, zig-zag ordered list of Points in lon/lat
        waypoints: list[Point] = []
        for idx, line_ll in enumerate(transects_ll):
            coords = list(line_ll.coords)  # each LineString is just two points: [(lon0, lat0), (lon1, lat1)]
            # Reverse every other transect so that we “zig-zag”
            if idx % 2 == 1:
                coords = coords[::-1]

            # Add both endpoints in order
            for (x, y) in coords:
                waypoints.append(Point(x, y))

        self._waypoints_ll = waypoints
        return self._waypoints_ll

    def get_waypoints(self):
        """
        :return: List of Shapely Point(lon, lat) in the exact order to be traversed.
        """
        return self._waypoints_ll