"""Chandrayaan-2 OHRC dataset layer: labels, memmap rasters, geometry, sun geometry.

Nothing in this package writes to the dataset tree, and no accessor loads a whole
raster into memory.
"""
from .geometry import (GeometryGrid, MOON_RADIUS_M, footprint_overlap,
                       ground_distance_m, lonlat_to_south_stereo,
                       south_stereo_to_lonlat)
from .label import OhrcLabel, parse_label, verify_raster_size
from .product import (OhrcProduct, by_timestamp, default_dataset_root, discover,
                      overlap_matrix)
from .sun import SunSeries, line_time_seconds, shadow_length_per_metre

__all__ = [
    "GeometryGrid", "MOON_RADIUS_M", "footprint_overlap", "ground_distance_m",
    "lonlat_to_south_stereo", "south_stereo_to_lonlat",
    "OhrcLabel", "parse_label", "verify_raster_size",
    "OhrcProduct", "by_timestamp", "default_dataset_root", "discover",
    "overlap_matrix",
    "SunSeries", "line_time_seconds", "shadow_length_per_metre",
]
