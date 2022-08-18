import numpy as np
import math
import pymap3d as pm

# The local coordinate origin (Zermatt, Switzerland)
lat0 = 46.021  # deg
lon0 = 117.660  # deg
h0 = 0  # meters

# The point of interest
lat = 46.019  # deg
lon = 117.658  # deg
h = 0  # meters

a = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, deg=True)
b = pm.enu2geodetic(20, 20, 20, lat, lon, h)
print(a)
print(b)
CONSTANTS_RADIUS_OF_EARTH = 6371000.  # meters (m)


# def GPStoXY(lat, lon, ref_lat, ref_lon):
#     # input GPS and Reference GPS in degrees
#     # output XY in meters (m) X:North Y:East
#     lat_rad = math.radians(lat)
#     lon_rad = math.radians(lon)
#     ref_lat_rad = math.radians(ref_lat)
#     ref_lon_rad = math.radians(ref_lon)
#
#     sin_lat = math.sin(lat_rad)
#     cos_lat = math.cos(lat_rad)
#     ref_sin_lat = math.sin(ref_lat_rad)
#     ref_cos_lat = math.cos(ref_lat_rad)
#
#     cos_d_lon = math.cos(lon_rad - ref_lon_rad)
#
#     arg = np.clip(ref_sin_lat * sin_lat + ref_cos_lat * cos_lat * cos_d_lon, -1.0, 1.0)
#     c = math.acos(arg)
#
#     k = 1.0
#     if abs(c) > 0:
#         k = (c / math.sin(c))
#
#     x = float(k * (ref_cos_lat * sin_lat - ref_sin_lat * cos_lat * cos_d_lon) * CONSTANTS_RADIUS_OF_EARTH)
#     y = float(k * cos_lat * math.sin(lon_rad - ref_lon_rad) * CONSTANTS_RADIUS_OF_EARTH)
#
#     return x, y
#
#
# def XYtoGPS(x, y, ref_lat, ref_lon):
#     x_rad = float(x) / CONSTANTS_RADIUS_OF_EARTH
#     y_rad = float(y) / CONSTANTS_RADIUS_OF_EARTH
#     c = math.sqrt(x_rad * x_rad + y_rad * y_rad)
#
#     ref_lat_rad = math.radians(ref_lat)
#     ref_lon_rad = math.radians(ref_lon)
#
#     ref_sin_lat = math.sin(ref_lat_rad)
#     ref_cos_lat = math.cos(ref_lat_rad)
#
#     if abs(c) > 0:
#         sin_c = math.sin(c)
#         cos_c = math.cos(c)
#
#         lat_rad = math.asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c)
#         lon_rad = (ref_lon_rad + math.atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c))
#
#         lat = math.degrees(lat_rad)
#         lon = math.degrees(lon_rad)
#
#     else:
#         lat = math.degrees(ref_lat)
#         lon = math.degrees(ref_lon)
#
#     return lat, lon
#
#
# print(GPStoXY(lat, lon, lat0, lon0, ))
