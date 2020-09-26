import math
import pandas as pd
import numpy as np


def deg2rad(x):
    return math.pi * x / 180.


def cor2dist(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = deg2rad(lat2 - lat1)
    dLon = deg2rad(lon2 - lon1)

    lat1 = deg2rad(lat1)
    lat2 = deg2rad(lat2)

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_tube_cords():
    with open('utils/tube.txt') as f:
        return eval(f.read())


def closest_tube(data: pd.Series):
    """
    Gets pd.Series with cordinates (Lat, Lon),
    returns distance to the closest tube station and color of the line.
    """
    tube = list(get_tube_cords().items())
    res = []
    for line in data:
        tube.sort(key=lambda x: cor2dist(x[0][0], x[0][1], line[0], line[1]))
        res += [[cor2dist(tube[0][0][0], tube[0][0][1], line[0], line[1]), str(tube[0][1])]]
    return pd.DataFrame(res, columns=['dist_to_closest_tube', 'line_color'])


def angle_from_coordinate(lat1, long1, lat2, long2):
    dLon = (long2 - long1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    brng = math.atan2(y, x)

    brng = math.degrees(brng)
    brng = 360 - (brng + 360) % 360

    return brng


def dist_to_center(data: pd.Series):
    """
    Gets pd.Series with cordinates (Lat, Lon),
    returns distance to Moscow center.
    """

    x, y = 55.751388, 37.618841

    angle = data.apply(lambda c: angle_from_coordinate(x, y, c[0], c[1]))
    dist = data.apply(lambda c: (cor2dist(c[0], c[1], x, y)))
    return pd.DataFrame(np.hstack([angle.values.reshape(-1, 1), dist.values.reshape(-1, 1)]),
                        columns=['angle', 'dist'])
