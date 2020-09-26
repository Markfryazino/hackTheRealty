import math
import copy


def get_config():
    """
    Return the copy of current config
    :returns: dict of current params
    """
    return copy.deepcopy(_config)


def set_config(config):
    """
    Updates some params of config
    :param config: params to be updated
    :type config: dict
    :returns:
    """
    global _config
    for param in config:
        if param in _config:
            _config[param] = config[param]
    _init_constants()


def _init_constants():
    equator_length = _config["earth_semimajor_axis"] * 2 * math.pi
    max_latitude = 2 * math.atan(math.exp(math.pi)) * 180.0 / math.pi - 90.0
    min_latitude = -max_latitude
    e2 = _config["eccentricity"] * _config["eccentricity"]
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    d2 = e2 / 2 + 5 * e4 / 24 + e6 / 12 + 13 * e8 / 360
    d4 = 7 * e4 / 48 + 29 * e6 / 240 + 811 * e8 / 11520
    d6 = 7 * e6 / 120 + 81 * e8 / 1120
    d8 = 4279 * e8 / 161280
    global _constants
    _constants = {
        "equator_length": equator_length,
        "max_latitude": max_latitude,
        "min_latitude": min_latitude,
        "e2": e2,
        "e4": e4,
        "e6": e6,
        "e8": e8,
        "d2": d2,
        "d4": d4,
        "d6": d6,
        "d8": d8,
    }


_config = {
    "eccentricity": 0.0818191908426,
    "tile_size": 256,
    "earth_semimajor_axis": 6378137,
}
_constants = {}
_init_constants()


def restrict(value, min_, max_):
    return max(min(value, max_), min_)


def cycle_restrict(value, min_, max_):
    return value - math.floor((value - min_) / (max_ - min_)) * (max_ - min_)


def get_world_size(zoom):
    return 2 ** (zoom + 8)


def latlon2pixelXY(lat, lon, zoom=15):
    epsilon = 1e-10
    latitude = restrict(lat, -90 + epsilon, 90 - epsilon) * math.pi / 180
    tan = math.tan(math.pi * 0.25 + latitude * 0.5)
    world_size = get_world_size(zoom)
    pow_ = math.tan(
        math.pi * 0.25 + math.asin(_config["eccentricity"] * math.sin(latitude)) * 0.5
    ) ** _config["eccentricity"]
    longitude = cycle_restrict(lon, -180, 180 - epsilon)
    return (
        (longitude / 360 + 0.5) * world_size,
        (0.5 - math.log(tan / pow_) / (2 * math.pi)) * world_size
    )


def latlon2quadkey(lat, lon, zoom=15):
    pixels = latlon2pixelXY(lat, lon, zoom)
    tiles = pixelXY2tileXY(*pixels)
    return tileXY2quadkey(
        *(tiles + (zoom,))
    )


def latlon2tileXY(lat, lon, zoom=15):
    pixels = latlon2pixelXY(lat, lon, zoom)
    return pixelXY2tileXY(*pixels)


def pixelXY2latlon(pixelX, pixelY, zoom=15):
    longitude_rad = cycle_restrict(
        math.pi * pixelX / 2 ** (zoom + 7) - math.pi,
        -math.pi,
        math.pi,
    )

    pixels_per_meter = get_world_size(zoom) / _constants["equator_length"]
    y = _constants["equator_length"] * 0.5 - pixelY / pixels_per_meter
    xphi = math.pi * 0.5 - 2 * math.atan(1 / math.exp(y / _config["earth_semimajor_axis"]))
    latitudeRad = (xphi + _constants["d2"] * math.sin(2 * xphi) + _constants["d4"] * math.sin(4 * xphi) +
                   _constants["d6"] * math.sin(6 * xphi) + _constants["d8"] * math.sin(8 * xphi))

    return latitudeRad * 180 / math.pi, longitude_rad * 180 / math.pi


def pixelXY2quadkey(pixelX, pixelY, zoom=15):
    tiles = pixelXY2tileXY(pixelX, pixelY)
    return tileXY2quadkey(
        *(tiles + (zoom,))
    )


def pixelXY2tileXY(pixelX, pixelY):
    return int(math.floor(pixelX / _config["tile_size"])), int(math.floor(pixelY / _config["tile_size"]))


def quadkey2latlon(quadkey):
    pixels = quadkey2pixelXY(quadkey)
    return pixelXY2latlon(
        *(pixels + (len(quadkey),))
    )


def quadkey2pixelXY(quadkey):
    tiles = quadkey2tileXY(quadkey)
    return tileXY2pixelXY(*tiles)


def quadkey2tileXY(quadkey):
    tileX = 0
    tileY = 0
    level_of_detail = len(quadkey)
    for i in range(level_of_detail, 0, -1):
        mask = 1 << (i - 1)
        if quadkey[level_of_detail - i] == '0':
            pass
        if quadkey[level_of_detail - i] == '1':
            tileX |= mask
        if quadkey[level_of_detail - i] == '2':
            tileY |= mask
        if quadkey[level_of_detail - i] == '3':
            tileX |= mask
            tileY |= mask
    return tileX, tileY


def tileXY2latlon(tileX, tileY, zoom=15):
    pixels = tileXY2pixelXY(tileX, tileY)
    return pixelXY2latlon(
        *(pixels + (zoom,))
    )


def tileXY2pixelXY(tileX, tileY):
    return float(tileX * _config["tile_size"]), float(tileY * _config["tile_size"])


def tileXY2quadkey(tileX, tileY, zoom=15):
    quadkey = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 1
            digit += 1
        quadkey.append(str(digit))
    return "".join(quadkey)


def get_upper_quadkey(quadkey, rewind=False):
    tileX, tileY = quadkey2tileXY(quadkey)
    zoom = len(quadkey)
    max_tile = get_world_size(zoom) / _config["tile_size"]
    if rewind:
        tileY = (tileY + 1) % max_tile
    else:
        tileY = min(tileY + 1, max_tile)
    return tileXY2quadkey(tileX, tileY, zoom)


def get_lower_quadkey(quadkey, rewind=False):
    tileX, tileY = quadkey2tileXY(quadkey)
    zoom = len(quadkey)
    max_tile = get_world_size(zoom) / _config["tile_size"]
    if rewind:
        tileY = (tileY - 1) % max_tile
    else:
        tileY = max(tileY - 1, 0)
    return tileXY2quadkey(tileX, tileY, zoom)


def get_left_quadkey(quadkey, rewind=False):
    tileX, tileY = quadkey2tileXY(quadkey)
    zoom = len(quadkey)
    max_tile = get_world_size(zoom) / _config["tile_size"]
    if rewind:
        tileX = (tileX - 1) % max_tile
    else:
        tileX = max(tileX - 1, 0)
    return tileXY2quadkey(tileX, tileY, zoom)


def get_right_quadkey(quadkey, rewind=False):
    tileX, tileY = quadkey2tileXY(quadkey)
    zoom = len(quadkey)
    max_tile = get_world_size(zoom) - 1 / _config["tile_size"]
    if rewind:
        tileX = (tileX + 1) % max_tile
    else:
        tileX = max(tileX + 1, 0)
    return tileXY2quadkey(tileX, tileY, zoom)


def get_tile_bounds(tileX, tileY, zoom=15, print_=False):
    epsilon = 1e-8
    pixelX_min = tileX * _config["tile_size"]
    pixelX_max = (tileX + 1) * _config["tile_size"] - epsilon
    pixelY_min = tileY * _config["tile_size"]
    pixelY_max = (tileY + 1) * _config["tile_size"] - epsilon

    return pixelXY2latlon(pixelX_min, pixelY_max, zoom) + pixelXY2latlon(pixelX_max, pixelY_min, zoom)


def get_tile_boundary(tileX, tileY, zoom=15):
    bounds = get_tile_bounds(tileX, tileY, zoom)
    return (bounds[2], bounds[3]), (bounds[0], bounds[3]), (bounds[0], bounds[1]), (bounds[2], bounds[1])


def get_tile_center(tileX, tileY, zoom=15):
    bounds = get_tile_bounds(tileX, tileY, zoom)
    return (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2


def get_quadkey_bounds(quadkey):
    tileX, tileY = quadkey2tileXY(quadkey)
    zoom = len(quadkey)
    return get_tile_bounds(tileX, tileY, zoom)


def get_quadkey_boundary(quadkey):
    tileX, tileY = quadkey2tileXY(quadkey)
    zoom = len(quadkey)
    return get_tile_boundary(tileX, tileY, zoom)


def get_quadkey_center(quadkey):
    tileX, tileY = quadkey2tileXY(quadkey)
    zoom = len(quadkey)
    return get_tile_center(tileX, tileY, zoom)

def get_quadkey(lat, lon, zoom):
    try:
        check = lat + lon + zoom
    except TypeError:
        return None
    return latlon2quadkey(lat, lon, zoom)