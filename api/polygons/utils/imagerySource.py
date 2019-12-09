"""The imagery sourcing module."""

import requests
import json
import api.settings as settings
from datetime import datetime, timedelta
import math
from django.contrib.gis.geos import Polygon
from io import BytesIO
from PIL import Image

def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat, lng, zoom to tile indexes."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)

    ytile = int((1.0 - math.log(
                math.tan(lat_rad) + (1 / math.cos(lat_rad))
                ) / math.pi) / 2.0 * n)

    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    """Get the NW-corner of the tile square."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def getTileByXYZoom(x, y, zoom, destination='memory'):
    """Get a map tile by x, y, zoom."""
    requestUrl = '%s/%s/%s/%s@2x.jpg90?access_token=%s' % (
        settings.MAPBOX_HOST, str(zoom), str(x), str(y), settings.MAPBOX_TOKEN
    )

    headers = {

    }

    print(requestUrl)
    if destination == 'file':
        filePath = './temp/tile-%s-%s-%s.jpg' % (str(x), str(y), str(zoom))
        r = requests.get(requestUrl, headers=headers, stream=True)
        if r.status_code == 200:
            with open(filePath, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
        return filePath
    else:
        r = requests.get(requestUrl)
        img = Image.open(BytesIO(r.content))
        return img


def getTilesOfBoundery(boundery, zoom):
    """Get all tiles that covers boundery."""
    box = list(list(boundery.polygon.envelope)[0])
    startPoint = box[1]
    endPoint = box[3]

    firstTile = deg2num(startPoint[1], startPoint[0], zoom)
    lastTile = deg2num(endPoint[1], endPoint[0], zoom)

    results = []
    startingPoint = [firstTile[0], firstTile[1]]
    while startingPoint[0] >= lastTile[0]:
        while startingPoint[1] >= lastTile[1]:
            tile = (startingPoint[0], startingPoint[1])

            tileBoxNW = num2deg(tile[0], tile[1], zoom)
            tileBoxSE = num2deg(tile[0] + 1, tile[1] + 1, zoom)

            tileWindow = [tileBoxNW, tileBoxSE]

            tileBox = [
                (tileBoxNW[1], tileBoxNW[0]),
                (tileBoxNW[1], tileBoxSE[0]),
                (tileBoxSE[1], tileBoxNW[0]),
                (tileBoxSE[1], tileBoxSE[0]),
                (tileBoxNW[1], tileBoxNW[0])
            ]

            tilePolygon = Polygon(tileBox)

            if tilePolygon.intersects(boundery.polygon):
                results.append({
                    'box': tileWindow,
                    'image': getTileByXYZoom(tile[0], tile[1], zoom)
                })

            startingPoint[1] -= 1
        startingPoint[1] = firstTile[1]
        startingPoint[0] -= 1
    return results
