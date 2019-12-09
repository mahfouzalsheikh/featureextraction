"""Google maps services."""


import googlemaps
import api.settings as settings


def getAddressesByCoordinates(lat, lng):
    """Find the closest address of lat, lng pair."""
    gmaps = googlemaps.Client(key=settings.GOOGLE_MAPS_KEY)

    addresses = gmaps.reverse_geocode((lat, lng))

    if len(addresses) > 0:
        return addresses[0]
    return None
