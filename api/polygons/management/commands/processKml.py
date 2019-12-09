"""Django command module to slice and ingest KML files."""

from django.core.management.base import BaseCommand
from polygons.models import Boundery
import os
from multiprocessing import Pool
from functools import partial
import pyproj
from fastkml import kml
from shapely.geometry import box, shape
from shapely.ops import transform
import shapely
from django.contrib.gis.geos import Polygon
from datetime import datetime
import uuid

# A function that slices a polygon using fishnet method.


def fishnet(geometry, threshold):
    """Apply a fishnet to the geometry by a threshold."""
    bounds = geometry.bounds
    xmin = int(bounds[0] // threshold)
    xmax = int(bounds[2] // threshold)
    ymin = int(bounds[1] // threshold)
    ymax = int(bounds[3] // threshold)
    result = []
    for i in range(xmin, xmax + 1):
        for j in range(ymin, ymax + 1):
            b = box(
                i * threshold,
                j * threshold,
                (i + 1) * threshold,
                (j + 1) * threshold
            )
            g = geometry.intersection(b)
            if g.is_empty:
                continue
            result.append(g)
    return result


def calculateArea(s):
    """Calculate the projected area of a shape."""
    proj = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),
        pyproj.Proj(proj='laea', lat1=s.bounds[1], lat2=s.bounds[3])
    )
    projectedArea = transform(proj, s).area
    return projectedArea


def insertBoundery(s, state, isMain):
    """Insert a geometry in the polygons table."""
    coords = list(s.exterior.coords)
    coords2d = [xy[0:2] for xy in coords]
    polygon = Polygon(coords2d)

    projectedArea = calculateArea(s)

    boundery = Boundery(polygon=polygon, area=projectedArea,
                        state=state, isMain=isMain)

    if isMain:
        print("Polygon", s)

    return boundery


def slicePlacemarks(placemarks, state):
    """Slice the source placemark geometry and ingest it."""
    outputPlacemarks = []
    outputPolygons = []
    totalPolygons = 0
    ns = '{http://www.opengis.net/kml/2.2}'
    for placemark in placemarks:
        s = shape(placemark.geometry)

        if isinstance(s, shapely.geometry.polygon.Polygon):
            boundery = insertBoundery(s, state, True)
            outputPolygons.append(boundery)

        elif isinstance(s, shapely.geometry.multipolygon.MultiPolygon):
            subSs = list(s)
            for subS in subSs:
                boundery = insertBoundery(subS, state, True)
                outputPolygons.append(boundery)

        slices = fishnet(placemark.geometry, 0.01)
        totalPolygons += len(slices)

        for polygonSlice in slices:
            matchingId = str(uuid.uuid4())
            newPlacemark = kml.Placemark(ns, 'id', matchingId, 'description')
            newPlacemark.geometry = polygonSlice
            newPlacemark.styleUrl = "KMLStyler"
            if isinstance(
                newPlacemark.geometry,
                shapely.geometry.polygon.Polygon
            ):
                outputPlacemarks.append(newPlacemark)
                boundery = insertBoundery(newPlacemark.geometry, state, False)
                boundery.matchingId = matchingId
                outputPolygons.append(boundery)

            elif isinstance(
                newPlacemark.geometry,
                shapely.geometry.multipolygon.MultiPolygon
            ):
                subPolygons = list(newPlacemark.geometry)
                for subPolygon in subPolygons:
                    matchingId = str(uuid.uuid4())
                    newSubPlacemark = kml.Placemark(
                        ns, 'id', matchingId, 'description')
                    newSubPlacemark.geometry = subPolygon
                    newSubPlacemark.styleUrl = "KMLStyler"
                    outputPlacemarks.append(newSubPlacemark)
                    boundery = insertBoundery(subPolygon, state, False)
                    boundery.matchingId = matchingId
                    outputPolygons.append(boundery)

    return outputPlacemarks, totalPolygons, outputPolygons


class Command(BaseCommand):
    """The KML processing command class."""

    help = "Import the source KML files"

    def add_arguments(self, parser):
        """Define the required arguments for the command."""
        parser.add_argument('--input', type=str, help="kml file path")

    def handle(self, *args, **options):
        """Slice and import the KML file."""
        # Open the input file and get its features.

        startTime = datetime.now()

        print("Starting at:", startTime)

        with open(options['input'], 'rt') as kmlFile:
            doc = kmlFile.read()

        fileName, fileExtension = os.path.splitext(options['input'])

        k = kml.KML()
        k.from_string(doc.replace("xsd:", "").encode('UTF-8'))
        features = list(k.features())
        features1 = list(features[0].features())
        inputStyles = list(features[0].styles())
        placemarks = list(features1[0].features())

        # Prepare the output file.
        ns = '{http://www.opengis.net/kml/2.2}'
        output = kml.KML()
        d = kml.Document(ns, 'docid', 'doc name', 'doc description')
        d._styles = inputStyles
        output.append(d)
        f = kml.Folder(ns, 'fid', 'f name', 'f description')
        d.append(f)
        nf = kml.Folder(ns, 'nested-fid', 'nested f name',
                        'nested f description')
        f.append(nf)

        # remove old entries of the state:
        toDelete = Boundery.objects.filter(state=fileName)
        toDelete.delete()

        # Parallelize the processing.
        pool = Pool(processes=8)
        print("Processing ", len(placemarks), " Placemarks in the file.")
        chunks = [placemarks[i::10] for i in range(10)]
        result = pool.map(partial(slicePlacemarks, state=fileName), chunks)

        # summarize the results.
        for pm in result:
            for pl in pm[0]:
                print(pl)
                nf.append(pl)

            Boundery.objects.bulk_create(pm[2])

        # dump the results into the output file.
        print("%s.%s" % (fileName, fileExtension))
        text_file = open("%s_sliced%s" % (fileName, fileExtension), "w")
        outputString = output.to_string()
        text_file.write(outputString)
        text_file.close()

        # print the totals.
        print("Done within:", datetime.now() - startTime)
