"""Django command module to slice and ingest KML files."""

from django.core.management.base import BaseCommand
from polygons.models import Boundery
from fastkml import kml, styles
from shapely.geometry import Polygon, Point
from datetime import datetime


class Command(BaseCommand):
    """The KML exporting command class."""

    help = "Exports results into KML files."

    def add_arguments(self, parser):
        """Define the required arguments for the command."""
        parser.add_argument('--addresses', type=str, help="Export Addresses")

    def handle(self, *args, **options):
        """Extract the results and write them into a KML files."""
        startTime = datetime.now()

        boundaries = Boundery.objects.filter(isMain=False, processed=True)

        k = kml.KML()
        ns = "{http://www.opengis.net/kml/2.2}"
        d = kml.Document(
            ns,
            'docid',
            "Results %s" % (startTime),
            'doc description'
        )
        k.append(d)
        f = kml.Folder(ns, 'fid', 'all objects', 'main folder')
        d.append(f)

        style = kml.Style(ns=ns, id="KMLStyler")

        polyStyle = styles.PolyStyle(id="polystyle", color="7fe1ca9e")

        style.append_style(polyStyle)

        k._features[0]._styles.append(style)

        print(list(k.features()))

        for boundary in boundaries:

            boundaryFolder = kml.Folder(
                ns, 'fid', boundary.matchingId,
                "Found features: %s" % (boundary.address_set.count())
            )

            boundaryPolygon = kml.Placemark(
                ns,
                boundary.matchingId,
                boundary.matchingId,
                "Found features: %s" % (boundary.address_set.count())
            )

            boundaryPolygon.geometry = Polygon(list(boundary.polygon.coords[0]))
            boundaryPolygon.styleUrl = "KMLStyler"
            boundaryFolder.append(boundaryPolygon)

            if (options['addresses']):
                for address in boundary.address_set.all():
                    p = kml.Placemark(
                        ns,
                        str(address.id),
                        address.formattedAddress,
                        "confidence: %s" % (str(address.confidence))
                    )
                    p.geometry = Point(address.point.coords)
                    p.styleUrl = "KMLStyler"
                    p.visibility = 0
                    boundaryFolder.append(p)

            f.append(boundaryFolder)

        text_file = open("./processed.kml", "w")
        outputString = k.to_string()
        text_file.write(outputString)
        text_file.close()

        print("Exporting:", boundaries.count())

        print("Done within:", datetime.now() - startTime)
