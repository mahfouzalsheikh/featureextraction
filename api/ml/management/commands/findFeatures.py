"""Django command warpper for the overall process of finding features."""


from django.core.management.base import BaseCommand
from ml.test import testImageInMemory
import os
import tensorflow as tf
from datetime import datetime
from polygons.models import Boundery, Address
from ml.images.processing import convertTiffToJpeg, sliceImage
from polygons.utils.imagerySource import getTilesOfBoundery
from polygons.utils.googleMaps import getAddressesByCoordinates
from django.contrib.gis.geos import Point
from termcolor import colored


def load_graph(model_file):
    """Load the trained graph from file."""
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


class Command(BaseCommand):
    """The find features command class."""

    help = "Find features"

    def handle(self, *args, **options):
        """Run the overall process."""
        LARGE_GRAPH_PB_FILE_LOC = "%s/api/ml/retrained_graph_large.pb" % (
            os.getcwd()
        )

        SMALL_GRAPH_PB_FILE_LOC = "%s/api/ml/retrained_graph_small.pb" % (
            os.getcwd()
        )

        print("starting program . . .")

        startTime = datetime.now()

        # load the graphs
        largeGraph = load_graph(LARGE_GRAPH_PB_FILE_LOC)
        smallGraph = load_graph(SMALL_GRAPH_PB_FILE_LOC)

        # Opne tensorflow sessions.
        largeSess = tf.Session(graph=largeGraph)
        smallSess = tf.Session(graph=smallGraph)

        input_name = 'import/Placeholder'
        output_name = "import/results"

        large_input_operation = largeGraph.get_operation_by_name(input_name)
        large_output_operation = largeGraph.get_operation_by_name(output_name)

        small_input_operation = smallGraph.get_operation_by_name(input_name)
        small_output_operation = smallGraph.get_operation_by_name(output_name)

        startTime = datetime.now()
        # Get boundaries.
        container = Boundery.objects.filter(
            matchingId='houston-area-manually-created'
        )[0]
        bounderies = Boundery.objects.filter(
            polygon__contained=container.polygon,
            processed=False,
            isMain=False
        )

        for boundery in bounderies:
            print("Processing Boundary:", boundery)
            
            featuresResults = {}

            tiffInfo, sourceImage = boundery.getImageFromSource()
            jpegImage = convertTiffToJpeg(sourceImage)

            jpegImages = getTilesOfBoundery(boundery, 17)

            for image in jpegImages:
                jpegImage = image['image']
                firstSlicingLevel = sliceImage(
                    jpegImage,
                    image['box'],
                    {'width': 30, 'height': 30}
                )

                for slicedImageLevel1 in firstSlicingLevel:
                    slicedImage = slicedImageLevel1['image']
                    yesNoResult, confidence = testImageInMemory(
                        slicedImage, largeSess,
                        large_input_operation, large_output_operation
                    )
                    if yesNoResult:
                        # A feature or a set of features have been detected on a large
                        # slice, so slice to small and find those features.
                        secondSlicingLevel = sliceImage(
                            slicedImage,
                            slicedImageLevel1['box'],
                            {'width': 7, 'height': 7}
                        )

                        featuresInLevel2 = False
                        for slicedImageLevel2 in secondSlicingLevel:
                            slicedImage2 = slicedImageLevel2['image']
                            yesNoResult2, confidence2 = testImageInMemory(
                                slicedImage2, smallSess,
                                small_input_operation, small_output_operation
                            )

                            if yesNoResult2:
                                featuresInLevel2 = True

                                point = [
                                    (slicedImageLevel2['box'][0][0]
                                     + slicedImageLevel2['box'][1][0]) / 2,
                                    (slicedImageLevel2['box'][0][1]
                                     + slicedImageLevel2['box'][1][1]) / 2,
                                ]
                                address = getAddressesByCoordinates(
                                    point[0],
                                    point[1]
                                )

                                if address[
                                    'formatted_address'
                                    
                                ] not in featuresResults:
                                
                                    featuresResults[
                                        address['formatted_address']
                                    ] = confidence2
                                    print(colored(
                                        address['formatted_address'], 'green'),
                                        confidence2
                                    )

                                    resultAddress = Address(
                                        formattedAddress=address[
                                            'formatted_address'
                                        ],
                                        fullAddress=address,
                                        boundery=boundery,
                                        confidence=confidence2,
                                        point=Point((
                                            point[1],
                                            point[0]
                                        ))
                                    )

                                    if not Address.objects.filter(
                                        formattedAddress=address[
                                            'formatted_address'
                                        ]
                                    ).exists():
                                        resultAddress.save()
                                    else:
                                        print(
                                            colored('Already Exists:', 'red'),
                                            colored(
                                                address['formatted_address'],
                                                'green'
                                            )
                                        )

                            else:
                                print(
                                    colored('       No feature small.', 'red'),
                                    slicedImage2,
                                    end="\r"
                                )
                        if featuresInLevel2:
                            slicedImageLevel1['image'].save(
                                './temp/results/large-%s-%s.jpeg' % (
                                    datetime.now(),
                                    confidence
                                )
                            )
                    else:
                        print(
                            colored('No feature Large.', 'red'),
                            slicedImage)

            # Update the boundery object.
            boundery.processed = True
            boundery.processedAt = datetime.now()
            boundery.save()

        largeSess.close()
        smallSess.close()
        print("Done processing in:", datetime.now() - startTime)
