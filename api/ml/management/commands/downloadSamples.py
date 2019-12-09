"""Django command warpper for the overall process of identifying features."""


from django.core.management.base import BaseCommand
from datetime import datetime
from polygons.models import Boundery
from ml.images.processing import sliceImage
from polygons.utils.imagerySource import getTilesOfBoundery


class Command(BaseCommand):
    """The find feature command class."""

    help = "Find features"

    def handle(self, *args, **options):
        """Run the overall process."""
        # Get boundaries.
        bounderies = Boundery.objects.filter(
            processed=False,
            matchingId__in=[
                '5b9f6c3e-0e05-430d-b49f-35cf3819dd94'
            ]
        )

        for boundery in bounderies:
            images = getTilesOfBoundery(boundery, 17)
            print(images)

            for image in images:
                firstSlicingLevel = sliceImage(
                    image,
                    None,
                    boundery.polygon,
                    {'width': 30, 'height': 30}
                )
                for slicedImage in firstSlicingLevel:
                    slicedImage.save(
                        './temp/large/%s.jpg' % (str(datetime.now()))
                    )
                    secondSlicingLevel = sliceImage(
                        slicedImage,
                        None,
                        boundery.polygon,
                        {'width': 7, 'height': 7}
                    )
                    for slicedImage2 in secondSlicingLevel:
                        slicedImage2.save(
                            './temp/small/%s.jpg' % (str(datetime.now()))
                        )
