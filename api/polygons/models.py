"""The django models of the polygons app."""

from django.contrib.gis.db import models
from django.contrib.postgres.fields import JSONField
import tifffile
import os


class Boundery(models.Model):
    """The boundery model class."""

    polygon = models.PolygonField(blank=True, null=True)
    isMain = models.BooleanField(default=False)
    state = models.CharField(max_length=256, blank=True, null=True)
    matchingId = models.CharField(max_length=256, blank=True, null=True)
    area = models.FloatField(blank=True, null=True, default=0.0)
    processed = models.BooleanField(default=False)
    processedAt = models.DateTimeField(blank=True, null=True)
    createdAt = models.DateTimeField(auto_now_add=True)
    updatedAt = models.DateTimeField(auto_now=True)

    class Meta:
        """The boundery model meta class."""

        app_label = 'polygons'

    def __str__(self):
        """Get the string format of the boundary object."""
        return '%s - (%s)' % (self.matchingId, str(self.polygon))

    def getImageFromSource(self):
        """Get the image from the imagery source."""
        # A placeholder test image
        path = os.getcwd() + "/api/polygons/scripts/data/images/sw-55.tiff"

        # Get the resolution information here.
        with tifffile.TiffFile(path) as tiff:
            xResolution = tiff.pages[0].tags['XResolution'].value
            yResolution = tiff.pages[0].tags['YResolution'].value
            resolutionUnit = tiff.pages[0].tags['ResolutionUnit'].value

        tiffInfo = {
            xResolution: xResolution,
            yResolution: yResolution,
            resolutionUnit: resolutionUnit
        }

        return tiffInfo, tifffile.imread(path)


class Address(models.Model):
    """The result address model class."""

    formattedAddress = models.CharField(
        max_length=256,
        blank=True,
        null=True,
        unique=True
    )
    fullAddress = JSONField()
    point = models.PointField(blank=True, null=True)
    boundery = models.ForeignKey('Boundery', on_delete=models.CASCADE)
    confidence = models.FloatField(blank=True, null=True, default=0.0)
    processed = models.BooleanField(default=False)
    processedAt = models.DateTimeField(blank=True, null=True)
    createdAt = models.DateTimeField(auto_now_add=True)
    updatedAt = models.DateTimeField(auto_now=True)

    def __str__(self):
        """Get the string format of the address object."""
        return self.formattedAddress

    class Meta:
        """The result address model meta class."""

        app_label = 'polygons'
