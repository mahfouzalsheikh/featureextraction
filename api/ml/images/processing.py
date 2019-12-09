"""Image processing functions."""


import numpy as np
from PIL import Image, ImageOps, ImageEnhance


def sliceImage(image, coordinates, size, offset=0):
    """Slice the image into 5X5 meters here."""
    # resoltion is hard coded for now, 55cm = 0.55m per pixel
    resolution = 0.53

    slicedImages = []
    originalWidth, originalHeight = image.size

    offsetRange = 0
    if offset != 0:
        offsetRange = 1

    width = size['width']
    height = size['height']

    xRange = int(originalWidth // (width // resolution)) + offsetRange
    yRange = int(originalHeight // (height // resolution)) + offsetRange

    for x in range(0, xRange + 1):
        for y in range(0, yRange + 1):
            topLeftX = x * width // resolution - offset // resolution
            topLeftY = y * height // resolution - offset // resolution
            bottomRightX = (x + 1) * width // resolution - offset // resolution
            bottomRightY = (y + 1) * height // resolution - offset // resolution

            slicedImage = image.crop(
                (
                    topLeftX,
                    topLeftY,
                    bottomRightX,
                    bottomRightY
                )
            )

            box = [
                (
                    coordinates[0][0]
                    - ((coordinates[0][0] - coordinates[1][0]) / xRange) * y,
                    coordinates[0][1]
                    - ((coordinates[0][1] - coordinates[1][1]) / yRange) * x
                ),
                (
                    coordinates[0][0]
                    - ((coordinates[0][0] - coordinates[1][0]) / xRange)
                    * (y + 1),
                    coordinates[0][1]
                    - ((coordinates[0][1] - coordinates[1][1]) / yRange)
                    * (x + 1)
                )
            ]
            slicedImages.append(
                {
                    'image': slicedImage,
                    'box': box,
                    'index': [x, y]
                })

    return slicedImages


def processColors(jpegIm):
    """Apply the enhancments."""
    # fix the histogram
    jpegIm = ImageOps.equalize(jpegIm)

    # increase saturation
    saturation = ImageEnhance.Color(jpegIm)
    jpegIm = saturation.enhance(1.5)

    # increase sharpness
    sharpness = ImageEnhance.Sharpness(jpegIm)
    jpegIm = sharpness.enhance(2.0)
    # return the jpeg image object.
    return jpegIm


def convertTiffToJpeg(tiffImage):
    """Convert the tiff image into a jpeg one."""
    # merge the RGB tiff bands into JPEG structure.
    jpegArray = np.dstack((tiffImage[0], tiffImage[1], tiffImage[2]))
    # correct the colors range
    jpegArray = jpegArray * 255 / 2.6868713

    # convert all the values to unsigned integers.
    jpegArray = jpegArray.astype(np.uint8)

    # create the jpeg Image object.
    jpegIm = Image.fromarray(jpegArray)

    return processColors(jpegIm)
