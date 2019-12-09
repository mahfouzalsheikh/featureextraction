"""Testing against the trained model module."""


import numpy as np
import tensorflow as tf


def readTensorFromImage(jpegImage):
    """Read jpeg image from memory and return normalized tensor image."""
    tfImage = np.array(jpegImage)[:, :, 0:3]
    float_caster = tf.cast(tfImage, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [299, 299])
    normalized = tf.divide(tf.subtract(resized, [0]), [255])
    sess = tf.Session()
    result = sess.run(normalized)
    return result


def testImageInMemory(jpegImage, sess, input_operation, output_operation):
    """Run the tensor image into the graph and get predictions."""
    imageData = readTensorFromImage(jpegImage)
    predictions = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: imageData
    })

    yesNoResult = predictions[0][0] < predictions[0][1]
    confidence = np.amax(predictions[0]) * 100.0
    return yesNoResult, confidence
