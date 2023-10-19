"""
U-Net model builder in TensorFlow 2.12.0
"""

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Conv2DTranspose,
    Concatenate,
)
from tensorflow.keras.models import Sequential


def doubleConvBlock(inputLayer, numFilters: int, kernelSize: int = 3):
    """
    Returns a double convolution block consisting of two convolutional layers,
    each followed by a batch normalization layer and a ReLU activation layer.

    inputLayer: the input layer to the block
    numFilters: the number of filters to use in the convolutional layers
    kernelSize: the size of the convolutional kernels to use
    """

    convolved = Sequential(
        [
            Conv2D(numFilters, kernelSize, padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(numFilters, kernelSize, padding="same"),
            BatchNormalization(),
            Activation("relu"),
        ]
    )(inputLayer)

    return convolved


def encoderBlock(
    inputLayer, numFilters: int, convKernelSize: int = 3, poolKernelSize: int = 2
):
    """
    Returns an encoder block consisting of a double convolution block and a
    max pooling layer.

    inputLayer: the input layer to the block
    numFilters: the number of filters to use in the convolutional layers
    convKernelSize: the size of the convolutional kernels to use
    poolKernelSize: the size of the max pooling kernels to use
    """

    convolved = doubleConvBlock(inputLayer, numFilters, convKernelSize)
    pooled = MaxPooling2D(poolKernelSize)(convolved)

    return convolved, pooled


def decoderBlock(
    inputLayer,
    skipFeatures,
    numFilters: int,
    kernelSize: int = 3,
    poolKernelSize: int = 2,
):
    """
    Returns a decoder block consisting of an upsampling layer, a concatenation
    layer, and a double convolution block.

    inputLayer: the input layer to the block
    skipFeatures: the features to concatenate with the upsampled input layer
    numFilters: the number of filters to use in the convolutional layers
    """

    upsampled = Conv2DTranspose(
        numFilters, poolKernelSize, strides=poolKernelSize, padding="same"
    )(inputLayer)
    concat = Concatenate()([upsampled, skipFeatures])
    convolved = doubleConvBlock(concat, numFilters, kernelSize)

    return convolved


def UNet(
    inputShape: tuple,
    numClasses: int,
    filters: list[int],
    convKernelSize: int = 3,
    poolKernelSize: int = 2,
    depth: int = 4,
):
    """
    Returns a UNet model with the given parameters.

    inputShape: the shape of the input to the model
    numClasses: the number of classes to predict
    filters: the number of filters to use in each convolutional layer
    convKernelSize: the size of the convolutional kernels to use
    poolKernelSize: the size of the max pooling kernels to use
    depth: the number of encoder/decoder blocks to use
    """

    inputs = Input(shape=inputShape)

    # Encoder
    encoderOutputs = []
    for i in range(depth):
        if i == 0:
            encoderOutputs.append(
                encoderBlock(inputs, filters[i], convKernelSize, poolKernelSize)
            )
        else:
            encoderOutputs.append(
                encoderBlock(
                    encoderOutputs[-1][1], filters[i], convKernelSize, poolKernelSize
                )
            )

    # Middle
    middle = doubleConvBlock(encoderOutputs[-1][1], filters[-1], convKernelSize)

    # Decoder
    decoderOutputs = []
    for i in range(depth - 1, -1, -1):
        if i == depth - 1:
            decoderOutputs.append(
                decoderBlock(
                    middle,
                    encoderOutputs[i][0],
                    filters[i],
                    convKernelSize,
                    poolKernelSize,
                )
            )
        else:
            decoderOutputs.append(
                decoderBlock(
                    decoderOutputs[-1],
                    encoderOutputs[i][0],
                    filters[i],
                    convKernelSize,
                    poolKernelSize,
                )
            )

    # Output
    outputs = Conv2D(numClasses, 1, padding="same", activation="sigmoid")(
        decoderOutputs[-1]
    )

    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model = UNet((512, 512, 3), [64, 128, 256, 512, 1024], 3, 2, 4)
    model.summary()
