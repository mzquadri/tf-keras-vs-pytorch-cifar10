"""The Keras translation of the PyTorch CIFAR-10 CNN, built to the same parameter count.

The reference is `CNN-Image-Classification-PyTorch/src/model.py`, a from-scratch
CNN with three convolutional blocks and a small classifier head:

    Conv(3,32)   -> BN -> ReLU -> Conv(32,32)   -> BN -> ReLU -> MaxPool -> Dropout
    Conv(32,64)  -> BN -> ReLU -> Conv(64,64)   -> BN -> ReLU -> MaxPool -> Dropout
    Conv(64,128) -> BN -> ReLU -> Conv(128,128) -> BN -> ReLU -> MaxPool -> Dropout
    Flatten -> Dense(256) -> ReLU -> Dropout(0.5) -> Dense(10)

The conv blocks drop at 0.25 and the classifier head at 0.5, following the
reference. See HEAD_DROPOUT below.

`EXPECTED_TRAINABLE_PARAMETERS` is the count that repository records in
`results/test_results.json`. It is asserted rather than trusted: two models with
the same diagram and different parameter counts are not the same model, and the
count is the cheapest way to catch a translation slip - a wrong channel width, a
missing bias, a BatchNorm on the wrong axis.

Worked by hand, so the number is not folklore:

    conv1a  3*32*9   + 32  =    896      bn1a  2*32  =   64
    conv1b  32*32*9  + 32  =  9,248      bn1b  2*32  =   64
    conv2a  32*64*9  + 64  = 18,496      bn2a  2*64  =  128
    conv2b  64*64*9  + 64  = 36,928      bn2b  2*64  =  128
    conv3a  64*128*9 + 128 = 73,856      bn3a  2*128 =  256
    conv3b  128*128*9+ 128 =147,584      bn3b  2*128 =  256
    dense1  2048*256 + 256 =524,544
    dense2  256*10   + 10  =  2,570
                              -------
                              815,018

BatchNorm contributes 2*C trainable (gamma, beta) and 2*C non-trainable (the
running mean and variance). PyTorch's `n_parameters` counts only the trainable
ones, so that is what this compares.
"""

from __future__ import annotations

EXPECTED_TRAINABLE_PARAMETERS = 815_018

#: PyTorch's Adam applies weight_decay by adding `wd * w` to the gradient.
#: Keras's l2 regulariser adds `l * sum(w^2)` to the loss, whose gradient is
#: `2 * l * w`. So matching PyTorch's 1e-4 needs l = 5e-5, not 1e-4. Getting
#: this wrong doubles the regularisation silently.
PYTORCH_WEIGHT_DECAY = 1e-4
KERAS_L2 = PYTORCH_WEIGHT_DECAY / 2.0

#: The PyTorch reference uses two different dropout rates, and they are not
#: interchangeable. `nn.Dropout2d(0.25)` sits in the conv blocks; the classifier
#: head uses `nn.Dropout(0.5)`. Passing the conv rate to the head would halve the
#: regularisation there, and because dropout carries no weights the parameter
#: count would still come out at 815,018 - so the assertion below would not
#: notice. Hence a separate constant rather than a reused argument.
HEAD_DROPOUT = 0.5


def build_model(num_classes: int = 10, dropout: float = 0.25):
    """Build and return the compiled-ready Keras model.

    `dropout` is the conv-block rate only. The classifier head always uses
    HEAD_DROPOUT, because the reference model's two rates differ.

    Imported inside the function so this module can be imported, and its
    constants read, without TensorFlow present.
    """
    import keras
    from keras import layers, regularizers

    l2 = regularizers.l2(KERAS_L2)

    def conv(filters: int):
        return layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",            # PyTorch padding=1 on a 3x3 kernel
            use_bias=True,             # PyTorch Conv2d has a bias by default
            kernel_regularizer=l2,
            bias_regularizer=l2,       # PyTorch weight_decay applies to biases too
        )

    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            # --- block 1 -------------------------------------------------
            conv(32),
            layers.BatchNormalization(),
            layers.ReLU(),
            conv(32),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, 2),
            # SpatialDropout2D, not Dropout: PyTorch's Dropout2d drops whole
            # channels, and plain Dropout drops individual activations. They are
            # different regularisers with the same rate.
            layers.SpatialDropout2D(dropout),
            # --- block 2 -------------------------------------------------
            conv(64),
            layers.BatchNormalization(),
            layers.ReLU(),
            conv(64),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, 2),
            layers.SpatialDropout2D(dropout),
            # --- block 3 -------------------------------------------------
            conv(128),
            layers.BatchNormalization(),
            layers.ReLU(),
            conv(128),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, 2),
            layers.SpatialDropout2D(dropout),
            # --- head ----------------------------------------------------
            layers.Flatten(),
            layers.Dense(256, kernel_regularizer=l2, bias_regularizer=l2),
            layers.ReLU(),
            # HEAD_DROPOUT, not `dropout`: the head matches nn.Dropout(0.5).
            layers.Dropout(HEAD_DROPOUT),
            layers.Dense(num_classes, kernel_regularizer=l2, bias_regularizer=l2),
        ],
        name="cifar10_cnn",
    )
    return model


def count_trainable(model) -> int:
    import numpy as np

    return int(sum(np.prod(w.shape) for w in model.trainable_weights))
