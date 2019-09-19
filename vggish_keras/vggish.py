"""VGGish model for Keras. A VGG-like model for audio classification

# Reference

- [CNN Architectures for Large-Scale Audio Classification](ICASSP 2017)

"""

import os
from functools import partial

from tensorflow.keras.models import Model
import tensorflow.keras.layers as tfkl
import tensorflow.keras.backend as K

from . import params


def VGGish(pump_op=None,
           input_shape=None,
           include_top=False,
           pooling='avg',
           weights='audioset',
           name='vggish'):
    '''A Keras implementation of the VGGish architecture.

    Arguments:
        input_shape (tuple): the model input shape. If ``include_top``,
            ``input_shape`` will be set to ``(params.NUM_FRAMES, params.NUM_BANDS, 1)``,
            otherwise it will be ``(None, None, 1)`` to accomodate variable sized
            inputs.

        include_top (bool): whether to include the fully connected layers. Default is False.

        pooling (str): what type of global pooling should be applied if no top? Default is 'avg'

        weights (str, None): the weights to use (see WEIGHTS_PATHS). Currently, there is
            only 'audioset'. Can also be a path to a keras weights file.

        name (str): the name for the model.

    Returns:
        A Keras model instance.
    '''

    if input_shape:
        pass
    elif include_top:
        input_shape = params.NUM_FRAMES, params.NUM_BANDS, 1

    elif pump_op:
        import pumpp
        inputs = pumpp.Pump(pump_op).layers('keras')[pump_op.name]

    else:
        input_shape = None, None, 1

    if input_shape:
        inputs = tfkl.Input(shape=input_shape, name='input_1')

    # setup layer params
    conv = partial(
        tfkl.Conv2D,
        kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')

    maxpool = partial(
        tfkl.MaxPooling2D, pool_size=(2, 2), strides=(2, 2), padding='same')

    # Block 1
    x = conv(64, name='conv1')(inputs)
    x = maxpool(name='pool1')(x)

    # Block 2
    x = conv(128, name='conv2')(x)
    x = maxpool(name='pool2')(x)

    # Block 3
    x = conv(256, name='conv3/conv3_1')(x)
    x = conv(256, name='conv3/conv3_2')(x)
    x = maxpool(name='pool3')(x)

    # Block 4
    x = conv(512, name='conv4/conv4_1')(x)
    x = conv(512, name='conv4/conv4_2')(x)
    x = maxpool(name='pool4')(x)

    if include_top:
        dense = partial(tfkl.Dense, activation='relu')

        # FC block
        x = tfkl.Flatten(name='flatten_')(x)
        x = dense(4096, name='vggish_fc1/fc1_1')(x)
        x = dense(4096, name='vggish_fc1/fc1_2')(x)
        x = dense(params.EMBEDDING_SIZE, name='vggish_fc2')(x)

    else:
        globalpool = (
            tfkl.GlobalAveragePooling2D() if pooling == 'avg' else
            tfkl.GlobalMaxPooling2D() if pooling == 'max' else None)

        if globalpool:
            x = globalpool(x)

    # Create model
    model = Model(inputs, x, name=name)

    # lookup weights location
    if weights in params.WEIGHTS_PATHS:
        w = params.WEIGHTS_PATHS[weights]
        weight_type = 'top' if include_top else 'no_top'
        if weight_type in w:
            weights = w[weight_type]

    # load weights
    if weights and os.path.isfile(weights):
        model.load_weights(weights)

    return model
