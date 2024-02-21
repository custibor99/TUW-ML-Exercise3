from utils import *

from tensorflow import keras
from keras import layers


@custom_logger
def eccv16() -> keras.Model:
    inputs = layers.Input(shape= [256,256,1])
    model1 = keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.BatchNormalization(),

        layers.Conv2D(128, kernel_size=3, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.BatchNormalization(), 

        layers.Conv2D(256, kernel_size=3, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(256, kernel_size=3, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(256, kernel_size=3, strides=2, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.BatchNormalization(),

        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.BatchNormalization(),

        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.BatchNormalization(),

        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.BatchNormalization(),

        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.BatchNormalization(),

        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(256, kernel_size=3, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2D(256, kernel_size=3, strides=1, padding="same", data_format="channels_last"),
        layers.ReLU(True),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", data_format="channels_last"),
        layers.Softmax(),
        layers.Conv2D(2, kernel_size=1, padding="valid", dilation_rate=1, strides=1, use_bias=False, data_format="channels_last"),
        layers.UpSampling2D((2,2)),
        layers.Activation("sigmoid")
    ])

    output = model1(inputs)
    output = (output * 2) - 1 
    return keras.Model(inputs=inputs, outputs=output)

def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample


def up(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add(keras.layers.LeakyReLU())
    return upsample

def unet():
    inputs = layers.Input(shape= [256,256,1])
    d1 = down(128,(3,3),False)(inputs)
    d2 = down(128,(3,3),False)(d1)
    d3 = down(256,(3,3),True)(d2)
    d4 = down(512,(3,3),True)(d3)
    d5 = down(512,(3,3),True)(d4)
    u1 = up(512,(3,3),False)(d5)
    u1 = layers.concatenate([u1,d4])
    u2 = up(256,(3,3),False)(u1)
    u2 = layers.concatenate([u2,d3])
    u3 = up(128,(3,3),False)(u2)
    u3 = layers.concatenate([u3,d2])
    u4 = up(128,(3,3),False)(u3)
    u4 = layers.concatenate([u4,d1])
    u5 = up(3,(3,3),False)(u4)
    u5 = layers.concatenate([u5,inputs])
    output = layers.Conv2D(2,(2,2),strides = 1, padding = 'same')(u5)
    output = layers.Activation("sigmoid")(output)
    output = (output * 2) - 1
    return tf.keras.Model(inputs=inputs, outputs=output)
