import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

# Get inputs and return outputs
# Don't forget to squeeze output

def conv4_b2_0(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def conv16_b4_0(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def conv16_b4_1(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def res_4_2_0_noBN(inputs):
    # To make the same filter size
    xi = layers.Conv2D(32, 1, padding='same', activation='linear')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.ReLU()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def res_4_2_0_BN(inputs):
    # To make the same filter size
    xi = layers.Conv2D(32, 1, padding='same', activation='linear')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    xi = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(xi)
    x = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    x = x + xi
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs
