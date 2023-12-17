# Inspired by Deep Learning for Computer Vision with Python [Rosebrock]
# MiniVGGNet

import tensorflow as tf

def mini_vgg_net(height=32, width=32, depth=3, classes=10):
    # build the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(height, width, depth))) # input layer

    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=-1)) # channels last
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=-1)) # channels last
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=-1)) # channels last
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=-1)) # channels last
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # softmax classifier
    model.add(tf.keras.layers.Dense(classes))
    model.add(tf.keras.layers.Activation('softmax'))

    return model

if __name__ == "__main__":
    model = mini_vgg_net()
    model.summary()
