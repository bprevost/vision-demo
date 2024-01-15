# Inspired by Deep Learning for Computer Vision with Python [Rosebrock]
# LeNet

import tensorflow as tf

def lenet(height=28, width=28, depth=1, classes=10):
    # build the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(height, width, depth))) # input layer

    # first set of CONV => RELU => POOL layers
    model.add(tf.keras.layers.Conv2D(20, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(tf.keras.layers.Conv2D(50, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500))
    model.add(tf.keras.layers.Activation('relu'))

    # softmax classifier
    model.add(tf.keras.layers.Dense(classes))
    model.add(tf.keras.layers.Activation('softmax'))

    return model

if __name__ == "__main__":
    model = lenet()
    model.summary()
