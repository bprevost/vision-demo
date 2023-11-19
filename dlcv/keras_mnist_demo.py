#!/usr/bin/env python3

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# grab the MNIST dataset of 70,000 hand-written digits
# 60,000 for training and 10,000 for testing
((trainX, trainY), (testX, testY)) = tf.keras.datasets.mnist.load_data()

# flatten each 28x28x1 image to a simple list of 28x28=784 pixels
trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))

# scale data to the range of [0, 1]
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# build the model
num_features = trainX.shape[1]
print(f'{num_features=}')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(num_features,))) # input layer
model.add(tf.keras.layers.Dense(256, activation='sigmoid')) # hidden layer
model.add(tf.keras.layers.Dense(128, activation='sigmoid')) # hidden layer
model.add(tf.keras.layers.Dense(10, activation='softmax')) # output layer
model.summary()

# initialize the gradient descent optimizer
sgd = tf.keras.optimizers.SGD(0.01)

# compile the model
model.compile(loss="categorical_crossentropy", optimizer=sgd,  metrics=["accuracy"])

# train the model
EPOCHS = 100
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=128)

# evaluate the network
predictions = model.predict(testX, batch_size=128)
report = classification_report(
    y_true=testY.argmax(axis=1),
    y_pred=predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_])
print(report)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('keras_mnist_demo.png')
