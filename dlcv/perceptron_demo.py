#!/usr/bin/env python3

import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N) # weight matrix
        self.alpha = alpha # learning rate

    def step(self, x):
        # step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # bias trick
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop over the epochs
        for epoch in np.arange(0, epochs):
            # loop over the data points
            for (x, target) in zip(X, y):
                # obtain the prediction
                pred = self.step(np.dot(x, self.W))
                # update the weight matrix if the prediction does not match the target
                if pred != target:
                    error = pred - target
                    self.W += -self.alpha * error * x

    def predict(self, X):
        # ensure input is a matrix
        X = np.atleast_2d(X)
        # bias trick
        X = np.c_[X, np.ones((X.shape[0]))]
        # obtain the prediction
        pred = self.step(np.dot(X, self.W))
        return pred

def perceptron_demo(y, epochs=20):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # create the perceptron
    p = Perceptron(X.shape[1], alpha=0.1)
    # train the perceptron
    p.fit(X, y, epochs=epochs)
    # test the perceptron
    for (x, target) in zip(X, y):
        # obtain the prediction
        pred = p.predict(x)
        print(f'data={x}, ground-truth={target[0]}, pred={pred}')

if __name__ == '__main__':
    print('Perceptron Demo - AND')
    y = np.array([[0], [0], [0], [1]])
    perceptron_demo(y)

    print('Perceptron Demo - OR')
    y = np.array([[0], [1], [1], [1]])
    perceptron_demo(y)

    print('Perceptron Demo - XOR')
    y = np.array([[0], [1], [1], [0]])
    perceptron_demo(y)
