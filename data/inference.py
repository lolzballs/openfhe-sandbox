#!/usr/bin/env python

import argparse

import numpy as np

import mnist


def transform_to_onehot(target: np.ndarray) -> np.ndarray:
    return target == 3


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', default='.data/mnist_trimmed')
    parser.add_argument('weights', help='logistic regression weights')
    parser.add_argument('image', help='index of test image to predict',
                        type=int, nargs='?', default=None)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    weights = np.fromfile(args.weights, dtype=np.double)
    bias = weights[-1]
    weights = weights[:-1]

    (train_x, train_y), (test_x, test_y) = mnist.load_data(args.location)
    train_x, test_x = train_x.reshape(-1, 28 * 28), test_x.reshape(-1, 28 * 28)
    train_y, test_y = transform_to_onehot(train_y), transform_to_onehot(test_y)

    if args.image is not None:
        dot = test_x[args.image, :] @ weights
        prediction = sigmoid(dot + bias)

        print(f'dot: {dot}, prediction: {prediction}, '
              f'ground truth: {test_y[0]}')
    else:
        dot = test_x @ weights
        prediction = sigmoid(dot + bias) >= 0.5

        print(f'accuracy: {(prediction == test_y).sum() / test_x.shape[0]}')
