#!/usr/bin/env python

import argparse

import numpy as np
import sklearn.metrics
import sklearn.linear_model

import mnist


def transform_to_onehot(target: np.ndarray) -> np.ndarray:
    return target == 3


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', default='.data/mnist_trimmed')
    parser.add_argument('-i', '--max-iter', default=100, type=int)
    parser.add_argument('-v', '--verbose', default=False, type=bool)
    parser.add_argument('output', default=None, nargs='?',
                        help='file to serialize trained coefficients to')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    (train_x, train_y), (test_x, test_y) = mnist.load_data(args.location)
    train_x, test_x = train_x.reshape(-1, 28 * 28), test_x.reshape(-1, 28 * 28)
    train_y, test_y = transform_to_onehot(train_y), transform_to_onehot(test_y)

    lr = sklearn.linear_model.LogisticRegression(penalty=None,
                                                 max_iter=args.max_iter,
                                                 verbose=args.verbose)
    lr.fit(train_x, train_y)
    prediction = lr.predict(test_x)

    accuracy = sklearn.metrics.accuracy_score(test_y, prediction)
    print(f'accuracy was {accuracy}')

    if args.output:
        with open(args.output, 'wb') as f:
            f.write(lr.coef_.tobytes())
