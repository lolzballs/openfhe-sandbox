#!/usr/bin/env python

import argparse
import os

import mnist


def filter_set(s: mnist.Set) -> mnist.Set:
    images, labels = s

    indices = [*(labels == 3).nonzero()[0].tolist(),
               *(labels == 8).nonzero()[0].tolist()]
    return images[indices], labels[indices]


def write_idx(idx: bytearray, name: str):
    with open(name, 'wb') as f:
        f.write(idx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', default='.data/mnist')
    parser.add_argument('--output', default='.data/mnist_trimmed')
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    location = args.location
    if not mnist.check_files_present(location)[0]:
        mnist.download_data(location)
    train_set, test_set = mnist.load_data(location)

    output = args.output
    os.mkdir(output)

    train_set_filtered = filter_set(train_set)
    write_idx(mnist.to_idx(train_set_filtered[0]),
              os.path.join(output, mnist.TRAIN_FILES[0]))
    write_idx(mnist.to_idx(train_set_filtered[1]),
              os.path.join(output, mnist.TRAIN_FILES[1]))
    print(f'trimmed from {len(train_set[1])} to {len(train_set_filtered[1])}')

    test_set_filtered = filter_set(test_set)
    write_idx(mnist.to_idx(test_set_filtered[0]),
              os.path.join(output, mnist.TEST_FILES[0]))
    write_idx(mnist.to_idx(test_set_filtered[1]),
              os.path.join(output, mnist.TEST_FILES[1]))
    print(f'trimmed from {len(test_set[1])} to {len(test_set_filtered[1])}')
