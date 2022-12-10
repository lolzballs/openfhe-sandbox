import gzip
import itertools
import os
from typing import Optional, Tuple
import urllib.request

import numpy as np


MNIST_URL = 'http://yann.lecun.com/exdb/mnist'
TRAIN_FILES = ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
TEST_FILES = ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

Set = Tuple[np.ndarray, np.ndarray]


def check_files_present(location: str) -> Tuple[bool, Optional[str]]:
    for file in itertools.chain(TRAIN_FILES, TEST_FILES):
        if not os.path.isfile(os.path.join(location, file)):
            return False, file
    return True, None


def _load_idx(idx: np.ndarray) -> np.ndarray:
    if not (idx[0] == 0x0 and idx[1] == 0x0):
        raise RuntimeError('idx file does not start with two 0 bytes')

    if idx[2] != 0x08:
        raise RuntimeError('idx file does not indicate uint8')

    dim = idx[3]

    # read lengths, which are formatted in 4 bytes
    length_size = dim * 4
    if idx[4:].size < length_size:
        raise RuntimeError(f'idx file does not include sizes for all {dim} dimensions')

    data_start = 4 + length_size
    lengths = idx[4:data_start].view(dtype='>u4')

    data = idx[data_start:]
    if data.size != np.prod(lengths):
        raise RuntimeError(f'idx file does not have {np.prod(lengths)} bytes of data')

    return data.reshape(lengths)


def _load_set(images_file: str, labels_file: str) -> Set:
    images = _load_idx(np.fromfile(images_file, dtype=np.uint8))
    labels = _load_idx(np.fromfile(labels_file, dtype=np.uint8))
    return images, labels


def to_idx(array: np.ndarray) -> bytearray:
    assert array.dtype == np.uint8

    idx = bytearray()

    # magic number:
    #  [0:1] 0x0000
    #  [2]   0x08 (uint8)
    #  [3]   number of dimensions
    idx.extend([0x00, 0x00, 0x08, len(array.shape)])

    # add dimension sizes
    for dim in array.shape:
        idx.extend(dim.to_bytes(4, 'big'))

    # insert actual data
    idx.extend(array.tobytes())

    return idx


def load_data(location: str) -> Tuple[Set, Set]:
    present, file = check_files_present(location)
    if not present:
        raise RuntimeError(f'{file} not in {location}')

    add_location = lambda f: os.path.join(location, f)
    return _load_set(*map(add_location, TRAIN_FILES)), \
        _load_set(*map(add_location, TEST_FILES))


def download_data(location: str):
    download_urls = map(lambda f: (f, f'{MNIST_URL}/{f}.gz'),
                        itertools.chain(TRAIN_FILES, TEST_FILES))
    for name, url in download_urls:
        with urllib.request.urlopen(url) as f:
            with open(os.path.join(location, name), 'wb') as out:
                out.write(gzip.decompress(f.read()))
