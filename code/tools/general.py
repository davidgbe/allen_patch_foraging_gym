import numpy as np
from pathlib import Path
from copy import deepcopy as copy
import matplotlib
import warnings
import pickle
import blosc
import os


def zero_pad(s, n):
    s_str = str(s)
    pad = n - len(s_str)
    zero_padding = '0' * pad
    return zero_padding + s_str


def make_path_if_not_exists(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def ordered_colors_from_cmap(cmap_name, n, cmap_range=(0, 1)):
    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(np.linspace(cmap_range[0], cmap_range[1], n))
    return colors


def compressed_write(data, dest):
    pickled_data = pickle.dumps(data)
    compressed_pickle = blosc.compress(pickled_data)
    with open(dest, 'wb') as f:
        f.write(compressed_pickle)


def compressed_read(source):
    with open(source, 'rb') as f:
        compressed_pickle = f.read()
    inflated_pickle = blosc.decompress(compressed_pickle)
    return pickle.loads(inflated_pickle)


def logical_and(*args):
    v = None
    for i in range(len(args)):
        if v is None:
            v = args[i]
        else:
            v = np.logical_and(v, args[i])
    return v