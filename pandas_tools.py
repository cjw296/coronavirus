from itertools import product

import numpy as np


def multi_index_add(df, name, data):
    df[[(name, level) for level in df.columns.levels[1]]] = data


def tuple_product_array(x, y):
    result = np.empty(len(x), dtype=object)
    result[:] = list(product(x, (y,)))
    result.shape = (len(x), 1)
    return result
