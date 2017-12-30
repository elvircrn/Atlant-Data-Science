import numpy as np


def invert_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


def flatten(image_matrix):
    return [item for sublist in image_matrix for item in sublist]


# TODO: Fix
def perc_split(elements, percentages):
    n = len(elements)
    perc_cum = np.cumsum(percentages)
    indices = [int(n * perc) for perc in perc_cum]
    split_elements = []
    for i in range(len(perc_cum)):
        if not i:
            ranges = np.array(range(0, indices[0]))
        else:
            ranges = np.array(range(indices[i - 1], indices[i]))

        split_elements.append(elements[ranges])
    return split_elements


def merge(groups):
    return np.concatenate(tuple(group for group in groups))


def shuffle(a):
    p = np.random.permutation(len(a))
    return a[p]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

