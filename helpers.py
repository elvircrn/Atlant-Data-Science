import numpy as np


def invert_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


def flatten(image_matrix):
    return [item for sublist in image_matrix for item in sublist]


def perc_split(elements, percentages):
    n = len(elements)
    perc_cum = np.cumsum(percentages)
    indices = [int(n * perc) for perc in perc_cum]
    ranges = []
    for i in range(len(perc_cum)):
        if not i:
            ranges.append(list(range(0, indices[0])))
        else:
            ranges.append(list(range(indices[i - 1], indices[i])))
    return np.take(elements, ranges)
