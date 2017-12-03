import numpy as np


def flatten(image_matrix):
    return [item for sublist in image_matrix for item in sublist]


def perc_split(elements, percentages):
    n = len(elements)
    perc_cum = np.cumsum(percentages)
    indices = [int(n * perc) for perc in perc_cum]
    ranges = []
    ret = []
    buff = []
    prev = 0
    for indice in indices:
        ret.append(elements[range(prev, indice)])
        prev = indice
    return ret
