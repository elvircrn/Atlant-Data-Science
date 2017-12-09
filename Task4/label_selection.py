import numpy as np

import data

def probabilistic_label_drawing(label):
    return np.random.choice(data.N_CLASSES, p=[vote_cnt data.N_CLASSES for vote_cnt in labels])


def majority_voting(label):
    return np.argmax(label)

