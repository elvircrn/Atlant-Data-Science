import pandas as pd
import numpy as np

from helpers import perc_split


def to_vector(pixels):
    return np.fromstring(pixels, dtype=int, sep=' ')


def labels_to_vector(entry):
    return np.array(
        [entry['neutral'].astype(int), entry['happiness'].astype(int), entry['surprise'].astype(int),
         entry['sadness'].astype(int), entry['anger'].astype(int),
         entry['disgust'].astype(int), entry['fear'].astype(int), entry['contempt'].astype(int),
         entry['unknown'].astype(int)])


def majority_voting(labels, n_classes):
    np.zeros((labels.size, n_classes))
    max_ids = np.argmax(labels, axis=1)
    labels[np.arange(max_ids.size), max_ids] = 1
    return labels


def split(input, labels):
    set_distribution = [0.4, 0.3, 0.3]
    datasets = list(zip(perc_split(input, set_distribution), perc_split(labels, set_distribution)))
    return datasets


def get_data():
    fer2013 = pd.read_csv('Data/FERPlus/fer2013.csv')
    fer2013new = pd.read_csv('Data/FERPlus/fer2013new.csv')

    del fer2013['emotion']
    del fer2013['Usage']

    ferplus = pd.concat([fer2013, fer2013new], axis=1)

    ferplus = ferplus.dropna()

    input = ferplus['pixels'].apply(to_vector).values
    input = np.concatenate(input).reshape((input.size, input[0].size)).astype(np.float32)

    labels = pd.DataFrame(
        [ferplus['neutral'].astype(int), ferplus['happiness'].astype(int), ferplus['surprise'].astype(int),
         ferplus['sadness'].astype(int), ferplus['anger'].astype(int),
         ferplus['disgust'].astype(int), ferplus['fear'].astype(int), ferplus['contempt'].astype(int),
         ferplus['unknown'].astype(int)]).as_matrix().T.astype(np.float32)
    
    labels = majority_voting(labels, 9)

    return split(input, labels)


def get_quick_train():
    pass
