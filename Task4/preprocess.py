import pandas as pd
import numpy as np

from helpers import perc_split
from sklearn.preprocessing import normalize


def to_vector(pixels):
    return np.fromstring(pixels, dtype=int, sep=' ')


def labels_to_vector(entry):
    return np.array(
        [entry['neutral'].astype(int), entry['happiness'].astype(int), entry['surprise'].astype(int),
         entry['sadness'].astype(int), entry['anger'].astype(int),
         entry['disgust'].astype(int), entry['fear'].astype(int), entry['contempt'].astype(int),
         entry['unknown'].astype(int)])


def majority_voting(labels, n_classes):
    major_labels = np.zeros((len(labels), n_classes), dtype=np.float32)
    major_labels.fill(1e-8)
    max_ids = np.argmax(labels, axis=1)
    major_labels[np.arange(max_ids.size), max_ids] = 1
    return major_labels


def split(faces, labels):
    set_distribution = [0.7, 0.15, 0.15]
    datasets = list(zip(perc_split(faces, set_distribution), perc_split(labels, set_distribution)))
    return datasets


def normalize_pixels(faces):
    faces = normalize(faces, axis=0, norm='max')
    return faces


def delete_unknown(data):
    data = data[data['NF'] != 10]
    del data['NF']
    del data['unknown']
    return data


def get_data(split_data=False):
    fer2013 = pd.read_csv('Data/FERPlus/fer2013.csv')
    fer2013new = pd.read_csv('Data/FERPlus/fer2013new.csv')

    del fer2013['emotion']
    del fer2013['Usage']

    ferplus = pd.concat([fer2013, fer2013new], axis=1)
    ferplus = ferplus.dropna()
    ferplus = delete_unknown(ferplus)

    faces = ferplus['pixels'].apply(to_vector).values
    faces = np.concatenate(faces).reshape((faces.size, faces[0].size)).astype(np.float32)
    faces = normalize_pixels(faces)

    labels = pd.DataFrame(
        [ferplus['neutral'].astype(int), ferplus['happiness'].astype(int), ferplus['surprise'].astype(int),
         ferplus['sadness'].astype(int), ferplus['anger'].astype(int),
         ferplus['disgust'].astype(int), ferplus['fear'].astype(int),
         ferplus['contempt'].astype(int)]).as_matrix().T.astype(np.float32)

    labels = majority_voting(labels, n_classes=8) / 10

    if split_data:
        return split(faces, labels)
    else:
        return faces, labels


# TODO: Refactor as soon as model training is fully implemented
def get_training():
    return get_data(split_data=True)[0]


def get_test():
    return get_data(split_data=True)[1]


def get_validation():
    return get_data(split_data=True)[2]


