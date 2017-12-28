import pandas as pd
import numpy as np

import cohn_kanade as ck
import data

from helpers import perc_split, unison_shuffled_copies
from sklearn.preprocessing import normalize


def to_vector(pixels):
    return np.fromstring(pixels, dtype=int, sep=' ')


def extract_labels(ferplus):
    return pd.DataFrame(
        [ferplus['neutral'].astype(int), ferplus['happiness'].astype(int), ferplus['surprise'].astype(int),
         ferplus['sadness'].astype(int), ferplus['anger'].astype(int),
         ferplus['disgust'].astype(int), ferplus['fear'].astype(int),
         ferplus['contempt'].astype(int)]).as_matrix().T.astype(np.float32)


def majority_voting(labels, n_classes):
    major_labels = np.zeros((len(labels), n_classes), dtype=np.float32)
    major_labels.fill(1e-8)
    max_ids = np.argmax(labels, axis=1)
    major_labels[np.arange(max_ids.size), max_ids] = 1
    return major_labels


def split(faces, labels):
    set_distribution = [0.94, 0.03, 0.03]
    total_size = len(faces)
    label_ids = np.argmax(labels, axis=1)
    labels_ids_cnt = np.array([np.sum(label_ids == label) for label in range(8)])
    distribution = labels_ids_cnt / total_size
    buckets = [faces[np.array(label_ids == label_id)] for label_id in range(8)]
    datasets = list(zip(perc_split(faces, set_distribution), perc_split(labels, set_distribution)))
    return datasets


def normalize_pixels(faces):
    faces = normalize(faces, axis=0, norm='max')
    return faces


def scale_labels(labels):
    return labels / data.N_VOTES


def delete_non_face(data):
    data = data[data['NF'] != 10]
    del data['NF']
    return data


def delete_unknown(data):
    del data['unknown']
    return data


def rebalance_labels(labels):
    return labels + np.repeat(((10 - labels.sum(axis=1)) / 8), 8).reshape(-1, 8)


def add_eps(labels):
    EPS = 1e-8
    return labels + EPS


def get_data(split_data=False, include_ck=False):
    majority_vote = False

    fer2013 = pd.read_csv('Data/FERPlus/fer2013.csv')
    fer2013new = pd.read_csv('Data/FERPlus/fer2013new.csv')

    del fer2013['emotion']
    del fer2013['Usage']

    ferplus = pd.concat([fer2013, fer2013new], axis=1)
    # noinspection PyUnresolvedReferences
    ferplus = ferplus.dropna()
    ferplus = delete_non_face(ferplus)
    ferplus = delete_unknown(ferplus)

    faces = ferplus['pixels'].apply(to_vector).values
    faces = np.concatenate(faces).reshape((faces.size, faces[0].size)).astype(np.float32)

    labels = extract_labels(ferplus)
    labels = rebalance_labels(labels)

    if include_ck:
        ck_faces, ck_labels = ck.load_meta()
        faces = np.concatenate((faces, ck_faces), axis=0)
        labels = np.concatenate((labels, ck_labels), axis=0)

    faces = normalize_pixels(faces)

    labels = scale_labels(labels)
    labels = add_eps(labels)

    if majority_vote:
        labels = majority_voting(labels, n_classes=8)

    if split_data:
        return split(faces, labels)
    else:
        return faces, labels


def preprocess_and_save():
    features, labels = get_data(split_data=False, include_ck=True)
    save(features, labels)


def save(features, labels):
    np.save(data.FEATURES_FILE, features)
    np.save(data.LABELS_FILE, labels)


def load_from_npy(split_data=False, features_loc=data.FEATURES_FILE, labels_loc=data.LABELS_FILE, shuffle_data=True):
    features, labels = np.load(features_loc), np.load(labels_loc)

    if shuffle_data:
        features, labels = unison_shuffled_copies(np.load(features_loc), np.load(labels_loc))

    if split_data:
        return split(features, labels)
    else:
        return features, labels


# TODO: Refactor as soon as model training is fully implemented
def get_training():
    return get_data(split_data=True)[0]


def get_test():
    return get_data(split_data=True)[1]


def get_validation():
    return get_data(split_data=True)[2]
