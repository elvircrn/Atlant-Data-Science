import pandas as pd
import facedetect as fd
from facedetect import FaceExtractor
from data import LABELS, N_CLASSES
import numpy as np


def label_to_one_vs_all(label):
    if label == 'happy':
        label = 'happiness'
    idx = LABELS.index(label)
    ret = np.zeros(N_CLASSES)
    ret[idx] = 10
    return ret.astype(np.float32)


def reduce_img_path(img_path):
    return 'Data/extracted_cohn/' + img_path[28:]


def img_path_to_feature_vector(img_path):
    reduced_path = reduce_img_path(img_path)
    image = fd.load_img(reduced_path)
    return FaceExtractor.extract_face(image)


def to_matrix(arrays):
    return np.concatenate(arrays).reshape((arrays.size, arrays[0].size)).astype(np.float32)


def load_meta():
    data = pd.read_csv('Data/cohn_kanade.csv')
    features = data['filename'].apply(img_path_to_feature_vector).values
    features = to_matrix(features)
    labels = data['label'].apply(label_to_one_vs_all).values
    labels = to_matrix(labels)
    return features, labels
