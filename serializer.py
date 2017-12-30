import preprocess
import helpers
import augment
import data
import numpy as np


class Serializer:
    _datasets = None

    @staticmethod
    def serialize_datasets():
        features, labels = preprocess.load_from_npy(split_data=False, shuffle_data=True)
        features, labels = augment.augment_data(features, labels)
        datasets = preprocess.split(features, labels)

        np.save(data.TRAIN_FEATURES_FILE, datasets[0][0])
        np.save(data.TRAIN_LABELS_FILE, datasets[0][1])
        np.save(data.TEST_FEATURES_FILE, datasets[1][0])
        np.save(data.TEST_LABELS_FILE, datasets[1][1])
        np.save(data.CV_FEATURES_FILE, datasets[2][0])
        np.save(data.CV_LABELS_FILE, datasets[2][1])

        Serializer._datasets = None

    @staticmethod
    def load_npy_datasets():
        if Serializer._datasets is None:
            Serializer._datasets = [[np.load(data.TRAIN_FEATURES_FILE).astype(np.float32),
                                     np.load(data.TRAIN_LABELS_FILE)],
                                    [np.load(data.TEST_FEATURES_FILE).astype(np.float32),
                                     np.load(data.TEST_LABELS_FILE)],
                                    [np.load(data.CV_FEATURES_FILE).astype(np.float32),
                                     np.load(data.CV_LABELS_FILE)]]
        return Serializer._datasets
