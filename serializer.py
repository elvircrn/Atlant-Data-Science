import preprocess
import helpers
import augment
import data
import numpy as np

import cost_functions as cf


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
    def shuffle_and_serialize(datasets):
        datasets[0] = helpers.unison_shuffled_copies(datasets[0][0], datasets[0][1])
        np.save(data.TRAIN_FEATURES_FILE, datasets[0][0])
        np.save(data.TRAIN_LABELS_FILE, datasets[0][1])
        np.save(data.TEST_FEATURES_FILE, datasets[1][0])
        np.save(data.TEST_LABELS_FILE, datasets[1][1])
        np.save(data.CV_FEATURES_FILE, datasets[2][0])
        np.save(data.CV_LABELS_FILE, datasets[2][1])
        Serializer._datasets = datasets

    @staticmethod
    def load_npy_datasets(label_type, shuffle_dataset=True, downsample=True):
        if Serializer._datasets is None:
            Serializer._datasets = [[np.load(data.TRAIN_FEATURES_FILE).astype(np.float32),
                                     np.load(data.TRAIN_LABELS_FILE)],
                                    [np.load(data.TEST_FEATURES_FILE).astype(np.float32),
                                     np.load(data.TEST_LABELS_FILE)],
                                    [np.load(data.CV_FEATURES_FILE).astype(np.float32),
                                     np.load(data.CV_LABELS_FILE)]]
        if shuffle_dataset:
            for i in range(3):
                Serializer._datasets[i] = helpers.unison_shuffled_copies(Serializer._datasets[i][0],
                                                                         Serializer._datasets[i][1])

        # TODO: Move this to the preprocessing pipeline
        if label_type == cf.LabelType.majority_vote:
            Serializer._datasets = Serializer.apply_majority_voting(Serializer._datasets)
        elif label_type == cf.LabelType.cross_entropy:
            Serializer._datasets = Serializer.rescale_labels(Serializer._datasets)
        else:
            raise Exception("Invalid label type passed")

        datasets = Serializer._datasets

        if downsample:
            datasets[0] = preprocess.downsample(*datasets[0])

        return datasets

    # TODO: Move this to the preprocessing pipeline
    @staticmethod
    def rescale_labels(datasets):
        for i in range(3):
            datasets[i] = datasets[i][0], np.divide(datasets[i][1],
                                                    np.stack([np.sum(datasets[i][1], axis=1)] * data.N_CLASSES, axis=1))
        return datasets

    @staticmethod
    def majority_voting(labels):
        major_labels = np.zeros(labels.shape, dtype=np.float32)
        major_labels.fill(1e-8)
        max_ids = np.argmax(labels, axis=1)
        major_labels[np.arange(max_ids.size), max_ids] = 1
        return major_labels

    @staticmethod
    def apply_majority_voting(datasets):
        for i in range(3):
            datasets[i] = datasets[i][0], Serializer.majority_voting(datasets[i][1])
        return datasets
