import tensorflow as tf
import numpy as np

import data

from enum import Enum


def cross_entropy_loss(label, logits):
    loss = tf.losses.softmax_cross_entropy(label, logits)
    return loss


def probabilistic_label_drawing(label):
    distribution = label / data.N_CLASSES
    rand_id = np.random.choice(data.N_CLASSES, p=distribution)
    label = np.zeros(data.N_CLASSES)
    label[rand_id] = 1


def majority_voting(label):
    return np.argmax(label)


class LabelType(Enum):
    majority_vote = None
    cross_entropy = None
