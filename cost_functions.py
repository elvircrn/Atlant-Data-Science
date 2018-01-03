from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

import data


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


def weighted_cross_entropy_with_logits(targets, logits, pos_weight, name=None):
    """Computes a weighted cross entropy.
    This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
    allows one to trade off recall and precision by up- or down-weighting the
    cost of a positive error relative to a negative error.
    The usual cross-entropy cost is defined as:
        targets * -log(sigmoid(logits)) +
            (1 - targets) * -log(1 - sigmoid(logits))
    The argument `pos_weight` is used as a multiplier for the positive targets:
        targets * -log(sigmoid(logits)) * pos_weight +
            (1 - targets) * -log(1 - sigmoid(logits))
    For brevity, let `x = logits`, `z = targets`, `q = pos_weight`.
    The loss is:
          qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
        = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
    Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
    the implementation uses
        (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    `logits` and `targets` must have the same type and shape.
    Args:
      targets: A `Tensor` of the same type and shape as `logits`.
      logits: A `Tensor` of type `float32` or `float64`.
      pos_weight: A coefficient to use on the positive examples.
      name: A name for the operation (optional).
    Returns:
      A `Tensor` of the same shape as `logits` with the componentwise
      weighted logistic losses.
    Raises:
      ValueError: If `logits` and `targets` do not have the same shape.
    """
    with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        try:
            targets.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError(
                "logits and targets must have the same shape (%s vs %s)" %
                (logits.get_shape(), targets.get_shape()))

        # The logistic loss formula from above is
        #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
        # To avoid branching, we use the combined version
        #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
        log_weight = 1 + (pos_weight - 1) * targets
        return math_ops.add(
            (1 - targets) * logits,
            log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                          nn_ops.relu(-logits)),
            name=name)
