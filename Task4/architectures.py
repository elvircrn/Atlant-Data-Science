import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner

import preprocess
import data


def mini_vgg(inputs, is_training, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            # -4
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding='VALID', scope='conv1')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, is_training=is_training, scope='dropout1')

            # -6
            net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], padding='VALID', scope='conv2')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.dropout(net, is_training=is_training, scope='dropout2')

            # -6
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='VALID', scope='conv3')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            # / 2
            net = slim.dropout(net, is_training=is_training, scope='dropout3')

            net = slim.flatten(net)
            net = slim.fully_connected(net, data.N_CLASSES, activation_fn=None, scope='fc1')

            net = slim.softmax(net, scope='sm1')
        return net


def small_vgg(inputs, is_training, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, is_training=is_training, scope='dropout1')

            net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], padding='VALID', scope='conv2')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.dropout(net, is_training=is_training, scope='dropout2')

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='VALID', scope='conv3')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')

            # TODO: Add L2 on last layer
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='VALID', scope='conv4')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            net = slim.dropout(net, is_training=is_training, scope='dropout4')

            net = slim.flatten(net)
            net = slim.fully_connected(net, data.N_CLASSES, activation_fn=None, scope='fc1')

            net = slim.softmax(net, scope='sm1')
        return net


_LAYERS = []
_INPUTS = []


def get_layers():
    return _LAYERS


def get_inputs():
    return _INPUTS[0]


def padded_mini_vgg(inputs, is_training, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            _INPUTS.append(inputs)
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
            _LAYERS.append(net)
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            _LAYERS.append(net)
            net = slim.dropout(net, is_training=is_training, scope='dropout1')
            _LAYERS.append(net)

            net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv2')
            _LAYERS.append(net)
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            _LAYERS.append(net)
            net = slim.dropout(net, is_training=is_training, scope='dropout2')
            _LAYERS.append(net)

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv3')
            _LAYERS.append(net)
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            _LAYERS.append(net)
            net = slim.dropout(net, is_training=is_training, scope='dropout3')
            _LAYERS.append(net)

            # TODO: Add L2 on last layer
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv4')
            _LAYERS.append(net)
            net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            _LAYERS.append(net)
            net = slim.dropout(net, is_training=is_training, scope='dropout4')
            _LAYERS.append(net)

            net = slim.flatten(net)
            net = slim.fully_connected(net, data.N_CLASSES, activation_fn=None, scope='fc1')

            net = slim.softmax(net, scope='sm1')
        return net
